"""Service REST FastAPI MLOps pour la Supply Chain connecte a MLflow Model Registry."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from supply_chain.agent.decision_store import (
    DecisionConflictError,
    DecisionNotFoundError,
    SQLDecisionStore,
)
from supply_chain.agent.llm_client import build_llm_client_from_settings
from supply_chain.agent.orchestrator import BoundedSupplyChainOrchestrator
from supply_chain.agent.schemas import (
    DecisionRecord,
    IncidentTrigger,
    InvestigateRequest,
    InvestigateResponse,
    OperatorDecisionRequest,
    OperatorDecisionResponse,
)
from supply_chain.agent.telemetry import record_execution_trace
from supply_chain.agent.tools import SupplyChainToolRegistry
from supply_chain.config import SupplyChainSettings, settings
from supply_chain.data_pipeline import (
    FEATURE_COLUMNS,
    build_causal_feature_Frame,
    load_raw_tables_from_sql,
)
from supply_chain.mlflow_loader import (
    ChampionModelBundle,
    ChampionModelLoadError,
    load_champion_bundle,
)
from supply_chain.models import SupplyChainAnomalyDetector, enforce_quantile_monotonicity
from supply_chain.schemas import (
    AnomalyPayload,
    AnomalyRequest,
    CriticalityPayload,
    ErrorResponse,
    ForecastPayload,
    HealthResponse,
    InventoryRecord,
    QuantileForecast,
)

logger = logging.getLogger(__name__)


class PlatformAppState:
    """Etat partage de l'application FastAPI (Modeles @champion S6 + Agent Gouverne S8)."""

    def __init__(
        self,
        cfg: SupplyChainSettings,
        model_service: Any,
        orchestrator: BoundedSupplyChainOrchestrator,
        decision_store: SQLDecisionStore,
    ) -> None:
        self.cfg = cfg
        self.model_service = model_service
        self.orchestrator = orchestrator
        self.decision_store = decision_store


class SupplyChainDataRepository:
    """Acces aux donnees SQLite et aux features causales pre-calculees pour l'inference rapide."""

    def __init__(
        self,
        cfg: SupplyChainSettings = settings,
        preloaded_features_df: pd.DataFrame | None = None,
        preloaded_skus_df: pd.DataFrame | None = None,
    ) -> None:
        self.cfg = cfg
        self._features_df: pd.DataFrame | None = preloaded_features_df
        self._skus_df: pd.DataFrame | None = preloaded_skus_df

    def ensure_loaded(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Charge en memoire les tables SQL et construit les features causales si necessaire."""
        if self._features_df is not None and self._skus_df is not None:
            return self._features_df, self._skus_df

        if not self.cfg.sqlite_db_path.is_file():
            raise FileNotFoundError(
                f"Base SQLite introuvable: '{self.cfg.sqlite_db_path}'. "
                "Executez `make db-init` ou `python -m supply_chain.data_pipeline --init-db`."
            )

        with sqlite3.connect(str(self.cfg.sqlite_db_path)) as conn:
            skus_df = pd.read_sql_query("SELECT * FROM skus", conn)

        daily_df, deliv_df = load_raw_tables_from_sql(self.cfg.sqlite_db_path)
        features_df = build_causal_feature_Frame(
            daily_df=daily_df,
            deliveries_df=deliv_df,
            horizon_days=self.cfg.default_horizon_days,
        )
        self._skus_df = skus_df
        self._features_df = features_df
        return features_df, skus_df

    def get_latest_sku_row(self, sku_id: str, obs_date: str | None = None) -> pd.Series | None:
        """Retourne la ligne de features causales la plus recente pour un SKU donne."""
        features_df, _ = self.ensure_loaded()
        sku_rows = features_df[features_df["sku_id"] == sku_id]
        if sku_rows.empty:
            return None

        if obs_date is not None:
            target_dt = pd.Timestamp(obs_date)
            filtered = sku_rows[pd.to_datetime(sku_rows["obs_date"]) <= target_dt]
            if filtered.empty:
                return None
            return filtered.iloc[-1]

        return sku_rows.iloc[-1]

    def get_inventory_record(
        self, sku_id: str, obs_date: str | None = None
    ) -> InventoryRecord | None:
        """Construit un objet InventoryRecord complet pour un SKU."""
        _, skus_df = self.ensure_loaded()
        sku_meta = skus_df[skus_df["sku_id"] == sku_id]
        row = self.get_latest_sku_row(sku_id, obs_date=obs_date)
        if sku_meta.empty or row is None:
            return None

        m = sku_meta.iloc[0]
        on_hand = max(0.0, float(row["on_hand_qty"]))
        in_transit = max(0.0, float(row["in_transit_qty"]))
        obs_str = pd.Timestamp(row["obs_date"]).strftime("%Y-%m-%d")

        return InventoryRecord(
            sku_id=sku_id,
            category=str(m["category"]),
            obs_date=obs_str,
            on_hand_qty=round(on_hand, 4),
            in_transit_qty=round(in_transit, 4),
            effective_stock_qty=round(on_hand + in_transit, 4),
            reorder_point_s=round(float(row["reorder_point_s"]), 4),
            order_up_to_S=round(float(row["order_up_to_S"]), 4),
            moq_units=int(m["moq_units"]),
            contractual_lead_time_days=int(m["contractual_lead_time_days"]),
            empirical_lead_time_mean=round(float(row["causal_lead_time_mean"]), 4),
            empirical_lead_time_std=round(float(row["causal_lead_time_std"]), 4),
            unit_cost_eur=round(float(m["unit_cost_eur"]), 4),
            holding_cost_per_day_eur=round(float(m["holding_cost_per_day_eur"]), 4),
            stockout_penalty_per_unit_eur=round(float(m["stockout_penalty_per_unit_eur"]), 4),
        )


def _error_json(
    status_code: int, code: str, msg: str, detail: str, sku_id: str | None = None
) -> JSONResponse:
    err = ErrorResponse(error_code=code, message=msg, detail=detail, sku_id=sku_id)
    return JSONResponse(status_code=status_code, content=err.model_dump())


def create_app(
    cfg: SupplyChainSettings = settings,
    custom_bundle: ChampionModelBundle | None = None,
    custom_repo: SupplyChainDataRepository | None = None,
    disable_auto_load: bool = False,
    custom_state: PlatformAppState | None = None,
) -> FastAPI:
    """Usine d'application FastAPI permettant le chargement @champion ou l'injection de mocks."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if custom_state is not None:
            app.state.platform = custom_state
            app.state.cfg = custom_state.cfg
            app.state.models_loaded = True
            yield
            return

        repo = custom_repo if custom_repo is not None else SupplyChainDataRepository(cfg=cfg)
        app.state.repo = repo
        app.state.cfg = cfg

        if custom_bundle is not None:
            app.state.champion_bundle = custom_bundle
            app.state.models_loaded = True
        elif disable_auto_load:
            app.state.champion_bundle = None
            app.state.models_loaded = False
        else:
            try:
                bundle = await run_in_threadpool(load_champion_bundle, cfg)
                await run_in_threadpool(repo.ensure_loaded)
                app.state.champion_bundle = bundle
                app.state.models_loaded = True
                logger.info("Modeles @champion charges avec succes: %s", bundle.versions)
            except (ChampionModelLoadError, Exception) as exc:
                logger.warning("Demarrage en mode degrade (modeles @champion absents): %s", exc)
                app.state.champion_bundle = None
                app.state.models_loaded = False

        decision_store = SQLDecisionStore(db_path=cfg.database_path)
        tool_registry = SupplyChainToolRegistry(cfg=cfg, use_in_memory_fallback=True)
        llm_client = build_llm_client_from_settings(cfg)
        orchestrator = BoundedSupplyChainOrchestrator(
            llm_client=llm_client,
            tool_registry=tool_registry,
            cfg=cfg,
        )
        app.state.platform = PlatformAppState(
            cfg=cfg,
            model_service=getattr(app.state, "champion_bundle", None),
            orchestrator=orchestrator,
            decision_store=decision_store,
        )
        yield

    app = FastAPI(
        title="SupplyChain-Decision-Platform API",
        version="1.2.0",
        description=(
            "API REST MLOps connectee a MLflow (@champion) et Agent d'aide a la decision "
            "avec garde-fous deterministes et workflow Human-in-the-Loop."
        ),
        lifespan=lifespan,
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["System"],
    )
    async def health_check(request: Request) -> HealthResponse | JSONResponse:
        state: PlatformAppState | None = getattr(request.app.state, "platform", None)
        bundle: ChampionModelBundle | None = getattr(request.app.state, "champion_bundle", None)
        models_ok = bool(getattr(request.app.state, "models_loaded", False) and bundle is not None)
        if state is not None:
            if state.model_service is not None:
                models_ok = getattr(state.model_service, "is_ready", True)
            elif custom_state is not None:
                models_ok = True

        repo: SupplyChainDataRepository | None = getattr(request.app.state, "repo", None)

        db_ok = True
        if repo is not None:
            try:
                await run_in_threadpool(repo.ensure_loaded)
            except Exception:
                db_ok = False

        if not models_ok:
            unavail = HealthResponse(
                status="unavailable",
                models_loaded=False,
                db_connected=db_ok,
                champion_alias=cfg.champion_alias,
                champion_versions={},
            )
            return JSONResponse(status_code=503, content=unavail.model_dump())

        v_dict = bundle.versions if bundle is not None else {}
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            db_connected=db_ok,
            champion_alias=cfg.champion_alias,
            champion_versions=v_dict,
        )

    @app.get(
        "/v1/skus/{sku_id}/forecast",
        response_model=ForecastPayload,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["MLOps Inference"],
    )
    async def get_sku_forecast(
        sku_id: str,
        request: Request,
        horizon_days: int = Query(default=7, ge=1, le=56, description="Horizon H en jours"),
    ) -> ForecastPayload | JSONResponse:
        bundle: ChampionModelBundle | None = getattr(request.app.state, "champion_bundle", None)
        if not getattr(request.app.state, "models_loaded", False) or bundle is None:
            return _error_json(
                503,
                "MODELS_NOT_LOADED",
                "Service indisponible",
                "Les modeles @champion ne sont pas charges dans MLflow.",
                sku_id=sku_id,
            )

        repo: SupplyChainDataRepository = request.app.state.repo

        def _compute_forecast_sync() -> ForecastPayload | None:
            row = repo.get_latest_sku_row(sku_id, obs_date=cfg.default_observation_date)
            if row is None:
                return None

            X_df = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()], columns=FEATURE_COLUMNS).astype(
                np.float64
            )
            raw_pred = bundle.forecaster_model.predict(X_df)
            if isinstance(raw_pred, pd.DataFrame):
                arr = raw_pred[["q10", "q50", "q90"]].to_numpy(dtype=np.float64)
            else:
                arr = np.asarray(raw_pred, dtype=np.float64)

            mono = enforce_quantile_monotonicity(arr)[0]
            q10, q50, q90 = (
                round(float(mono[0]), 4),
                round(float(mono[1]), 4),
                round(float(mono[2]), 4),
            )
            q50 = max(q10, q50)
            q90 = max(q50, q90)

            rel_uncert = round((q90 - q10) / max(q50, 1.0), 4)
            q_obj = QuantileForecast(
                q10=q10,
                q50=q50,
                q90=q90,
                relative_uncertainty=rel_uncert,
                high_uncertainty_flag=bool(rel_uncert > cfg.uncertainty_Threshold_tau),
            )

            probs = bundle.classifier_model.predict_proba(X_df.to_numpy(dtype=np.float64))
            cal_prob = float(np.clip(np.asarray(probs, dtype=np.float64)[0, 1], 0.0, 1.0))

            roll_std = float(row.get("demand_roll_std_28", 1.0))
            roll_mean = max(float(row.get("demand_roll_mean_28", 1.0)), 1.0)
            local_mase = round(min(2.5, max(0.15, roll_std / roll_mean)), 4)

            v_str = bundle.versions.get(cfg.registered_forecaster_name, "1")
            return ForecastPayload(
                sku_id=sku_id,
                obs_date=pd.Timestamp(row["obs_date"]).strftime("%Y-%m-%d"),
                horizon_days=horizon_days,
                quantiles=q_obj,
                total_demand_q10=round(q10 * horizon_days, 4),
                total_demand_q50=round(q50 * horizon_days, 4),
                total_demand_q90=round(q90 * horizon_days, 4),
                calibrated_criticality_prob=round(cal_prob, 4),
                historical_mase_4w=local_mase,
                model_version=f"{cfg.registered_forecaster_name}@v{v_str}",
                top_shap_drivers=[],
            )

        payload = await run_in_threadpool(_compute_forecast_sync)
        if payload is None:
            return _error_json(
                404,
                "SKU_NOT_FOUND",
                f"Reference produit '{sku_id}' introuvable.",
                "Aucun historique disponible dans la base relationnelle SQL.",
                sku_id=sku_id,
            )
        return payload

    @app.get(
        "/v1/skus/{sku_id}/criticality",
        response_model=CriticalityPayload,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["MLOps Inference"],
    )
    async def get_sku_criticality(
        sku_id: str,
        request: Request,
        horizon_days: int = Query(default=7, ge=1, le=56),
    ) -> CriticalityPayload | JSONResponse:
        bundle: ChampionModelBundle | None = getattr(request.app.state, "champion_bundle", None)
        if not getattr(request.app.state, "models_loaded", False) or bundle is None:
            return _error_json(
                503,
                "MODELS_NOT_LOADED",
                "Service indisponible",
                "Les modeles @champion ne sont pas charges.",
                sku_id=sku_id,
            )

        repo: SupplyChainDataRepository = request.app.state.repo

        def _compute_criticality_sync() -> CriticalityPayload | None:
            row = repo.get_latest_sku_row(sku_id, obs_date=cfg.default_observation_date)
            if row is None:
                return None

            X_arr = np.asarray([row[FEATURE_COLUMNS].to_numpy(dtype=np.float64)], dtype=np.float64)
            cal_probs = bundle.classifier_model.predict_proba(X_arr)
            cal_p = float(np.clip(np.asarray(cal_probs, dtype=np.float64)[0, 1], 0.0, 1.0))

            raw_p = cal_p
            if hasattr(bundle.classifier_model, "calibrated_classifiers_"):
                inner = bundle.classifier_model.calibrated_classifiers_[0].estimator
                if hasattr(inner, "predict_proba"):
                    raw_p = float(np.clip(inner.predict_proba(X_arr)[0, 1], 0.0, 1.0))

            raw_cal_method = str(getattr(bundle.classifier_model, "method", "isotonic"))
            cal_method: Literal["isotonic", "sigmoid", "uncalibrated"] = (
                "sigmoid" if raw_cal_method == "sigmoid" else "isotonic"
            )

            v_str = bundle.versions.get(cfg.registered_classifier_name, "1")
            return CriticalityPayload(
                sku_id=sku_id,
                obs_date=pd.Timestamp(row["obs_date"]).strftime("%Y-%m-%d"),
                horizon_days=horizon_days,
                raw_uncalibrated_prob=round(raw_p, 4),
                calibrated_criticality_prob=round(cal_p, 4),
                calibration_method=cal_method,
                is_critical_alert=bool(cal_p >= 0.50),
                model_version=f"{cfg.registered_classifier_name}@v{v_str}",
                top_shap_drivers=[],
            )

        payload = await run_in_threadpool(_compute_criticality_sync)
        if payload is None:
            return _error_json(
                404,
                "SKU_NOT_FOUND",
                f"Reference produit '{sku_id}' introuvable.",
                "SKU inconnu dans la base relationnelle SQL.",
                sku_id=sku_id,
            )
        return payload

    @app.post(
        "/v1/anomalies/evaluate",
        response_model=AnomalyPayload,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["MLOps Inference"],
    )
    async def evaluate_sku_anomalies(
        body: AnomalyRequest,
        request: Request,
    ) -> AnomalyPayload | JSONResponse:
        bundle: ChampionModelBundle | None = getattr(request.app.state, "champion_bundle", None)
        if not getattr(request.app.state, "models_loaded", False) or bundle is None:
            return _error_json(
                503,
                "MODELS_NOT_LOADED",
                "Service indisponible",
                "Les modeles @champion ne sont pas charges.",
                sku_id=body.sku_id,
            )

        repo: SupplyChainDataRepository = request.app.state.repo

        def _evaluate_anomaly_sync() -> AnomalyPayload | None:
            row = repo.get_latest_sku_row(body.sku_id, obs_date=body.obs_date)
            if row is None:
                return None

            eval_row = row.copy()
            if body.observed_demand_qty is not None:
                eval_row["demand_qty"] = float(body.observed_demand_qty)
            if body.on_hand_qty is not None:
                eval_row["on_hand_qty"] = float(body.on_hand_qty)
            if body.unit_price_eur is not None:
                eval_row["unit_price_eur"] = float(body.unit_price_eur)
            if body.observed_lead_time_days is not None:
                contractual_lt = float(eval_row.get("causal_lead_time_mean", 5.0))
                eval_row["lead_time_drift"] = float(body.observed_lead_time_days) - contractual_lt

            row_df = pd.DataFrame([eval_row])
            pred_q50_arr = (
                np.array([float(body.predicted_q50_qty)], dtype=np.float64)
                if body.predicted_q50_qty is not None
                else None
            )
            X_anom = SupplyChainAnomalyDetector.build_anomaly_features(
                row_df, predicted_q50=pred_q50_arr
            )

            raw_score = float(-bundle.anomaly_model.decision_function(X_anom)[0])
            pred_label = int(bundle.anomaly_model.predict(X_anom)[0])

            reasons: list[str] = []
            corrupted = False
            on_hand_val = float(eval_row["on_hand_qty"])
            price_val = float(eval_row["unit_price_eur"])
            demand_val = float(eval_row["demand_qty"])

            if on_hand_val < 0.0 or not np.isfinite(on_hand_val):
                corrupted = True
                reasons.append("DATA_CORRUPTION: negative_or_nan_inventory_on_hand")
            if price_val <= 0.0 or not np.isfinite(price_val):
                corrupted = True
                reasons.append("DATA_CORRUPTION: non_positive_unit_price")
            if demand_val < 0.0 or not np.isfinite(demand_val):
                corrupted = True
                reasons.append("DATA_CORRUPTION: negative_or_nan_demand")

            res_val, lt_z, spike_val, fill_def, stock_ratio = X_anom[0]
            if res_val > 0.75:
                reasons.append(f"HIGH_FORECAST_RESIDUAL: rel_error={res_val:.2f}")
            if lt_z > 2.2:
                reasons.append(f"SUPPLIER_LEAD_TIME_SHOCK: z_score={lt_z:.2f}")
            if spike_val > 2.3:
                reasons.append(f"ATYPICAL_DEMAND_SPIKE: ratio={spike_val:.2f}x")
            if fill_def > 0.12:
                reasons.append(f"SUPPLIER_UNDER_DELIVERY: deficit={fill_def:.2%}")

            is_anom = bool(pred_label == -1 or corrupted)
            if is_anom and not reasons:
                reasons.append("ISOLATION_FOREST_MULTIVARIATE_OUTLIER")

            diagnostics_dict = {
                "abs_relative_forecast_residual": round(float(res_val), 4),
                "lead_time_zscore": round(float(lt_z), 4),
                "demand_spike_ratio": round(float(spike_val), 4),
                "supplier_fill_rate_deficit": round(float(fill_def), 4),
                "stock_to_reorder_ratio": round(float(stock_ratio), 4),
            }

            v_str = bundle.versions.get(cfg.registered_anomaly_name, "1")
            return AnomalyPayload(
                sku_id=body.sku_id,
                obs_date=pd.Timestamp(eval_row["obs_date"]).strftime("%Y-%m-%d"),
                window_days=body.window_days,
                anomaly_score=round(raw_score, 4),
                is_anomaly=is_anom,
                data_corruption_detected=corrupted,
                triggered_reasons=reasons,
                feature_diagnostics=diagnostics_dict,
                model_version=f"{cfg.registered_anomaly_name}@v{v_str}",
            )

        payload = await run_in_threadpool(_evaluate_anomaly_sync)
        if payload is None:
            return _error_json(
                404,
                "SKU_NOT_FOUND",
                f"Reference produit '{body.sku_id}' introuvable.",
                "SKU inconnu dans la base relationnelle SQL.",
                sku_id=body.sku_id,
            )
        return payload

    # --- Endpoints Agentic AI & Human-in-the-Loop ---

    @app.post("/v1/agent/investigate", response_model=InvestigateResponse, tags=["Agentic HITL"])
    async def investigate_incident(
        payload: InvestigateRequest, request: Request
    ) -> InvestigateResponse:
        """Declenche l'agent gouverne sur un incident SKU, persiste la decision
        et trace en JSONL.
        """

        state: PlatformAppState = request.app.state.platform
        trigger = IncidentTrigger(
            sku_id=payload.sku_id,
            trigger_event=payload.trigger_event,
            horizon_days=payload.horizon_days,
        )
        run_res = await run_in_threadpool(state.orchestrator.run, trigger)
        saved_rec = await run_in_threadpool(
            state.decision_store.save_decision,
            run_res.trace_id,
            run_res.report,
        )
        return InvestigateResponse(
            decision_id=saved_rec.id,
            trace_id=run_res.trace_id,
            report=run_res.report,
        )

    @app.get("/v1/decisions/pending", response_model=list[DecisionRecord], tags=["Agentic HITL"])
    async def get_pending_decisions(request: Request) -> list[DecisionRecord]:
        """Liste toutes les decisions en attente de validation humaine (PENDING_HUMAN_APPROVAL)."""
        state: PlatformAppState = request.app.state.platform
        return await run_in_threadpool(state.decision_store.list_pending)

    @app.post(
        "/v1/decisions/{decision_id}/approve_or_reject",
        response_model=OperatorDecisionResponse,
        tags=["Agentic HITL"],
    )
    async def approve_or_reject_decision(
        decision_id: str,
        payload: OperatorDecisionRequest,
        request: Request,
    ) -> OperatorDecisionResponse:
        """Approuve ou rejette une decision HITL (idempotent : 409 Conflict si deja traitee)."""
        state: PlatformAppState = request.app.state.platform
        try:
            updated_rec, prev_status = await run_in_threadpool(
                state.decision_store.approve_or_reject,
                decision_id,
                payload.operator_id,
                payload.decision,
                payload.justification,
            )
        except DecisionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DecisionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        # Journalisation de la decision humaine dans logs/agent_traces.jsonl
        await run_in_threadpool(
            record_execution_trace,
            updated_rec.sku_id,
            f"hitl_operator_{payload.decision}",
            [],
            0,
            [],
            updated_rec.status,
            updated_rec.report_json.guardrails_triggered,
            updated_rec.trace_id,
            {
                "decision_id": updated_rec.id,
                "operator_id": payload.operator_id,
                "decision": payload.decision,
                "justification": payload.justification,
                "previous_status": prev_status.value,
                "new_status": updated_rec.status.value,
            },
            state.orchestrator.traces_path,
        )

        return OperatorDecisionResponse(
            decision_id=updated_rec.id,
            trace_id=updated_rec.trace_id,
            previous_status=prev_status,
            new_status=updated_rec.status,
            operator_id=payload.operator_id,
            decided_at=updated_rec.decided_at or "",
        )

    return app


app = create_app()
