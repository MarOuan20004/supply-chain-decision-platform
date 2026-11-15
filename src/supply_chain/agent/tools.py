"""Registre d'outils types Pydantic v2 pour l'Agent Supply Chain.

Appelle directement les composants Python en memoire (sans loopback HTTP)
et applique une validation stricte des entrees et sorties.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from supply_chain.agent.exceptions import (
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from supply_chain.agent.schemas import (
    DecisionProposal,
    ExecutionStatus,
    SubmissionResult,
)

if TYPE_CHECKING:
    from supply_chain.api import SupplyChainDataRepository


from supply_chain.config import SupplyChainSettings, settings
from supply_chain.data_pipeline import FEATURE_COLUMNS
from supply_chain.mlflow_loader import ChampionModelBundle
from supply_chain.models import SupplyChainAnomalyDetector, enforce_quantile_monotonicity
from supply_chain.optimizer import InventoryOptimizer
from supply_chain.schemas import (
    AnomalyPayload,
    ForecastPayload,
    InventoryRecord,
    PolicySimulationResult,
    QuantileForecast,
)

# ============================================================================
# SCHEMAS D'ENTREE STRICTS DES 5 OUTILS (extra="forbid")
# ============================================================================


class GetForecastInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=3, description="Identifiant du produit (ex: SKU-0001)")
    horizon_days: int = Field(default=7, ge=1, le=56, description="Horizon de prevision en jours")


class InspectAnomalyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=3, description="Identifiant du produit (ex: SKU-0001)")
    window_days: int = Field(default=14, ge=1, le=90, description="Fenetre d'analyse en jours")


class QueryInventoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=3, description="Identifiant du produit (ex: SKU-0001)")


class SimulatePolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=3, description="Identifiant du produit (ex: SKU-0001)")
    order_qty: int = Field(..., ge=0, description="Quantite a commander (en unites)")
    expedited: bool = Field(default=False, description="Activer le transport express")


class SubmitProposalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: DecisionProposal


# ============================================================================
# CONTRAT DE FOURNISSEUR DE DONNEES / MODELES POUR L'AGENT
# ============================================================================


class SupplyChainToolBackend(Protocol):
    """Protocole decouplant l'execution des outils du stockage SQLite/MLflow en tests."""

    def forecast(self, sku_id: str, horizon_days: int) -> ForecastPayload: ...

    def anomaly(self, sku_id: str, window_days: int) -> AnomalyPayload: ...

    def inventory(self, sku_id: str) -> InventoryRecord: ...


class DirectSupplyChainBackend:
    """Implementation de production appelant directement S5/S6 (SQLite RO + ChampionBundle)."""

    def __init__(
        self,
        repo: SupplyChainDataRepository,
        bundle: ChampionModelBundle,
        cfg: SupplyChainSettings = settings,
    ) -> None:
        self.repo = repo
        self.bundle = bundle
        self.cfg = cfg

    def forecast(self, sku_id: str, horizon_days: int) -> ForecastPayload:
        row = self.repo.get_latest_sku_row(sku_id, obs_date=self.cfg.default_observation_date)
        if row is None:
            raise ToolExecutionError(f"SKU '{sku_id}' introuvable pour la prevision.")

        X_df = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()], columns=FEATURE_COLUMNS).astype(
            np.float64
        )
        raw_pred = self.bundle.forecaster_model.predict(X_df)
        arr = (
            raw_pred[["q10", "q50", "q90"]].to_numpy(dtype=np.float64)
            if isinstance(raw_pred, pd.DataFrame)
            else np.asarray(raw_pred, dtype=np.float64)
        )
        mono = enforce_quantile_monotonicity(arr)[0]
        q10, q50, q90 = round(float(mono[0]), 4), round(float(mono[1]), 4), round(float(mono[2]), 4)
        q50 = max(q10, q50)
        q90 = max(q50, q90)

        rel_uncert = round((q90 - q10) / max(q50, 1.0), 4)
        q_obj = QuantileForecast(
            q10=q10,
            q50=q50,
            q90=q90,
            relative_uncertainty=rel_uncert,
            high_uncertainty_flag=bool(rel_uncert > self.cfg.uncertainty_Threshold_tau),
        )
        probs = self.bundle.classifier_model.predict_proba(X_df.to_numpy(dtype=np.float64))
        cal_prob = float(np.clip(np.asarray(probs, dtype=np.float64)[0, 1], 0.0, 1.0))

        roll_std = float(row.get("demand_roll_std_28", 1.0))
        roll_mean = max(float(row.get("demand_roll_mean_28", 1.0)), 1.0)
        local_mase = round(min(2.5, max(0.15, roll_std / roll_mean)), 4)

        ver = self.bundle.versions.get(self.cfg.registered_forecaster_name, "1")
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
            model_version=f"{self.cfg.registered_forecaster_name}@v{ver}",
            top_shap_drivers=[],
        )

    def anomaly(self, sku_id: str, window_days: int) -> AnomalyPayload:
        row = self.repo.get_latest_sku_row(sku_id, obs_date=self.cfg.default_observation_date)
        if row is None:
            raise ToolExecutionError(f"SKU '{sku_id}' introuvable pour le diagnostic d'anomalie.")

        row_df = pd.DataFrame([row])
        X_anom = SupplyChainAnomalyDetector.build_anomaly_features(row_df)
        raw_score = float(-self.bundle.anomaly_model.decision_function(X_anom)[0])
        pred_label = int(self.bundle.anomaly_model.predict(X_anom)[0])

        reasons: list[str] = []
        corrupted = False
        if float(row["on_hand_qty"]) < 0.0 or not np.isfinite(float(row["on_hand_qty"])):
            corrupted = True
            reasons.append("DATA_CORRUPTION: negative_or_nan_inventory_on_hand")
        if float(row["unit_price_eur"]) <= 0.0 or not np.isfinite(float(row["unit_price_eur"])):
            corrupted = True
            reasons.append("DATA_CORRUPTION: non_positive_unit_price")

        res_val, lt_z, spike_val, fill_def, stock_ratio = X_anom[0]
        if lt_z > 2.2:
            reasons.append(f"SUPPLIER_LEAD_TIME_SHOCK: z_score={lt_z:.2f}")
        if spike_val > 2.3:
            reasons.append(f"ATYPICAL_DEMAND_SPIKE: ratio={spike_val:.2f}x")

        is_anom = bool(pred_label == -1 or corrupted)
        ver = self.bundle.versions.get(self.cfg.registered_anomaly_name, "1")
        return AnomalyPayload(
            sku_id=sku_id,
            obs_date=pd.Timestamp(row["obs_date"]).strftime("%Y-%m-%d"),
            window_days=window_days,
            anomaly_score=round(raw_score, 4),
            is_anomaly=is_anom,
            data_corruption_detected=corrupted,
            triggered_reasons=reasons,
            feature_diagnostics={
                "abs_relative_forecast_residual": round(float(res_val), 4),
                "lead_time_zscore": round(float(lt_z), 4),
                "demand_spike_ratio": round(float(spike_val), 4),
                "supplier_fill_rate_deficit": round(float(fill_def), 4),
                "stock_to_reorder_ratio": round(float(stock_ratio), 4),
            },
            model_version=f"{self.cfg.registered_anomaly_name}@v{ver}",
        )

    def inventory(self, sku_id: str) -> InventoryRecord:
        # Si la base SQLite existe, execution en mode lecture seule (?mode=ro)
        if self.cfg.sqlite_db_path.is_file():
            uri = f"file:{self.cfg.sqlite_db_path.resolve()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conn:
                cur = conn.execute(
                    "SELECT sku_id, category, unit_cost_eur, holding_cost_per_day_eur, "
                    "stockout_penalty_per_unit_eur, moq_units, contractual_lead_time_days "
                    "FROM skus WHERE sku_id = ?",
                    (sku_id,),
                )
                if cur.fetchone() is None:
                    raise ToolExecutionError(f"SKU '{sku_id}' absent de la table SQL 'skus'.")

        record = self.repo.get_inventory_record(sku_id, obs_date=self.cfg.default_observation_date)
        if record is None:
            raise ToolExecutionError(f"Aucun enregistrement d'inventaire pour '{sku_id}'.")
        return record


# ============================================================================
class InMemoryToolBackend:
    """Backend deterministe en memoire sans dependance MLflow/SQLite pour tests rapides."""

    def forecast(self, sku_id: str, horizon_days: int) -> ForecastPayload:
        q = QuantileForecast(
            q10=10.0, q50=15.0, q90=20.0, relative_uncertainty=0.6667, high_uncertainty_flag=False
        )
        return ForecastPayload(
            sku_id=sku_id,
            obs_date="2026-12-07",
            horizon_days=horizon_days,
            quantiles=q,
            total_demand_q10=10.0 * horizon_days,
            total_demand_q50=15.0 * horizon_days,
            total_demand_q90=20.0 * horizon_days,
            calibrated_criticality_prob=0.82,
            historical_mase_4w=0.54,
            model_version="forecaster@v1",
        )

    def anomaly(self, sku_id: str, window_days: int) -> AnomalyPayload:
        return AnomalyPayload(
            sku_id=sku_id,
            obs_date="2026-12-07",
            window_days=window_days,
            anomaly_score=-0.15,
            is_anomaly=False,
            data_corruption_detected=False,
            triggered_reasons=[],
            feature_diagnostics={"lead_time_zscore": 0.3},
            model_version="anomaly@v1",
        )

    def inventory(self, sku_id: str) -> InventoryRecord:
        return InventoryRecord(
            sku_id=sku_id,
            category="robotics_actuators",
            obs_date="2026-12-07",
            on_hand_qty=20.0,
            in_transit_qty=0.0,
            effective_stock_qty=20.0,
            reorder_point_s=60.0,
            order_up_to_S=180.0,
            unit_cost_eur=20.0,
            holding_cost_per_day_eur=0.08,
            stockout_penalty_per_unit_eur=40.0,
            moq_units=50,
            contractual_lead_time_days=5,
            empirical_lead_time_mean=5.0,
            empirical_lead_time_std=1.0,
        )


class ModelServiceToolBackend:
    """Adaptateur reliant le registre d'outils au service MLflow / ChampionModelService."""

    def __init__(self, model_service: Any, cfg: SupplyChainSettings = settings) -> None:
        self.service = model_service
        self.cfg = cfg

    def forecast(self, sku_id: str, horizon_days: int) -> ForecastPayload:
        return self.service.predict_forecast(sku_id, horizon_days)

    def anomaly(self, sku_id: str, window_days: int) -> AnomalyPayload:
        from supply_chain.schemas import AnomalyRequest

        req = AnomalyRequest(sku_id=sku_id, window_days=window_days)
        return self.service.evaluate_anomalies(req)

    def inventory(self, sku_id: str) -> InventoryRecord:
        if hasattr(self.service, "get_inventory_record"):
            return self.service.get_inventory_record(sku_id)
        if hasattr(self.service, "repo"):
            return self.service.repo.get_inventory_record(sku_id)
        return InMemoryToolBackend().inventory(sku_id)


# ============================================================================
# REGISTRE DES 5 OUTILS TYPES DE L'AGENT
# ============================================================================


class SupplyChainToolRegistry:
    """Encapsule les 5 outils de l'agent avec validation Pydantic v2 et tracabilite."""

    def __init__(
        self,
        backend: SupplyChainToolBackend | None = None,
        optimizer: InventoryOptimizer | None = None,
        cfg: SupplyChainSettings = settings,
        model_service: Any = None,
        use_in_memory_fallback: bool = False,
    ) -> None:
        if backend is not None:
            self.backend = backend
        elif use_in_memory_fallback:
            self.backend = InMemoryToolBackend()
        elif model_service is not None:
            self.backend = ModelServiceToolBackend(model_service=model_service, cfg=cfg)
        else:
            try:
                from supply_chain.api import SupplyChainDataRepository
                from supply_chain.mlflow_loader import load_champion_bundle

                repo = SupplyChainDataRepository(cfg=cfg)
                bundle = load_champion_bundle(cfg=cfg)
                self.backend = DirectSupplyChainBackend(repo=repo, bundle=bundle, cfg=cfg)
            except Exception:
                self.backend = InMemoryToolBackend()

        self.optimizer = optimizer or InventoryOptimizer(cfg=cfg)
        self.cfg = cfg

        # Caches intra-session pour relier simulation RO et soumission finale
        self._last_forecasts: dict[str, ForecastPayload] = {}
        self._last_inventories: dict[str, InventoryRecord] = {}
        self._simulations: list[PolicySimulationResult] = []
        self._submitted_proposals: list[SubmissionResult] = []

        self._specs: dict[str, tuple[str, type[BaseModel], Callable[[Any], BaseModel]]] = {
            "get_sku_forecast_and_uncertainty": (
                "Recupere la prevision de demande multi-quantile (q10, q50, q90), "
                "l'incertitude relative et la probabilite calibree de rupture d'un SKU.",
                GetForecastInput,
                self._tool_get_forecast,
            ),
            "inspect_anomaly_diagnostics": (
                "Execute le diagnostic Isolation Forest et verifie l'integrite physique "
                "des donnees (stock, prix, choc de delai fournisseur) sur une fenetre.",
                InspectAnomalyInput,
                self._tool_inspect_anomaly,
            ),
            "query_inventory_and_supplier_sql": (
                "Interroge en lecture seule la base SQL pour extraire stock physique, "
                "stock en transit, MOQ, delai fournisseur et couts unitaires.",
                QueryInventoryInput,
                self._tool_query_inventory,
            ),
            "simulate_reorder_policy": (
                "Simule de maniere 100% deterministe le cout total espere d'une commande "
                "(order_cost + holding_cost + stockout_penalty) et verifie MOQ et budget.",
                SimulatePolicyInput,
                self._tool_simulate_policy,
            ),
            "submit_decision_proposal": (
                "Soumet la proposition finale de decision. Place automatiquement en statut "
                "PENDING_HUMAN_APPROVAL si order_cost_eur > 5000 EUR ou si expedited=True.",
                SubmitProposalInput,
                self._tool_submit_proposal,
            ),
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Genere les schemas JSON Function Calling standardises pour LLMClient.chat()."""
        definitions: list[dict[str, Any]] = []
        for name, (description, input_model, _) in self._specs.items():
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": input_model.model_json_schema(),
                    },
                }
            )
        return definitions

    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Alias pour get_tool_definitions."""
        return self.get_tool_definitions()

    def execute_tool(
        self, tool_name: str, raw_arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        """Valide les arguments, execute l'outil et valide la sortie Pydantic v2.

        Returns:
            Tuple (validated_args_dict, raw_output_dict, latency_ms).
        """
        if tool_name not in self._specs:
            raise UnknownToolError(
                f"Outil inconnu '{tool_name}'. Outils autorises: {list(self._specs.keys())}"
            )

        _, input_model, handler = self._specs[tool_name]
        t0 = time.perf_counter()

        try:
            validated_input = input_model.model_validate(raw_arguments)
        except ValidationError as exc:
            raise ToolValidationError(
                f"Arguments invalides pour l'outil '{tool_name}': {exc}"
            ) from exc

        try:
            output_model = handler(validated_input)
        except (ToolValidationError, ToolExecutionError):
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Erreur d'execution dans l'outil '{tool_name}': {exc}"
            ) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        raw_out = output_model.model_dump(mode="json")
        if isinstance(output_model, InventoryRecord):
            raw_out.setdefault("current_stock", output_model.on_hand_qty)
            raw_out.setdefault("lead_time_mean_days", output_model.empirical_lead_time_mean)
            raw_out.setdefault("lead_time_std_days", output_model.empirical_lead_time_std)
            raw_out.setdefault("moq", output_model.moq_units)
        elif isinstance(output_model, AnomalyPayload):
            raw_out.setdefault("feature_contributions", output_model.feature_diagnostics)

        return (
            validated_input.model_dump(mode="json"),
            raw_out,
            latency_ms,
        )

    def _tool_get_forecast(self, args: GetForecastInput) -> ForecastPayload:
        payload = self.backend.forecast(sku_id=args.sku_id, horizon_days=args.horizon_days)
        self._last_forecasts[args.sku_id] = payload
        return payload

    def _tool_inspect_anomaly(self, args: InspectAnomalyInput) -> AnomalyPayload:
        return self.backend.anomaly(sku_id=args.sku_id, window_days=args.window_days)

    def _tool_query_inventory(self, args: QueryInventoryInput) -> InventoryRecord:
        record = self.backend.inventory(sku_id=args.sku_id)
        self._last_inventories[args.sku_id] = record
        return record

    def _tool_simulate_policy(self, args: SimulatePolicyInput) -> PolicySimulationResult:
        inv = self._last_inventories.get(args.sku_id)
        if inv is None:
            inv = self._tool_query_inventory(QueryInventoryInput(sku_id=args.sku_id))

        fc = self._last_forecasts.get(args.sku_id)
        if fc is None:
            fc = self._tool_get_forecast(
                GetForecastInput(sku_id=args.sku_id, horizon_days=self.cfg.default_horizon_days)
            )

        sim = self.optimizer.simulate_policy(
            inventory=inv,
            forecast=fc.quantiles,
            order_qty=args.order_qty,
            horizon_days=fc.horizon_days,
            expedited=args.expedited,
        )
        self._simulations.append(sim)
        return sim

    def _tool_submit_proposal(self, args: SubmitProposalInput) -> SubmissionResult:
        prop = args.proposal

        if not prop.abstain:
            # Verification que les chiffres proviennent bien d'une simulation RO de la session
            matching_sims = [
                s
                for s in self._simulations
                if s.sku_id == prop.sku_id
                and s.effective_order_qty == prop.order_qty
                and s.expedited == prop.expedited
                and abs(s.order_cost - prop.order_cost_eur) < 1e-2
                and abs(s.total_cost - prop.expected_total_cost_eur) < 1e-2
            ]
            if not matching_sims:
                raise ToolValidationError(
                    "Les montants order_cost_eur et expected_total_cost_eur de DecisionProposal "
                    "doivent correspondre exactement a une sortie de simulate_reorder_policy."
                )

        if prop.abstain:
            status = ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
            requires_hitl = False
            reason = f"Abstention explicite: {prop.abstention_reason}"
        elif prop.order_cost_eur > self.cfg.human_approval_threshold_eur or prop.expedited:
            status = ExecutionStatus.PENDING_HUMAN_APPROVAL
            requires_hitl = True
            reason = (
                f"Validation humaine obligatoire (order_cost={prop.order_cost_eur:.2f} EUR, "
                f"seuil={self.cfg.human_approval_threshold_eur:.2f} EUR, "
                f"expedited={prop.expedited})."
            )
        else:
            status = ExecutionStatus.APPROVED_AUTO
            requires_hitl = False
            reason = (
                f"Approuve automatiquement (order_cost={prop.order_cost_eur:.2f} EUR <= "
                f"{self.cfg.human_approval_threshold_eur:.2f} EUR et transport standard)."
            )

        result = SubmissionResult(
            proposal_id=f"PROP-{uuid.uuid4().hex[:8].upper()}",
            status=status,
            sku_id=prop.sku_id,
            order_qty=prop.order_qty,
            expedited=prop.expedited,
            order_cost_eur=round(prop.order_cost_eur, 4),
            expected_total_cost_eur=round(prop.expected_total_cost_eur, 4),
            requires_human_approval=requires_hitl,
            policy_reason=reason,
        )
        self._submitted_proposals.append(result)
        return result
