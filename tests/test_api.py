"""Tests d'integration de l'API REST FastAPI avec TestClient et mocking MLflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from supply_chain.api import SupplyChainDataRepository, create_app
from supply_chain.config import SupplyChainSettings
from supply_chain.data_pipeline import FEATURE_COLUMNS
from supply_chain.mlflow_loader import ChampionModelBundle


class _MockForecasterPyfunc:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame(
            {"q10": [8.5] * n, "q50": [12.0] * n, "q90": [17.5] * n},
            index=df.index,
        )


class _MockCalibratedClassifier:
    method = "isotonic"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.tile(np.array([[0.28, 0.72]], dtype=np.float64), (n, 1))


class _MockIsolationForest:
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Score negatif lorsque le z-score de delai > 2.0, positif sinon
        scores = np.where(X[:, 1] > 2.0, -0.35, 0.18)
        return scores.astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, 1] > 2.0, -1, 1)


def _build_in_memory_repo(cfg: SupplyChainSettings) -> SupplyChainDataRepository:
    row_dict: dict[str, object] = {col: 1.0 for col in FEATURE_COLUMNS}
    row_dict.update(
        {
            "sku_id": "SKU-0001",
            "obs_date": pd.Timestamp("2026-11-30"),
            "demand_qty": 11.0,
            "demand_roll_mean_14": 10.5,
            "demand_roll_mean_28": 10.0,
            "demand_roll_std_28": 2.2,
            "on_hand_qty": 35.0,
            "in_transit_qty": 10.0,
            "reorder_point_s": 50.0,
            "order_up_to_S": 150.0,
            "unit_price_eur": 29.9,
            "lead_time_drift": 0.2,
            "causal_lead_time_mean": 4.0,
            "causal_lead_time_std": 0.5,
            "causal_fill_rate_mean": 0.98,
            "stock_to_reorder_ratio": 0.70,
        }
    )
    feat_df = pd.DataFrame([row_dict])
    skus_df = pd.DataFrame(
        [
            {
                "sku_id": "SKU-0001",
                "category": "robotics",
                "unit_cost_eur": 15.0,
                "holding_cost_per_day_eur": 0.08,
                "stockout_penalty_per_unit_eur": 40.0,
                "moq_units": 30,
                "contractual_lead_time_days": 4,
                "safety_stock_units": 20,
            }
        ]
    )
    return SupplyChainDataRepository(
        cfg=cfg, preloaded_features_df=feat_df, preloaded_skus_df=skus_df
    )


def test_health_returns_503_when_models_missing_and_200_when_loaded() -> None:
    """Verifie GET /health -> 503 sans modeles @champion et 200 lorsque charges."""
    cfg = SupplyChainSettings()
    repo = _build_in_memory_repo(cfg)

    # 1. Cas sans modeles charges -> 503
    app_unavail = create_app(cfg=cfg, custom_bundle=None, custom_repo=repo, disable_auto_load=True)
    with TestClient(app_unavail) as client_503:
        r_503 = client_503.get("/health")
        assert r_503.status_code == 503
        assert r_503.json()["models_loaded"] is False

    # 2. Cas avec modeles @champion charges -> 200
    bundle = ChampionModelBundle(
        forecaster_model=_MockForecasterPyfunc(),
        classifier_model=_MockCalibratedClassifier(),
        anomaly_model=_MockIsolationForest(),
        versions={
            cfg.registered_forecaster_name: "1",
            cfg.registered_classifier_name: "1",
            cfg.registered_anomaly_name: "1",
        },
    )
    app_ok = create_app(cfg=cfg, custom_bundle=bundle, custom_repo=repo)
    with TestClient(app_ok) as client_200:
        r_200 = client_200.get("/health")
        assert r_200.status_code == 200
        body = r_200.json()
        assert body["status"] == "healthy"
        assert body["models_loaded"] is True


def test_forecast_criticality_and_anomaly_endpoints_nominal_and_errors() -> None:
    """Verifie les endpoints REST nominaux, 404 (SKU inconnu), 422 et la schema OpenAPI."""
    cfg = SupplyChainSettings()
    repo = _build_in_memory_repo(cfg)
    bundle = ChampionModelBundle(
        forecaster_model=_MockForecasterPyfunc(),
        classifier_model=_MockCalibratedClassifier(),
        anomaly_model=_MockIsolationForest(),
        versions={
            cfg.registered_forecaster_name: "2",
            cfg.registered_classifier_name: "2",
            cfg.registered_anomaly_name: "2",
        },
    )
    app = create_app(cfg=cfg, custom_bundle=bundle, custom_repo=repo)

    with TestClient(app) as client:
        # 1. GET /v1/skus/SKU-0001/forecast
        r_fc = client.get("/v1/skus/SKU-0001/forecast?horizon_days=7")
        assert r_fc.status_code == 200
        fc_data = r_fc.json()
        q = fc_data["quantiles"]
        assert 0.0 <= q["q10"] <= q["q50"] <= q["q90"]
        assert 0.0 <= fc_data["calibrated_criticality_prob"] <= 1.0

        # 2. GET /v1/skus/SKU-0001/criticality
        r_crit = client.get("/v1/skus/SKU-0001/criticality?horizon_days=14")
        assert r_crit.status_code == 200
        crit_data = r_crit.json()
        assert 0.0 <= crit_data["calibrated_criticality_prob"] <= 1.0
        assert crit_data["is_critical_alert"] is True

        # 3. POST /v1/anomalies/evaluate (cas nominal + cas corruption physique)
        r_anom_ok = client.post(
            "/v1/anomalies/evaluate", json={"sku_id": "SKU-0001", "window_days": 14}
        )
        assert r_anom_ok.status_code == 200
        assert r_anom_ok.json()["is_anomaly"] is False

        r_anom_corrupt = client.post(
            "/v1/anomalies/evaluate",
            json={"sku_id": "SKU-0001", "window_days": 14, "on_hand_qty": -10.0},
        )
        assert r_anom_corrupt.status_code == 200
        assert r_anom_corrupt.json()["data_corruption_detected"] is True
        assert r_anom_corrupt.json()["is_anomaly"] is True

        # 4. Erreurs HTTP 404 (SKU inconnu) et 422 (horizon invalide)
        assert client.get("/v1/skus/SKU-UNKNOWN/forecast").status_code == 404
        assert client.get("/v1/skus/SKU-0001/forecast?horizon_days=0").status_code == 422

        # 5. Verification de la completude du schema OpenAPI
        openapi_paths = client.get("/openapi.json").json()["paths"]
        assert "/health" in openapi_paths
        assert "/v1/skus/{sku_id}/forecast" in openapi_paths
        assert "/v1/skus/{sku_id}/criticality" in openapi_paths
        assert "/v1/anomalies/evaluate" in openapi_paths
