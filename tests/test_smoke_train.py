"""Smoke test de bout en bout : SQL -> CatBoost -> MLflow @champion -> FastAPI."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow.models import infer_signature

from scripts.train_and_register_mlflow import MonotonicCatBoostPyfuncWrapper
from supply_chain.api import create_app
from supply_chain.config import SupplyChainSettings
from supply_chain.data_pipeline import (
    FEATURE_COLUMNS,
    build_causal_feature_Frame,
    generate_synthetic_supply_chain_db,
    load_raw_tables_from_sql,
)
from supply_chain.mlflow_loader import promote_latest_to_champion
from supply_chain.models import (
    CalibratedCriticalityClassifier,
    MultiHorizonQuantileForecaster,
    SupplyChainAnomalyDetector,
)


@pytest.mark.smoke
def test_end_to_end_smoke_train_promote_and_serve(tmp_path: Path) -> None:
    """Valide en < 15s le cycle SQL -> Entrainement -> MLflow @champion -> FastAPI TestClient."""
    db_path = tmp_path / "smoke_supply.db"
    mlflow_db = tmp_path / "smoke_mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db}"

    cfg = SupplyChainSettings(
        sqlite_db_path=db_path,
        num_skus=6,
        num_days=140,
        catboost_iterations=35,
        catboost_depth=3,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_name="smoke_exp_s6",
        registered_forecaster_name="smoke_forecaster",
        registered_classifier_name="smoke_classifier",
        registered_anomaly_name="smoke_anomaly",
    )

    generate_synthetic_supply_chain_db(
        db_path=db_path, num_skus=cfg.num_skus, num_days=cfg.num_days, seed=cfg.random_seed
    )
    daily_df, deliv_df = load_raw_tables_from_sql(db_path)
    feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=7)

    forecaster = MultiHorizonQuantileForecaster(horizons=(7,), cfg=cfg)
    forecaster.fit_horizon(feat_df, horizon_days=7)

    classifier = CalibratedCriticalityClassifier(cfg=cfg)
    classifier.fit(feat_df, horizon_days=7, preferred_method="sigmoid")

    detector = SupplyChainAnomalyDetector(cfg=cfg)
    detector.fit(feat_df)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    X_ex = feat_df[FEATURE_COLUMNS].head(5).astype(np.float64)

    with mlflow.start_run(run_name="smoke_run"):
        pf_model = MonotonicCatBoostPyfuncWrapper(forecaster.models_by_horizon[7])
        mlflow.pyfunc.log_model(
            artifact_path="forecaster",
            python_model=pf_model,
            signature=infer_signature(X_ex, pf_model.predict(None, X_ex)),
            registered_model_name=cfg.registered_forecaster_name,
        )
        cal_est = classifier.sigmoid_calibrator_
        assert cal_est is not None
        mlflow.sklearn.log_model(
            sk_model=cal_est,
            artifact_path="classifier",
            signature=infer_signature(
                X_ex.to_numpy(dtype=np.float64),
                cal_est.predict_proba(X_ex.to_numpy(dtype=np.float64)),
            ),
            registered_model_name=cfg.registered_classifier_name,
        )
        X_anom_ex = SupplyChainAnomalyDetector.build_anomaly_features(feat_df.head(5))
        mlflow.sklearn.log_model(
            sk_model=detector.iforest,
            artifact_path="anomaly",
            signature=infer_signature(X_anom_ex, detector.iforest.decision_function(X_anom_ex)),
            registered_model_name=cfg.registered_anomaly_name,
        )

    promoted = promote_latest_to_champion(cfg)
    assert len(promoted) == 3

    app = create_app(cfg=cfg)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        first_sku = str(feat_df["sku_id"].iloc[0])
        resp = client.get(f"/v1/skus/{first_sku}/forecast?horizon_days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["quantiles"]["q10"] <= body["quantiles"]["q50"] <= body["quantiles"]["q90"]
