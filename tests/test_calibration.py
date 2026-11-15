"""Tests unitaires et d'integration pour CatBoost MultiQuantile, Calibration et SHAP (S5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supply_chain.config import SupplyChainSettings
from supply_chain.data_pipeline import (
    build_causal_feature_Frame,
    generate_synthetic_supply_chain_db,
    load_raw_tables_from_sql,
)
from supply_chain.models import (
    MultiHorizonQuantileForecaster,
    SupplyChainAnomalyDetector,
    enforce_quantile_monotonicity,
    evaluate_catboost_across_splits,
)
from supply_chain.validation import (
    ExpandingWindowTimeSplit,
    SeasonalNaiveBaseline,
    evaluate_baseline_across_splits,
)


@pytest.fixture(scope="module")
def calibration_test_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, pd.DataFrame, SupplyChainSettings]:
    """Genere une base SQLite de 12 SKUs sur 280 jours et une configuration rapide."""
    db_file: Path = tmp_path_factory.mktemp("s5_db") / "s5_supply.db"
    cfg = SupplyChainSettings(
        sqlite_db_path=db_file,
        num_skus=12,
        num_days=280,
        catboost_iterations=180,
        catboost_depth=4,
        catboost_learning_rate=0.05,
        calibration_holdout_frac=0.18,
        n_splits=4,
        min_train_days=110,
        val_window_days=21,
    )
    generate_synthetic_supply_chain_db(
        db_path=db_file,
        num_skus=cfg.num_skus,
        num_days=cfg.num_days,
        seed=cfg.random_seed,
    )
    daily_df, deliv_df = load_raw_tables_from_sql(db_file)
    return daily_df, deliv_df, cfg


def test_enforce_quantile_monotonicity_fixes_crossing_and_negatives() -> None:
    """Verifie que enforce_quantile_monotonicity corrige tout croisement et valeurs < 0."""
    raw = np.array(
        [
            [25.0, 18.0, 12.0],  # Croisement total inverse
            [-5.0, 10.0, 8.0],  # Quantile negatif + croisement q50 > q90
            [5.0, 15.0, 30.0],  # Deja monotone
        ],
        dtype=np.float64,
    )
    fixed = enforce_quantile_monotonicity(raw)
    assert fixed.shape == (3, 3)
    assert np.all(fixed >= 0.0)
    assert np.all(fixed[:, 0] <= fixed[:, 1])
    assert np.all(fixed[:, 1] <= fixed[:, 2])
    np.testing.assert_allclose(fixed[0], [12.0, 18.0, 25.0])
    np.testing.assert_allclose(fixed[1], [0.0, 8.0, 10.0])


def test_multi_horizon_catboost_quantiles_monotonicity_and_shap_additivity(
    calibration_test_data: tuple[pd.DataFrame, pd.DataFrame, SupplyChainSettings],
) -> None:
    """Verifie q10 <= q50 <= q90 sur 100% des predictions pour H in {7, 14, 28}."""
    daily_df, deliv_df, cfg = calibration_test_data
    forecaster = MultiHorizonQuantileForecaster(horizons=(7, 14, 28), cfg=cfg)

    for h in (7, 14, 28):
        feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=h)
        split_idx = int(len(feat_df) * 0.80)
        train_df = feat_df.iloc[:split_idx].reset_index(drop=True)
        val_df = feat_df.iloc[split_idx:].reset_index(drop=True)

        forecaster.fit_horizon(train_df, horizon_days=h)
        q10, q50, q90 = forecaster.predict_quantiles_matrix(val_df, horizon_days=h)

        assert np.all(q10 >= 0.0)
        assert np.all(q10 <= q50 + 1e-12)
        assert np.all(q50 <= q90 + 1e-12)

    feat_h7 = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=7)
    sample_val = feat_h7.tail(25).reset_index(drop=True)
    structured_preds = forecaster.build_structured_predictions(
        sample_val, horizon_days=7, include_shap=True, top_k_shap=5
    )
    assert len(structured_preds) == 25
    assert all(len(p.top_shap_drivers) == 5 for p in structured_preds)

    shap_vals, expected_vals = forecaster.explain_local_shap(sample_val, horizon_days=7)
    raw_model_preds = forecaster.models_by_horizon[7].predict(
        sample_val[forecaster.feature_columns].to_numpy(dtype=np.float64)
    )[:, 1]
    reconstructed = np.sum(shap_vals, axis=1) + expected_vals
    np.testing.assert_allclose(reconstructed, raw_model_preds, rtol=1e-4, atol=1e-4)


def test_probability_calibration_bounds_and_brier_ece_reduction(
    calibration_test_data: tuple[pd.DataFrame, pd.DataFrame, SupplyChainSettings],
) -> None:
    """Verifie que les probabilites calibrees sont dans [0, 1] et reduisent le Brier / ECE."""
    daily_df, deliv_df, cfg = calibration_test_data
    feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=7)

    splitter = ExpandingWindowTimeSplit(
        n_splits=cfg.n_splits,
        val_window_days=cfg.val_window_days,
        purge_gap_days=7,
        min_train_days=cfg.min_train_days,
    )

    naive_summary = evaluate_baseline_across_splits(
        feat_df, splitter, SeasonalNaiveBaseline(), horizon_days=7
    )
    (
        cb_summary,
        calib_comp,
        _,
        classifier,
        _,
        prob_uncal,
        prob_sig,
        prob_iso,
    ) = evaluate_catboost_across_splits(feat_df, splitter, horizon_days=7, cfg=cfg)

    for arr in (prob_uncal, prob_sig, prob_iso):
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)
        assert np.all(np.isfinite(arr))

    assert cb_summary.mase_mean < naive_summary.mase_mean
    assert cb_summary.pr_auc_mean >= naive_summary.pr_auc_mean

    uncal_m = next(c for c in calib_comp if c.method == "uncalibrated")
    sel_m = next(c for c in calib_comp if c.method == classifier.selected_method_)
    assert sel_m.brier_score < uncal_m.brier_score
    assert sel_m.ece < uncal_m.ece


def test_isolation_forest_anomaly_detector_flags_corruptions_and_shocks(
    calibration_test_data: tuple[pd.DataFrame, pd.DataFrame, SupplyChainSettings],
) -> None:
    """Verifie que SupplyChainAnomalyDetector detecte les anomalies et corruptions."""
    daily_df, deliv_df, cfg = calibration_test_data
    feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=7)

    detector = SupplyChainAnomalyDetector(cfg=cfg)
    detector.fit(feat_df)

    test_slice = feat_df.head(5).copy().reset_index(drop=True)
    test_slice.loc[0, "on_hand_qty"] = -15.0
    test_slice.loc[1, "lead_time_drift"] = 25.0
    test_slice.loc[1, "causal_lead_time_std"] = 1.0
    test_slice.loc[1, "demand_qty"] = test_slice.loc[1, "demand_roll_mean_14"] * 5.0

    diagnostics = detector.inspect_records(test_slice)
    assert len(diagnostics) == 5

    assert diagnostics[0].is_anomaly is True
    assert diagnostics[0].data_corruption_detected is True
    assert any("DATA_CORRUPTION" in r for r in diagnostics[0].triggered_reasons)

    assert diagnostics[1].is_anomaly is True
    assert diagnostics[1].data_corruption_detected is False
    assert any("SUPPLIER_LEAD_TIME_SHOCK" in r for r in diagnostics[1].triggered_reasons)
