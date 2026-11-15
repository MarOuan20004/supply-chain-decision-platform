"""Tests automatises anti-fuite temporelle par mutation synthetique du futur."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supply_chain.data_pipeline import (
    FEATURE_COLUMNS,
    build_causal_feature_Frame,
    generate_synthetic_supply_chain_db,
    load_raw_tables_from_sql,
)
from supply_chain.schemas import (
    DailyDemandRecord,
    InventorySnapshotRecord,
    SKUMasterRecord,
    SupplierDeliveryRecord,
)
from supply_chain.validation import (
    CausalMovingAverageBaseline,
    ExpandingWindowTimeSplit,
    SeasonalNaiveBaseline,
    evaluate_baseline_across_splits,
)


@pytest.fixture(scope="module")
def sample_sql_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cree une base SQLite temporaire de 8 SKUs sur 260 jours pour des tests deterministes."""
    db_file: Path = tmp_path_factory.mktemp("supply_db") / "test_supply_chain.db"
    counts = generate_synthetic_supply_chain_db(
        db_path=db_file,
        num_skus=8,
        num_days=260,
        seed=123,
    )
    assert counts["skus"] == 8
    assert counts["daily_demand"] == 8 * 260
    return load_raw_tables_from_sql(db_file)


def test_future_demand_and_stock_mutation_does_not_alter_past_features(
    sample_sql_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """PREUVE: Mutater violemment y_{t+k} et stock_{t+k} pour k >= 1 n'altere aucune feature X_t."""
    daily_df, deliv_df = sample_sql_data

    orig_feat_df = build_causal_feature_Frame(
        daily_df, deliv_df, horizon_days=7, drop_incomplete_targets=False
    )

    cutoff_date = pd.Timestamp("2025-05-15")

    mutated_daily = daily_df.copy()
    future_mask = mutated_daily["obs_date"] > cutoff_date
    assert int(future_mask.sum()) > 100

    mutated_daily.loc[future_mask, "demand_qty"] = (
        mutated_daily.loc[future_mask, "demand_qty"] * 1000.0 + 1_000_000.0
    )
    mutated_daily.loc[future_mask, "on_hand_qty"] = 999_999.0
    mutated_daily.loc[future_mask, "in_transit_qty"] = 888_888.0
    mutated_daily.loc[future_mask, "is_promo"] = 1
    mutated_daily.loc[future_mask, "unit_price_eur"] = 0.01

    mutated_feat_df = build_causal_feature_Frame(
        mutated_daily, deliv_df, horizon_days=7, drop_incomplete_targets=False
    )

    past_orig = orig_feat_df[orig_feat_df["obs_date"] <= cutoff_date].reset_index(drop=True)
    past_mut = mutated_feat_df[mutated_feat_df["obs_date"] <= cutoff_date].reset_index(drop=True)

    assert len(past_orig) == len(past_mut) and len(past_orig) > 0

    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            past_orig[col].to_numpy(dtype=np.float64),
            past_mut[col].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"Fuite temporelle detectee sur la feature '{col}' avant {cutoff_date}!",
        )


def test_future_supplier_delivery_mutation_does_not_leak_into_causal_lead_time(
    sample_sql_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifie qu'une commande livree apres t ne fuite pas son delai reel a la date t."""
    daily_df, deliv_df = sample_sql_data
    cutoff_date = pd.Timestamp("2025-05-15")

    orig_feat_df = build_causal_feature_Frame(
        daily_df, deliv_df, horizon_days=7, drop_incomplete_targets=False
    )

    mutated_deliv = deliv_df.copy()
    future_arrival_mask = mutated_deliv["actual_delivery_date"] > cutoff_date
    assert int(future_arrival_mask.sum()) > 10

    mutated_deliv.loc[future_arrival_mask, "actual_lead_time_days"] = 9999
    mutated_deliv.loc[future_arrival_mask, "delivered_qty"] = 0

    mutated_feat_df = build_causal_feature_Frame(
        daily_df, mutated_deliv, horizon_days=7, drop_incomplete_targets=False
    )

    past_orig = orig_feat_df[orig_feat_df["obs_date"] <= cutoff_date].reset_index(drop=True)
    past_mut = mutated_feat_df[mutated_feat_df["obs_date"] <= cutoff_date].reset_index(drop=True)

    eval_cols = (
        "causal_lead_time_mean",
        "causal_lead_time_std",
        "causal_fill_rate_mean",
        "lead_time_drift",
    )
    for col in eval_cols:
        np.testing.assert_allclose(
            past_orig[col].to_numpy(dtype=np.float64),
            past_mut[col].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"Fuite de delai fournisseur futur detectee sur '{col}'!",
        )


def test_expanding_window_time_split_strict_chronology_and_purge_gap(
    sample_sql_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifie que les 4 plis sont strictement croissants et respectent le gap H=7 jours."""
    daily_df, deliv_df = sample_sql_data
    horizon = 7
    feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=horizon)

    splitter = ExpandingWindowTimeSplit(
        n_splits=4,
        val_window_days=21,
        purge_gap_days=horizon,
        min_train_days=100,
    )

    splits = list(splitter.split(feat_df, date_col="obs_date"))
    assert len(splits) == 4

    prev_train_size = 0
    for fold in splits:
        overlap = np.intersect1d(fold.train_indices, fold.val_indices)
        assert overlap.size == 0

        gap_days = (fold.val_start_date - fold.train_end_date).days
        assert gap_days > horizon, (
            f"Embargo temporel viole sur le pli {fold.fold_index}: gap={gap_days} <= {horizon}"
        )

        assert fold.train_indices.size > prev_train_size
        prev_train_size = fold.train_indices.size


def test_baselines_evaluation_end_to_end_produces_valid_metrics(
    sample_sql_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifie que les deux baselines s'entrainent et produisent des metriques valides."""
    daily_df, deliv_df = sample_sql_data
    feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=7)

    splitter = ExpandingWindowTimeSplit(
        n_splits=4,
        val_window_days=21,
        purge_gap_days=7,
        min_train_days=100,
    )

    for baseline in (SeasonalNaiveBaseline(), CausalMovingAverageBaseline()):
        summary = evaluate_baseline_across_splits(
            df=feat_df,
            splitter=splitter,
            baseline_model=baseline,
            horizon_days=7,
        )
        assert summary.n_splits == 4
        assert summary.mase_mean > 0.0
        assert 0.0 < summary.wape_mean < 2.0
        assert 0.0 <= summary.pr_auc_mean <= 1.0
        assert 0.0 <= summary.brier_score_mean <= 1.0
        assert 0.0 <= summary.ece_mean <= 1.0
        for fold in summary.folds:
            assert fold.pinball_q10 >= 0.0
            assert fold.pinball_q50 >= 0.0
            assert fold.pinball_q90 >= 0.0


def test_expanding_window_edge_cases() -> None:
    """Verifie la gestion des cas limites et erreurs d'ExpandingWindowTimeSplit."""
    with pytest.raises(ValueError, match="n_splits doit etre >= 2"):
        ExpandingWindowTimeSplit(n_splits=1)

    with pytest.raises(ValueError, match="val_window_days doit etre >= 1"):
        ExpandingWindowTimeSplit(val_window_days=0)

    with pytest.raises(ValueError, match="purge_gap_days doit etre >= 0"):
        ExpandingWindowTimeSplit(purge_gap_days=-1)

    splitter = ExpandingWindowTimeSplit(
        n_splits=4, val_window_days=30, purge_gap_days=7, min_train_days=100
    )
    tiny_df = pd.DataFrame({"obs_date": pd.date_range("2025-01-01", periods=50)})

    with pytest.raises(ValueError, match="Historique insuffisant"):
        list(splitter.split(tiny_df))

    with pytest.raises(KeyError, match="Colonne de date 'missing_col' absente"):
        list(splitter.split(tiny_df, date_col="missing_col"))


def test_data_pipeline_validation_errors(tmp_path: Path) -> None:
    """Verifie les leves d'exceptions dans les fonctions de pipeline de donnees."""
    missing_db = tmp_path / "non_existent.db"
    with pytest.raises(FileNotFoundError, match="Base SQLite introuvable"):
        load_raw_tables_from_sql(missing_db)

    dummy_daily = pd.DataFrame({"obs_date": [pd.Timestamp("2025-01-01")]})
    dummy_deliv = pd.DataFrame({"actual_delivery_date": [pd.Timestamp("2025-01-01")]})
    with pytest.raises(ValueError, match="horizon_days doit etre >= 1"):
        build_causal_feature_Frame(dummy_daily, dummy_deliv, horizon_days=0)


def test_pydantic_schema_contracts() -> None:
    """Verifie les contrats de types Pydantic v2 pour les enregistrements SQL."""
    sku = SKUMasterRecord(
        sku_id="SKU-001",
        category="Electronics",
        unit_cost_eur=50.0,
        holding_cost_per_day_eur=0.1,
        stockout_penalty_per_unit_eur=25.0,
        moq_units=20,
        contractual_lead_time_days=5,
        safety_stock_units=30,
    )
    assert sku.sku_id == "SKU-001"
    assert sku.unit_cost_eur == 50.0

    demand = DailyDemandRecord(
        sku_id="SKU-001",
        obs_date=date(2025, 1, 1),
        demand_qty=15.5,
        is_promo=0,
        unit_price_eur=72.5,
    )
    assert demand.demand_qty == 15.5

    inv = InventorySnapshotRecord(
        sku_id="SKU-001",
        obs_date=date(2025, 1, 1),
        on_hand_qty=100.0,
        in_transit_qty=50.0,
        reorder_point_s=60.0,
        order_up_to_S=150.0,
    )
    assert inv.on_hand_qty == 100.0

    deliv = SupplierDeliveryRecord(
        order_id="ORD-000001",
        sku_id="SKU-001",
        order_date=date(2025, 1, 1),
        expected_delivery_date=date(2025, 1, 6),
        actual_delivery_date=date(2025, 1, 7),
        ordered_qty=50,
        delivered_qty=50,
        actual_lead_time_days=6,
    )
    assert deliv.actual_lead_time_days == 6
