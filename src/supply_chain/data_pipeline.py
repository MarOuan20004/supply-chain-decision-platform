"""Pipeline de donnees SQL et Feature Engineering causal sans fuite temporelle."""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain.config import settings

logger = logging.getLogger(__name__)

SQL_SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS skus (
    sku_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    unit_cost_eur REAL NOT NULL,
    holding_cost_per_day_eur REAL NOT NULL,
    stockout_penalty_per_unit_eur REAL NOT NULL,
    moq_units INTEGER NOT NULL,
    contractual_lead_time_days INTEGER NOT NULL,
    safety_stock_units INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_demand (
    sku_id TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    demand_qty REAL NOT NULL,
    is_promo INTEGER NOT NULL,
    unit_price_eur REAL NOT NULL,
    PRIMARY KEY (sku_id, obs_date),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    sku_id TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    on_hand_qty REAL NOT NULL,
    in_transit_qty REAL NOT NULL,
    reorder_point_s REAL NOT NULL,
    order_up_to_S REAL NOT NULL,
    PRIMARY KEY (sku_id, obs_date),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
);

CREATE TABLE IF NOT EXISTS supplier_deliveries (
    order_id TEXT PRIMARY KEY,
    sku_id TEXT NOT NULL,
    order_date TEXT NOT NULL,
    expected_delivery_date TEXT NOT NULL,
    actual_delivery_date TEXT NOT NULL,
    ordered_qty INTEGER NOT NULL,
    delivered_qty INTEGER NOT NULL,
    actual_lead_time_days INTEGER NOT NULL,
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
);

CREATE INDEX IF NOT EXISTS idx_demand_date ON daily_demand(obs_date);
CREATE INDEX IF NOT EXISTS idx_inv_date ON inventory_snapshots(obs_date);
CREATE INDEX IF NOT EXISTS idx_deliv_sku_actual
ON supplier_deliveries(sku_id, actual_delivery_date);
"""


def generate_synthetic_supply_chain_db(
    db_path: Path = settings.sqlite_db_path,
    num_skus: int = settings.num_skus,
    num_days: int = settings.num_days,
    seed: int = settings.random_seed,
) -> dict[str, int]:
    """Initialise la base SQLite et insere un jeu de donnees Supply Chain deterministe."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = np.random.default_rng(seed)
    start_date = date(2025, 1, 1)
    categories = [
        "Industrial_Sensors",
        "ColdChain_Pharma",
        "Power_Electronics",
        "Precision_Optics",
    ]

    sku_rows: list[tuple] = []
    demand_rows: list[tuple] = []
    inventory_rows: list[tuple] = []
    delivery_rows: list[tuple] = []

    order_counter = 1

    for s_idx in range(1, num_skus + 1):
        sku_id = f"SKU-{s_idx:03d}"
        cat = categories[(s_idx - 1) % len(categories)]
        unit_cost = round(float(rng.uniform(15.0, 220.0)), 2)
        holding_cost = round(unit_cost * 0.002, 4)
        penalty_cost = round(unit_cost * 0.45, 2)
        lead_time_nom = int(rng.integers(3, 10))
        base_daily_demand = float(rng.uniform(12.0, 65.0))
        moq = int(max(20, round(base_daily_demand * lead_time_nom * 0.6)))
        safety_stock = int(round(base_daily_demand * (lead_time_nom**0.5) * 1.4))
        reorder_s = float(safety_stock + base_daily_demand * lead_time_nom)
        order_up_to_S = float(reorder_s + max(moq, base_daily_demand * 8.0))

        sku_rows.append(
            (
                sku_id,
                cat,
                unit_cost,
                holding_cost,
                penalty_cost,
                moq,
                lead_time_nom,
                safety_stock,
            )
        )

        on_hand = float(order_up_to_S * 0.85)
        active_shipments: list[tuple[int, int]] = []

        for d_idx in range(num_days):
            curr_date = start_date + timedelta(days=d_idx)
            curr_date_str = curr_date.isoformat()

            arrived_qty = sum(qty for arr_day, qty in active_shipments if arr_day == d_idx)
            active_shipments = [
                (arr_day, qty) for arr_day, qty in active_shipments if arr_day > d_idx
            ]
            on_hand += float(arrived_qty)

            dow = curr_date.weekday()
            weekly_factor = 1.20 if dow in (0, 1, 2) else (0.65 if dow >= 5 else 1.0)
            yearly_factor = 1.0 + 0.18 * math.sin(2.0 * math.pi * d_idx / 365.25)
            is_promo = 1 if (rng.random() < 0.08) else 0
            promo_multiplier = 1.65 if is_promo == 1 else 1.0
            unit_price = round(unit_cost * (1.25 if is_promo == 1 else 1.45), 2)

            expected_d = base_daily_demand * weekly_factor * yearly_factor * promo_multiplier
            observed_d = float(rng.gamma(shape=4.5, scale=expected_d / 4.5))
            observed_d = round(max(0.0, observed_d), 2)

            on_hand = round(max(0.0, on_hand - observed_d), 2)
            in_transit = float(sum(qty for _, qty in active_shipments))

            inventory_position = on_hand + in_transit
            if inventory_position <= reorder_s:
                raw_order = max(moq, int(round(order_up_to_S - inventory_position)))
                delay_shock = int(
                    rng.choice([0, 0, 0, 1, 2, 4], p=[0.50, 0.15, 0.10, 0.12, 0.08, 0.05])
                )
                actual_lt = max(1, lead_time_nom + delay_shock)
                deliv_day_idx = d_idx + actual_lt
                expected_deliv_date = (curr_date + timedelta(days=lead_time_nom)).isoformat()
                actual_deliv_date = (curr_date + timedelta(days=actual_lt)).isoformat()
                delivered_qty = int(round(raw_order * float(rng.uniform(0.95, 1.0))))

                order_id = f"ORD-{order_counter:06d}"
                order_counter += 1
                delivery_rows.append(
                    (
                        order_id,
                        sku_id,
                        curr_date_str,
                        expected_deliv_date,
                        actual_deliv_date,
                        raw_order,
                        delivered_qty,
                        actual_lt,
                    )
                )
                active_shipments.append((deliv_day_idx, delivered_qty))
                in_transit += float(delivered_qty)

            demand_rows.append((sku_id, curr_date_str, observed_d, is_promo, unit_price))
            inventory_rows.append(
                (sku_id, curr_date_str, on_hand, round(in_transit, 2), reorder_s, order_up_to_S)
            )

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(SQL_SCHEMA_DDL)
        conn.executemany(
            "INSERT INTO skus VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sku_rows,
        )
        conn.executemany(
            "INSERT INTO daily_demand VALUES (?, ?, ?, ?, ?)",
            demand_rows,
        )
        conn.executemany(
            "INSERT INTO inventory_snapshots VALUES (?, ?, ?, ?, ?, ?)",
            inventory_rows,
        )
        conn.executemany(
            "INSERT INTO supplier_deliveries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            delivery_rows,
        )
        conn.commit()

    counts = {
        "skus": len(sku_rows),
        "daily_demand": len(demand_rows),
        "inventory_snapshots": len(inventory_rows),
        "supplier_deliveries": len(delivery_rows),
    }
    logger.info("Base SQLite initialisee dans %s : %s", db_path, counts)
    return counts


def load_raw_tables_from_sql(
    db_path: Path = settings.sqlite_db_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrait par requete SQL jointe la serie journaliere et les livraisons."""
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Base SQLite introuvable: {db_path}. Executez generate_synthetic_supply_chain_db()."
        )

    daily_query = """
    SELECT
        d.sku_id,
        d.obs_date,
        d.demand_qty,
        d.is_promo,
        d.unit_price_eur,
        i.on_hand_qty,
        i.in_transit_qty,
        i.reorder_point_s,
        i.order_up_to_S,
        s.category,
        s.unit_cost_eur,
        s.holding_cost_per_day_eur,
        s.stockout_penalty_per_unit_eur,
        s.moq_units,
        s.contractual_lead_time_days,
        s.safety_stock_units
    FROM daily_demand d
    INNER JOIN inventory_snapshots i
        ON d.sku_id = i.sku_id AND d.obs_date = i.obs_date
    INNER JOIN skus s
        ON d.sku_id = s.sku_id
    ORDER BY d.sku_id ASC, d.obs_date ASC;
    """

    deliveries_query = """
    SELECT
        order_id,
        sku_id,
        order_date,
        expected_delivery_date,
        actual_delivery_date,
        ordered_qty,
        delivered_qty,
        actual_lead_time_days
    FROM supplier_deliveries
    ORDER BY sku_id ASC, actual_delivery_date ASC;
    """

    with sqlite3.connect(str(db_path)) as conn:
        daily_df = pd.read_sql_query(daily_query, conn)
        deliveries_df = pd.read_sql_query(deliveries_query, conn)

    daily_df["obs_date"] = pd.to_datetime(daily_df["obs_date"])
    deliveries_df["order_date"] = pd.to_datetime(deliveries_df["order_date"])
    deliveries_df["expected_delivery_date"] = pd.to_datetime(
        deliveries_df["expected_delivery_date"]
    )
    deliveries_df["actual_delivery_date"] = pd.to_datetime(deliveries_df["actual_delivery_date"])

    return daily_df, deliveries_df


def compute_causal_supplier_features(
    daily_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule les statistiques de delais fournisseurs strictement causales."""
    out_frames: list[pd.DataFrame] = []

    deliveries_by_sku = {
        sku: grp.sort_values("actual_delivery_date").reset_index(drop=True)
        for sku, grp in deliveries_df.groupby("sku_id", sort=False)
    }

    for sku_id, sku_daily in daily_df.groupby("sku_id", sort=False):
        sku_daily = sku_daily.sort_values("obs_date").copy()
        nominal_lt = float(sku_daily["contractual_lead_time_days"].iloc[0])

        deliv = deliveries_by_sku.get(sku_id)
        if deliv is None or deliv.empty:
            sku_daily["causal_lead_time_mean"] = nominal_lt
            sku_daily["causal_lead_time_std"] = 0.0
            sku_daily["causal_fill_rate_mean"] = 1.0
            out_frames.append(sku_daily)
            continue

        deliv = deliv.copy()
        deliv["fill_rate"] = deliv["delivered_qty"] / deliv["ordered_qty"].clip(lower=1)

        actual_dates = deliv["actual_delivery_date"].to_numpy()
        lt_values = deliv["actual_lead_time_days"].to_numpy(dtype=np.float64)
        fr_values = deliv["fill_rate"].to_numpy(dtype=np.float64)

        obs_dates = sku_daily["obs_date"].to_numpy()
        cutoff_indices = np.searchsorted(actual_dates, obs_dates, side="right")

        lt_means = np.empty(len(obs_dates), dtype=np.float64)
        lt_stds = np.empty(len(obs_dates), dtype=np.float64)
        fr_means = np.empty(len(obs_dates), dtype=np.float64)

        for i, c_idx in enumerate(cutoff_indices):
            if c_idx == 0:
                lt_means[i] = nominal_lt
                lt_stds[i] = 0.0
                fr_means[i] = 1.0
            else:
                start_w = max(0, int(c_idx) - 10)
                window_lt = lt_values[start_w:c_idx]
                window_fr = fr_values[start_w:c_idx]
                lt_means[i] = float(np.mean(window_lt))
                lt_stds[i] = float(np.std(window_lt)) if len(window_lt) > 1 else 0.0
                fr_means[i] = float(np.mean(window_fr))

        sku_daily["causal_lead_time_mean"] = np.round(lt_means, 4)
        sku_daily["causal_lead_time_std"] = np.round(lt_stds, 4)
        sku_daily["causal_fill_rate_mean"] = np.round(fr_means, 4)
        out_frames.append(sku_daily)

    return pd.concat(out_frames, axis=0, ignore_index=True)


FEATURE_COLUMNS: list[str] = [
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_roll_mean_7",
    "demand_roll_std_7",
    "demand_roll_mean_14",
    "demand_roll_std_14",
    "demand_roll_mean_28",
    "demand_roll_std_28",
    "promo_roll_sum_7",
    "is_promo",
    "unit_price_eur",
    "on_hand_qty",
    "in_transit_qty",
    "inventory_position",
    "days_of_supply_est",
    "stock_to_reorder_ratio",
    "causal_lead_time_mean",
    "causal_lead_time_std",
    "causal_fill_rate_mean",
    "lead_time_drift",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]


def build_causal_feature_Frame(
    daily_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    horizon_days: int = settings.default_horizon_days,
    drop_incomplete_targets: bool = True,
) -> pd.DataFrame:
    """Construit la matrice de features causales X_t et les cibles a horizon H."""
    if horizon_days < 1:
        raise ValueError(f"horizon_days doit etre >= 1 (recu: {horizon_days})")

    df = compute_causal_supplier_features(daily_df, deliveries_df)
    df = df.sort_values(["sku_id", "obs_date"]).reset_index(drop=True)

    grp = df.groupby("sku_id", sort=False)["demand_qty"]

    df["demand_lag_1"] = grp.shift(0)
    df["demand_lag_7"] = grp.shift(6)
    df["demand_lag_14"] = grp.shift(13)

    for w in (7, 14, 28):
        df[f"demand_roll_mean_{w}"] = (
            grp.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"demand_roll_std_{w}"] = (
            grp.rolling(window=w, min_periods=2).std().fillna(0.0).reset_index(level=0, drop=True)
        )

    df["promo_roll_sum_7"] = (
        df.groupby("sku_id", sort=False)["is_promo"]
        .rolling(window=7, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["inventory_position"] = df["on_hand_qty"] + df["in_transit_qty"]
    safe_daily_rate = df["demand_roll_mean_14"].clip(lower=0.5)
    df["days_of_supply_est"] = df["inventory_position"] / safe_daily_rate
    df["stock_to_reorder_ratio"] = df["inventory_position"] / df["reorder_point_s"].clip(lower=1.0)
    df["lead_time_drift"] = df["causal_lead_time_mean"] - df["contractual_lead_time_days"]

    dow = df["obs_date"].dt.dayofweek.to_numpy(dtype=np.float64)
    doy = df["obs_date"].dt.dayofyear.to_numpy(dtype=np.float64)
    df["dow_sin"] = np.sin(2.0 * math.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * math.pi * dow / 7.0)
    df["doy_sin"] = np.sin(2.0 * math.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * math.pi * doy / 365.25)

    for col in ("demand_lag_7", "demand_lag_14"):
        df[col] = df[col].fillna(df["demand_roll_mean_7"])

    future_sum_series: list[pd.Series] = []
    for _, sku_grp in df.groupby("sku_id", sort=False):
        d_vals = sku_grp["demand_qty"]
        fut_sum = sum(d_vals.shift(-step) for step in range(1, horizon_days + 1))
        future_sum_series.append(fut_sum)

    df["future_cum_demand_h"] = pd.concat(future_sum_series).sort_index()
    df["target_demand_next_h"] = df["future_cum_demand_h"] / float(horizon_days)

    effective_h = min(float(horizon_days), 7.0)
    projected_ending_stock = df["inventory_position"] - df["target_demand_next_h"] * effective_h
    df["target_stockout_criticality"] = (projected_ending_stock < df["safety_stock_units"]).astype(
        int
    )

    if drop_incomplete_targets:
        df = df.dropna(subset=["target_demand_next_h"]).reset_index(drop=True)

    return df


build_causal_feature_frame = build_causal_feature_Frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialisation SQL et verification du pipeline Supply Chain."
    )
    parser.add_argument(
        "--init-db", action="store_true", help="Genere la base SQLite supply_chain.db"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    if args.init_db or not settings.sqlite_db_path.is_file():
        generate_synthetic_supply_chain_db()

    daily_df, deliv_df = load_raw_tables_from_sql()
    feat_df = build_causal_feature_Frame(
        daily_df, deliv_df, horizon_days=settings.default_horizon_days
    )
    logger.info(
        "Dataset causal : %d lignes, %d features, taux criticite=%.2f%%",
        len(feat_df),
        len(FEATURE_COLUMNS),
        100.0 * float(feat_df["target_stockout_criticality"].mean()),
    )


if __name__ == "__main__":
    main()
