"""Validation temporelle ExpandingWindowTimeSplit, metriques MLOps et Baselines causales."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from supply_chain.data_pipeline import FEATURE_COLUMNS
from supply_chain.schemas import BaselineComparisonSummary, FoldMetrics


@dataclass(frozen=True)
class TemporalFoldSplit:
    """Represente les indices et bornes de dates d'un pli temporel sans fuite."""

    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    val_start_date: pd.Timestamp
    val_end_date: pd.Timestamp


class ExpandingWindowTimeSplit:
    """Decoupage temporel a fenetre croissante avec embargo (purge_gap_days) anti-chevauchement."""

    def __init__(
        self,
        n_splits: int = 4,
        val_window_days: int = 28,
        purge_gap_days: int = 7,
        min_train_days: int = 120,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits doit etre >= 2.")
        if val_window_days < 1:
            raise ValueError("val_window_days doit etre >= 1.")
        if purge_gap_days < 0:
            raise ValueError("purge_gap_days doit etre >= 0.")
        self.n_splits = int(n_splits)
        self.val_window_days = int(val_window_days)
        self.purge_gap_days = int(purge_gap_days)
        self.min_train_days = int(min_train_days)

    def split(self, df: pd.DataFrame, date_col: str = "obs_date") -> Iterator[TemporalFoldSplit]:
        """Genere les `n_splits` plis temporels ordonnes chronologiquement."""
        if date_col not in df.columns:
            raise KeyError(f"Colonne de date '{date_col}' absente du DataFrame.")

        unique_dates = np.sort(pd.to_datetime(df[date_col]).unique())
        total_days = len(unique_dates)
        required_days = (
            self.min_train_days + self.purge_gap_days + self.n_splits * self.val_window_days
        )
        if total_days < required_days:
            raise ValueError(
                f"Historique insuffisant ({total_days} jours uniques) pour {self.n_splits} plis "
                f"(requis >= {required_days} jours)."
            )

        first_val_start_idx = total_days - self.n_splits * self.val_window_days
        obs_dates_series = pd.to_datetime(df[date_col])

        for fold_idx in range(self.n_splits):
            val_start_pos = first_val_start_idx + fold_idx * self.val_window_days
            val_end_pos = val_start_pos + self.val_window_days - 1
            train_end_pos = val_start_pos - self.purge_gap_days - 1

            if train_end_pos < self.min_train_days - 1:
                raise ValueError(f"Pli {fold_idx}: train_end_pos={train_end_pos} < min_train_days.")

            train_start_dt = pd.Timestamp(unique_dates[0])
            train_end_dt = pd.Timestamp(unique_dates[train_end_pos])
            val_start_dt = pd.Timestamp(unique_dates[val_start_pos])
            val_end_dt = pd.Timestamp(unique_dates[val_end_pos])

            train_mask = (obs_dates_series >= train_start_dt) & (obs_dates_series <= train_end_dt)
            val_mask = (obs_dates_series >= val_start_dt) & (obs_dates_series <= val_end_dt)

            train_idx = np.flatnonzero(train_mask.to_numpy())
            val_idx = np.flatnonzero(val_mask.to_numpy())

            yield TemporalFoldSplit(
                fold_index=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                train_start_date=train_start_dt,
                train_end_date=train_end_dt,
                val_start_date=val_start_dt,
                val_end_date=val_end_dt,
            )


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_naive_diffs: np.ndarray,
) -> float:
    """Calcule le Mean Absolute Scaled Error (MASE)."""
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    scale = float(np.mean(np.abs(y_train_naive_diffs)))
    if scale < 1e-9:
        return 0.0
    return mae_val / scale


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le Weighted Absolute Percentage Error (WAPE)."""
    denom = float(np.sum(np.abs(y_true)))
    if denom < 1e-9:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred))) / denom


def compute_pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Calcule la Pinball Loss (Quantile Loss) pour le quantile d'ordre alpha in (0, 1)."""
    residual = y_true - q_pred
    loss = np.maximum(alpha * residual, (alpha - 1.0) * residual)
    return float(np.mean(loss))


def compute_expected_calibration_error(
    y_true_bin: np.ndarray,
    prob_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcule l'Expected Calibration Error (ECE) sur n_bins bacs uniformes dans [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true_bin)
    if n_total == 0:
        return 0.0

    for b_idx in range(n_bins):
        low, high = bins[b_idx], bins[b_idx + 1]
        if b_idx == n_bins - 1:
            mask = (prob_pred >= low) & (prob_pred <= high)
        else:
            mask = (prob_pred >= low) & (prob_pred < high)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(prob_pred[mask]))
            avg_acc = float(np.mean(y_true_bin[mask]))
            ece += (count / n_total) * abs(avg_acc - avg_conf)
    return float(ece)


class SeasonalNaiveBaseline:
    """Baseline 1 : Seasonal Naive (lag 7 jours) + Regresseur Logistique L2 sans fuite."""

    def __init__(self) -> None:
        self.model_name = "baseline_seasonal_naive_lag7"
        self.residual_q10_offset_: float = -5.0
        self.residual_q90_offset_: float = 5.0
        self.clf_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> SeasonalNaiveBaseline:
        y_train = train_df["target_demand_next_h"].to_numpy(dtype=np.float64)
        pred_train = train_df["demand_lag_7"].to_numpy(dtype=np.float64)
        residuals = y_train - pred_train
        self.residual_q10_offset_ = float(np.percentile(residuals, 10))
        self.residual_q90_offset_ = float(np.percentile(residuals, 90))

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        c_train = train_df["target_stockout_criticality"].to_numpy(dtype=np.int32)
        if len(np.unique(c_train)) < 2:
            self.single_class_proba_: float | None = float(c_train[0]) if len(c_train) > 0 else 1.0
        else:
            self.single_class_proba_ = None
            self.clf_pipeline.fit(X_train, c_train)
        return self

    def predict_quantiles(self, val_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q50 = val_df["demand_lag_7"].to_numpy(dtype=np.float64)
        q10 = np.maximum(0.0, q50 + self.residual_q10_offset_)
        q90 = np.maximum(q50, q50 + self.residual_q90_offset_)
        return q10, q50, q90

    def predict_criticality_proba(self, val_df: pd.DataFrame) -> np.ndarray:
        if getattr(self, "single_class_proba_", None) is not None:
            return np.full(len(val_df), self.single_class_proba_, dtype=np.float64)
        X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        return self.clf_pipeline.predict_proba(X_val)[:, 1]


class CausalMovingAverageBaseline:
    """Baseline 2 : Moyenne Mobile Causale 14j + Regresseur Logistique L2 sans fuite."""

    def __init__(self) -> None:
        self.model_name = "baseline_moving_average_14d"
        self.single_class_proba_: float | None = None
        self.clf_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> CausalMovingAverageBaseline:
        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        c_train = train_df["target_stockout_criticality"].to_numpy(dtype=np.int32)
        if len(np.unique(c_train)) < 2:
            self.single_class_proba_ = float(c_train[0]) if len(c_train) > 0 else 1.0
        else:
            self.single_class_proba_ = None
            self.clf_pipeline.fit(X_train, c_train)
        return self

    def predict_quantiles(self, val_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q50 = val_df["demand_roll_mean_14"].to_numpy(dtype=np.float64)
        roll_std = val_df["demand_roll_std_14"].to_numpy(dtype=np.float64)
        q10 = np.maximum(0.0, q50 - 1.28155 * roll_std)
        q90 = np.maximum(q50, q50 + 1.28155 * roll_std)
        return q10, q50, q90

    def predict_criticality_proba(self, val_df: pd.DataFrame) -> np.ndarray:
        if getattr(self, "single_class_proba_", None) is not None:
            return np.full(len(val_df), self.single_class_proba_, dtype=np.float64)
        X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        return self.clf_pipeline.predict_proba(X_val)[:, 1]


def evaluate_baseline_across_splits(
    df: pd.DataFrame,
    splitter: ExpandingWindowTimeSplit,
    baseline_model: SeasonalNaiveBaseline | CausalMovingAverageBaseline,
    horizon_days: int = 7,
) -> BaselineComparisonSummary:
    """Evalue un modele baseline sur les plis temporels d'ExpandingWindowTimeSplit."""
    fold_results: list[FoldMetrics] = []

    for split_info in splitter.split(df, date_col="obs_date"):
        train_df = df.iloc[split_info.train_indices]
        val_df = df.iloc[split_info.val_indices]

        baseline_model.fit(train_df)
        q10, q50, q90 = baseline_model.predict_quantiles(val_df)
        prob_crit = baseline_model.predict_criticality_proba(val_df)

        y_val = val_df["target_demand_next_h"].to_numpy(dtype=np.float64)
        c_val = val_df["target_stockout_criticality"].to_numpy(dtype=np.int32)

        naive_diffs = (
            train_df.groupby("sku_id", sort=False)["target_demand_next_h"]
            .diff()
            .dropna()
            .to_numpy(dtype=np.float64)
        )

        mase = compute_mase(y_val, q50, naive_diffs)
        wape = compute_wape(y_val, q50)
        p10 = compute_pinball_loss(y_val, q10, alpha=0.10)
        p50 = compute_pinball_loss(y_val, q50, alpha=0.50)
        p90 = compute_pinball_loss(y_val, q90, alpha=0.90)
        mean_pinball = (p10 + p50 + p90) / 3.0

        pr_auc = float(average_precision_score(c_val, prob_crit)) if np.any(c_val == 1) else 0.0
        brier = float(brier_score_loss(c_val, prob_crit))
        ece = compute_expected_calibration_error(c_val, prob_crit, n_bins=10)

        fold_results.append(
            FoldMetrics(
                fold_index=split_info.fold_index,
                train_start_date=split_info.train_start_date.strftime("%Y-%m-%d"),
                train_end_date=split_info.train_end_date.strftime("%Y-%m-%d"),
                val_start_date=split_info.val_start_date.strftime("%Y-%m-%d"),
                val_end_date=split_info.val_end_date.strftime("%Y-%m-%d"),
                num_train_rows=len(train_df),
                num_val_rows=len(val_df),
                mase=round(mase, 4),
                wape=round(wape, 4),
                pinball_q10=round(p10, 4),
                pinball_q50=round(p50, 4),
                pinball_q90=round(p90, 4),
                mean_pinball=round(mean_pinball, 4),
                pr_auc=round(pr_auc, 4),
                brier_score=round(brier, 4),
                ece=round(ece, 4),
            )
        )

    mase_vals = [f.mase for f in fold_results]
    wape_vals = [f.wape for f in fold_results]
    pin_vals = [f.mean_pinball for f in fold_results]
    prauc_vals = [f.pr_auc for f in fold_results]
    brier_vals = [f.brier_score for f in fold_results]
    ece_vals = [f.ece for f in fold_results]

    return BaselineComparisonSummary(
        model_name=baseline_model.model_name,
        horizon_days=horizon_days,
        n_splits=len(fold_results),
        mase_mean=round(float(np.mean(mase_vals)), 4),
        mase_std=round(float(np.std(mase_vals)), 4),
        wape_mean=round(float(np.mean(wape_vals)), 4),
        wape_std=round(float(np.std(wape_vals)), 4),
        mean_pinball_avg=round(float(np.mean(pin_vals)), 4),
        pr_auc_mean=round(float(np.mean(prauc_vals)), 4),
        pr_auc_std=round(float(np.std(prauc_vals)), 4),
        brier_score_mean=round(float(np.mean(brier_vals)), 4),
        ece_mean=round(float(np.mean(ece_vals)), 4),
        folds=fold_results,
    )
