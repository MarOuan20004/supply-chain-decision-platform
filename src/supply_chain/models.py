"""Modelisation Multi-Horizon CatBoost, Calibration, SHAP et IsolationForest."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, brier_score_loss

from supply_chain.config import SupplyChainSettings, settings
from supply_chain.data_pipeline import FEATURE_COLUMNS
from supply_chain.schemas import (
    AnomalyDiagnosticsPayload,
    BaselineComparisonSummary,
    CalibrationMethodMetrics,
    FoldMetrics,
    MultiQuantilePrediction,
    ShapFeatureContribution,
)
from supply_chain.validation import (
    ExpandingWindowTimeSplit,
    compute_expected_calibration_error,
    compute_mase,
    compute_pinball_loss,
    compute_wape,
)

logger = logging.getLogger(__name__)


def enforce_quantile_monotonicity(raw_quantiles: np.ndarray) -> np.ndarray:
    """Applique la projection isotonique de Chernozhukov (tri par ligne + troncature a 0).

    Garantit 0.0 <= q10 <= q50 <= q90 pour 100% des lignes predites.
    """
    arr = np.asarray(raw_quantiles, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    clipped = np.maximum(0.0, arr)
    return np.sort(clipped, axis=1)


def _extract_top_shap_contributions(
    feature_names: Sequence[str],
    feature_row: np.ndarray,
    shap_row: np.ndarray,
    top_k: int = 5,
) -> list[ShapFeatureContribution]:
    """Ordonne les contributions TreeSHAP par valeur absolue decroissante."""
    indices = np.argsort(np.abs(shap_row))[::-1][:top_k]
    return [
        ShapFeatureContribution(
            feature_name=str(feature_names[idx]),
            feature_value=round(float(feature_row[idx]), 4),
            shap_value=round(float(shap_row[idx]), 4),
        )
        for idx in indices
    ]


class MultiHorizonQuantileForecaster:
    """Regresseur Multi-Horizon / Multi-Quantile CatBoost (alpha = 0.10, 0.50, 0.90)."""

    def __init__(
        self,
        horizons: Sequence[int] = (7, 14, 28),
        cfg: SupplyChainSettings = settings,
        iterations: int | None = None,
        depth: int | None = None,
        learning_rate: float | None = None,
    ) -> None:
        self.horizons = tuple(sorted(set(int(h) for h in horizons)))
        self.cfg = cfg
        self.iterations = int(iterations or cfg.catboost_iterations)
        self.depth = int(depth or cfg.catboost_depth)
        self.learning_rate = float(learning_rate or cfg.catboost_learning_rate)
        self.models_by_horizon: dict[int, CatBoostRegressor] = {}
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)

    def fit_horizon(
        self,
        train_df: pd.DataFrame,
        horizon_days: int,
        val_df: pd.DataFrame | None = None,
    ) -> CatBoostRegressor:
        """Entraine un CatBoostRegressor MultiQuantile pour un horizon H donne."""
        X_train = train_df[self.feature_columns].to_numpy(dtype=np.float64)
        y_train = train_df["target_demand_next_h"].to_numpy(dtype=np.float64)

        loss_spec = "MultiQuantile:alpha=0.1,0.5,0.9"
        model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=loss_spec,
            eval_metric=loss_spec,
            random_seed=self.cfg.random_seed,
            verbose=False,
            allow_writing_files=False,
        )

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_columns].to_numpy(dtype=np.float64)
            y_val = val_df["target_demand_next_h"].to_numpy(dtype=np.float64)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=35,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)

        self.models_by_horizon[int(horizon_days)] = model
        return model

    def predict_quantiles_matrix(
        self,
        df: pd.DataFrame | np.ndarray,
        horizon_days: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne (q10, q50, q90) garantis monotones pour un DataFrame ou ndarray."""
        if horizon_days not in self.models_by_horizon:
            raise KeyError(
                f"Aucun modele CatBoost entraine pour l'horizon H={horizon_days}j. "
                f"Horizons disponibles: {list(self.models_by_horizon.keys())}"
            )
        model = self.models_by_horizon[horizon_days]
        if isinstance(df, pd.DataFrame):
            X = df[self.feature_columns].to_numpy(dtype=np.float64)
        else:
            X = np.asarray(df, dtype=np.float64)

        raw_preds = model.predict(X)
        monotonic_preds = enforce_quantile_monotonicity(raw_preds)
        return monotonic_preds[:, 0], monotonic_preds[:, 1], monotonic_preds[:, 2]

    def explain_local_shap(
        self,
        df: pd.DataFrame,
        horizon_days: int = 7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcule les valeurs TreeSHAP exactes pour la mediane q50."""
        if horizon_days not in self.models_by_horizon:
            raise KeyError(f"Modele non entraine pour H={horizon_days}")
        model = self.models_by_horizon[horizon_days]
        X = df[self.feature_columns].to_numpy(dtype=np.float64)
        pool = Pool(X, feature_names=self.feature_columns)
        raw_shap = np.asarray(model.get_feature_importance(pool, type="ShapValues"))
        if raw_shap.ndim == 3:
            q50_shap = raw_shap[:, 1, :]
        else:
            q50_shap = raw_shap
        return q50_shap[:, :-1], q50_shap[:, -1]

    def build_structured_predictions(
        self,
        df: pd.DataFrame,
        horizon_days: int = 7,
        include_shap: bool = True,
        top_k_shap: int = 5,
    ) -> list[MultiQuantilePrediction]:
        """Produit les objets Pydantic MultiQuantilePrediction avec garde-fou d'incertitude."""
        q10_arr, q50_arr, q90_arr = self.predict_quantiles_matrix(df, horizon_days=horizon_days)
        X = df[self.feature_columns].to_numpy(dtype=np.float64)

        if include_shap:
            shap_mat, _ = self.explain_local_shap(df, horizon_days=horizon_days)
        else:
            shap_mat = np.zeros_like(X)

        results: list[MultiQuantilePrediction] = []
        sku_list = df["sku_id"].astype(str).tolist()
        date_list = pd.to_datetime(df["obs_date"]).dt.strftime("%Y-%m-%d").tolist()

        for idx in range(len(df)):
            q10 = round(float(q10_arr[idx]), 4)
            q50 = max(q10, round(float(q50_arr[idx]), 4))
            q90 = max(q50, round(float(q90_arr[idx]), 4))
            rel_uncert = round((q90 - q10) / max(q50, 1.0), 4)
            drivers = (
                _extract_top_shap_contributions(
                    self.feature_columns, X[idx], shap_mat[idx], top_k=top_k_shap
                )
                if include_shap
                else []
            )
            results.append(
                MultiQuantilePrediction(
                    sku_id=sku_list[idx],
                    obs_date=date_list[idx],
                    horizon_days=horizon_days,
                    q10=q10,
                    q50=q50,
                    q90=q90,
                    relative_uncertainty=rel_uncert,
                    high_uncertainty_flag=bool(rel_uncert > self.cfg.uncertainty_Threshold_tau),
                    top_shap_drivers=drivers,
                )
            )
        return results


def _wrap_prefit_calibrator(
    base_estimator: CatBoostClassifier,
    method: Literal["isotonic", "sigmoid"],
) -> CalibratedClassifierCV:
    """Cree un CalibratedClassifierCV compatible toutes versions scikit-learn."""
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_estimator),
            method=method,
        )
    except Exception:
        return CalibratedClassifierCV(
            estimator=base_estimator,
            method=method,
            cv="prefit",
        )


class CalibratedCriticalityClassifier:
    """Classifieur de criticite CatBoost avec calibration temporelle sans fuite."""

    def __init__(
        self,
        cfg: SupplyChainSettings = settings,
        iterations: int | None = None,
        depth: int | None = None,
        learning_rate: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.iterations = int(iterations or cfg.catboost_iterations)
        self.depth = int(depth or cfg.catboost_depth)
        self.learning_rate = float(learning_rate or cfg.catboost_learning_rate)
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)

        self.base_model_: CatBoostClassifier | None = None
        self.isotonic_calibrator_: CalibratedClassifierCV | None = None
        self.sigmoid_calibrator_: CalibratedClassifierCV | None = None
        self.selected_method_: Literal["isotonic", "sigmoid"] = "sigmoid"

    @staticmethod
    def _split_train_into_fit_and_calib_chronological(
        train_df: pd.DataFrame,
        holdout_frac: float = 0.22,
        purge_gap_days: int = 7,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Decoupe chronologiquement le pli d'entrainement en (T_fit, embargo H, T_cal)."""
        unique_dates = np.sort(pd.to_datetime(train_df["obs_date"]).unique())
        n_dates = len(unique_dates)
        n_cal_dates = max(14, int(round(n_dates * holdout_frac)))
        cal_start_idx = max(30, n_dates - n_cal_dates)
        fit_end_idx = max(15, cal_start_idx - purge_gap_days - 1)

        fit_end_dt = pd.Timestamp(unique_dates[fit_end_idx])
        cal_start_dt = pd.Timestamp(unique_dates[cal_start_idx])

        obs_dt = pd.to_datetime(train_df["obs_date"])
        fit_sub = train_df[obs_dt <= fit_end_dt].reset_index(drop=True)
        cal_sub = train_df[obs_dt >= cal_start_dt].reset_index(drop=True)

        if len(cal_sub) < 20 or cal_sub["target_stockout_criticality"].nunique() < 2:
            split_pt = int(len(train_df) * (1.0 - holdout_frac))
            fit_sub = train_df.iloc[:split_pt].reset_index(drop=True)
            cal_sub = train_df.iloc[split_pt:].reset_index(drop=True)

        return fit_sub, cal_sub

    def fit(
        self,
        train_df: pd.DataFrame,
        horizon_days: int = 7,
        preferred_method: Literal["auto", "isotonic", "sigmoid"] = "auto",
    ) -> CalibratedCriticalityClassifier:
        """Entraine CatBoostClassifier sur T_fit puis calibre Isotonique et Platt sur T_cal."""
        fit_df, cal_df = self._split_train_into_fit_and_calib_chronological(
            train_df,
            holdout_frac=self.cfg.calibration_holdout_frac,
            purge_gap_days=horizon_days,
        )

        X_fit = fit_df[self.feature_columns].to_numpy(dtype=np.float64)
        y_fit = fit_df["target_stockout_criticality"].to_numpy(dtype=np.int32)

        if len(np.unique(y_fit)) < 2:
            self.single_class_proba_: float | None = float(y_fit[0]) if len(y_fit) > 0 else 1.0
            return self
        self.single_class_proba_ = None

        X_cal = cal_df[self.feature_columns].to_numpy(dtype=np.float64)
        y_cal = cal_df["target_stockout_criticality"].to_numpy(dtype=np.int32)

        base_clf = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=self.cfg.random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        base_clf.fit(
            X_fit,
            y_fit,
            eval_set=(X_cal, y_cal),
            early_stopping_rounds=35,
            verbose=False,
        )
        self.base_model_ = base_clf

        iso_cal = _wrap_prefit_calibrator(base_clf, method="isotonic")
        iso_cal.fit(X_cal, y_cal)
        self.isotonic_calibrator_ = iso_cal

        sig_cal = _wrap_prefit_calibrator(base_clf, method="sigmoid")
        sig_cal.fit(X_cal, y_cal)
        self.sigmoid_calibrator_ = sig_cal

        if preferred_method in ("isotonic", "sigmoid"):
            self.selected_method_ = preferred_method
        else:
            prob_iso = np.clip(iso_cal.predict_proba(X_cal)[:, 1], 0.0, 1.0)
            prob_sig = np.clip(sig_cal.predict_proba(X_cal)[:, 1], 0.0, 1.0)
            score_iso = float(brier_score_loss(y_cal, prob_iso)) + 0.5 * (
                compute_expected_calibration_error(y_cal, prob_iso)
            )
            score_sig = float(brier_score_loss(y_cal, prob_sig)) + 0.5 * (
                compute_expected_calibration_error(y_cal, prob_sig)
            )
            if len(y_cal) >= 1200 and score_iso < 0.85 * score_sig:
                self.selected_method_ = "isotonic"
            else:
                self.selected_method_ = "sigmoid"

        return self

    def predict_proba_by_method(
        self,
        df_or_arr: pd.DataFrame | np.ndarray,
        method: Literal["selected", "uncalibrated", "isotonic", "sigmoid"] = "selected",
    ) -> np.ndarray:
        """Retourne les probabilites de rupture dans [0, 1] selon la methode choisie."""
        if isinstance(df_or_arr, pd.DataFrame):
            X = df_or_arr[self.feature_columns].to_numpy(dtype=np.float64)
        else:
            X = np.asarray(df_or_arr, dtype=np.float64)

        if getattr(self, "single_class_proba_", None) is not None:
            return np.full(len(X), self.single_class_proba_, dtype=np.float64)

        if (
            self.base_model_ is None
            or self.isotonic_calibrator_ is None
            or self.sigmoid_calibrator_ is None
        ):
            raise RuntimeError("Le classifieur doit etre entraine via fit() avant predict.")

        effective_method = self.selected_method_ if method == "selected" else method

        if effective_method == "uncalibrated":
            probs = self.base_model_.predict_proba(X)[:, 1]
        elif effective_method == "isotonic":
            probs = self.isotonic_calibrator_.predict_proba(X)[:, 1]
        elif effective_method == "sigmoid":
            probs = self.sigmoid_calibrator_.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Methode de calibration inconnue: {effective_method}")

        return np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)

    def evaluate_calibration_methods(self, val_df: pd.DataFrame) -> list[CalibrationMethodMetrics]:
        """Compare quantitativement Uncalibrated, Sigmoid (Platt) et Isotonic."""
        y_true = val_df["target_stockout_criticality"].to_numpy(dtype=np.int32)
        emp_rate = float(np.mean(y_true))
        reports: list[CalibrationMethodMetrics] = []

        for m_name in ("uncalibrated", "sigmoid", "isotonic"):
            probs = self.predict_proba_by_method(val_df, method=m_name)
            pr_auc = float(average_precision_score(y_true, probs)) if np.any(y_true == 1) else 0.0
            brier = float(brier_score_loss(y_true, probs))
            ece = compute_expected_calibration_error(y_true, probs, n_bins=10)
            reports.append(
                CalibrationMethodMetrics(
                    method=m_name,
                    pr_auc=round(pr_auc, 4),
                    brier_score=round(brier, 4),
                    ece=round(ece, 4),
                    mean_predicted_prob=round(float(np.mean(probs)), 4),
                    empirical_positive_rate=round(emp_rate, 4),
                )
            )
        return reports

    def explain_local_shap(self, df: pd.DataFrame) -> np.ndarray:
        """Retourne la matrice TreeSHAP (N, d) en log-odds pour le classifieur CatBoost."""
        if self.base_model_ is None:
            raise RuntimeError("Classifieur non entraine.")
        X = df[self.feature_columns].to_numpy(dtype=np.float64)
        pool = Pool(X, feature_names=self.feature_columns)
        shap_all = np.asarray(self.base_model_.get_feature_importance(pool, type="ShapValues"))
        return shap_all[:, :-1]


class SupplyChainAnomalyDetector:
    """Detecteur d'anomalies IsolationForest sur les residus et delais fournisseurs."""

    ANOMALY_FEATURE_NAMES: tuple[str, ...] = (
        "abs_relative_forecast_residual",
        "lead_time_zscore",
        "demand_spike_ratio",
        "supplier_fill_rate_deficit",
        "stock_to_reorder_ratio",
    )

    def __init__(self, cfg: SupplyChainSettings = settings) -> None:
        self.cfg = cfg
        self.iforest = IsolationForest(
            n_estimators=cfg.isolation_forest_estimators,
            contamination=cfg.isolation_forest_contamination,
            random_state=cfg.random_seed,
        )
        self.is_fitted_: bool = False

    @staticmethod
    def build_anomaly_features(
        df: pd.DataFrame,
        predicted_q50: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construit les 5 indicateurs diagnostiques d'anomalies pour chaque observation."""
        demand = df["demand_qty"].to_numpy(dtype=np.float64)
        roll_mean_14 = df["demand_roll_mean_14"].to_numpy(dtype=np.float64).clip(min=0.5)
        ref_pred = (
            np.asarray(predicted_q50, dtype=np.float64)
            if predicted_q50 is not None
            else roll_mean_14
        )

        abs_rel_residual = np.abs(demand - ref_pred) / np.maximum(ref_pred, 1.0)

        lt_drift = df["lead_time_drift"].to_numpy(dtype=np.float64)
        lt_std = df["causal_lead_time_std"].to_numpy(dtype=np.float64).clip(min=0.5)
        lt_zscore = np.abs(lt_drift) / lt_std

        demand_spike = demand / roll_mean_14
        fill_deficit = np.maximum(0.0, 1.0 - df["causal_fill_rate_mean"].to_numpy(dtype=np.float64))
        stock_ratio = df["stock_to_reorder_ratio"].to_numpy(dtype=np.float64)

        return np.column_stack(
            [abs_rel_residual, lt_zscore, demand_spike, fill_deficit, stock_ratio]
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        predicted_q50: np.ndarray | None = None,
    ) -> SupplyChainAnomalyDetector:
        X_anom = self.build_anomaly_features(train_df, predicted_q50=predicted_q50)
        self.iforest.fit(X_anom)
        self.is_fitted_ = True
        return self

    def inspect_records(
        self,
        df: pd.DataFrame,
        predicted_q50: np.ndarray | None = None,
    ) -> list[AnomalyDiagnosticsPayload]:
        """Evalue chaque ligne et retourne un diagnostic structure explicable."""
        if not self.is_fitted_:
            raise RuntimeError("IsolationForest non entraine. Appelez fit() d'abord.")

        X_anom = self.build_anomaly_features(df, predicted_q50=predicted_q50)
        raw_scores = -self.iforest.decision_function(X_anom)
        preds = self.iforest.predict(X_anom)

        sku_list = df["sku_id"].astype(str).tolist()
        date_list = pd.to_datetime(df["obs_date"]).dt.strftime("%Y-%m-%d").tolist()
        on_hand_arr = df["on_hand_qty"].to_numpy(dtype=np.float64)
        price_arr = df["unit_price_eur"].to_numpy(dtype=np.float64)
        demand_arr = df["demand_qty"].to_numpy(dtype=np.float64)

        diagnostics: list[AnomalyDiagnosticsPayload] = []
        for idx in range(len(df)):
            reasons: list[str] = []
            corrupted = False

            if on_hand_arr[idx] < 0.0 or not np.isfinite(on_hand_arr[idx]):
                corrupted = True
                reasons.append("DATA_CORRUPTION: negative_or_nan_inventory_on_hand")
            if price_arr[idx] <= 0.0 or not np.isfinite(price_arr[idx]):
                corrupted = True
                reasons.append("DATA_CORRUPTION: non_positive_unit_price")
            if demand_arr[idx] < 0.0 or not np.isfinite(demand_arr[idx]):
                corrupted = True
                reasons.append("DATA_CORRUPTION: negative_or_nan_demand")

            res_val, lt_z, spike_val, fill_def, _ = X_anom[idx]
            if res_val > 0.75:
                reasons.append(f"HIGH_FORECAST_RESIDUAL: rel_error={res_val:.2f}")
            if lt_z > 2.2:
                reasons.append(f"SUPPLIER_LEAD_TIME_SHOCK: z_score={lt_z:.2f}")
            if spike_val > 2.3:
                reasons.append(f"ATYPICAL_DEMAND_SPIKE: ratio={spike_val:.2f}x")
            if fill_def > 0.12:
                reasons.append(f"SUPPLIER_UNDER_DELIVERY: deficit={fill_def:.2%}")

            is_anom = bool(preds[idx] == -1 or corrupted)
            if is_anom and not reasons:
                reasons.append("ISOLATION_FOREST_MULTIVARIATE_OUTLIER")

            diagnostics.append(
                AnomalyDiagnosticsPayload(
                    sku_id=sku_list[idx],
                    obs_date=date_list[idx],
                    anomaly_score=round(float(raw_scores[idx]), 4),
                    is_anomaly=is_anom,
                    data_corruption_detected=corrupted,
                    triggered_reasons=reasons,
                )
            )
        return diagnostics


def evaluate_catboost_across_splits(
    df: pd.DataFrame,
    splitter: ExpandingWindowTimeSplit,
    horizon_days: int = 7,
    cfg: SupplyChainSettings = settings,
) -> tuple[
    BaselineComparisonSummary,
    list[CalibrationMethodMetrics],
    MultiHorizonQuantileForecaster,
    CalibratedCriticalityClassifier,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evalue CatBoost MultiQuantile et Classifier Calibre sur les K plis temporels."""
    fold_metrics_list: list[FoldMetrics] = []
    last_calib_comparison: list[CalibrationMethodMetrics] = []
    last_forecaster: MultiHorizonQuantileForecaster | None = None
    last_classifier: CalibratedCriticalityClassifier | None = None

    last_y_bin = np.empty(0, dtype=np.int32)
    last_prob_uncal = np.empty(0, dtype=np.float64)
    last_prob_sig = np.empty(0, dtype=np.float64)
    last_prob_iso = np.empty(0, dtype=np.float64)

    for split_info in splitter.split(df, date_col="obs_date"):
        train_df = df.iloc[split_info.train_indices].reset_index(drop=True)
        val_df = df.iloc[split_info.val_indices].reset_index(drop=True)

        forecaster = MultiHorizonQuantileForecaster(horizons=(horizon_days,), cfg=cfg)
        forecaster.fit_horizon(train_df, horizon_days=horizon_days, val_df=None)
        q10, q50, q90 = forecaster.predict_quantiles_matrix(val_df, horizon_days=horizon_days)

        classifier = CalibratedCriticalityClassifier(cfg=cfg)
        classifier.fit(train_df, horizon_days=horizon_days, preferred_method="auto")
        prob_cal = classifier.predict_proba_by_method(val_df, method="selected")

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

        pr_auc = float(average_precision_score(c_val, prob_cal)) if np.any(c_val == 1) else 0.0
        brier = float(brier_score_loss(c_val, prob_cal))
        ece = compute_expected_calibration_error(c_val, prob_cal, n_bins=10)

        fold_metrics_list.append(
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

        last_calib_comparison = classifier.evaluate_calibration_methods(val_df)
        last_forecaster = forecaster
        last_classifier = classifier
        last_y_bin = c_val
        last_prob_uncal = classifier.predict_proba_by_method(val_df, method="uncalibrated")
        last_prob_sig = classifier.predict_proba_by_method(val_df, method="sigmoid")
        last_prob_iso = classifier.predict_proba_by_method(val_df, method="isotonic")

    assert last_forecaster is not None and last_classifier is not None

    mase_vals = [f.mase for f in fold_metrics_list]
    wape_vals = [f.wape for f in fold_metrics_list]
    pin_vals = [f.mean_pinball for f in fold_metrics_list]
    prauc_vals = [f.pr_auc for f in fold_metrics_list]
    brier_vals = [f.brier_score for f in fold_metrics_list]
    ece_vals = [f.ece for f in fold_metrics_list]

    summary = BaselineComparisonSummary(
        model_name=f"catboost_multiquantile_calibrated_{last_classifier.selected_method_}",
        horizon_days=horizon_days,
        n_splits=len(fold_metrics_list),
        mase_mean=round(float(np.mean(mase_vals)), 4),
        mase_std=round(float(np.std(mase_vals)), 4),
        wape_mean=round(float(np.mean(wape_vals)), 4),
        wape_std=round(float(np.std(wape_vals)), 4),
        mean_pinball_avg=round(float(np.mean(pin_vals)), 4),
        pr_auc_mean=round(float(np.mean(prauc_vals)), 4),
        pr_auc_std=round(float(np.std(prauc_vals)), 4),
        brier_score_mean=round(float(np.mean(brier_vals)), 4),
        ece_mean=round(float(np.mean(ece_vals)), 4),
        folds=fold_metrics_list,
    )

    return (
        summary,
        last_calib_comparison,
        last_forecaster,
        last_classifier,
        last_y_bin,
        last_prob_uncal,
        last_prob_sig,
        last_prob_iso,
    )
