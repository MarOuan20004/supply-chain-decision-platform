#!/usr/bin/env python3
"""Entrainement Multi-Horizon CatBoost, Calibration, IsolationForest et Model Registry (S5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from supply_chain.config import settings  # noqa: E402
from supply_chain.data_pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    build_causal_feature_Frame,
    generate_synthetic_supply_chain_db,
    load_raw_tables_from_sql,
)
from supply_chain.models import (  # noqa: E402
    SupplyChainAnomalyDetector,
    enforce_quantile_monotonicity,
    evaluate_catboost_across_splits,
)
from supply_chain.schemas import HorizonEvaluationReport  # noqa: E402
from supply_chain.validation import (  # noqa: E402
    CausalMovingAverageBaseline,
    ExpandingWindowTimeSplit,
    SeasonalNaiveBaseline,
    evaluate_baseline_across_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("train_and_register_mlflow")


class MonotonicCatBoostPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Enveloppe MLflow pyfunc garantissant la sortie DataFrame (q10, q50, q90) sans croisement."""

    def __init__(self, cb_model: Any) -> None:
        self.cb_model = cb_model

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        X = model_input[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        raw = self.cb_model.predict(X)
        mono = enforce_quantile_monotonicity(raw)
        return pd.DataFrame(
            {
                "q10": mono[:, 0],
                "q50": mono[:, 1],
                "q90": mono[:, 2],
            },
            index=model_input.index,
        )


def save_reliability_diagram(
    y_true: np.ndarray,
    prob_uncal: np.ndarray,
    prob_sig: np.ndarray,
    prob_iso: np.ndarray,
    output_path: Path,
) -> None:
    """Genere et sauvegarde la figure Reliability Diagram (Avant vs Apres Calibration)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    ax1.plot([0, 1], [0, 1], "k--", label="Calibration parfaite (diagonale)")
    for probs, label, marker in (
        (prob_uncal, "CatBoost Brut (Balanced)", "o"),
        (prob_sig, "Calibre Platt (Sigmoid)", "s"),
        (prob_iso, "Calibre Isotonique", "^"),
    ):
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=8, strategy="uniform")
        ax1.plot(mean_pred, frac_pos, marker=marker, linewidth=1.8, label=label)

    ax1.set_title("Reliability Diagram (Hors-Echantillon Temporel)")
    ax1.set_xlabel("Probabilite predite moyenne par bac")
    ax1.set_ylabel("Frequence empirique de rupture observee")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.hist(prob_uncal, bins=15, alpha=0.45, label="Brut (non calibre)", density=True)
    ax2.hist(prob_sig, bins=15, alpha=0.55, label="Calibre Platt", density=True)
    ax2.hist(prob_iso, bins=15, alpha=0.55, label="Calibre Isotonique", density=True)
    ax2.set_title("Distribution des Probabilites de Criticite")
    ax2.set_xlabel("Probabilite de rupture p_crit")
    ax2.set_ylabel("Densite")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Reliability diagram sauvegarde dans %s", output_path)


def format_catboost_markdown_report(
    horizon_reports: list[HorizonEvaluationReport],
    shap_top_features: list[tuple[str, float]],
    anomaly_summary: dict[str, Any],
) -> str:
    """Genere le rapport Markdown comparatif officiel Baselines vs CatBoost Calibre."""
    lines = [
        "# SupplyChain-Decision-Platform - Rapport Experimental CatBoost MLOps",
        "## Comparaison Baselines vs CatBoost MultiQuantile & Calibration",
        "",
        "### 1. Synthese Multi-Horizon sur 4 Plis Temporels",
        "",
        "| Horizon | Modele | MASE (Mean +/- Std) | WAPE | Pinball Moy. "
        "| PR-AUC (Mean +/- Std) | Brier Score | ECE |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for hr in horizon_reports:
        for s in (hr.seasonal_naive_summary, hr.moving_average_summary, hr.catboost_summary):
            row = (
                f"| **{hr.horizon_days}j** | `{s.model_name}` | "
                f"**{s.mase_mean:.4f}** +/- {s.mase_std:.4f} | "
                f"{s.wape_mean:.4f} | {s.mean_pinball_avg:.4f} | "
                f"**{s.pr_auc_mean:.4f}** +/- {s.pr_auc_std:.4f} | "
                f"**{s.brier_score_mean:.4f}** | **{s.ece_mean:.4f}** |"
            )
            lines.append(row)

    lines.extend(
        [
            "",
            "### 2. Impact de la Calibration Probabiliste Hors-Echantillon (H = 7 jours)",
            "",
            "| Methode | PR-AUC | Brier Score | ECE (10 bacs) "
            "| Prob. Moyenne Predite | Taux Empirique Reel |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    h7_report = horizon_reports[0]
    for cm in h7_report.calibration_comparison:
        row = (
            f"| `{cm.method}` | {cm.pr_auc:.4f} | **{cm.brier_score:.4f}** | **{cm.ece:.4f}** | "
            f"{cm.mean_predicted_prob:.4f} | {cm.empirical_positive_rate:.4f} |"
        )
        lines.append(row)

    lines.extend(
        [
            "",
            "### 3. Importance Globale et Locale TreeSHAP (H = 7 jours, Top 8 Variables)",
            "",
            "| Rang | Variable Causale | Importance Moyenne Abs(SHAP) |",
            "| :---: | :--- | :---: |",
        ]
    )
    for rank_idx, (fname, val) in enumerate(shap_top_features[:8], start=1):
        lines.append(f"| {rank_idx} | `{fname}` | {val:.4f} |")

    lines.extend(
        [
            "",
            "### 4. Diagnostic du Detecteur d'Anomalies IsolationForest",
            "",
            f"- **Observations analysees** : `{anomaly_summary['total_rows']}`",
            f"- **Anomalies detectees** : `{anomaly_summary['anomalies_detected']}` "
            f"(`{anomaly_summary['anomaly_rate_pct']:.2f}%`)",
            "- **Modeles enregistres dans MLflow Model Registry** : "
            f"`{settings.registered_forecaster_name}`, "
            f"`{settings.registered_classifier_name}`, "
            f"`{settings.registered_anomaly_name}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not settings.sqlite_db_path.is_file():
        logger.info("Initialisation de la base SQLite %s...", settings.sqlite_db_path)
        generate_synthetic_supply_chain_db()

    daily_df, deliv_df = load_raw_tables_from_sql(settings.sqlite_db_path)

    Path("mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    horizon_reports: list[HorizonEvaluationReport] = []
    shap_top_features: list[tuple[str, float]] = []
    anomaly_summary: dict[str, Any] = {}

    for h_days in settings.forecast_horizons:
        logger.info("=== Evaluation Multi-Horizon H = %d jours ===", h_days)
        feat_df = build_causal_feature_Frame(daily_df, deliv_df, horizon_days=h_days)
        splitter = ExpandingWindowTimeSplit(
            n_splits=settings.n_splits,
            val_window_days=settings.val_window_days,
            purge_gap_days=h_days,
            min_train_days=settings.min_train_days,
        )

        s_naive = evaluate_baseline_across_splits(
            feat_df, splitter, SeasonalNaiveBaseline(), horizon_days=h_days
        )
        m_avg = evaluate_baseline_across_splits(
            feat_df, splitter, CausalMovingAverageBaseline(), horizon_days=h_days
        )

        (
            cb_summary,
            calib_comp,
            last_forecaster,
            last_classifier,
            last_y_bin,
            last_prob_uncal,
            last_prob_sig,
            last_prob_iso,
        ) = evaluate_catboost_across_splits(feat_df, splitter, horizon_days=h_days, cfg=settings)

        best_base_mase = min(s_naive.mase_mean, m_avg.mase_mean)
        best_base_prauc = max(s_naive.pr_auc_mean, m_avg.pr_auc_mean)
        mase_gain_pct = (best_base_mase - cb_summary.mase_mean) / best_base_mase * 100.0
        prauc_gain_pct = (
            (cb_summary.pr_auc_mean - best_base_prauc) / max(best_base_prauc, 1e-6) * 100.0
        )

        uncal_metrics = next(c for c in calib_comp if c.method == "uncalibrated")
        sel_metrics = next(c for c in calib_comp if c.method == last_classifier.selected_method_)
        brier_red_pct = (
            (uncal_metrics.brier_score - sel_metrics.brier_score)
            / uncal_metrics.brier_score
            * 100.0
        )
        ece_red_pct = (uncal_metrics.ece - sel_metrics.ece) / max(uncal_metrics.ece, 1e-6) * 100.0

        horizon_reports.append(
            HorizonEvaluationReport(
                horizon_days=h_days,
                catboost_summary=cb_summary,
                seasonal_naive_summary=s_naive,
                moving_average_summary=m_avg,
                calibration_comparison=calib_comp,
                selected_calibration_method=last_classifier.selected_method_,
                mase_improvement_pct_vs_best_baseline=round(mase_gain_pct, 2),
                pr_auc_improvement_pct_vs_best_baseline=round(prauc_gain_pct, 2),
                brier_reduction_pct_after_calibration=round(brier_red_pct, 2),
                ece_reduction_pct_after_calibration=round(ece_red_pct, 2),
            )
        )

        if h_days == settings.default_horizon_days:
            fig_path = settings.results_dir / "reliability_diagram.png"
            save_reliability_diagram(
                last_y_bin, last_prob_uncal, last_prob_sig, last_prob_iso, fig_path
            )

            sample_df = feat_df.tail(200).reset_index(drop=True)
            shap_mat, _ = last_forecaster.explain_local_shap(sample_df, horizon_days=h_days)
            mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)
            shap_top_features = sorted(
                zip(FEATURE_COLUMNS, [float(v) for v in mean_abs_shap], strict=True),
                key=lambda item: item[1],
                reverse=True,
            )

            _, q50_all, _ = last_forecaster.predict_quantiles_matrix(feat_df, horizon_days=h_days)
            anom_detector = SupplyChainAnomalyDetector(cfg=settings)
            anom_detector.fit(feat_df, predicted_q50=q50_all)
            diag_list = anom_detector.inspect_records(sample_df)
            anom_count = sum(1 for d in diag_list if d.is_anomaly)
            anomaly_summary = {
                "total_rows": len(diag_list),
                "anomalies_detected": anom_count,
                "anomaly_rate_pct": round(100.0 * anom_count / max(len(diag_list), 1), 2),
            }

            with mlflow.start_run(run_name=f"catboost_mlops_suite_H{h_days}d"):
                mlflow.log_params(
                    {
                        "horizon_days": h_days,
                        "catboost_iterations": settings.catboost_iterations,
                        "catboost_depth": settings.catboost_depth,
                        "catboost_learning_rate": settings.catboost_learning_rate,
                        "selected_calibration": last_classifier.selected_method_,
                        "isolation_forest_contamination": settings.isolation_forest_contamination,
                    }
                )
                mlflow.log_metrics(
                    {
                        "catboost_mase_mean": cb_summary.mase_mean,
                        "catboost_wape_mean": cb_summary.wape_mean,
                        "catboost_pinball_mean": cb_summary.mean_pinball_avg,
                        "catboost_pr_auc_mean": cb_summary.pr_auc_mean,
                        "uncalibrated_brier": uncal_metrics.brier_score,
                        "calibrated_brier": sel_metrics.brier_score,
                        "uncalibrated_ece": uncal_metrics.ece,
                        "calibrated_ece": sel_metrics.ece,
                        "mase_improvement_pct": round(mase_gain_pct, 2),
                        "brier_reduction_pct": round(brier_red_pct, 2),
                    }
                )
                mlflow.log_artifact(str(fig_path))

                # 1. Enregistrement du Regresseur Multi-Quantile avec signature typee
                X_example = sample_df[FEATURE_COLUMNS].head(5).astype(np.float64)
                pyfunc_forecaster = MonotonicCatBoostPyfuncWrapper(
                    last_forecaster.models_by_horizon[h_days]
                )
                pred_example = pyfunc_forecaster.predict(None, X_example)
                sig_forecaster = infer_signature(X_example, pred_example)
                mlflow.pyfunc.log_model(
                    artifact_path="forecaster_model",
                    python_model=pyfunc_forecaster,
                    signature=sig_forecaster,
                    input_example=X_example,
                    registered_model_name=settings.registered_forecaster_name,
                )

                # 2. Enregistrement du Classifieur de Criticite Calibre avec signature typee
                cal_estimator = (
                    last_classifier.isotonic_calibrator_
                    if last_classifier.selected_method_ == "isotonic"
                    else last_classifier.sigmoid_calibrator_
                )
                prob_example = cal_estimator.predict_proba(X_example.to_numpy(dtype=np.float64))
                sig_classifier = infer_signature(X_example.to_numpy(dtype=np.float64), prob_example)
                mlflow.sklearn.log_model(
                    sk_model=cal_estimator,
                    artifact_path="criticality_classifier",
                    signature=sig_classifier,
                    registered_model_name=settings.registered_classifier_name,
                )

                # 3. Enregistrement du Detecteur d'Anomalies IsolationForest avec signature typee
                X_anom_example = SupplyChainAnomalyDetector.build_anomaly_features(
                    sample_df.head(5)
                )
                anom_pred_example = anom_detector.iforest.decision_function(X_anom_example)
                sig_anomaly = infer_signature(X_anom_example, anom_pred_example)
                mlflow.sklearn.log_model(
                    sk_model=anom_detector.iforest,
                    artifact_path="anomaly_detector",
                    signature=sig_anomaly,
                    registered_model_name=settings.registered_anomaly_name,
                )

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    json_out = settings.results_dir / "catboost_vs_baselines.json"
    md_out = settings.results_dir / "catboost_vs_baselines.md"

    json_out.write_text(
        json.dumps([r.model_dump() for r in horizon_reports], indent=2),
        encoding="utf-8",
    )
    md_text = format_catboost_markdown_report(horizon_reports, shap_top_features, anomaly_summary)
    md_out.write_text(md_text, encoding="utf-8")

    print("\n" + md_text)
    logger.info("Rapports d'evaluation sauvegardes dans %s et %s", json_out, md_out)


if __name__ == "__main__":
    main()
