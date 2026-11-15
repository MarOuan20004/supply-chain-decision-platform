#!/usr/bin/env python3
"""Execution des baselines causales et journalisation dans MLflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

from supply_chain.config import settings
from supply_chain.data_pipeline import (
    FEATURE_COLUMNS,
    build_causal_feature_Frame,
    generate_synthetic_supply_chain_db,
    load_raw_tables_from_sql,
)
from supply_chain.schemas import BaselineComparisonSummary
from supply_chain.validation import (
    CausalMovingAverageBaseline,
    ExpandingWindowTimeSplit,
    SeasonalNaiveBaseline,
    evaluate_baseline_across_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("run_baselines_mlflow")


def format_baselines_markdown(summaries: list[BaselineComparisonSummary]) -> str:
    """Genere le rapport Markdown comparatif des baselines sur les 4 plis temporels."""
    lines = [
        "# SupplyChain-Decision-Platform -- Rapport des Baselines MLOps",
        "",
        "## 1. Synthese Globale sur 4 Plis Temporels (`ExpandingWindowTimeSplit`)",
        "",
        "| Modele Baseline | Horizon | MASE (Mean +/- Std) | WAPE (Mean +/- Std) | Pinball Moy. "
        "| PR-AUC (Mean +/- Std) | Brier Score | ECE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        row = (
            f"| `{s.model_name}` | {s.horizon_days}j | "
            f"**{s.mase_mean:.4f}** +/- {s.mase_std:.4f} | "
            f"**{s.wape_mean:.4f}** +/- {s.wape_std:.4f} | {s.mean_pinball_avg:.4f} | "
            f"**{s.pr_auc_mean:.4f}** +/- {s.pr_auc_std:.4f} | {s.brier_score_mean:.4f} | "
            f"{s.ece_mean:.4f} |"
        )
        lines.append(row)

    lines.extend(["", "## 2. Detail par Pli Temporel (Verification de Non-Chevauchement)", ""])
    for s in summaries:
        lines.append(f"### Modele : `{s.model_name}`")
        lines.append("")
        lines.append(
            "| Pli | Periode Train | Periode Validation | MASE | WAPE | Pinball (q10/q50/q90) "
            "| PR-AUC | Brier |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for f in s.folds:
            row = (
                f"| Fold {f.fold_index} | `{f.train_start_date}` -> `{f.train_end_date}` | "
                f"`{f.val_start_date}` -> `{f.val_end_date}` | {f.mase:.4f} | {f.wape:.4f} | "
                f"{f.pinball_q10:.2f} / {f.pinball_q50:.2f} / {f.pinball_q90:.2f} | "
                f"{f.pr_auc:.4f} | {f.brier_score:.4f} |"
            )
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    if not settings.sqlite_db_path.is_file():
        logger.info("Initialisation de la base SQL %s...", settings.sqlite_db_path)
        generate_synthetic_supply_chain_db()

    daily_df, deliv_df = load_raw_tables_from_sql(settings.sqlite_db_path)
    feat_df = build_causal_feature_Frame(
        daily_df, deliv_df, horizon_days=settings.default_horizon_days
    )

    splitter = ExpandingWindowTimeSplit(
        n_splits=settings.n_splits,
        val_window_days=settings.val_window_days,
        purge_gap_days=settings.default_horizon_days,
        min_train_days=settings.min_train_days,
    )

    Path("mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    baselines = [
        SeasonalNaiveBaseline(),
        CausalMovingAverageBaseline(),
    ]

    summaries: list[BaselineComparisonSummary] = []

    for model in baselines:
        logger.info("Evaluation et journalisation MLflow de %s...", model.model_name)
        summary = evaluate_baseline_across_splits(
            df=feat_df,
            splitter=splitter,
            baseline_model=model,
            horizon_days=settings.default_horizon_days,
        )
        summaries.append(summary)

        with mlflow.start_run(run_name=model.model_name):
            mlflow.log_params(
                {
                    "model_name": model.model_name,
                    "horizon_days": settings.default_horizon_days,
                    "n_splits": settings.n_splits,
                    "val_window_days": settings.val_window_days,
                    "purge_gap_days": settings.default_horizon_days,
                    "num_features": len(FEATURE_COLUMNS),
                    "num_skus": int(feat_df["sku_id"].nunique()),
                }
            )
            mlflow.log_metrics(
                {
                    "mase_mean": summary.mase_mean,
                    "mase_std": summary.mase_std,
                    "wape_mean": summary.wape_mean,
                    "mean_pinball_avg": summary.mean_pinball_avg,
                    "pr_auc_mean": summary.pr_auc_mean,
                    "brier_score_mean": summary.brier_score_mean,
                    "ece_mean": summary.ece_mean,
                }
            )
            for fold in summary.folds:
                mlflow.log_metrics(
                    {
                        "fold_mase": fold.mase,
                        "fold_wape": fold.wape,
                        "fold_pr_auc": fold.pr_auc,
                        "fold_brier": fold.brier_score,
                    },
                    step=fold.fold_index,
                )

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.results_dir / "baselines_evaluation.json"
    md_path = settings.results_dir / "baselines_evaluation.md"

    json_path.write_text(
        json.dumps([s.model_dump() for s in summaries], indent=2),
        encoding="utf-8",
    )
    md_report = format_baselines_markdown(summaries)
    md_path.write_text(md_report, encoding="utf-8")

    print("\n" + md_report)
    logger.info("Rapports sauvegardes dans %s et %s", json_path, md_path)


if __name__ == "__main__":
    main()
