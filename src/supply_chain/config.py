"""Centralized configuration for Supply Chain Analytics & MLOps platform via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupplyChainSettings(BaseSettings):
    """Platform configuration: SQL database, temporal cross-validation, CatBoost and MLflow."""

    model_config = SettingsConfigDict(
        env_prefix="SUPPLY_CHAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Relational Database (SQLite) ---
    sqlite_db_path: Path = Field(
        default=Path("data/supply_chain.db"),
        description="Chemin de la base relationnelle SQLite locale (4 tables metier)",
    )
    random_seed: int = Field(default=42, description="Graine deterministe globale")
    num_skus: int = Field(
        default=50,
        ge=5,
        le=1000,
        description="Nombre de references produits (SKUs) dans l'echantillon stratifie",
    )
    num_days: int = Field(
        default=400,
        ge=120,
        le=2000,
        description="Profondeur historique en jours (ex: 50 SKUs x 400 jours = 20 000 lignes)",
    )

    # --- Feature Engineering & Multi-Horizon Parameters ---
    default_horizon_days: int = Field(
        default=7,
        ge=1,
        le=56,
        description="Horizon de prevision par defaut H (en jours)",
    )
    forecast_horizons: tuple[int, int, int] = Field(
        default=(7, 14, 28),
        description="Horizons multi-pas H predictifs (7, 14 et 28 jours)",
    )
    quantiles_alpha: tuple[float, float, float] = Field(
        default=(0.10, 0.50, 0.90),
        description="Ordres des quantiles predits par CatBoost MultiQuantile",
    )
    rolling_windows: tuple[int, int, int] = Field(
        default=(7, 14, 28),
        description="Fenetres glissantes causales (en jours) pour la demande",
    )
    seasonal_period_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Periode saisonniere pour la baseline Seasonal Naive (7 = hebdomadaire)",
    )

    # --- Parametres de Validation Temporelle (ExpandingWindowTimeSplit) ---
    n_splits: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Nombre de plis temporels pour la validation croisee a fenetre croissante",
    )
    min_train_days: int = Field(
        default=180,
        ge=60,
        description="Nombre minimal de jours dans le premier pli d'entrainement",
    )
    val_window_days: int = Field(
        default=28,
        ge=7,
        description="Duree en jours de chaque pli de validation temporelle",
    )

    # --- Hyperparametres CatBoost, Calibration & Isolation Forest ---
    catboost_iterations: int = Field(
        default=280,
        ge=30,
        le=3000,
        description="Nombre d'arbres pour CatBoostRegressor et CatBoostClassifier",
    )
    catboost_depth: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Profondeur des arbres symetriques (Oblivious Trees) de CatBoost",
    )
    catboost_learning_rate: float = Field(
        default=0.06,
        gt=0.001,
        le=0.5,
        description="Pas d'apprentissage du gradient boosting",
    )
    calibration_holdout_frac: float = Field(
        default=0.22,
        gt=0.05,
        lt=0.50,
        description="Fraction chronologique finale du pli reservee a la calibration",
    )
    uncertainty_Threshold_tau: float = Field(
        default=0.80,
        gt=0.10,
        le=5.0,
        description="Seuil critique d'incertitude relative (q90 - q10) / q50 pour l'Agent",
    )
    isolation_forest_contamination: float = Field(
        default=0.05,
        gt=0.001,
        lt=0.30,
        description="Proportion attendue d'anomalies dans IsolationForest",
    )
    isolation_forest_estimators: int = Field(
        default=150,
        ge=20,
        le=1000,
        description="Nombre d'arbres d'isolement dans IsolationForest",
    )

    # --- Suivi experimental MLflow & Model Registry ---
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns/mlflow.db",
        description="URI de suivi MLflow (fichier SQLite local compatible Model Registry)",
    )
    mlflow_experiment_name: str = Field(
        default="supply_chain_catboost_mlops",
        description="Nom de l'experience MLflow pour le projet MLOps",
    )
    registered_forecaster_name: str = Field(
        default="supply_chain_catboost_forecaster",
        description="Nom du modele de prevision multi-quantile dans MLflow Model Registry",
    )
    registered_classifier_name: str = Field(
        default="supply_chain_criticality_classifier",
        description="Nom du classifieur de criticite calibre dans MLflow Model Registry",
    )
    registered_anomaly_name: str = Field(
        default="supply_chain_anomaly_detector",
        description="Nom du detecteur d'anomalies IsolationForest dans MLflow Model Registry",
    )
    champion_alias: str = Field(
        default="champion",
        description="Alias MLflow Model Registry charge par l'API FastAPI",
    )

    # --- Parametres API FastAPI & Optimiseur Deterministe ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    default_budget_eur: float = Field(
        default=15000.0,
        gt=0.0,
        description="Plafond budgetaire par defaut pour une commande de reapprovisionnement",
    )
    human_approval_threshold_eur: float = Field(
        default=5000.0,
        gt=0.0,
        description="Seuil d'engagement financier declenchant PENDING_HUMAN_APPROVAL",
    )
    expedited_cost_multiplier: float = Field(
        default=1.35,
        ge=1.0,
        description="Multiplicateur de cout unitaire en cas de transport express",
    )
    fixed_order_setup_cost_eur: float = Field(
        default=45.0,
        ge=0.0,
        description="Cout administratif fixe par commande non nulle",
    )
    default_observation_date: str | None = Field(
        default=None,
        description="Date de reference ISO optionnelle (defaut: derniere date en base SQLite)",
    )
    # --- Parametres Agent Decisionnel Supply Chain ---
    agent_max_steps: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Nombre maximal d'iterations de la boucle d'orchestration (MAX_STEPS = 5)",
    )
    agent_timeout_sec: float = Field(
        default=15.0,
        gt=0.1,
        le=120.0,
        description="Timeout global d'une session d'investigation agentique en secondes",
    )
    agent_llm_provider: str = Field(
        default="mock",
        description="Fournisseur LLM actif : 'mock' (CI), 'ollama' (local) ou 'openai' (API)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL du serveur local Ollama",
    )
    ollama_model: str = Field(
        default="qwen2.5:3b-instruct",
        description="Modele local utilise par OllamaClient",
    )
    openai_base_url: str = Field(
        default="https://api.mistral.ai/v1",
        description="URL de base compatible OpenAI / Mistral",
    )
    openai_api_key: str = Field(
        default="not-configured",
        description="Cle API pour OpenAIClient (via SUPPLY_CHAIN_OPENAI_API_KEY)",
    )
    openai_model: str = Field(
        default="mistral-small-latest",
        description="Modele cible pour OpenAIClient",
    )
    # --- Garde-Fous Deterministes, HITL & Telemetrie ---
    database_path: Path = Field(
        default=Path("data/supply_chain.db"),
        description="Chemin vers la base SQLite (alias retrocompatible pour sqlite_db_path)",
    )
    agent_uncertainty_threshold: float = Field(
        default=0.80,
        gt=0.0,
        le=5.0,
        description="Seuil maximal admissible pour (q90 - q10) / q50 avant abstention",
    )
    hitl_approval_threshold_eur: float = Field(
        default=5000.0,
        ge=0.0,
        description="Plafond financier EUR declenchant obligatoirement PENDING_HUMAN_APPROVAL",
    )
    numeric_grounding_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        description="Tolerance absolue (1e-2 EUR) pour le validateur d'ancrage numerique",
    )
    agent_traces_path: Path = Field(
        default=Path("logs/agent_traces.jsonl"),
        description="Chemin du journal d'audit JSONL append-only",
    )
    llm_provider: str = Field(
        default="mock",
        description="Alias pour agent_llm_provider ('mock', 'ollama', 'openai')",
    )
    openai_api_base: str = Field(
        default="https://api.mistral.ai/v1",
        description="Alias pour openai_base_url",
    )

    results_dir: Path = Field(
        default=Path("benchmarks/results"),
        description="Repertoire d'export des rapports JSON, Markdown et figures de calibration",
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "database_path" in data and "sqlite_db_path" not in data:
                data["sqlite_db_path"] = data["database_path"]
            elif "sqlite_db_path" in data and "database_path" not in data:
                data["database_path"] = data["sqlite_db_path"]
            if "llm_provider" in data and "agent_llm_provider" not in data:
                data["agent_llm_provider"] = data["llm_provider"]
            elif "agent_llm_provider" in data and "llm_provider" not in data:
                data["llm_provider"] = data["agent_llm_provider"]
            if "openai_api_base" in data and "openai_base_url" not in data:
                data["openai_base_url"] = data["openai_api_base"]
            elif "openai_base_url" in data and "openai_api_base" not in data:
                data["openai_api_base"] = data["openai_base_url"]
        return data

    @property
    def horizons_days(self) -> tuple[int, int, int]:
        """Alias retrocompatible pour forecast_horizons."""
        return self.forecast_horizons


settings = SupplyChainSettings()
