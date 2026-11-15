"""Contrats de donnees Pydantic v2 pour les schemas SQL, predictions et outils agentiques."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SKUMasterRecord(BaseModel):
    """Enregistrement de la table relationnelle `skus`."""

    model_config = ConfigDict(frozen=True)

    sku_id: str = Field(..., min_length=3, description="Identifiant unique du produit")
    category: str = Field(..., description="Famille logistique")
    unit_cost_eur: float = Field(..., gt=0.0, description="Cout d'achat unitaire en EUR")
    holding_cost_per_day_eur: float = Field(..., gt=0.0, description="Cout de possession par jour")
    stockout_penalty_per_unit_eur: float = Field(..., gt=0.0, description="Penalite unitaire")
    moq_units: int = Field(..., ge=1, description="Quantite minimale contractuelle (MOQ)")
    contractual_lead_time_days: int = Field(..., ge=1, description="Delai fournisseur nominal")
    safety_stock_units: int = Field(..., ge=0, description="Seuil de stock de securite")


class DailyDemandRecord(BaseModel):
    """Enregistrement de la table relationnelle `daily_demand`."""

    model_config = ConfigDict(frozen=True)

    sku_id: str
    obs_date: date
    demand_qty: float = Field(..., ge=0.0, description="Demande journaliere observee")
    is_promo: int = Field(..., ge=0, le=1, description="Indicateur binaire de promotion")
    unit_price_eur: float = Field(..., gt=0.0, description="Prix de vente unitaire actif")


class InventorySnapshotRecord(BaseModel):
    """Enregistrement de la table relationnelle `inventory_snapshots`."""

    model_config = ConfigDict(frozen=True)

    sku_id: str
    obs_date: date
    on_hand_qty: float = Field(..., ge=0.0, description="Stock physique disponible")
    in_transit_qty: float = Field(..., ge=0.0, description="Stock en transit commande non livre")
    reorder_point_s: float = Field(..., ge=0.0, description="Seuil s de la politique (s, S)")
    order_up_to_S: float = Field(..., ge=0.0, description="Niveau de recompletement S")


class SupplierDeliveryRecord(BaseModel):
    """Enregistrement de la table relationnelle `supplier_deliveries`."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    sku_id: str
    order_date: date
    expected_delivery_date: date
    actual_delivery_date: date
    ordered_qty: int = Field(..., ge=1)
    delivered_qty: int = Field(..., ge=0)
    actual_lead_time_days: int = Field(..., ge=1, description="actual_delivery_date - order_date")


class FoldMetrics(BaseModel):
    """Metriques de regression et de classification mesurees sur un pli temporel."""

    model_config = ConfigDict(frozen=True)

    fold_index: int = Field(..., ge=0)
    train_start_date: str
    train_end_date: str
    val_start_date: str
    val_end_date: str
    num_train_rows: int
    num_val_rows: int
    mase: float = Field(..., ge=0.0, description="Mean Absolute Scaled Error")
    wape: float = Field(..., ge=0.0, description="Weighted Absolute Percentage Error")
    pinball_q10: float = Field(..., ge=0.0, description="Pinball Loss quantile 0.10")
    pinball_q50: float = Field(..., ge=0.0, description="Pinball Loss quantile 0.50 (mediane)")
    pinball_q90: float = Field(..., ge=0.0, description="Pinball Loss quantile 0.90")
    mean_pinball: float = Field(..., ge=0.0, description="Moyenne des Pinball Losses")
    pr_auc: float = Field(..., ge=0.0, le=1.0, description="Area Under PR Curve")
    brier_score: float = Field(..., ge=0.0, le=1.0, description="Brier Score")
    ece: float = Field(..., ge=0.0, le=1.0, description="Expected Calibration Error (10 bacs)")


class BaselineComparisonSummary(BaseModel):
    """Synthese moyenne et ecart-type sur les K plis temporels pour un modele."""

    model_name: str
    horizon_days: int
    n_splits: int
    mase_mean: float
    mase_std: float
    wape_mean: float
    wape_std: float
    mean_pinball_avg: float
    pr_auc_mean: float
    pr_auc_std: float
    brier_score_mean: float
    ece_mean: float
    folds: list[FoldMetrics]


# ============================================================================
# CONTRATS DE PREVISION MULTI-QUANTILE ET CALIBRATION
# ============================================================================


class ShapFeatureContribution(BaseModel):
    """Contribution locale TreeSHAP d'une variable explicative pour une prediction."""

    model_config = ConfigDict(frozen=True)

    feature_name: str
    feature_value: float
    shap_value: float = Field(..., description="Contribution marginale signee a la prediction")


class MultiQuantilePrediction(BaseModel):
    """Prevision de demande multi-quantile sans croisement (q10 <= q50 <= q90)."""

    model_config = ConfigDict(frozen=True)

    sku_id: str
    obs_date: str
    horizon_days: int = Field(..., ge=1)
    q10: float = Field(..., ge=0.0, description="Quantile 10% de la demande sur H")
    q50: float = Field(..., ge=0.0, description="Quantile 50% (mediane) de la demande sur H")
    q90: float = Field(..., ge=0.0, description="Quantile 90% de la demande sur H")
    relative_uncertainty: float = Field(
        ...,
        ge=0.0,
        description="Largeur relative de l'intervalle (q90 - q10) / max(q50, 1.0)",
    )
    high_uncertainty_flag: bool = Field(
        ...,
        description="Vaut True si relative_uncertainty > tau_uncert (declenche abstention)",
    )
    top_shap_drivers: list[ShapFeatureContribution] = Field(
        default_factory=list,
        description="Top variables explicatives locales (TreeSHAP)",
    )

    @model_validator(mode="after")
    def _check_quantile_monotonicity(self) -> MultiQuantilePrediction:
        if not (self.q10 <= self.q50 + 1e-9 and self.q50 <= self.q90 + 1e-9):
            raise ValueError(
                f"Violation de monotonie des quantiles pour {self.sku_id}: "
                f"q10={self.q10}, q50={self.q50}, q90={self.q90}"
            )
        return self


class CriticalityPrediction(BaseModel):
    """Prediction probabiliste calibree du risque de rupture de stock a horizon H."""

    model_config = ConfigDict(frozen=True)

    sku_id: str
    obs_date: str
    horizon_days: int = Field(..., ge=1)
    raw_uncalibrated_prob: float = Field(..., ge=0.0, le=1.0)
    calibrated_criticality_prob: float = Field(..., ge=0.0, le=1.0)
    calibration_method: Literal["isotonic", "sigmoid", "uncalibrated"]
    is_critical_alert: bool = Field(..., description="True si calibrated_criticality_prob >= 0.50")
    top_shap_drivers: list[ShapFeatureContribution] = Field(default_factory=list)


class AnomalyDiagnosticsPayload(BaseModel):
    """Diagnostic d'anomalie produit par IsolationForest (contrat d'outil agentique)."""

    model_config = ConfigDict(frozen=True)

    sku_id: str
    obs_date: str
    anomaly_score: float = Field(
        ...,
        description="Score d'anomalie continu (positif = anormal, negatif = normal)",
    )
    is_anomaly: bool = Field(..., description="True si detecte comme anomalie ou donnee corrompue")
    data_corruption_detected: bool = Field(
        ...,
        description="True si stock < 0, prix <= 0 ou demande NaN/negative",
    )
    triggered_reasons: list[str] = Field(
        default_factory=list,
        description="Variables explicatives ayant declenche l'alerte",
    )


class CalibrationMethodMetrics(BaseModel):
    """Comparaison quantitative avant et apres calibration sur un pli hors-echantillon."""

    model_config = ConfigDict(frozen=True)

    method: Literal["uncalibrated", "sigmoid", "isotonic"]
    pr_auc: float = Field(..., ge=0.0, le=1.0)
    brier_score: float = Field(..., ge=0.0, le=1.0)
    ece: float = Field(..., ge=0.0, le=1.0)
    mean_predicted_prob: float = Field(..., ge=0.0, le=1.0)
    empirical_positive_rate: float = Field(..., ge=0.0, le=1.0)


class HorizonEvaluationReport(BaseModel):
    """Rapport complet pour un horizon H in {7, 14, 28} comparant Baselines et CatBoost."""

    horizon_days: int
    catboost_summary: BaselineComparisonSummary
    seasonal_naive_summary: BaselineComparisonSummary
    moving_average_summary: BaselineComparisonSummary
    calibration_comparison: list[CalibrationMethodMetrics]
    selected_calibration_method: Literal["isotonic", "sigmoid"]
    mase_improvement_pct_vs_best_baseline: float
    pr_auc_improvement_pct_vs_best_baseline: float
    brier_reduction_pct_after_calibration: float
    ece_reduction_pct_after_calibration: float


# ============================================================================
# CONTRATS PYDANTIC V2 STRICTS ET SCHEMAS D'OUTILS TYPES (extra="forbid")
# ============================================================================


class QuantileForecast(BaseModel):
    """Triplet de quantiles journaliers avec validation stricte de monotonie q10 <= q50 <= q90."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    q10: float = Field(..., ge=0.0, description="Quantile 10% de la demande journaliere")
    q50: float = Field(..., ge=0.0, description="Quantile 50% (mediane) de la demande journaliere")
    q90: float = Field(..., ge=0.0, description="Quantile 90% de la demande journaliere")
    relative_uncertainty: float = Field(
        ...,
        ge=0.0,
        description="Incertitude relative (q90 - q10) / max(q50, 1.0)",
    )
    high_uncertainty_flag: bool = Field(
        ...,
        description="True si relative_uncertainty > tau_uncert (declenche l'abstention de l'agent)",
    )

    @model_validator(mode="after")
    def _validate_monotonicity(self) -> QuantileForecast:
        if not (self.q10 <= self.q50 + 1e-9 and self.q50 <= self.q90 + 1e-9):
            raise ValueError(
                f"Sequence de quantiles non monotone rejetee: "
                f"q10={self.q10}, q50={self.q50}, q90={self.q90}"
            )
        return self


class ForecastPayload(BaseModel):
    """Reponse de GET /v1/skus/{sku_id}/forecast et contrat d'outil agentique."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sku_id: str = Field(..., min_length=3)
    obs_date: str
    horizon_days: int = Field(..., ge=1, le=56)
    quantiles: QuantileForecast
    total_demand_q10: float = Field(..., ge=0.0, description="Demande cumulee q10 sur H jours")
    total_demand_q50: float = Field(..., ge=0.0, description="Demande cumulee q50 sur H jours")
    total_demand_q90: float = Field(..., ge=0.0, description="Demande cumulee q90 sur H jours")
    calibrated_criticality_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilite calibree de rupture de stock dans [0, 1]",
    )
    historical_mase_4w: float = Field(
        ...,
        ge=0.0,
        description="Erreur historique locale MASE sur les 4 dernieres semaines",
    )
    model_version: str = Field(..., min_length=1)
    top_shap_drivers: list[ShapFeatureContribution] = Field(default_factory=list)


class CriticalityPayload(BaseModel):
    """Reponse de GET /v1/skus/{sku_id}/criticality avec probabilite calibree dans [0, 1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sku_id: str = Field(..., min_length=3)
    obs_date: str
    horizon_days: int = Field(..., ge=1, le=56)
    raw_uncalibrated_prob: float = Field(..., ge=0.0, le=1.0)
    calibrated_criticality_prob: float = Field(..., ge=0.0, le=1.0)
    calibration_method: Literal["isotonic", "sigmoid", "uncalibrated"]
    is_critical_alert: bool
    model_version: str = Field(..., min_length=1)
    top_shap_drivers: list[ShapFeatureContribution] = Field(default_factory=list)


class AnomalyRequest(BaseModel):
    """Corps de requete pour POST /v1/anomalies/evaluate."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=3, description="Identifiant du SKU a inspecter")
    window_days: int = Field(default=14, ge=1, le=90, description="Fenetre d'observation en jours")
    obs_date: str | None = Field(default=None, description="Date ISO optionnelle")
    observed_demand_qty: float | None = Field(
        default=None, description="Surcharge optionnelle de la demande observee"
    )
    predicted_q50_qty: float | None = Field(
        default=None, description="Surcharge optionnelle de la prevision mediane"
    )
    on_hand_qty: float | None = Field(
        default=None, description="Surcharge optionnelle du stock physique (teste corruption < 0)"
    )
    unit_price_eur: float | None = Field(
        default=None, description="Surcharge optionnelle du prix unitaire (teste corruption <= 0)"
    )
    observed_lead_time_days: float | None = Field(
        default=None, description="Surcharge optionnelle du delai fournisseur observe"
    )


class AnomalyPayload(BaseModel):
    """Reponse de POST /v1/anomalies/evaluate et contrat d'outil agentique."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sku_id: str = Field(..., min_length=3)
    obs_date: str
    window_days: int = Field(..., ge=1)
    anomaly_score: float = Field(
        ..., description="Score continu (positif = anomalie, oppose de decision_function)"
    )
    is_anomaly: bool
    data_corruption_detected: bool
    triggered_reasons: list[str] = Field(default_factory=list)
    feature_diagnostics: dict[str, float] = Field(default_factory=dict)
    model_version: str = Field(..., min_length=1)


class InventoryRecord(BaseModel):
    """Etat SQL d'inventaire et parametres fournisseur (contrat d'outil SQL)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sku_id: str = Field(..., min_length=3)
    category: str
    obs_date: str
    on_hand_qty: float = Field(..., ge=0.0)
    in_transit_qty: float = Field(..., ge=0.0)
    effective_stock_qty: float = Field(..., ge=0.0)
    reorder_point_s: float = Field(..., ge=0.0)
    order_up_to_S: float = Field(..., ge=0.0)
    moq_units: int = Field(..., ge=1)
    contractual_lead_time_days: int = Field(..., ge=1)
    empirical_lead_time_mean: float = Field(..., gt=0.0)
    empirical_lead_time_std: float = Field(..., ge=0.0)
    unit_cost_eur: float = Field(..., gt=0.0)
    holding_cost_per_day_eur: float = Field(..., gt=0.0)
    stockout_penalty_per_unit_eur: float = Field(..., gt=0.0)


class PolicySimulationResult(BaseModel):
    """Resultat deterministe de simulation de politique de commande (100% NumPy, sans LLM)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sku_id: str = Field(..., min_length=3)
    horizon_days: int = Field(..., ge=1)
    requested_qty: int = Field(..., ge=0)
    effective_order_qty: int = Field(..., ge=0)
    expedited: bool
    moq_units: int = Field(..., ge=1)
    moq_compliant: bool
    budget_limit_eur: float = Field(..., gt=0.0)
    within_budget: bool
    order_cost: float = Field(..., ge=0.0)
    holding_cost: float = Field(..., ge=0.0)
    stockout_penalty: float = Field(..., ge=0.0)
    total_cost: float = Field(..., ge=0.0)
    expected_ending_inventory: float = Field(..., ge=0.0)
    expected_stockout_units: float = Field(..., ge=0.0)
    requires_human_approval: bool

    @model_validator(mode="after")
    def _validate_cost_additive_invariant(self) -> PolicySimulationResult:
        recomputed = self.order_cost + self.holding_cost + self.stockout_penalty
        if abs(recomputed - self.total_cost) > 1e-2:
            raise ValueError(
                f"Violation de l'invariant de cout: "
                f"{self.order_cost} + {self.holding_cost} + {self.stockout_penalty} "
                f"!= {self.total_cost}"
            )
        return self


class ErrorResponse(BaseModel):
    """Schema d'erreur HTTP standardise (404, 422, 503)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error_code: str
    message: str
    detail: str
    sku_id: str | None = None


class HealthResponse(BaseModel):
    """Reponse de GET /health (200 si modeles @champion charges, 503 sinon)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["healthy", "unavailable"]
    models_loaded: bool
    db_connected: bool
    champion_alias: str
    champion_versions: dict[str, str] = Field(default_factory=dict)
