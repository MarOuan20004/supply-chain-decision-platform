"""Tests unitaires des contrats Pydantic v2 stricts et des contrats d'outils agentiques."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from supply_chain.schemas import (
    AnomalyPayload,
    AnomalyRequest,
    CriticalityPayload,
    ErrorResponse,
    ForecastPayload,
    InventoryRecord,
    PolicySimulationResult,
    QuantileForecast,
)


def test_quantile_forecast_rejects_non_monotonic_sequences() -> None:
    """Verifie que QuantileForecast rejette q10 > q50 et q50 > q90."""
    valid = QuantileForecast(
        q10=10.0, q50=15.0, q90=22.0, relative_uncertainty=0.80, high_uncertainty_flag=False
    )
    assert valid.q10 <= valid.q50 <= valid.q90

    with pytest.raises(ValidationError, match="non monotone"):
        QuantileForecast(
            q10=18.0, q50=15.0, q90=22.0, relative_uncertainty=0.26, high_uncertainty_flag=False
        )

    with pytest.raises(ValidationError, match="non monotone"):
        QuantileForecast(
            q10=10.0, q50=25.0, q90=22.0, relative_uncertainty=0.48, high_uncertainty_flag=False
        )


def test_probability_bounds_enforced_in_0_1() -> None:
    """Verifie que les probabilites calibrees hors de [0, 1] sont rejetees par Pydantic."""
    q_ok = QuantileForecast(
        q10=5.0, q50=10.0, q90=15.0, relative_uncertainty=1.0, high_uncertainty_flag=True
    )
    with pytest.raises(ValidationError):
        ForecastPayload(
            sku_id="SKU-001",
            obs_date="2026-11-30",
            horizon_days=7,
            quantiles=q_ok,
            total_demand_q10=35.0,
            total_demand_q50=70.0,
            total_demand_q90=105.0,
            calibrated_criticality_prob=1.05,
            historical_mase_4w=0.62,
            model_version="forecaster@v1",
        )

    with pytest.raises(ValidationError):
        CriticalityPayload(
            sku_id="SKU-001",
            obs_date="2026-11-30",
            horizon_days=7,
            raw_uncalibrated_prob=0.4,
            calibrated_criticality_prob=-0.01,
            calibration_method="isotonic",
            is_critical_alert=False,
            model_version="clf@v1",
        )


def test_extra_forbid_and_agent_schemas_importable() -> None:
    """Verifie extra='forbid' et l'instanciation des schemas requis par l'Agent."""
    with pytest.raises(ValidationError):
        AnomalyRequest(sku_id="SKU-001", unexpected_field="injection")  # type: ignore[call-arg]

    inv = InventoryRecord(
        sku_id="SKU-001",
        category="electronics",
        obs_date="2026-11-30",
        on_hand_qty=120.0,
        in_transit_qty=30.0,
        effective_stock_qty=150.0,
        reorder_point_s=80.0,
        order_up_to_S=250.0,
        moq_units=50,
        contractual_lead_time_days=5,
        empirical_lead_time_mean=5.2,
        empirical_lead_time_std=0.8,
        unit_cost_eur=12.5,
        holding_cost_per_day_eur=0.05,
        stockout_penalty_per_unit_eur=25.0,
    )
    assert inv.effective_stock_qty == 150.0

    anom = AnomalyPayload(
        sku_id="SKU-001",
        obs_date="2026-11-30",
        window_days=14,
        anomaly_score=-0.12,
        is_anomaly=False,
        data_corruption_detected=False,
        triggered_reasons=[],
        feature_diagnostics={"lead_time_zscore": 0.4},
        model_version="anom@v1",
    )
    assert anom.is_anomaly is False

    sim = PolicySimulationResult(
        sku_id="SKU-001",
        horizon_days=7,
        requested_qty=100,
        effective_order_qty=100,
        expedited=False,
        moq_units=50,
        moq_compliant=True,
        budget_limit_eur=15000.0,
        within_budget=True,
        order_cost=1295.0,
        holding_cost=42.0,
        stockout_penalty=0.0,
        total_cost=1337.0,
        expected_ending_inventory=145.0,
        expected_stockout_units=0.0,
        requires_human_approval=False,
    )
    assert sim.total_cost == 1337.0

    err = ErrorResponse(
        error_code="SKU_NOT_FOUND",
        message="Not found",
        detail="Missing SKU",
        sku_id="SKU-999",
    )
    assert err.error_code == "SKU_NOT_FOUND"
