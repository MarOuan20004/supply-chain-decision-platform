"""Tests unitaires du registre des 5 outils types de l'Agent Supply Chain."""

from __future__ import annotations

import pytest

from supply_chain.agent.exceptions import ToolValidationError, UnknownToolError
from supply_chain.agent.schemas import DecisionProposal, ExecutionStatus
from supply_chain.agent.tools import SupplyChainToolRegistry
from supply_chain.schemas import (
    AnomalyPayload,
    ForecastPayload,
    InventoryRecord,
    QuantileForecast,
)


class DeterministicFakeBackend:
    """Backend en memoire fournissant des payloads deterministes pour tester les 5 outils."""

    def forecast(self, sku_id: str, horizon_days: int) -> ForecastPayload:
        q = QuantileForecast(
            q10=10.0, q50=15.0, q90=20.0, relative_uncertainty=0.6667, high_uncertainty_flag=False
        )
        return ForecastPayload(
            sku_id=sku_id,
            obs_date="2026-12-07",
            horizon_days=horizon_days,
            quantiles=q,
            total_demand_q10=10.0 * horizon_days,
            total_demand_q50=15.0 * horizon_days,
            total_demand_q90=20.0 * horizon_days,
            calibrated_criticality_prob=0.82,
            historical_mase_4w=0.54,
            model_version="forecaster@v1",
        )

    def anomaly(self, sku_id: str, window_days: int) -> AnomalyPayload:
        return AnomalyPayload(
            sku_id=sku_id,
            obs_date="2026-12-07",
            window_days=window_days,
            anomaly_score=-0.15,
            is_anomaly=False,
            data_corruption_detected=False,
            triggered_reasons=[],
            feature_diagnostics={"lead_time_zscore": 0.3},
            model_version="anomaly@v1",
        )

    def inventory(self, sku_id: str) -> InventoryRecord:
        return InventoryRecord(
            sku_id=sku_id,
            category="robotics_actuators",
            obs_date="2026-12-07",
            on_hand_qty=20.0,
            in_transit_qty=0.0,
            effective_stock_qty=20.0,
            reorder_point_s=60.0,
            order_up_to_S=180.0,
            moq_units=40,
            contractual_lead_time_days=3,
            empirical_lead_time_mean=3.0,
            empirical_lead_time_std=0.4,
            unit_cost_eur=50.0,
            holding_cost_per_day_eur=0.10,
            stockout_penalty_per_unit_eur=90.0,
        )


def test_five_tools_chain_and_hitl_threshold_5000_eur() -> None:
    """Verifie l'execution des 5 outils et le blocage PENDING_HUMAN_APPROVAL > 5000 EUR."""
    registry = SupplyChainToolRegistry(backend=DeterministicFakeBackend())

    _, fc_out, _ = registry.execute_tool(
        "get_sku_forecast_and_uncertainty", {"sku_id": "SKU-0001", "horizon_days": 7}
    )
    assert fc_out["calibrated_criticality_prob"] == 0.82

    _, anom_out, _ = registry.execute_tool(
        "inspect_anomaly_diagnostics", {"sku_id": "SKU-0001", "window_days": 14}
    )
    assert anom_out["data_corruption_detected"] is False

    _, inv_out, _ = registry.execute_tool(
        "query_inventory_and_supplier_sql", {"sku_id": "SKU-0001"}
    )
    assert inv_out["moq_units"] == 40

    # Commande de 120 unites a 50 EUR/u + 45 EUR fixe = 6045 EUR (> 5000 EUR)
    _, sim_out, _ = registry.execute_tool(
        "simulate_reorder_policy",
        {"sku_id": "SKU-0001", "order_qty": 120, "expedited": False},
    )
    assert sim_out["order_cost"] == pytest.approx(6045.0)
    assert sim_out["requires_human_approval"] is True

    proposal = DecisionProposal(
        sku_id="SKU-0001",
        horizon_days=7,
        order_qty=120,
        expedited=False,
        order_cost_eur=sim_out["order_cost"],
        expected_total_cost_eur=sim_out["total_cost"],
        confidence_level="HIGH",
        reasoning_summary="Probabilite de rupture 0.82, commande de 120 unites simulee.",
        evidence_tool_calls=[
            '"calibrated_criticality_prob": 0.82',
            f'"total_cost": {sim_out["total_cost"]}',
        ],
    )
    _, sub_out, _ = registry.execute_tool(
        "submit_decision_proposal", {"proposal": proposal.model_dump(mode="json")}
    )
    assert sub_out["status"] == ExecutionStatus.PENDING_HUMAN_APPROVAL.value
    assert sub_out["requires_human_approval"] is True


def test_tool_registry_rejects_unknown_tool_and_fabricated_numbers() -> None:
    """Verifie UnknownToolError, ToolValidationError et le rejet de chiffres non simules."""
    registry = SupplyChainToolRegistry(backend=DeterministicFakeBackend())

    with pytest.raises(UnknownToolError):
        registry.execute_tool("delete_database_table", {"sku_id": "SKU-0001"})

    with pytest.raises(ToolValidationError):
        registry.execute_tool(
            "get_sku_forecast_and_uncertainty",
            {"sku_id": "SKU-0001", "horizon_days": "not_an_int"},
        )

    # Tentative de soumission d'un cout invente sans appel prealable a simulate_reorder_policy
    fabricated_prop = DecisionProposal(
        sku_id="SKU-0001",
        horizon_days=7,
        order_qty=80,
        expedited=False,
        order_cost_eur=1234.56,
        expected_total_cost_eur=1500.00,
        confidence_level="MEDIUM",
        reasoning_summary="Tentative avec montant non issu du simulateur.",
        evidence_tool_calls=['"sku_id": "SKU-0001"'],
    )
    with pytest.raises(ToolValidationError, match="simulate_reorder_policy"):
        registry.execute_tool(
            "submit_decision_proposal", {"proposal": fabricated_prop.model_dump(mode="json")}
        )
