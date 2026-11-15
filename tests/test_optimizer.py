"""Tests unitaires deterministes de InventoryOptimizer."""

from __future__ import annotations

import pytest

from supply_chain.optimizer import InventoryOptimizer
from supply_chain.schemas import InventoryRecord, QuantileForecast


def _make_inventory(
    on_hand: float = 50.0,
    in_transit: float = 0.0,
    moq: int = 40,
    unit_cost: float = 20.0,
    holding_cost: float = 0.10,
    penalty: float = 35.0,
    lead_time: float = 3.0,
) -> InventoryRecord:
    return InventoryRecord(
        sku_id="SKU-OPT-01",
        category="industrial_sensors",
        obs_date="2026-11-30",
        on_hand_qty=on_hand,
        in_transit_qty=in_transit,
        effective_stock_qty=on_hand + in_transit,
        reorder_point_s=60.0,
        order_up_to_S=200.0,
        moq_units=moq,
        contractual_lead_time_days=int(lead_time),
        empirical_lead_time_mean=lead_time,
        empirical_lead_time_std=0.5,
        unit_cost_eur=unit_cost,
        holding_cost_per_day_eur=holding_cost,
        stockout_penalty_per_unit_eur=penalty,
    )


@pytest.mark.parametrize(
    "on_hand,in_transit,order_qty,horizon,expedited,q10,q50,q90,unit_cost,holding,penalty,lead_time",
    [
        (50.0, 0.0, 0, 7, False, 5.0, 10.0, 15.0, 20.0, 0.10, 35.0, 3.0),
        (50.0, 0.0, 40, 7, False, 5.0, 10.0, 15.0, 20.0, 0.10, 35.0, 3.0),
        (50.0, 0.0, 100, 7, True, 5.0, 10.0, 15.0, 20.0, 0.10, 35.0, 4.0),
        (10.0, 5.0, 120, 14, False, 8.0, 12.0, 20.0, 15.0, 0.08, 50.0, 5.0),
        (200.0, 50.0, 0, 7, False, 2.0, 4.0, 6.0, 10.0, 0.25, 15.0, 2.0),
        (0.0, 0.0, 80, 7, True, 10.0, 15.0, 25.0, 45.0, 0.15, 80.0, 6.0),
        (30.0, 10.0, 25, 7, False, 6.0, 9.0, 14.0, 12.0, 0.05, 30.0, 2.0),
        (80.0, 0.0, 300, 28, False, 4.0, 7.0, 11.0, 25.0, 0.12, 40.0, 7.0),
        (15.0, 0.0, 60, 14, True, 9.0, 14.0, 19.0, 30.0, 0.20, 60.0, 8.0),
        (120.0, 20.0, 50, 7, False, 12.0, 18.0, 24.0, 8.0, 0.04, 22.0, 1.0),
        (5.0, 0.0, 150, 14, False, 15.0, 22.0, 30.0, 18.0, 0.09, 45.0, 4.0),
        (60.0, 30.0, 0, 28, False, 3.0, 5.0, 8.0, 50.0, 0.30, 90.0, 5.0),
    ],
)
def test_cost_additive_invariant_across_12_scenarios(
    on_hand: float,
    in_transit: float,
    order_qty: int,
    horizon: int,
    expedited: bool,
    q10: float,
    q50: float,
    q90: float,
    unit_cost: float,
    holding: float,
    penalty: float,
    lead_time: float,
) -> None:
    """Verifie l'invariant de conservation des couts a 10^-2 EUR pres."""
    opt = InventoryOptimizer()
    inv = _make_inventory(
        on_hand=on_hand,
        in_transit=in_transit,
        unit_cost=unit_cost,
        holding_cost=holding,
        penalty=penalty,
        lead_time=lead_time,
    )
    rel_u = (q90 - q10) / max(q50, 1.0)
    fc = QuantileForecast(
        q10=q10, q50=q50, q90=q90, relative_uncertainty=rel_u, high_uncertainty_flag=rel_u > 0.8
    )

    res = opt.simulate_policy(
        inventory=inv,
        forecast=fc,
        order_qty=order_qty,
        horizon_days=horizon,
        expedited=expedited,
        budget_limit_eur=25000.0,
    )

    recomputed_total = res.order_cost + res.holding_cost + res.stockout_penalty
    assert abs(recomputed_total - res.total_cost) < 1e-2


def test_moq_budget_and_human_approval_guardrails() -> None:
    """Verifie la detection MOQ, le plafond budgetaire et l'escalade humaine > 5000 EUR."""
    opt = InventoryOptimizer()
    inv = _make_inventory(on_hand=10.0, moq=50, unit_cost=60.0, penalty=90.0, lead_time=2.0)
    fc = QuantileForecast(
        q10=8.0, q50=12.0, q90=16.0, relative_uncertainty=0.66, high_uncertainty_flag=False
    )

    # 1. Commande inferieure au MOQ (20 < 50) -> moq_compliant=False
    sim_low_moq = opt.simulate_policy(inv, fc, order_qty=20, horizon_days=7, expedited=False)
    assert sim_low_moq.moq_compliant is False

    # 2. Commande > 5000 EUR (100 * 60 + 45 = 6045 EUR) -> requires_human_approval=True
    sim_high_eur = opt.simulate_policy(
        inv, fc, order_qty=100, horizon_days=7, expedited=False, budget_limit_eur=10000.0
    )
    assert sim_high_eur.moq_compliant is True
    assert sim_high_eur.within_budget is True
    assert sim_high_eur.requires_human_approval is True

    # 3. Commande depassant le budget (budget = 3000 EUR) -> within_budget=False
    sim_over_budget = opt.simulate_policy(
        inv, fc, order_qty=100, horizon_days=7, expedited=False, budget_limit_eur=3000.0
    )
    assert sim_over_budget.within_budget is False

    # 4. Optimisation deterministe sous contrainte MOQ et budget
    optimal = opt.recommend_optimal_policy(
        inv, fc, horizon_days=7, budget_limit_eur=4500.0, allow_expedited=False
    )
    assert optimal.moq_compliant is True
    assert optimal.within_budget is True
