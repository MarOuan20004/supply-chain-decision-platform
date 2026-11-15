"""Simulateur deterministe NumPy pur d'optimisation de politique de reapprovisionnement (0% LLM)."""

from __future__ import annotations

import numpy as np

from supply_chain.config import SupplyChainSettings, settings
from supply_chain.schemas import InventoryRecord, PolicySimulationResult, QuantileForecast

QUANTILE_WEIGHTS = np.array([0.25, 0.50, 0.25], dtype=np.float64)


class InventoryOptimizer:
    """Evalue et optimise les politiques de commande sous incertitude quantile (100% NumPy)."""

    def __init__(self, cfg: SupplyChainSettings = settings) -> None:
        self.cfg = cfg

    def simulate_policy(
        self,
        inventory: InventoryRecord,
        forecast: QuantileForecast,
        order_qty: int,
        horizon_days: int = 7,
        expedited: bool = False,
        budget_limit_eur: float | None = None,
    ) -> PolicySimulationResult:
        """Simule de maniere deterministe le cout total espere d'une commande donnee."""
        if order_qty < 0:
            raise ValueError(f"order_qty doit etre >= 0 (recu: {order_qty})")
        if horizon_days < 1:
            raise ValueError(f"horizon_days doit etre >= 1 (recu: {horizon_days})")

        budget = float(
            budget_limit_eur if budget_limit_eur is not None else self.cfg.default_budget_eur
        )
        if budget <= 0.0:
            raise ValueError(f"budget_limit_eur doit etre > 0 (recu: {budget})")

        effective_qty = int(order_qty)
        moq_compliant = bool(effective_qty == 0 or effective_qty >= inventory.moq_units)

        # Calcul du delai effectif et de la fraction de commande recue dans l'horizon H
        lead_time = float(inventory.empirical_lead_time_mean)
        if expedited:
            lead_time = max(1.0, lead_time * 0.5)
        arrival_fraction = float(
            np.clip((horizon_days - lead_time) / float(horizon_days), 0.0, 1.0)
        )

        available_supply = (
            float(inventory.on_hand_qty)
            + float(inventory.in_transit_qty)
            + arrival_fraction * float(effective_qty)
        )

        # Scenarios de demande cumulee sur H jours pour q10, q50, q90
        daily_quantiles = np.array([forecast.q10, forecast.q50, forecast.q90], dtype=np.float64)
        horizon_demand = daily_quantiles * float(horizon_days)

        ending_inv_scenarios = np.maximum(0.0, available_supply - horizon_demand)
        stockout_scenarios = np.maximum(0.0, horizon_demand - available_supply)
        avg_inv_scenarios = 0.5 * (float(inventory.on_hand_qty) + ending_inv_scenarios)

        expected_ending_inv = float(np.dot(QUANTILE_WEIGHTS, ending_inv_scenarios))
        expected_stockout = float(np.dot(QUANTILE_WEIGHTS, stockout_scenarios))
        expected_avg_inv = float(np.dot(QUANTILE_WEIGHTS, avg_inv_scenarios))

        # Calcul des trois postes de couts arrondis a 4 decimales
        unit_price = float(inventory.unit_cost_eur) * (
            self.cfg.expedited_cost_multiplier if expedited else 1.0
        )
        setup_cost = self.cfg.fixed_order_setup_cost_eur if effective_qty > 0 else 0.0

        order_cost = round(setup_cost + float(effective_qty) * unit_price, 4)
        holding_cost = round(
            expected_avg_inv * float(inventory.holding_cost_per_day_eur) * float(horizon_days),
            4,
        )
        stockout_penalty = round(
            expected_stockout * float(inventory.stockout_penalty_per_unit_eur),
            4,
        )
        # Invariant additif exact garanti
        total_cost = round(order_cost + holding_cost + stockout_penalty, 4)

        within_budget = bool(order_cost <= budget + 1e-9)
        requires_hitl = bool(expedited or (order_cost > self.cfg.human_approval_threshold_eur))

        return PolicySimulationResult(
            sku_id=inventory.sku_id,
            horizon_days=int(horizon_days),
            requested_qty=int(order_qty),
            effective_order_qty=effective_qty,
            expedited=bool(expedited),
            moq_units=inventory.moq_units,
            moq_compliant=moq_compliant,
            budget_limit_eur=round(budget, 2),
            within_budget=within_budget,
            order_cost=order_cost,
            holding_cost=holding_cost,
            stockout_penalty=stockout_penalty,
            total_cost=total_cost,
            expected_ending_inventory=round(expected_ending_inv, 4),
            expected_stockout_units=round(expected_stockout, 4),
            requires_human_approval=requires_hitl,
        )

    def recommend_optimal_policy(
        self,
        inventory: InventoryRecord,
        forecast: QuantileForecast,
        horizon_days: int = 7,
        budget_limit_eur: float | None = None,
        allow_expedited: bool = True,
    ) -> PolicySimulationResult:
        """Recherche deterministe de la quantite optimale respectant MOQ et budget."""
        budget = float(
            budget_limit_eur if budget_limit_eur is not None else self.cfg.default_budget_eur
        )
        max_target = int(
            max(
                inventory.order_up_to_S * 2.0,
                forecast.q90 * horizon_days * 2.0,
                float(inventory.moq_units),
            )
        )
        step = max(1, inventory.moq_units // 4)

        candidate_qtys = [0] + list(range(inventory.moq_units, max_target + step, step))
        expedited_options = [False, True] if allow_expedited else [False]

        best_result: PolicySimulationResult | None = None
        for exp_flag in expedited_options:
            for qty in candidate_qtys:
                sim = self.simulate_policy(
                    inventory=inventory,
                    forecast=forecast,
                    order_qty=qty,
                    horizon_days=horizon_days,
                    expedited=exp_flag,
                    budget_limit_eur=budget,
                )
                if not (sim.moq_compliant and sim.within_budget):
                    continue
                if best_result is None or sim.total_cost < best_result.total_cost:
                    best_result = sim

        if best_result is None:
            # Repli sur commande nulle (0 unite, toujours conforme MOQ et budget)
            best_result = self.simulate_policy(
                inventory=inventory,
                forecast=forecast,
                order_qty=0,
                horizon_days=horizon_days,
                expedited=False,
                budget_limit_eur=budget,
            )
        return best_result
