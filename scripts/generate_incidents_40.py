#!/usr/bin/env python3
"""Generateur deterministe du jeu d'evaluation de 40 scenarios Supply Chain.

Produit benchmarks/supply_chain_incidents_40.json (4 categories x 10 scenarios = 40 incidents)
avec les fixtures exactes des sorties d'outils et la decision oracle annotee.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compute_deterministic_policy_cost(
    current_stock: float,
    in_transit_stock: float,
    q10: float,
    q50: float,
    q90: float,
    unit_cost_eur: float,
    order_qty: int,
    expedited: bool,
    horizon_days: int = 14,
    holding_rate_per_day: float = 0.02,
    stockout_penalty_mult: float = 3.5,
) -> dict[str, float]:
    """Calcule le cout total espere pondere par quantiles (0.25*q10 + 0.50*q50 + 0.25*q90)."""
    safe_unit_cost = max(0.0, float(unit_cost_eur))
    shipping_cost = (450.0 if expedited else 120.0) if order_qty > 0 else 0.0
    order_cost = round(float(order_qty) * safe_unit_cost + shipping_cost, 2)

    available = (
        max(0.0, float(current_stock)) + max(0.0, float(in_transit_stock)) + float(order_qty)
    )
    weights = [(0.25, q10), (0.50, q50), (0.25, q90)]

    expected_holding = 0.0
    expected_penalty = 0.0
    for w, demand in weights:
        ending_inv = max(0.0, available - float(demand))
        shortage = max(0.0, float(demand) - available)
        expected_holding += w * ending_inv * safe_unit_cost * holding_rate_per_day * horizon_days
        expected_penalty += w * shortage * safe_unit_cost * stockout_penalty_mult

    holding_cost = round(expected_holding, 2)
    stockout_penalty = round(expected_penalty, 2)
    total_cost = round(order_cost + holding_cost + stockout_penalty, 2)

    return {
        "order_cost": order_cost,
        "holding_cost": holding_cost,
        "stockout_penalty": stockout_penalty,
        "total_cost": total_cost,
    }


def find_optimal_order_qty(
    current_stock: float,
    in_transit_stock: float,
    q10: float,
    q50: float,
    q90: float,
    unit_cost_eur: float,
    moq: int,
    expedited: bool,
    max_unit_cost_cap: float | None = None,
) -> tuple[int, dict[str, float]]:
    """Recherche par grille sur les multiples de 10 (>= moq) la quantite minimisant total_cost."""
    best_qty = max(moq, 10)
    best_sim = compute_deterministic_policy_cost(
        current_stock=current_stock,
        in_transit_stock=in_transit_stock,
        q10=q10,
        q50=q50,
        q90=q90,
        unit_cost_eur=unit_cost_eur,
        order_qty=best_qty,
        expedited=expedited,
    )

    upper_qty = max(moq + 200, int(math.ceil(q90 * 1.5)))
    for candidate_qty in range(moq, upper_qty + 1, 10):
        if max_unit_cost_cap is not None and (candidate_qty * unit_cost_eur) > max_unit_cost_cap:
            continue
        sim = compute_deterministic_policy_cost(
            current_stock=current_stock,
            in_transit_stock=in_transit_stock,
            q10=q10,
            q50=q50,
            q90=q90,
            unit_cost_eur=unit_cost_eur,
            order_qty=candidate_qty,
            expedited=expedited,
        )
        if sim["total_cost"] < best_sim["total_cost"]:
            best_qty = candidate_qty
            best_sim = sim

    return best_qty, best_sim


def build_40_scenarios() -> list[dict[str, Any]]:
    """Construit la liste deterministe des 40 scenarios repartis en 4 categories de 10."""
    scenarios: list[dict[str, Any]] = []

    # =========================================================================
    # CATEGORIE 1 : 10 ruptures standards previsibles -> APPROVED_AUTO
    # =========================================================================
    for idx in range(1, 11):
        sku_id = f"SKU-STD-{idx:03d}"
        q50 = float(100 + idx * 10)
        q10 = round(q50 * 0.85, 2)
        q90 = round(q50 * 1.18, 2)  # (q90 - q10)/q50 = 0.33 <= 0.80
        reorder_s = 60 + idx * 4
        current_stock = (reorder_s + 5) if idx % 2 == 0 else max(15, reorder_s - 25)
        in_transit = 0
        moq = 40 if idx % 3 != 0 else 60
        unit_cost = float(15 + (idx % 5) * 2)  # Entre 15 et 23 EUR
        order_up_to_S = reorder_s + 130

        opt_qty, opt_sim = find_optimal_order_qty(
            current_stock=current_stock,
            in_transit_stock=in_transit,
            q10=q10,
            q50=q50,
            q90=q90,
            unit_cost_eur=unit_cost,
            moq=moq,
            expedited=False,
            max_unit_cost_cap=4800.0,
        )

        scenarios.append(
            {
                "scenario_id": f"INC-{idx:02d}",
                "category": "standard_stockout",
                "sku_id": sku_id,
                "trigger_event": "stockout_risk_high",
                "horizon_days": 14,
                "tool_fixtures": {
                    "forecast": {
                        "sku_id": sku_id,
                        "forecasts": [{"horizon_days": 14, "q10": q10, "q50": q50, "q90": q90}],
                        "criticality_prob": round(0.74 + idx * 0.015, 3),
                        "historical_mase": round(0.58 + (idx % 4) * 0.03, 3),
                    },
                    "inventory": {
                        "sku_id": sku_id,
                        "current_stock": current_stock,
                        "in_transit_stock": in_transit,
                        "reorder_point_s": reorder_s,
                        "order_up_to_S": order_up_to_S,
                        "lead_time_mean_days": 4.0,
                        "lead_time_std_days": 0.8,
                        "unit_cost_eur": unit_cost,
                        "moq": moq,
                    },
                    "anomaly": {
                        "sku_id": sku_id,
                        "is_anomaly": False,
                        "anomaly_score": -0.14,
                        "feature_contributions": {
                            "demand_trend_7d": 0.62,
                            "stock_coverage_ratio": -0.31,
                        },
                    },
                },
                "oracle_decision": {
                    "status": "APPROVED_AUTO",
                    "order_qty": opt_qty,
                    "expedited": False,
                    "expected_total_cost_eur": opt_sim["total_cost"],
                },
                "rationale": (
                    f"Rupture previsible (p_crit={0.74 + idx * 0.015:.3f}), donnees saines, "
                    f"incertitude faible (0.33 <= 0.80) et engagement "
                    f"{opt_qty * unit_cost:.2f} EUR <= 5000 EUR : "
                    f"commande optimale standard de {opt_qty} unites."
                ),
            }
        )

    # =========================================================================
    # CATEGORIE 2 : 10 chocs de demande MOQ/delai -> PENDING_HUMAN_APPROVAL
    # =========================================================================
    for idx in range(11, 21):
        local_i = idx - 10
        sku_id = f"SKU-SHK-{local_i:03d}"
        q50 = float(220 + local_i * 25)
        q10 = round(q50 * 0.80, 2)
        q90 = round(q50 * 1.25, 2)  # (q90 - q10)/q50 = 0.45 <= 0.80
        current_stock = 35 + local_i * 3
        in_transit = 10
        reorder_s = 90
        order_up_to_S = 320
        needs_expedited = local_i % 2 == 1
        moq = 150 if not needs_expedited else 80
        unit_cost = 42.0 if not needs_expedited else 28.0
        lead_time_mean = 16.0 if needs_expedited else 6.0

        opt_qty, opt_sim = find_optimal_order_qty(
            current_stock=current_stock,
            in_transit_stock=in_transit,
            q10=q10,
            q50=q50,
            q90=q90,
            unit_cost_eur=unit_cost,
            moq=moq,
            expedited=needs_expedited,
        )

        scenarios.append(
            {
                "scenario_id": f"INC-{idx:02d}",
                "category": "demand_shock_moq_leadtime",
                "sku_id": sku_id,
                "trigger_event": "demand_shock_critical",
                "horizon_days": 14,
                "tool_fixtures": {
                    "forecast": {
                        "sku_id": sku_id,
                        "forecasts": [{"horizon_days": 14, "q10": q10, "q50": q50, "q90": q90}],
                        "criticality_prob": round(0.86 + local_i * 0.01, 3),
                        "historical_mase": 0.64,
                    },
                    "inventory": {
                        "sku_id": sku_id,
                        "current_stock": current_stock,
                        "in_transit_stock": in_transit,
                        "reorder_point_s": reorder_s,
                        "order_up_to_S": order_up_to_S,
                        "lead_time_mean_days": lead_time_mean,
                        "lead_time_std_days": 1.5,
                        "unit_cost_eur": unit_cost,
                        "moq": moq,
                    },
                    "anomaly": {
                        "sku_id": sku_id,
                        "is_anomaly": True,
                        "anomaly_score": 0.22,
                        "feature_contributions": {
                            "demand_spike_zscore": 2.45,
                            "supplier_lead_time_days": lead_time_mean,
                        },
                    },
                },
                "oracle_decision": {
                    "status": "PENDING_HUMAN_APPROVAL",
                    "order_qty": opt_qty,
                    "expedited": needs_expedited,
                    "expected_total_cost_eur": opt_sim["total_cost"],
                },
                "rationale": (
                    f"Choc de demande confirme (q50={q50}), commande de {opt_qty} unites "
                    f"(expedited={needs_expedited}, engagement={opt_qty * unit_cost:.2f} EUR) "
                    f"necessitant validation humaine obligatoire."
                ),
            }
        )

    # =========================================================================
    # CATEGORIE 3 : 10 fausses alertes (donnees corrompues) -> ABSTAIN
    # =========================================================================
    for idx in range(21, 31):
        local_i = idx - 20
        sku_id = f"SKU-COR-{local_i:03d}"
        q50 = 120.0
        q10 = 105.0
        q90 = 135.0

        if local_i <= 4:
            stock = -5 * local_i
            unit_cost = 25.0
            lt_mean, lt_std = 4.0, 0.8
            contribs: dict[str, float] = {"inventory_glitch": -3.2}
            defect_desc = f"current_stock={stock} negatif"
        elif local_i <= 7:
            stock = 20
            unit_cost = 0.0 if local_i < 7 else -12.0
            lt_mean, lt_std = 4.0, 0.8
            contribs = {"erp_price_missing": 1.9}
            defect_desc = f"unit_cost_eur={unit_cost} invalide"
        else:
            stock = 25
            unit_cost = 30.0
            lt_mean, lt_std = 3.0, 7.5 + local_i
            contribs = {}
            defect_desc = f"lead_time_std_days={lt_std} > lead_time_mean_days={lt_mean}"

        scenarios.append(
            {
                "scenario_id": f"INC-{idx:02d}",
                "category": "corrupted_sensor_false_alerts",
                "sku_id": sku_id,
                "trigger_event": "stockout_risk_high",
                "horizon_days": 14,
                "tool_fixtures": {
                    "forecast": {
                        "sku_id": sku_id,
                        "forecasts": [{"horizon_days": 14, "q10": q10, "q50": q50, "q90": q90}],
                        "criticality_prob": 0.89,
                        "historical_mase": 0.61,
                    },
                    "inventory": {
                        "sku_id": sku_id,
                        "current_stock": stock,
                        "in_transit_stock": 0,
                        "reorder_point_s": 65,
                        "order_up_to_S": 180,
                        "lead_time_mean_days": lt_mean,
                        "lead_time_std_days": lt_std,
                        "unit_cost_eur": unit_cost,
                        "moq": 40,
                    },
                    "anomaly": {
                        "sku_id": sku_id,
                        "is_anomaly": True,
                        "anomaly_score": 0.68,
                        "data_corrupted": True,
                        "feature_contributions": contribs,
                    },
                },
                "oracle_decision": {
                    "status": "ABSTAIN_INSUFFICIENT_DATA",
                    "order_qty": 0,
                    "expedited": False,
                    "expected_total_cost_eur": 0.0,
                },
                "rationale": (
                    f"Fausse alerte due a une corruption de donnees ({defect_desc}) : "
                    f"abstention obligatoire sans commande."
                ),
            }
        )

    # =========================================================================
    # CATEGORIE 4 : 10 situations a tres haute incertitude -> ABSTAIN
    # =========================================================================
    for idx in range(31, 41):
        local_i = idx - 30
        sku_id = f"SKU-UNC-{local_i:03d}"
        q50 = 100.0
        half_spread = 45.0 + local_i * 3.0
        q10 = round(q50 - half_spread, 2)
        q90 = round(q50 + half_spread, 2)
        rel_uncert = round((q90 - q10) / q50, 3)

        scenarios.append(
            {
                "scenario_id": f"INC-{idx:02d}",
                "category": "high_uncertainty_escalation",
                "sku_id": sku_id,
                "trigger_event": "volatility_alert",
                "horizon_days": 14,
                "tool_fixtures": {
                    "forecast": {
                        "sku_id": sku_id,
                        "forecasts": [{"horizon_days": 14, "q10": q10, "q50": q50, "q90": q90}],
                        "criticality_prob": 0.68,
                        "historical_mase": 1.35,
                    },
                    "inventory": {
                        "sku_id": sku_id,
                        "current_stock": 30,
                        "in_transit_stock": 0,
                        "reorder_point_s": 70,
                        "order_up_to_S": 200,
                        "lead_time_mean_days": 5.0,
                        "lead_time_std_days": 1.2,
                        "unit_cost_eur": 24.0,
                        "moq": 50,
                    },
                    "anomaly": {
                        "sku_id": sku_id,
                        "is_anomaly": False,
                        "anomaly_score": 0.05,
                        "feature_contributions": {
                            "regime_shift_entropy": 0.88,
                            "promo_calendar_overlap": 0.41,
                        },
                    },
                },
                "oracle_decision": {
                    "status": "ABSTAIN_INSUFFICIENT_DATA",
                    "order_qty": 0,
                    "expedited": False,
                    "expected_total_cost_eur": 0.0,
                },
                "rationale": (
                    f"Incertitude predictive excessive "
                    f"((q90 - q10)/q50 = {rel_uncert:.3f} > 0.80) : "
                    f"abstention et escalade humaine sans commande automatique."
                ),
            }
        )

    assert len(scenarios) == 40
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Genere benchmarks/supply_chain_incidents_40.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/supply_chain_incidents_40.json"),
        help="Chemin du fichier JSON de sortie.",
    )
    args = parser.parse_args()
    scenarios = build_40_scenarios()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Jeu de benchmark genere : {len(scenarios)} scenarios -> {args.output}")


if __name__ == "__main__":
    main()
