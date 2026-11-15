#!/usr/bin/env python3
"""Script de demonstration interactive (2 cas contrastes).

Cas 1 : Vraie rupture imminente -> enchainement des 5 outils -> montant > 5000 EUR
        -> statut force PENDING_HUMAN_APPROVAL (garde-fou GR-04-HITL-AMOUNT).
Cas 2 : Fausse alerte capteur -> stock negatif (-18) detecte -> abstention forcee
        -> statut ABSTAIN_INSUFFICIENT_DATA avec motif 'corrupted_input_data'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repository root and src/ are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.evaluate_agent_vs_baselines import FixtureBackedToolRegistry  # noqa: E402
from scripts.generate_incidents_40 import compute_deterministic_policy_cost  # noqa: E402
from supply_chain.agent.guardrails import DeterministicGuardrailEngine  # noqa: E402
from supply_chain.agent.llm_client import (  # noqa: E402
    MockLLMClient,
    MockLLMResponse,
    MockToolCall,
)
from supply_chain.agent.orchestrator import BoundedSupplyChainOrchestrator  # noqa: E402
from supply_chain.agent.schemas import IncidentTrigger  # noqa: E402
from supply_chain.config import SupplyChainSettings  # noqa: E402


def _build_case_1_true_stockout() -> tuple[dict[str, Any], MockLLMClient]:
    """Construit le Cas 1 : vraie rupture imminente avec commande > 5000 EUR."""
    sku_id = "SKU-ACT-101"
    sim = compute_deterministic_policy_cost(
        current_stock=22.0,
        in_transit_stock=0.0,
        q10=165.0,
        q50=190.0,
        q90=215.0,
        unit_cost_eur=35.0,
        order_qty=180,
        expedited=False,
        horizon_days=14,
    )
    scenario: dict[str, Any] = {
        "scenario_id": "DEMO-CASE-1",
        "category": "demand_shock_moq_leadtime",
        "sku_id": sku_id,
        "trigger_event": "stockout_risk_imminent",
        "horizon_days": 14,
        "tool_fixtures": {
            "forecast": {
                "sku_id": sku_id,
                "forecasts": [{"horizon_days": 14, "q10": 165.0, "q50": 190.0, "q90": 215.0}],
                "criticality_prob": 0.92,
                "historical_mase": 0.54,
            },
            "inventory": {
                "sku_id": sku_id,
                "current_stock": 22,
                "in_transit_stock": 0,
                "reorder_point_s": 85,
                "order_up_to_S": 240,
                "lead_time_mean_days": 5.0,
                "lead_time_std_days": 0.9,
                "unit_cost_eur": 35.0,
                "moq": 100,
            },
            "anomaly": {
                "sku_id": sku_id,
                "is_anomaly": False,
                "anomaly_score": -0.11,
                "feature_contributions": {
                    "demand_acceleration_7d": 0.74,
                    "stock_days_left": 1.6,
                },
            },
        },
    }

    draft = {
        "sku_id": sku_id,
        "status": "APPROVED_AUTO",
        "order_qty": 180,
        "expedited": False,
        "expected_total_cost_eur": sim["total_cost"],
        "confidence_level": 0.92,
        "evidence_tool_calls": [
            "q50=190.0, current_stock=22",
            f"order_qty=180, total_cost={sim['total_cost']}",
        ],
        "reasoning_summary": (
            f"Rupture imminente (p_crit=0.92, stock=22 vs q50=190.0). Commande de 180 unites "
            f"simulee a {sim['total_cost']} EUR (cout commande {sim['order_cost']} EUR)."
        ),
    }

    llm = MockLLMClient(
        scripted_responses=[
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="get_sku_forecast_and_uncertainty",
                        arguments={"sku_id": sku_id, "horizon_days": 14},
                    )
                ],
                tokens_used=42,
            ),
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="inspect_anomaly_diagnostics",
                        arguments={"sku_id": sku_id, "window_days": 14},
                    )
                ],
                tokens_used=38,
            ),
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="query_inventory_and_supplier_sql",
                        arguments={"sku_id": sku_id},
                    )
                ],
                tokens_used=35,
            ),
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="simulate_reorder_policy",
                        arguments={"sku_id": sku_id, "order_qty": 180, "expedited": False},
                    )
                ],
                tokens_used=46,
            ),
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="submit_decision_proposal",
                        arguments={
                            "sku_id": sku_id,
                            "action": "reorder",
                            "order_qty": 180,
                            "expedited": False,
                            "expected_total_cost_eur": sim["total_cost"],
                            "unit_cost_eur": 35.0,
                            "rationale": "Commande optimale de 180 unites pour couvrir q50=190.0.",
                        },
                    )
                ],
                tokens_used=52,
            ),
            MockLLMResponse(final_content=json.dumps(draft), tokens_used=68),
        ]
    )
    return scenario, llm


def _build_case_2_corrupted_sensor() -> tuple[dict[str, Any], MockLLMClient]:
    """Construit le Cas 2 : fausse alerte due a un stock negatif (-18) -> ABSTAIN."""
    sku_id = "SKU-SEN-404"
    scenario: dict[str, Any] = {
        "scenario_id": "DEMO-CASE-2",
        "category": "corrupted_sensor_false_alerts",
        "sku_id": sku_id,
        "trigger_event": "stockout_risk_high",
        "horizon_days": 14,
        "tool_fixtures": {
            "forecast": {
                "sku_id": sku_id,
                "forecasts": [{"horizon_days": 14, "q10": 80.0, "q50": 95.0, "q90": 110.0}],
                "criticality_prob": 0.94,
                "historical_mase": 0.59,
            },
            "inventory": {
                "sku_id": sku_id,
                "current_stock": -18,
                "in_transit_stock": 0,
                "reorder_point_s": 50,
                "order_up_to_S": 150,
                "lead_time_mean_days": 4.0,
                "lead_time_std_days": 0.7,
                "unit_cost_eur": 22.0,
                "moq": 40,
            },
            "anomaly": {
                "sku_id": sku_id,
                "is_anomaly": True,
                "anomaly_score": 0.79,
                "data_corrupted": True,
                "summary": "corrupted_input_data: negative physical stock (-18) from WMS sensor",
                "feature_contributions": {"wms_stock_discrepancy": -4.1},
            },
        },
    }

    draft = {
        "sku_id": sku_id,
        "status": "APPROVED_AUTO",
        "order_qty": 120,
        "expedited": False,
        "expected_total_cost_eur": 2760.0,
        "confidence_level": 0.85,
        "evidence_tool_calls": ["current_stock=-18"],
        "reasoning_summary": "Stock actuel -18, tentative de commande de 120 unites.",
    }

    llm = MockLLMClient(
        scripted_responses=[
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="get_sku_forecast_and_uncertainty",
                        arguments={"sku_id": sku_id, "horizon_days": 14},
                    )
                ],
                tokens_used=40,
            ),
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="inspect_anomaly_diagnostics",
                        arguments={"sku_id": sku_id, "window_days": 14},
                    )
                ],
                tokens_used=38,
            ),
            MockLLMResponse(
                tool_calls=[
                    MockToolCall(
                        name="query_inventory_and_supplier_sql",
                        arguments={"sku_id": sku_id},
                    )
                ],
                tokens_used=35,
            ),
            MockLLMResponse(final_content=json.dumps(draft), tokens_used=55),
        ]
    )
    return scenario, llm


def run_demo(traces_path: Path = Path("logs/agent_traces.jsonl")) -> list[dict[str, Any]]:
    """Execute les deux cas contrastes et imprime les rapports et traces JSONL."""
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = SupplyChainSettings(agent_traces_path=traces_path)
    cases = [
        ("CAS 1 : VRAIE RUPTURE IMMINENTE (ENGAGEMENT > 5000 EUR)", *_build_case_1_true_stockout()),
        ("CAS 2 : FAUSSE ALERTE CAPTEUR (DONNEES CORROMPUES)", *_build_case_2_corrupted_sensor()),
    ]

    outputs: list[dict[str, Any]] = []
    for title, scenario, llm_client in cases:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        registry = FixtureBackedToolRegistry(scenario=scenario, cfg=cfg)
        orchestrator = BoundedSupplyChainOrchestrator(
            llm_client=llm_client,
            tool_registry=registry,
            cfg=cfg,
            guardrail_engine=DeterministicGuardrailEngine(cfg=cfg),
            traces_path=traces_path,
        )
        res = orchestrator.run(
            IncidentTrigger(
                sku_id=scenario["sku_id"],
                trigger_event=scenario["trigger_event"],
                horizon_days=14,
            )
        )
        print(f"Trace ID           : {res.trace_id}")
        print(f"Statut Final       : {res.report.status.value}")
        print(f"Quantite Commandee : {res.report.order_qty}")
        print(f"Cout Total Espere  : {res.report.expected_total_cost_eur:.2f} EUR")
        print(f"Garde-fous Actifs  : {res.guardrails_triggered}")
        if res.report.abstention_reason:
            print(f"Motif Abstention   : {res.report.abstention_reason}")

        last_jsonl = traces_path.read_text(encoding="utf-8").splitlines()[-1]
        print("\n--- Derniere entree logs/agent_traces.jsonl ---")
        print(last_jsonl)
        outputs.append(
            {
                "trace_id": res.trace_id,
                "status": res.report.status.value,
                "order_qty": res.report.order_qty,
                "abstention_reason": res.report.abstention_reason,
                "guardrails_triggered": res.guardrails_triggered,
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstration live de l'Agent Supply Chain.")
    parser.add_argument(
        "--traces-path",
        type=Path,
        default=Path("logs/agent_traces.jsonl"),
    )
    args = parser.parse_args()
    run_demo(traces_path=args.traces_path)


if __name__ == "__main__":
    main()
