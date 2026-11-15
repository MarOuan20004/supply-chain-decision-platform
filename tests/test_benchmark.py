"""Tests automatises du jeu de 40 scenarios, du harnais comparatif et de la demo."""

from __future__ import annotations

from pathlib import Path

from benchmarks.evaluate_agent_vs_baselines import evaluate_suite
from scripts.demo_incident_response import run_demo
from scripts.generate_incidents_40 import build_40_scenarios


def test_40_scenarios_stratification_and_oracles() -> None:
    """Verifie la structure exacte des 40 scenarios (4 categories x 10)."""
    scenarios = build_40_scenarios()
    assert len(scenarios) == 40

    by_cat: dict[str, int] = {}
    for sc in scenarios:
        by_cat[sc["category"]] = by_cat.get(sc["category"], 0) + 1
        assert "sku_id" in sc
        assert "trigger_event" in sc
        assert "tool_fixtures" in sc
        assert "oracle_decision" in sc
        assert "rationale" in sc

    assert by_cat == {
        "standard_stockout": 10,
        "demand_shock_moq_leadtime": 10,
        "corrupted_sensor_false_alerts": 10,
        "high_uncertainty_escalation": 10,
    }


def test_evaluate_agent_vs_baselines_full_40_scenarios(tmp_path: Path) -> None:
    """Verifie l'execution complete sur 40 scenarios et les invariants des metriques."""
    scenarios_file = tmp_path / "incidents_40.json"
    results_dir = tmp_path / "results"
    traces_file = tmp_path / "bench_traces.jsonl"

    summary = evaluate_suite(
        scenarios_path=scenarios_file,
        results_dir=results_dir,
        llm_mode="mock",
        smoke=False,
        traces_path=traces_file,
    )

    assert summary["num_scenarios"] == 40
    b1 = summary["approaches"]["baseline_1_ss_rule"]
    b2 = summary["approaches"]["baseline_2_static_ml"]
    ag = summary["approaches"]["governed_agent"]

    # L'agent gouverne doit surpasser strictement les deux baselines en Decision Accuracy
    assert ag["decision_accuracy"] > b1["decision_accuracy"]
    assert ag["decision_accuracy"] > b2["decision_accuracy"]

    # Invariants critiques de gouvernance
    assert ag["numeric_hallucination_rate"] == 0.0
    assert ag["false_action_prevention_rate"] == 1.0
    assert ag["invalid_tool_call_rate"] == 0.0
    assert ag["total_simulated_cost_eur"] < b1["total_simulated_cost_eur"]


def test_demo_incident_response_produces_two_contrasted_decisions(tmp_path: Path) -> None:
    """Verifie que la demo live produit PENDING_HUMAN_APPROVAL et ABSTAIN_INSUFFICIENT_DATA."""
    traces_file = tmp_path / "demo_traces.jsonl"
    outputs = run_demo(traces_path=traces_file)

    assert len(outputs) == 2
    assert outputs[0]["status"] == "PENDING_HUMAN_APPROVAL"
    assert outputs[0]["order_qty"] == 180
    assert "GR-04-HITL-AMOUNT" in outputs[0]["guardrails_triggered"]

    assert outputs[1]["status"] == "ABSTAIN_INSUFFICIENT_DATA"
    assert outputs[1]["order_qty"] == 0
    assert "corrupted_input_data" in (outputs[1]["abstention_reason"] or "")

    lines = traces_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
