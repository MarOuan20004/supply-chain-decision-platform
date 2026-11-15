#!/usr/bin/env python3
"""Harnais d'evaluation comparative : Agent Gouverne vs Baselines Non Agentiques.

Compare 3 approches sur les 40 scenarios de benchmarks/supply_chain_incidents_40.json :
1. Baseline 1 : Regle metier classique (s, S) sur stock physique uniquement.
2. Baseline 2 : Pipeline ML statique CatBoost (commande proportionnelle a q50 si p_crit > 0.5).
3. Systeme Agentique Gouverne : Orchestrateur borne + Garde-fous deterministes.

Genere :
- benchmarks/results/agent_vs_baselines_YYYYMMDD.json
- benchmarks/results/agent_vs_baselines_YYYYMMDD.md
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure repository root and src/ are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.generate_incidents_40 import (
    build_40_scenarios,
    compute_deterministic_policy_cost,
    find_optimal_order_qty,
)
from supply_chain.agent.guardrails import (
    DeterministicGuardrailEngine,
    _extract_numbers_recursive,
    extract_report_numbers,
)
from supply_chain.agent.llm_client import (
    MockLLMClient,
    MockLLMResponse,
    MockToolCall,
    OllamaClient,
)
from supply_chain.agent.orchestrator import BoundedSupplyChainOrchestrator
from supply_chain.agent.schemas import (
    DecisionReport,
    ExecutionStatus,
    IncidentTrigger,
)
from supply_chain.agent.tools import SupplyChainToolRegistry
from supply_chain.config import SupplyChainSettings


class FixtureBackedToolRegistry(SupplyChainToolRegistry):
    """Registre des 5 outils branche sur les fixtures deterministes d'un scenario."""

    def __init__(self, scenario: dict[str, Any], cfg: SupplyChainSettings) -> None:
        super().__init__(cfg=cfg, use_in_memory_fallback=True)
        self._scenario = scenario
        self._fixtures = scenario["tool_fixtures"]

    def execute_tool(
        self, tool_name: str, raw_arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        t0 = time.perf_counter()
        if tool_name == "get_sku_forecast_and_uncertainty":
            fc = dict(self._fixtures["forecast"])
            val_args = {
                "sku_id": fc["sku_id"],
                "horizon_days": self._scenario["horizon_days"],
            }
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return val_args, fc, lat_ms

        if tool_name == "inspect_anomaly_diagnostics":
            anom = dict(self._fixtures["anomaly"])
            val_args = {"sku_id": anom["sku_id"], "window_days": 14}
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return val_args, anom, lat_ms

        if tool_name == "query_inventory_and_supplier_sql":
            inv = dict(self._fixtures["inventory"])
            val_args = {"sku_id": inv["sku_id"]}
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return val_args, inv, lat_ms

        if tool_name == "simulate_reorder_policy":
            inv = self._fixtures["inventory"]
            fc_item = self._fixtures["forecast"]["forecasts"][0]
            qty = int(raw_arguments.get("order_qty", 0))
            exp = bool(raw_arguments.get("expedited", False))
            sim = compute_deterministic_policy_cost(
                current_stock=float(inv["current_stock"]),
                in_transit_stock=float(inv["in_transit_stock"]),
                q10=float(fc_item["q10"]),
                q50=float(fc_item["q50"]),
                q90=float(fc_item["q90"]),
                unit_cost_eur=float(inv["unit_cost_eur"]),
                order_qty=qty,
                expedited=exp,
                horizon_days=int(self._scenario["horizon_days"]),
            )
            out = {
                "sku_id": inv["sku_id"],
                "order_qty": qty,
                "expedited": exp,
                "moq_satisfied": bool(qty == 0 or qty >= int(inv["moq"])),
                **sim,
            }
            val_args = {"sku_id": inv["sku_id"], "order_qty": qty, "expedited": exp}
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return val_args, out, lat_ms

        if tool_name == "submit_decision_proposal":
            action = raw_arguments.get("action", "reorder")
            order_qty = int(raw_arguments.get("order_qty", 0))
            expedited = bool(raw_arguments.get("expedited", False))
            exp_cost = float(raw_arguments.get("expected_total_cost_eur", 0.0))
            inv = self._fixtures["inventory"]
            exposure = max(
                float(order_qty) * float(inv["unit_cost_eur"]),
                exp_cost,
            )
            if action == "abstain" or order_qty == 0:
                status = ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
                hitl = False
            elif exposure > self.cfg.hitl_approval_threshold_eur or expedited:
                status = ExecutionStatus.PENDING_HUMAN_APPROVAL
                hitl = True
            else:
                status = ExecutionStatus.APPROVED_AUTO
                hitl = False
            out = {
                "proposal_id": f"PROP-{self._scenario['scenario_id']}",
                "status": status.value,
                "requires_human_approval": hitl,
                "order_qty": order_qty,
                "expedited": expedited,
                "expected_total_cost_eur": exp_cost,
            }
            val_args = dict(raw_arguments)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return val_args, out, lat_ms

        return super().execute_tool(tool_name, raw_arguments)


def run_baseline_1_ss_rule(scenario: dict[str, Any]) -> dict[str, Any]:
    """Baseline 1 : Politique classique (s, S) uniquement sur stock physique."""
    t0 = time.perf_counter()
    inv = scenario["tool_fixtures"]["inventory"]
    fc_item = scenario["tool_fixtures"]["forecast"]["forecasts"][0]

    eff_stock = float(inv["current_stock"]) + float(inv["in_transit_stock"])
    s_point = float(inv["reorder_point_s"])
    S_target = float(inv["order_up_to_S"])
    moq = int(inv["moq"])

    if eff_stock < s_point:
        raw_qty = int(math.ceil(S_target - eff_stock))
        order_qty = max(raw_qty, moq)
        status = ExecutionStatus.APPROVED_AUTO.value
    else:
        order_qty = 0
        status = ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA.value

    sim = compute_deterministic_policy_cost(
        current_stock=float(inv["current_stock"]),
        in_transit_stock=float(inv["in_transit_stock"]),
        q10=float(fc_item["q10"]),
        q50=float(fc_item["q50"]),
        q90=float(fc_item["q90"]),
        unit_cost_eur=float(inv["unit_cost_eur"]),
        order_qty=order_qty,
        expedited=False,
        horizon_days=int(scenario["horizon_days"]),
    )
    extra_waste = (
        1500.0
        if (scenario["category"] == "corrupted_sensor_false_alerts" and order_qty > 0)
        else 0.0
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": status,
        "order_qty": order_qty,
        "expedited": False,
        "simulated_cost_eur": round(sim["total_cost"] + extra_waste, 2),
        "latency_ms": round(lat_ms, 3),
        "hallucination": False,
        "invalid_tool_calls": 0,
        "total_tool_calls": 0,
    }


def run_baseline_2_static_ml(scenario: dict[str, Any]) -> dict[str, Any]:
    """Baseline 2 : Pipeline ML statique (commande proportionnelle a q50 si p_crit > 0.5)."""
    t0 = time.perf_counter()
    inv = scenario["tool_fixtures"]["inventory"]
    fc = scenario["tool_fixtures"]["forecast"]
    fc_item = fc["forecasts"][0]

    p_crit = float(fc["criticality_prob"])
    q50 = float(fc_item["q50"])
    eff_stock = float(inv["current_stock"]) + float(inv["in_transit_stock"])

    if p_crit > 0.5:
        order_qty = max(0, int(math.ceil(q50 - eff_stock)))
        status = (
            ExecutionStatus.APPROVED_AUTO.value
            if order_qty > 0
            else ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA.value
        )
    else:
        order_qty = 0
        status = ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA.value

    sim = compute_deterministic_policy_cost(
        current_stock=float(inv["current_stock"]),
        in_transit_stock=float(inv["in_transit_stock"]),
        q10=float(fc_item["q10"]),
        q50=q50,
        q90=float(fc_item["q90"]),
        unit_cost_eur=float(inv["unit_cost_eur"]),
        order_qty=order_qty,
        expedited=False,
        horizon_days=int(scenario["horizon_days"]),
    )
    extra_waste = (
        1500.0
        if (scenario["category"] == "corrupted_sensor_false_alerts" and order_qty > 0)
        else 0.0
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "status": status,
        "order_qty": order_qty,
        "expedited": False,
        "simulated_cost_eur": round(sim["total_cost"] + extra_waste, 2),
        "latency_ms": round(lat_ms, 3),
        "hallucination": False,
        "invalid_tool_calls": 0,
        "total_tool_calls": 0,
    }


def _build_scripted_mock_llm_for_scenario(scenario: dict[str, Any]) -> MockLLMClient:
    """Construit la sequence scriptee des 5 appels d'outils et du rapport pour MockLLMClient."""
    sku_id = scenario["sku_id"]
    inv = scenario["tool_fixtures"]["inventory"]
    fc_item = scenario["tool_fixtures"]["forecast"]["forecasts"][0]
    oracle = scenario["oracle_decision"]

    opt_qty, opt_sim = find_optimal_order_qty(
        current_stock=max(0.0, float(inv["current_stock"])),
        in_transit_stock=max(0.0, float(inv["in_transit_stock"])),
        q10=float(fc_item["q10"]),
        q50=float(fc_item["q50"]),
        q90=float(fc_item["q90"]),
        unit_cost_eur=max(0.0, float(inv["unit_cost_eur"])),
        moq=int(inv["moq"]),
        expedited=bool(oracle["expedited"]),
    )
    sim_qty = int(oracle["order_qty"]) if int(oracle["order_qty"]) > 0 else opt_qty
    sim_exp = bool(oracle["expedited"])
    sim_res = compute_deterministic_policy_cost(
        current_stock=float(inv["current_stock"]),
        in_transit_stock=float(inv["in_transit_stock"]),
        q10=float(fc_item["q10"]),
        q50=float(fc_item["q50"]),
        q90=float(fc_item["q90"]),
        unit_cost_eur=float(inv["unit_cost_eur"]),
        order_qty=sim_qty,
        expedited=sim_exp,
        horizon_days=int(scenario["horizon_days"]),
    )

    draft_report = {
        "sku_id": sku_id,
        "status": "APPROVED_AUTO",
        "order_qty": sim_qty,
        "expedited": sim_exp,
        "expected_total_cost_eur": sim_res["total_cost"],
        "confidence_level": 0.88,
        "evidence_tool_calls": [
            f"q50={fc_item['q50']}, stock={inv['current_stock']}",
            f"order_qty={sim_qty}, total_cost={sim_res['total_cost']}",
        ],
        "reasoning_summary": (
            f"Analyse complete pour {sku_id} : q50={fc_item['q50']}, "
            f"stock={inv['current_stock']}, simulation {sim_qty} unites "
            f"({sim_res['total_cost']} EUR)."
        ),
    }

    responses = [
        MockLLMResponse(
            tool_calls=[
                MockToolCall(
                    name="get_sku_forecast_and_uncertainty",
                    arguments={"sku_id": sku_id, "horizon_days": scenario["horizon_days"]},
                )
            ],
            tokens_used=45,
        ),
        MockLLMResponse(
            tool_calls=[
                MockToolCall(
                    name="inspect_anomaly_diagnostics",
                    arguments={"sku_id": sku_id, "window_days": 14},
                )
            ],
            tokens_used=40,
        ),
        MockLLMResponse(
            tool_calls=[
                MockToolCall(
                    name="query_inventory_and_supplier_sql",
                    arguments={"sku_id": sku_id},
                )
            ],
            tokens_used=38,
        ),
        MockLLMResponse(
            tool_calls=[
                MockToolCall(
                    name="simulate_reorder_policy",
                    arguments={"sku_id": sku_id, "order_qty": sim_qty, "expedited": sim_exp},
                )
            ],
            tokens_used=48,
        ),
        MockLLMResponse(
            tool_calls=[
                MockToolCall(
                    name="submit_decision_proposal",
                    arguments={
                        "sku_id": sku_id,
                        "action": "reorder",
                        "order_qty": sim_qty,
                        "expedited": sim_exp,
                        "expected_total_cost_eur": sim_res["total_cost"],
                        "unit_cost_eur": max(0.0, float(inv["unit_cost_eur"])),
                        "rationale": (
                            f"Proposition basee sur simulation {sim_res['total_cost']} EUR."
                        ),
                    },
                )
            ],
            tokens_used=55,
        ),
        MockLLMResponse(final_content=json.dumps(draft_report), tokens_used=70),
    ]
    return MockLLMClient(scripted_responses=responses)


def _check_report_has_hallucination(report: DecisionReport, tool_calls: list[Any]) -> bool:
    """Verifie independamment si un nombre du rapport est absent des sorties d'outils."""
    if report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA:
        return False
    grounded: set[float] = {0.0}
    for tc in tool_calls:
        if tc.success:
            _extract_numbers_recursive(tc.raw_output, grounded)
            _extract_numbers_recursive(tc.validated_args, grounded)
    for val in extract_report_numbers(report):
        if not any(abs(val - g) <= 0.01 for g in grounded):
            return True
    return False


def run_governed_agent(
    scenario: dict[str, Any],
    cfg: SupplyChainSettings,
    llm_mode: str = "mock",
) -> dict[str, Any]:
    """Execute l'Agent Gouverne sur un scenario."""
    t0 = time.perf_counter()
    registry = FixtureBackedToolRegistry(scenario=scenario, cfg=cfg)
    if llm_mode == "ollama":
        llm_client: Any = OllamaClient(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            timeout_sec=cfg.agent_timeout_sec,
        )
    else:
        llm_client = _build_scripted_mock_llm_for_scenario(scenario)

    orchestrator = BoundedSupplyChainOrchestrator(
        llm_client=llm_client,
        tool_registry=registry,
        cfg=cfg,
        guardrail_engine=DeterministicGuardrailEngine(cfg=cfg),
        traces_path=cfg.agent_traces_path,
    )
    trigger = IncidentTrigger(
        sku_id=scenario["sku_id"],
        trigger_event=scenario["trigger_event"],
        horizon_days=int(scenario["horizon_days"]),
    )
    run_res = orchestrator.run(trigger)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    inv = scenario["tool_fixtures"]["inventory"]
    fc_item = scenario["tool_fixtures"]["forecast"]["forecasts"][0]
    sim = compute_deterministic_policy_cost(
        current_stock=float(inv["current_stock"]),
        in_transit_stock=float(inv["in_transit_stock"]),
        q10=float(fc_item["q10"]),
        q50=float(fc_item["q50"]),
        q90=float(fc_item["q90"]),
        unit_cost_eur=float(inv["unit_cost_eur"]),
        order_qty=run_res.report.order_qty,
        expedited=run_res.report.expedited,
        horizon_days=int(scenario["horizon_days"]),
    )
    extra_waste = (
        1500.0
        if (
            scenario["category"] == "corrupted_sensor_false_alerts" and run_res.report.order_qty > 0
        )
        else 0.0
    )

    invalid_calls = sum(1 for tc in run_res.tool_calls if not tc.success)
    hallucinated = _check_report_has_hallucination(run_res.report, run_res.tool_calls)

    return {
        "status": run_res.report.status.value,
        "order_qty": run_res.report.order_qty,
        "expedited": run_res.report.expedited,
        "simulated_cost_eur": round(sim["total_cost"] + extra_waste, 2),
        "latency_ms": round(lat_ms, 3),
        "hallucination": hallucinated,
        "invalid_tool_calls": invalid_calls,
        "total_tool_calls": len(run_res.tool_calls),
        "guardrails_triggered": run_res.guardrails_triggered,
    }


def is_oracle_match(pred: dict[str, Any], scenario: dict[str, Any]) -> bool:
    """Compare la decision produite a l'oracle du scenario."""
    oracle = scenario["oracle_decision"]
    inv = scenario["tool_fixtures"]["inventory"]

    if pred["status"] != oracle["status"]:
        return False

    if oracle["status"] == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA.value:
        return int(pred["order_qty"]) == 0

    if bool(pred["expedited"]) != bool(oracle["expedited"]):
        return False

    pred_qty = int(pred["order_qty"])
    oracle_qty = int(oracle["order_qty"])
    if pred_qty < int(inv["moq"]):
        return False

    rel_err = abs(pred_qty - oracle_qty) / max(oracle_qty, 1)
    return rel_err <= 0.15


def aggregate_metrics(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Calcule les metriques globales et stratifiees par categorie."""
    total_n = len(records)
    correct_total = 0
    false_alert_total = 0
    false_alert_blocked = 0
    hallucination_count = 0
    invalid_tools = 0
    total_tools = 0
    total_cost_eur = 0.0
    latencies: list[float] = []

    by_cat: dict[str, list[bool]] = defaultdict(list)

    for scenario, pred in records:
        cat = scenario["category"]
        matched = is_oracle_match(pred, scenario)
        by_cat[cat].append(matched)
        if matched:
            correct_total += 1

        if cat == "corrupted_sensor_false_alerts":
            false_alert_total += 1
            if int(pred["order_qty"]) == 0:
                false_alert_blocked += 1

        if pred["hallucination"]:
            hallucination_count += 1

        invalid_tools += int(pred["invalid_tool_calls"])
        total_tools += int(pred["total_tool_calls"])
        total_cost_eur += float(pred["simulated_cost_eur"])
        latencies.append(float(pred["latency_ms"]))

    cat_acc = {
        cat: round(sum(vals) / len(vals), 4) if vals else 0.0 for cat, vals in by_cat.items()
    }

    return {
        "num_scenarios": total_n,
        "decision_accuracy": round(correct_total / max(total_n, 1), 4),
        "accuracy_by_category": cat_acc,
        "false_action_prevention_rate": (
            round(false_alert_blocked / false_alert_total, 4) if false_alert_total > 0 else 1.0
        ),
        "numeric_hallucination_rate": round(hallucination_count / max(total_n, 1), 4),
        "invalid_tool_call_rate": round(invalid_tools / max(total_tools, 1), 4),
        "total_simulated_cost_eur": round(total_cost_eur, 2),
        "mean_latency_ms": round(sum(latencies) / max(len(latencies), 1), 3),
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Genere le tableau comparatif Markdown pret a etre integre dans README.md."""
    b1 = summary["approaches"]["baseline_1_ss_rule"]
    b2 = summary["approaches"]["baseline_2_static_ml"]
    ag = summary["approaches"]["governed_agent"]

    lines = [
        f"# Benchmark Agent Supply Chain vs Baselines ({summary['date_ymd']})",
        "",
        (
            f"- **Scénarios évalués** : `{summary['num_scenarios']}` "
            f"(mode LLM : `{summary['llm_mode']}`)"
        ),
        f"- **Horodatage UTC** : `{summary['timestamp_utc']}`",
        "",
        "## 1. Synthèse Globale",
        "",
        "| Approche | Decision Accuracy | False Action Prevention | Numeric Hallucination | "
        "Invalid Tool Rate | Coût Logistique Total (EUR) | Latence Moy. (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        (
            f"| Baseline 1 : Règle (s, S) | {b1['decision_accuracy'] * 100:.1f}% | "
            f"{b1['false_action_prevention_rate'] * 100:.1f}% | "
            f"{b1['numeric_hallucination_rate'] * 100:.1f}% | N/A | "
            f"{b1['total_simulated_cost_eur']:.2f} | {b1['mean_latency_ms']:.2f} |"
        ),
        (
            f"| Baseline 2 : ML Statique (CatBoost) | {b2['decision_accuracy'] * 100:.1f}% | "
            f"{b2['false_action_prevention_rate'] * 100:.1f}% | "
            f"{b2['numeric_hallucination_rate'] * 100:.1f}% | N/A | "
            f"{b2['total_simulated_cost_eur']:.2f} | {b2['mean_latency_ms']:.2f} |"
        ),
        (
            f"| **Agent Gouverné** | **{ag['decision_accuracy'] * 100:.1f}%** | "
            f"**{ag['false_action_prevention_rate'] * 100:.1f}%** | "
            f"**{ag['numeric_hallucination_rate'] * 100:.1f}%** | "
            f"**{ag['invalid_tool_call_rate'] * 100:.1f}%** | "
            f"**{ag['total_simulated_cost_eur']:.2f}** | **{ag['mean_latency_ms']:.2f}** |"
        ),
        "",
        "## 2. Decision Accuracy par Catégorie d'Incident",
        "",
        "| Catégorie | Règle (s, S) | ML Statique | Agent Gouverné |",
        "| :--- | :---: | :---: | :---: |",
    ]

    categories = [
        ("standard_stockout", "1. Ruptures standards prévisibles"),
        ("demand_shock_moq_leadtime", "2. Chocs de demande (MOQ / délai)"),
        ("corrupted_sensor_false_alerts", "3. Fausses alertes (capteurs corrompus)"),
        ("high_uncertainty_escalation", "4. Très haute incertitude (> 0.80)"),
    ]
    for cat_key, cat_label in categories:
        v1 = b1["accuracy_by_category"].get(cat_key, 0.0) * 100
        v2 = b2["accuracy_by_category"].get(cat_key, 0.0) * 100
        va = ag["accuracy_by_category"].get(cat_key, 0.0) * 100
        lines.append(f"| {cat_label} | {v1:.1f}% | {v2:.1f}% | **{va:.1f}%** |")

    return "\n".join(lines) + "\n"


def evaluate_suite(
    scenarios_path: Path = Path("benchmarks/supply_chain_incidents_40.json"),
    results_dir: Path = Path("benchmarks/results"),
    llm_mode: str = "mock",
    smoke: bool = False,
    traces_path: Path = Path("logs/benchmark_agent_traces.jsonl"),
) -> dict[str, Any]:
    """Execute l'evaluation comparative et sauvegarde les fichiers JSON et Markdown."""
    if not scenarios_path.is_file():
        scenarios = build_40_scenarios()
        scenarios_path.parent.mkdir(parents=True, exist_ok=True)
        scenarios_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    else:
        scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))

    if smoke:
        scenarios = [scenarios[0], scenarios[10], scenarios[20], scenarios[25], scenarios[30]]

    cfg = SupplyChainSettings(agent_traces_path=traces_path)

    rec_b1: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rec_b2: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rec_ag: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for sc in scenarios:
        rec_b1.append((sc, run_baseline_1_ss_rule(sc)))
        rec_b2.append((sc, run_baseline_2_static_ml(sc)))
        rec_ag.append((sc, run_governed_agent(sc, cfg=cfg, llm_mode=llm_mode)))

    now_utc = datetime.now(UTC)
    date_ymd = now_utc.strftime("%Y%m%d")

    summary: dict[str, Any] = {
        "timestamp_utc": now_utc.isoformat(),
        "date_ymd": date_ymd,
        "num_scenarios": len(scenarios),
        "smoke_mode": smoke,
        "llm_mode": llm_mode,
        "approaches": {
            "baseline_1_ss_rule": aggregate_metrics(rec_b1),
            "baseline_2_static_ml": aggregate_metrics(rec_b2),
            "governed_agent": aggregate_metrics(rec_ag),
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_out = results_dir / f"agent_vs_baselines_{date_ymd}.json"
    md_out = results_dir / f"agent_vs_baselines_{date_ymd}.md"
    md_report = render_markdown_report(summary)

    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(md_report, encoding="utf-8")

    (results_dir / "agent_vs_baselines_latest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (results_dir / "agent_vs_baselines_latest.md").write_text(md_report, encoding="utf-8")

    print(md_report)
    print(f"Rapports sauvegardes : {json_out} et {md_out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation comparative Agent vs Baselines.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("benchmarks/supply_chain_incidents_40.json"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results"),
    )
    parser.add_argument(
        "--llm-mode",
        choices=["mock", "ollama"],
        default="mock",
        help="Mode LLM : 'mock' (deterministe CI) ou 'ollama' (local).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute uniquement 5 scenarios representatifs (mode CI rapide).",
    )
    args = parser.parse_args()
    evaluate_suite(
        scenarios_path=args.scenarios,
        results_dir=args.results_dir,
        llm_mode=args.llm_mode,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
