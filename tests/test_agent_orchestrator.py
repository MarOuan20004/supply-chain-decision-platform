"""Tests unitaires de l'orchestrateur a etats bornes avec MockLLMClient."""

from __future__ import annotations

import json

from supply_chain.agent.llm_client import MockLLMClient
from supply_chain.agent.orchestrator import SupplyChainDecisionOrchestrator
from supply_chain.agent.schemas import (
    DecisionProposal,
    ExecutionStatus,
    IncidentTrigger,
    LLMMessage,
    ToolCallRequest,
)
from supply_chain.agent.tools import SupplyChainToolRegistry
from tests.test_agent_tools import DeterministicFakeBackend


def test_orchestrator_nominal_5_tools_sequence_and_hitl_status() -> None:
    """Valide l'enchainement complet des 5 outils et la generation du DecisionReport."""
    registry = SupplyChainToolRegistry(backend=DeterministicFakeBackend())

    # Pre-calcul de la simulation attendue pour construire le script du MockLLMClient
    _, sim_expected, _ = registry.execute_tool(
        "simulate_reorder_policy", {"sku_id": "SKU-0001", "order_qty": 120, "expedited": False}
    )
    # Reinitialisation d'un registre vierge pour la session d'orchestration
    clean_registry = SupplyChainToolRegistry(backend=DeterministicFakeBackend())

    proposal = DecisionProposal(
        sku_id="SKU-0001",
        horizon_days=7,
        order_qty=120,
        expedited=False,
        order_cost_eur=sim_expected["order_cost"],
        expected_total_cost_eur=sim_expected["total_cost"],
        confidence_level="HIGH",
        reasoning_summary="Rupture probable (0.82), aucune anomalie capteur, commande 120 u.",
        evidence_tool_calls=[
            '"calibrated_criticality_prob": 0.82',
            '"data_corruption_detected": false',
            '"moq_units": 40',
            f'"total_cost": {sim_expected["total_cost"]}',
        ],
    )

    final_report_json = json.dumps(
        {
            "incident_id": "INC-2026-001",
            "sku_id": "SKU-0001",
            "status": "PENDING_HUMAN_APPROVAL",
            "order_qty": 120,
            "expedited": False,
            "expected_total_cost_eur": sim_expected["total_cost"],
            "confidence_level": "HIGH",
            "reasoning_summary": "Rupture probable (0.82), sans anomalie, commande 120 u.",
            "evidence_tool_calls": proposal.evidence_tool_calls,
            "abstention_reason": None,
        }
    )

    mock_llm = MockLLMClient(
        scripted_turns=[
            LLMMessage(
                tool_calls=[
                    ToolCallRequest(
                        call_id="c1",
                        tool_name="get_sku_forecast_and_uncertainty",
                        arguments={"sku_id": "SKU-0001", "horizon_days": 7},
                    )
                ]
            ),
            LLMMessage(
                tool_calls=[
                    ToolCallRequest(
                        call_id="c2",
                        tool_name="inspect_anomaly_diagnostics",
                        arguments={"sku_id": "SKU-0001", "window_days": 14},
                    )
                ]
            ),
            LLMMessage(
                tool_calls=[
                    ToolCallRequest(
                        call_id="c3",
                        tool_name="query_inventory_and_supplier_sql",
                        arguments={"sku_id": "SKU-0001"},
                    )
                ]
            ),
            LLMMessage(
                tool_calls=[
                    ToolCallRequest(
                        call_id="c4",
                        tool_name="simulate_reorder_policy",
                        arguments={"sku_id": "SKU-0001", "order_qty": 120, "expedited": False},
                    )
                ]
            ),
            LLMMessage(
                tool_calls=[
                    ToolCallRequest(
                        call_id="c5",
                        tool_name="submit_decision_proposal",
                        arguments={"proposal": proposal.model_dump(mode="json")},
                    )
                ]
            ),
            LLMMessage(content=final_report_json),
        ]
    )

    orchestrator = SupplyChainDecisionOrchestrator(
        llm_client=mock_llm,
        tool_registry=clean_registry,
        max_steps=6,
        timeout_sec=15.0,
    )
    incident = IncidentTrigger(incident_id="INC-2026-001", sku_id="SKU-0001")
    report = orchestrator.run(incident)

    assert report.status == ExecutionStatus.PENDING_HUMAN_APPROVAL
    assert report.order_qty == 120
    assert report.expected_total_cost_eur == sim_expected["total_cost"]
    assert len(report.evidence_tool_calls) == 4
    assert [r.tool_name for r in report.tool_trace] == [
        "get_sku_forecast_and_uncertainty",
        "inspect_anomaly_diagnostics",
        "query_inventory_and_supplier_sql",
        "simulate_reorder_policy",
        "submit_decision_proposal",
    ]


def test_orchestrator_intercepts_invalid_tool_and_enforces_max_steps() -> None:
    """Verifie l'interception d'un outil inexistant, d'un mauvais type et le plafond MAX_STEPS."""
    registry = SupplyChainToolRegistry(backend=DeterministicFakeBackend())

    # Le LLM appelle un outil inexistant au pas 1, un mauvais type au pas 2, puis boucle
    looping_turns = [
        LLMMessage(
            tool_calls=[
                ToolCallRequest(
                    call_id="bad_1", tool_name="non_existent_tool", arguments={"sku_id": "SKU-0001"}
                )
            ]
        ),
        LLMMessage(
            tool_calls=[
                ToolCallRequest(
                    call_id="bad_2",
                    tool_name="get_sku_forecast_and_uncertainty",
                    arguments={"sku_id": "SKU-0001", "horizon_days": -99},
                )
            ]
        ),
        LLMMessage(
            tool_calls=[
                ToolCallRequest(
                    call_id="ok_3",
                    tool_name="query_inventory_and_supplier_sql",
                    arguments={"sku_id": "SKU-0001"},
                )
            ]
        ),
    ]
    mock_llm = MockLLMClient(scripted_turns=looping_turns)
    orchestrator = SupplyChainDecisionOrchestrator(
        llm_client=mock_llm,
        tool_registry=registry,
        max_steps=5,
        timeout_sec=15.0,
    )

    report = orchestrator.run(IncidentTrigger(incident_id="INC-ERR-01", sku_id="SKU-0001"))

    # L'orchestrateur ne crashe pas, journalise les erreurs et abstient apres MAX_STEPS=5
    assert report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert report.steps_executed == 5
    assert report.abstention_reason is not None and "MAX_STEPS_EXCEEDED" in report.abstention_reason
    assert report.tool_trace[0].success is False
    assert "UnknownToolError" in (report.tool_trace[0].error_message or "")
    assert report.tool_trace[1].success is False
    assert "ToolValidationError" in (report.tool_trace[1].error_message or "")


def test_orchestrator_enforces_timeout_cleanly() -> None:
    """Verifie l'arret propre et l'abstention en cas de depassement de timeout_sec."""
    registry = SupplyChainToolRegistry(backend=DeterministicFakeBackend())
    slow_llm = MockLLMClient(
        scripted_turns=[
            LLMMessage(
                tool_calls=[
                    ToolCallRequest(
                        call_id="c1",
                        tool_name="query_inventory_and_supplier_sql",
                        arguments={"sku_id": "SKU-0001"},
                    )
                ]
            )
        ],
        per_call_delay_sec=0.06,
    )
    orchestrator = SupplyChainDecisionOrchestrator(
        llm_client=slow_llm,
        tool_registry=registry,
        max_steps=5,
        timeout_sec=0.03,
    )

    report = orchestrator.run(IncidentTrigger(incident_id="INC-TO-01", sku_id="SKU-0001"))
    assert report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert report.abstention_reason is not None
    assert "ORCHESTRATOR_TIMEOUT" in report.abstention_reason
