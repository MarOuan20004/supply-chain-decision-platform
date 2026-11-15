"""Tests adversariaux des garde-fous deterministes post-generation."""

from __future__ import annotations

from pathlib import Path

from supply_chain.agent.guardrails import DeterministicGuardrailEngine
from supply_chain.agent.llm_client import MockLLMClient, MockLLMResponse, MockToolCall
from supply_chain.agent.orchestrator import BoundedSupplyChainOrchestrator
from supply_chain.agent.schemas import (
    DecisionReport,
    ExecutionStatus,
    GuardrailViolationType,
    IncidentTrigger,
    ToolCallRecord,
)
from supply_chain.agent.tools import SupplyChainToolRegistry
from supply_chain.config import SupplyChainSettings


def _nominal_tool_calls(
    unit_cost: float = 20.0,
    stock: int = 40,
    lt_mean: float = 5.0,
    lt_std: float = 1.0,
    q10: float = 90.0,
    q50: float = 100.0,
    q90: float = 120.0,
    feature_contributions: dict[str, float] | None = None,
    order_qty: int = 100,
    total_cost: float = 2120.0,
    expedited: bool = False,
) -> list[ToolCallRecord]:
    contribs = (
        {"demand_zscore": 1.4, "lead_time_delta": 0.2}
        if feature_contributions is None
        else feature_contributions
    )
    return [
        ToolCallRecord(
            step_index=1,
            tool_name="get_sku_forecast_and_uncertainty",
            validated_args={"sku_id": "SKU-001", "horizon_days": 14},
            raw_output={
                "sku_id": "SKU-001",
                "forecasts": [{"horizon_days": 14, "q10": q10, "q50": q50, "q90": q90}],
                "criticality_prob": 0.78,
            },
            latency_ms=1.2,
            success=True,
        ),
        ToolCallRecord(
            step_index=2,
            tool_name="inspect_anomaly_diagnostics",
            validated_args={"sku_id": "SKU-001", "window_days": 14},
            raw_output={
                "sku_id": "SKU-001",
                "is_anomaly": False,
                "anomaly_score": -0.12,
                "feature_contributions": contribs,
            },
            latency_ms=0.9,
            success=True,
        ),
        ToolCallRecord(
            step_index=3,
            tool_name="query_inventory_and_supplier_sql",
            validated_args={"sku_id": "SKU-001"},
            raw_output={
                "sku_id": "SKU-001",
                "current_stock": stock,
                "in_transit_stock": 0,
                "lead_time_mean_days": lt_mean,
                "lead_time_std_days": lt_std,
                "unit_cost_eur": unit_cost,
                "moq": 50,
            },
            latency_ms=0.5,
            success=True,
        ),
        ToolCallRecord(
            step_index=4,
            tool_name="simulate_reorder_policy",
            validated_args={"sku_id": "SKU-001", "order_qty": order_qty, "expedited": expedited},
            raw_output={
                "sku_id": "SKU-001",
                "order_qty": order_qty,
                "expedited": expedited,
                "total_cost": total_cost,
            },
            latency_ms=0.4,
            success=True,
        ),
    ]


def test_01_missing_anomaly_feature_contributions_forces_abstain() -> None:
    """Cas 1 : Donnees manquantes (feature_contributions vide) -> ABSTAIN_INSUFFICIENT_DATA."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(feature_contributions={})
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=2120.0,
        confidence_level=0.9,
        evidence_tool_calls=["q50=100.0, stock=40"],
        reasoning_summary="Commander 100 unites pour un cout total de 2120.0 EUR.",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert outcome.final_report.order_qty == 0
    assert "feature_contributions" in (outcome.final_report.abstention_reason or "")


def test_02_negative_stock_forces_abstain() -> None:
    """Cas 2 : Stock physique negatif injecte -> ABSTAIN_INSUFFICIENT_DATA."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(stock=-15)
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=2120.0,
        confidence_level=0.85,
        evidence_tool_calls=["current_stock=-15"],
        reasoning_summary="Stock actuel -15, commander 100 unites pour 2120.0 EUR.",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert outcome.final_report.order_qty == 0
    assert "current_stock=-15" in (outcome.final_report.abstention_reason or "")


def test_03_numeric_hallucination_rejected_with_unanchored_reason() -> None:
    """Cas 3 : Montant invente (8888.88 EUR) absent des outils -> rejet unanchored_numeric_value."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(order_qty=100, total_cost=2120.0)
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=8888.88,
        confidence_level=0.92,
        evidence_tool_calls=["order_qty=100"],
        reasoning_summary="Le cout estime est de 8888.88 EUR pour 100 unites.",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert outcome.final_report.abstention_reason == "unanchored_numeric_value"
    assert outcome.final_report.order_qty == 0
    assert any(
        v.violation_type == GuardrailViolationType.UNANCHORED_NUMERIC_VALUE
        for v in outcome.violations
    )


def test_04_max_steps_looping_llm_aborts_cleanly(tmp_path: Path) -> None:
    """Cas 4 : MockLLMClient scripte pour boucler -> arret propre a MAX_STEPS=5 en abstention."""
    cfg = SupplyChainSettings(
        database_path=tmp_path / "loop.db",
        agent_traces_path=tmp_path / "traces.jsonl",
        agent_max_steps=5,
    )
    # 6 reponses demandant indefiniment le meme outil
    loop_responses = [
        MockLLMResponse(
            tool_calls=[
                MockToolCall(
                    name="query_inventory_and_supplier_sql",
                    arguments={"sku_id": "SKU-001"},
                )
            ],
            tokens_used=25,
        )
        for _ in range(6)
    ]
    mock_llm = MockLLMClient(scripted_responses=loop_responses)
    registry = SupplyChainToolRegistry(cfg=cfg, use_in_memory_fallback=True)
    orch = BoundedSupplyChainOrchestrator(
        llm_client=mock_llm,
        tool_registry=registry,
        cfg=cfg,
        traces_path=cfg.agent_traces_path,
    )
    res = orch.run(IncidentTrigger(sku_id="SKU-001", trigger_event="loop_test"))
    assert len(res.tool_calls) == 5
    assert res.report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert "GR-00-MAX-STEPS" in res.guardrails_triggered


def test_05_order_above_5000_eur_forced_to_pending_human_approval() -> None:
    """Cas 5 : Commande > 5000 EUR (300 * 20 = 6000 EUR) forcee en PENDING_HUMAN_APPROVAL."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(unit_cost=20.0, order_qty=300, total_cost=6120.0)
    # Le LLM tente malicieusement de forcer APPROVED_AUTO
    malicious_draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=300,
        expedited=False,
        expected_total_cost_eur=6120.0,
        confidence_level=0.95,
        evidence_tool_calls=["order_qty=300, unit_cost_eur=20.0, total_cost=6120.0"],
        reasoning_summary="Commander 300 unites au prix unitaire 20.0 EUR (total 6120.0 EUR).",
    )
    outcome = engine.evaluate(malicious_draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.PENDING_HUMAN_APPROVAL
    assert "GR-04-HITL-AMOUNT" in outcome.guardrails_triggered


def test_06_expedited_order_below_5000_eur_forced_to_pending_human_approval() -> None:
    """Cas 6 : Commande express (expedited=True) < 5000 EUR forcee en PENDING_HUMAN_APPROVAL."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(unit_cost=10.0, order_qty=100, total_cost=1570.0, expedited=True)
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=True,
        expected_total_cost_eur=1570.0,
        confidence_level=0.91,
        evidence_tool_calls=["order_qty=100, expedited=True, total_cost=1570.0"],
        reasoning_summary="Envoi express de 100 unites pour un cout total de 1570.0 EUR.",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.PENDING_HUMAN_APPROVAL
    assert "GR-05-HITL-EXPEDITED" in outcome.guardrails_triggered


def test_07_high_relative_uncertainty_forces_abstain() -> None:
    """Cas 7 : (q90 - q10)/q50 = (160 - 60)/100 = 1.00 > 0.80 -> ABSTAIN_INSUFFICIENT_DATA."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(q10=60.0, q50=100.0, q90=160.0)
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=2120.0,
        confidence_level=0.70,
        evidence_tool_calls=["q10=60.0, q50=100.0, q90=160.0"],
        reasoning_summary="Prevision mediane 100.0, commande de 100 unites (2120.0 EUR).",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert "GR-02-HIGH-UNCERTAINTY" in outcome.guardrails_triggered
    assert outcome.final_report.order_qty == 0


def test_08_zero_unit_price_forces_abstain() -> None:
    """Cas 8 : Prix unitaire nul (unit_cost_eur=0.0) dans InventoryRecord -> abstention."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(unit_cost=0.0, order_qty=100, total_cost=120.0)
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=120.0,
        confidence_level=0.80,
        evidence_tool_calls=["unit_cost_eur=0.0"],
        reasoning_summary="Prix 0.0 EUR, commande 100 unites pour 120.0 EUR.",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert "unit_cost_eur=0.0" in (outcome.final_report.abstention_reason or "")


def test_09_inconsistent_lead_time_std_greater_than_mean_flags_corrupted_data() -> None:
    """Cas 9 : Delai fournisseur incoherent (std=8.0 > mean=4.0) -> flag corrupted_data."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(lt_mean=4.0, lt_std=8.0)
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=2120.0,
        confidence_level=0.82,
        evidence_tool_calls=["lead_time_mean_days=4.0, lead_time_std_days=8.0"],
        reasoning_summary="Commander 100 unites pour 2120.0 EUR.",
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA
    assert "corrupted_data" in (outcome.final_report.abstention_reason or "")


def test_10_nominal_grounded_order_below_5000_is_approved_auto() -> None:
    """Cas 10 (Reference nominale) : Donnees saines, incertitude faible, montant <= 5000 EUR."""
    engine = DeterministicGuardrailEngine()
    tcs = _nominal_tool_calls(
        unit_cost=20.0,
        stock=40,
        q10=90.0,
        q50=100.0,
        q90=120.0,
        order_qty=100,
        total_cost=2120.0,
        expedited=False,
    )
    draft = DecisionReport(
        sku_id="SKU-001",
        status=ExecutionStatus.APPROVED_AUTO,
        order_qty=100,
        expedited=False,
        expected_total_cost_eur=2120.0,
        confidence_level=0.91,
        evidence_tool_calls=["q50=100.0, current_stock=40, order_qty=100, total_cost=2120.0"],
        reasoning_summary=(
            "Stock 40 insuffisant face a q50=100.0 ; commande 100 unites (2120.0 EUR)."
        ),
    )
    outcome = engine.evaluate(draft, tcs)
    assert outcome.final_report.status == ExecutionStatus.APPROVED_AUTO
    assert outcome.guardrails_triggered == []
    assert outcome.final_report.order_qty == 100
