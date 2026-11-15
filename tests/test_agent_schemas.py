"""Tests unitaires des schemas Pydantic v2 de l'Agent Supply Chain."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from supply_chain.agent.schemas import (
    DecisionProposal,
    DecisionReport,
    ExecutionStatus,
    IncidentTrigger,
)


def test_decision_proposal_requires_evidence_or_abstention_reason() -> None:
    """Verifie le rejet d'une proposition sans preuve ou d'une abstention sans motif."""
    # Action sans evidence_tool_calls -> rejetee
    with pytest.raises(ValidationError, match="evidence_tool_calls"):
        DecisionProposal(
            sku_id="SKU-0001",
            horizon_days=7,
            order_qty=50,
            expedited=False,
            order_cost_eur=1045.0,
            expected_total_cost_eur=1120.0,
            confidence_level="HIGH",
            reasoning_summary="Commande standard recommandee suite a la prevision.",
            evidence_tool_calls=[],
            abstain=False,
        )

    # Abstention sans abstention_reason -> rejetee
    with pytest.raises(ValidationError, match="abstention_reason"):
        DecisionProposal(
            sku_id="SKU-0001",
            horizon_days=7,
            order_qty=0,
            expedited=False,
            order_cost_eur=0.0,
            expected_total_cost_eur=0.0,
            confidence_level="LOW",
            reasoning_summary="Incertitude trop elevee pour commander.",
            abstain=True,
            abstention_reason=None,
        )


def test_decision_report_invariants_and_extra_forbid() -> None:
    """Verifie extra='forbid' et les invariants de DecisionReport."""
    with pytest.raises(ValidationError):
        IncidentTrigger(incident_id="INC-01", sku_id="SKU-0001", unknown_field=123)  # type: ignore[call-arg]

    report = DecisionReport(
        incident_id="INC-01",
        sku_id="SKU-0001",
        status=ExecutionStatus.PENDING_HUMAN_APPROVAL,
        order_qty=120,
        expedited=True,
        expected_total_cost_eur=6540.0,
        confidence_level="HIGH",
        reasoning_summary="Rupture imminente confirmee, commande express > 5000 EUR.",
        evidence_tool_calls=['"calibrated_criticality_prob": 0.84', '"total_cost": 6540.0'],
        steps_executed=5,
        total_latency_ms=12.4,
    )
    assert report.status == ExecutionStatus.PENDING_HUMAN_APPROVAL
    assert report.order_qty == 120
