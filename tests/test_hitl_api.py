"""Tests d'idempotence des endpoints HITL et integration FastAPI."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from supply_chain.agent.decision_store import SQLDecisionStore
from supply_chain.agent.llm_client import MockLLMClient, MockLLMResponse
from supply_chain.agent.orchestrator import BoundedSupplyChainOrchestrator
from supply_chain.agent.tools import SupplyChainToolRegistry
from supply_chain.api import PlatformAppState, create_app
from supply_chain.config import SupplyChainSettings


def test_hitl_workflow_and_double_approval_conflict_409(tmp_path: Path) -> None:
    """Cas 10bis : Cycle complet investigate -> pending -> approve
    -> double approve rejetee (409).
    """

    db_file = tmp_path / "hitl_test.db"
    traces_file = tmp_path / "traces.jsonl"
    cfg = SupplyChainSettings(
        database_path=db_file,
        agent_traces_path=traces_file,
    )

    # Le LLM propose une commande > 5000 EUR (ex: 400 unites * 25 EUR = 10000 EUR)
    draft_payload = {
        "sku_id": "SKU-001",
        "status": "APPROVED_AUTO",
        "order_qty": 400,
        "expedited": False,
        "expected_total_cost_eur": 10120.0,
        "confidence_level": 0.90,
        "evidence_tool_calls": ["order_qty=400, total_cost=10120.0"],
        "reasoning_summary": "Commander 400 unites pour un cout total de 10120.0 EUR.",
    }
    mock_llm = MockLLMClient(
        scripted_responses=[
            MockLLMResponse(final_content=json.dumps(draft_payload), tokens_used=85)
        ]
    )
    registry = SupplyChainToolRegistry(cfg=cfg, use_in_memory_fallback=True)
    orchestrator = BoundedSupplyChainOrchestrator(
        llm_client=mock_llm,
        tool_registry=registry,
        cfg=cfg,
        traces_path=traces_file,
    )
    store = SQLDecisionStore(db_path=db_file)
    state = PlatformAppState(
        cfg=cfg,
        model_service=None,
        orchestrator=orchestrator,
        decision_store=store,
    )
    app = create_app(cfg=cfg, custom_state=state)

    with TestClient(app) as client:
        # 1. Investigation forcee en PENDING_HUMAN_APPROVAL par GR-04-HITL-AMOUNT
        inv_resp = client.post(
            "/v1/agent/investigate",
            json={"sku_id": "SKU-001", "trigger_event": "stockout_risk_high"},
        )
        assert inv_resp.status_code == 200
        body = inv_resp.json()
        decision_id = body["decision_id"]
        assert body["report"]["status"] == "PENDING_HUMAN_APPROVAL"

        # 2. Verification dans GET /v1/decisions/pending
        pending_resp = client.get("/v1/decisions/pending")
        assert pending_resp.status_code == 200
        pending_list = pending_resp.json()
        assert len(pending_list) == 1
        assert pending_list[0]["id"] == decision_id

        # 3. Premiere approbation humaine -> 200 OK
        app_resp = client.post(
            f"/v1/decisions/{decision_id}/approve_or_reject",
            json={
                "operator_id": "op_marwane_01",
                "decision": "approve",
                "justification": "Validation budgetaire confirmee avec le fournisseur.",
            },
        )
        assert app_resp.status_code == 200
        assert app_resp.json()["new_status"] == "APPROVED_HUMAN"

        # 4. La file pending doit desormais etre vide
        assert client.get("/v1/decisions/pending").json() == []

        # 5. Double approbation sur la meme decision -> rejetee avec HTTP 409 Conflict
        dup_resp = client.post(
            f"/v1/decisions/{decision_id}/approve_or_reject",
            json={
                "operator_id": "op_marwane_01",
                "decision": "approve",
                "justification": "Tentative de double clic operateur.",
            },
        )
        assert dup_resp.status_code == 409

    # 6. Verification que le journal JSONL contient 2 traces (investigation + decision humaine)
    trace_lines = traces_file.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 2
