"""Tests de serialisation JSONL thread-safe pour l'audit trail de l'agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from supply_chain.agent.schemas import (
    AgentTrace,
    ExecutionStatus,
)
from supply_chain.agent.telemetry import log_agent_trace


def test_jsonl_telemetry_is_valid_and_thread_safe(tmp_path: Path) -> None:
    """Verifie que 20 ecritures concurrentes produisent 20 lignes JSONL valides sans corruption."""
    log_file = tmp_path / "agent_traces.jsonl"

    def _write_one(idx: int) -> None:
        trace = AgentTrace(
            sku_id=f"SKU-{idx:03d}",
            trigger_event="concurrent_load_test",
            tokens_consumed=120 + idx,
            decision_final_status=ExecutionStatus.PENDING_HUMAN_APPROVAL,
            guardrail_applied=["GR-04-HITL-AMOUNT"],
        )
        log_agent_trace(trace, log_path=log_file)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write_one, range(20)))

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    parsed_skus = set()
    for line in lines:
        obj = AgentTrace.model_validate_json(line)
        parsed_skus.add(obj.sku_id)
        assert obj.decision_final_status == ExecutionStatus.PENDING_HUMAN_APPROVAL
    assert len(parsed_skus) == 20
