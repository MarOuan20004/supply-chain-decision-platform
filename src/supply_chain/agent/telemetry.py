"""Logger structure JSONL append-only et thread-safe pour l'audit trail de l'agent."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from supply_chain.agent.schemas import (
    AgentTrace,
    ExecutionStatus,
    ToolCallRecord,
)
from supply_chain.config import settings

_TRACE_WRITE_LOCK = threading.Lock()


def log_agent_trace(
    trace: AgentTrace,
    log_path: Path | None = None,
) -> None:
    """Valide et ajoute une ligne JSON atomique dans logs/agent_traces.jsonl.

    Thread-safe via threading.Lock() et mode ouverture append ('a').
    """
    target_path = Path(log_path) if log_path is not None else settings.agent_traces_path
    validated_trace = AgentTrace.model_validate(trace)
    serialized_line = validated_trace.model_dump_json() + "\n"

    with _TRACE_WRITE_LOCK:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as fh:
            fh.write(serialized_line)
            fh.flush()


def record_execution_trace(
    sku_id: str,
    trigger_event: str,
    tool_calls: Sequence[ToolCallRecord],
    tokens_consumed: int,
    schema_validation_errors: Sequence[str],
    decision_final_status: ExecutionStatus,
    guardrail_applied: Sequence[str],
    trace_id: str | None = None,
    operator_event: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> AgentTrace:
    """Construit un AgentTrace complet, l'ecrit dans le fichier JSONL et le retourne."""
    kwargs: dict[str, Any] = {
        "sku_id": sku_id,
        "trigger_event": trigger_event,
        "tool_calls": list(tool_calls),
        "tokens_consumed": int(tokens_consumed),
        "schema_validation_errors": list(schema_validation_errors),
        "decision_final_status": decision_final_status,
        "guardrail_applied": list(guardrail_applied),
        "operator_event": operator_event,
    }
    if trace_id is not None:
        kwargs["trace_id"] = trace_id

    trace = AgentTrace(**kwargs)
    log_agent_trace(trace, log_path=log_path)
    return trace
