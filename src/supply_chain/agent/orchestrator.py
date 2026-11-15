"""Orchestrateur d'aide a la decision Supply Chain a etats bornes (MAX_STEPS=5, timeout=15s)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from supply_chain.agent.exceptions import (
    AgentError,
    MaxStepsExceededError,
    OrchestratorTimeoutError,
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from supply_chain.agent.guardrails import DeterministicGuardrailEngine
from supply_chain.agent.llm_client import LLMClient
from supply_chain.agent.schemas import (
    DecisionReport,
    ExecutionStatus,
    IncidentTrigger,
    ToolCallRecord,
)
from supply_chain.agent.telemetry import record_execution_trace
from supply_chain.agent.tools import SupplyChainToolRegistry
from supply_chain.config import SupplyChainSettings, settings

SYSTEM_PROMPT = (
    "You are a deterministic Supply Chain Incident Resolution Orchestrator. "
    "You must investigate the SKU alert using the provided typed tools in logical order: "
    "(1) get_sku_forecast_and_uncertainty, (2) inspect_anomaly_diagnostics, "
    "(3) query_inventory_and_supplier_sql, (4) simulate_reorder_policy, "
    "(5) submit_decision_proposal. "
    "Never perform mental arithmetic: every cost and quantity in your final JSON report "
    "must come verbatim from tool outputs. After submit_decision_proposal succeeds, "
    "output ONLY a valid JSON object matching DecisionReport."
)


@dataclass
class OrchestratorSessionState:
    """Etat interne observable d'une session d'execution de l'orchestrateur."""

    incident: IncidentTrigger
    steps_executed: int = 0
    tool_trace: list[ToolCallRecord] = field(default_factory=list)
    termination_reason: str = "IN_PROGRESS"


class SupplyChainDecisionOrchestrator:
    """Machine a etats finis bornee (MAX_STEPS = 5, timeout = 15s) pilotant les 5 outils."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: SupplyChainToolRegistry,
        cfg: SupplyChainSettings = settings,
        max_steps: int | None = None,
        timeout_sec: float | None = None,
        strict_exceptions: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.cfg = cfg
        self.max_steps = int(max_steps if max_steps is not None else cfg.agent_max_steps)
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else cfg.agent_timeout_sec)
        self.strict_exceptions = bool(strict_exceptions)
        self.last_state: OrchestratorSessionState | None = None

    def run(self, incident: IncidentTrigger) -> DecisionReport:
        """Execute la boucle d'investigation bornee et retourne un DecisionReport verifie."""
        t_session_start = time.perf_counter()
        state = OrchestratorSessionState(incident=incident)
        self.last_state = state

        tool_schemas = self.tool_registry.get_tool_definitions()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Investigate supply chain incident {incident.incident_id} for "
                    f"sku_id='{incident.sku_id}' (trigger={incident.trigger_type}, "
                    f"horizon_days={incident.horizon_days}, window_days={incident.window_days})."
                ),
            },
        ]

        while state.steps_executed < self.max_steps:
            elapsed = time.perf_counter() - t_session_start
            if elapsed > self.timeout_sec:
                state.termination_reason = (
                    f"ORCHESTRATOR_TIMEOUT: {elapsed:.2f}s > {self.timeout_sec:.2f}s"
                )
                if self.strict_exceptions:
                    raise OrchestratorTimeoutError(state.termination_reason)
                return self._build_abstention_report(
                    state=state,
                    reason=state.termination_reason,
                    t_start=t_session_start,
                )

            state.steps_executed += 1
            assistant_msg = self.llm_client.chat(messages=messages, tools=tool_schemas)

            # Verifier le timeout immediatement apres l'appel LLM
            elapsed_after_llm = time.perf_counter() - t_session_start
            if elapsed_after_llm > self.timeout_sec:
                state.termination_reason = (
                    f"ORCHESTRATOR_TIMEOUT: {elapsed_after_llm:.2f}s > {self.timeout_sec:.2f}s"
                )
                if self.strict_exceptions:
                    raise OrchestratorTimeoutError(state.termination_reason)
                return self._build_abstention_report(
                    state=state,
                    reason=state.termination_reason,
                    t_start=t_session_start,
                )

            # Cas A : Le LLM demande l'execution d'un ou plusieurs outils
            if assistant_msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_msg.content or "",
                        "tool_calls": [
                            tc.model_dump(mode="json") for tc in assistant_msg.tool_calls
                        ],
                    }
                )
                for tc in assistant_msg.tool_calls:
                    t_tool0 = time.perf_counter()
                    try:
                        val_args, raw_out, lat_ms = self.tool_registry.execute_tool(
                            tool_name=tc.tool_name,
                            raw_arguments=tc.arguments,
                        )
                        record = ToolCallRecord(
                            step_index=state.steps_executed,
                            tool_name=tc.tool_name,
                            validated_args=val_args,
                            raw_output=raw_out,
                            latency_ms=lat_ms,
                            success=True,
                            error_message=None,
                        )
                        state.tool_trace.append(record)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.call_id,
                                "name": tc.tool_name,
                                "content": json.dumps(raw_out, ensure_ascii=False),
                            }
                        )
                    except AgentError as exc:
                        err_lat_ms = round((time.perf_counter() - t_tool0) * 1000.0, 3)
                        err_text = f"{type(exc).__name__}: {exc}"
                        record = ToolCallRecord(
                            step_index=state.steps_executed,
                            tool_name=tc.tool_name,
                            validated_args=dict(tc.arguments),
                            raw_output={},
                            latency_ms=err_lat_ms,
                            success=False,
                            error_message=err_text,
                        )
                        state.tool_trace.append(record)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.call_id,
                                "name": tc.tool_name,
                                "content": json.dumps({"error": err_text}, ensure_ascii=False),
                            }
                        )
                continue

            # Cas B : Le LLM emet le rapport final DecisionReport en JSON
            if assistant_msg.content:
                try:
                    report = self._parse_and_verify_final_report(
                        raw_content=assistant_msg.content,
                        state=state,
                        t_start=t_session_start,
                    )
                    state.termination_reason = "COMPLETED_SUCCESSFULLY"
                    return report
                except (ValueError, ValidationError) as exc:
                    # Renvoie l'erreur de schema au LLM s'il reste des pas disponibles
                    messages.append({"role": "assistant", "content": assistant_msg.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Invalid DecisionReport format or grounding error: {exc}",
                        }
                    )

        # Depassement de MAX_STEPS sans rapport final valide
        state.termination_reason = (
            f"MAX_STEPS_EXCEEDED: limite de {self.max_steps} etapes atteinte sans decision valide."
        )
        if self.strict_exceptions:
            raise MaxStepsExceededError(state.termination_reason)
        return self._build_abstention_report(
            state=state,
            reason=state.termination_reason,
            t_start=t_session_start,
        )

    def _parse_and_verify_final_report(
        self,
        raw_content: str,
        state: OrchestratorSessionState,
        t_start: float,
    ) -> DecisionReport:
        """Valide le JSON final et controle son ancrage strict dans les sorties d'outils."""
        parsed = json.loads(raw_content)
        if not isinstance(parsed, dict):
            raise ValueError("La reponse finale doit etre un objet JSON.")

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        parsed["incident_id"] = state.incident.incident_id
        parsed["sku_id"] = state.incident.sku_id
        parsed["tool_trace"] = [r.model_dump(mode="json") for r in state.tool_trace]
        parsed["steps_executed"] = state.steps_executed
        parsed["total_latency_ms"] = total_ms

        # Si une proposition a ete soumise via submit_decision_proposal, aligner le statut
        submissions = [
            r.raw_output
            for r in state.tool_trace
            if r.tool_name == "submit_decision_proposal" and r.success
        ]
        if submissions:
            last_sub = submissions[-1]
            parsed["status"] = last_sub["status"]
            parsed["order_qty"] = last_sub["order_qty"]
            parsed["expedited"] = last_sub["expedited"]
            parsed["expected_total_cost_eur"] = last_sub["expected_total_cost_eur"]

        report = DecisionReport.model_validate(parsed)

        # Verification que chaque citation dans evidence_tool_calls existe dans les sorties d'outils
        successful_outputs_blob = json.dumps(
            [r.raw_output for r in state.tool_trace if r.success],
            ensure_ascii=False,
        )
        for evidence in report.evidence_tool_calls:
            if evidence not in successful_outputs_blob:
                raise ValueError(
                    f"Preuve non ancree dans les sorties d'outils de la session: '{evidence}'"
                )

        return report

    def _build_abstention_report(
        self,
        state: OrchestratorSessionState,
        reason: str,
        t_start: float,
    ) -> DecisionReport:
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        return DecisionReport(
            incident_id=state.incident.incident_id or f"INC-{uuid4().hex[:8].upper()}",
            sku_id=state.incident.sku_id,
            status=ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA,
            order_qty=0,
            expedited=False,
            expected_total_cost_eur=0.0,
            confidence_level=0.30,
            reasoning_summary=(
                f"Arret controle de l'orchestrateur apres {state.steps_executed} etapes: {reason}"
            ),
            evidence_tool_calls=[],
            abstention_reason=reason,
            tool_trace=list(state.tool_trace),
            steps_executed=state.steps_executed,
            total_latency_ms=total_ms,
        )


SYSTEM_PROMPT_SUPPLY_CHAIN_AGENT = (
    "You are a deterministic Supply Chain Incident Resolution Orchestrator. "
    "Investigate the SKU alert using the registered tools in order: "
    "1) get_sku_forecast_and_uncertainty, 2) inspect_anomaly_diagnostics, "
    "3) query_inventory_and_supplier_sql, 4) simulate_reorder_policy, "
    "5) submit_decision_proposal. "
    "Never perform mental arithmetic; copy exact numeric values from tool outputs into "
    "your final DecisionReport JSON."
)


@dataclass
class OrchestratorRunResult:
    """Resultat complet d'une execution de l'orchestrateur gouverne."""

    trace_id: str
    report: DecisionReport
    tool_calls: list[ToolCallRecord]
    tokens_consumed: int
    schema_validation_errors: list[str]
    guardrails_triggered: list[str]


class BoundedSupplyChainOrchestrator:
    """Machine a etats finis bornee (MAX_STEPS <= 5) avec garde-fous post-generation S8."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: SupplyChainToolRegistry,
        cfg: SupplyChainSettings = settings,
        guardrail_engine: DeterministicGuardrailEngine | None = None,
        traces_path: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.cfg = cfg
        self.max_steps = cfg.agent_max_steps
        self.timeout_sec = cfg.agent_timeout_sec
        self.guardrail_engine = guardrail_engine or DeterministicGuardrailEngine(cfg=cfg)
        self.traces_path = traces_path or cfg.agent_traces_path

    def run(self, trigger: IncidentTrigger) -> OrchestratorRunResult:
        """Execute la boucle d'investigation bornee, applique les garde-fous et logue la trace."""
        trace_id = str(uuid4())
        t_start = time.perf_counter()

        tool_calls: list[ToolCallRecord] = []
        schema_errors: list[str] = []
        total_tokens = 0
        draft_report: DecisionReport | None = None
        extra_guardrails: list[str] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_SUPPLY_CHAIN_AGENT},
            {
                "role": "user",
                "content": (
                    f"Investigate supply chain incident for sku_id='{trigger.sku_id}', "
                    f"trigger_event='{trigger.trigger_event}', "
                    f"horizon_days={trigger.horizon_days}."
                ),
            },
        ]
        tool_schemas = self.tool_registry.get_openai_tool_schemas()

        for step_idx in range(1, self.max_steps + 1):
            elapsed = time.perf_counter() - t_start
            if elapsed > self.timeout_sec:
                raise OrchestratorTimeoutError(
                    f"Timeout de {self.timeout_sec}s depasse a l'etape {step_idx}."
                )

            llm_response = self.llm_client.chat(messages=messages, tools=tool_schemas)
            total_tokens += int(getattr(llm_response, "tokens_used", 0))

            # Cas A : Le LLM demande l'execution d'un ou plusieurs outils
            resp_calls = getattr(llm_response, "tool_calls", None) or []
            if resp_calls:
                for tc_req in resp_calls:
                    t_tool0 = time.perf_counter()
                    tc_name: str = str(
                        getattr(tc_req, "name", None)
                        or getattr(tc_req, "tool_name", "")
                        or "unknown_tool"
                    )
                    raw_args = getattr(tc_req, "arguments", {})
                    tc_args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
                    try:
                        tool_res = self.tool_registry.execute_tool(
                            tool_name=tc_name,
                            raw_arguments=tc_args,
                        )
                        tool_lat_ms = round((time.perf_counter() - t_tool0) * 1000.0, 3)
                        if len(tool_res) == 3:
                            validated_args, raw_out, _ = tool_res
                        else:
                            raw_out, validated_args = tool_res

                        rec = ToolCallRecord(
                            step_index=step_idx,
                            tool_name=tc_name,
                            validated_args=validated_args,
                            raw_output=raw_out,
                            latency_ms=tool_lat_ms,
                            success=True,
                        )
                        tool_calls.append(rec)
                        messages.append(
                            {
                                "role": "tool",
                                "name": tc_name,
                                "content": json.dumps(raw_out, ensure_ascii=False),
                            }
                        )
                    except (UnknownToolError, ToolValidationError, ToolExecutionError) as exc:
                        tool_lat_ms = round((time.perf_counter() - t_tool0) * 1000.0, 3)
                        err_msg = f"{type(exc).__name__}: {exc}"
                        schema_errors.append(err_msg)
                        rec = ToolCallRecord(
                            step_index=step_idx,
                            tool_name=tc_name,
                            validated_args=tc_args if isinstance(tc_args, dict) else {},
                            raw_output={},
                            latency_ms=tool_lat_ms,
                            success=False,
                            error_message=err_msg,
                        )
                        tool_calls.append(rec)
                        messages.append(
                            {
                                "role": "tool",
                                "name": tc_name,
                                "content": json.dumps({"error": err_msg}, ensure_ascii=False),
                            }
                        )
                continue

            # Cas B : Le LLM emet son rapport final JSON
            content_val = getattr(llm_response, "final_content", None) or getattr(
                llm_response, "content", None
            )
            if content_val is not None:
                try:
                    raw_payload = (
                        json.loads(content_val) if isinstance(content_val, str) else content_val
                    )
                    draft_report = DecisionReport.model_validate(raw_payload)
                    break
                except (json.JSONDecodeError, ValidationError) as exc:
                    err_msg = f"DecisionReportValidationError: {exc}"
                    schema_errors.append(err_msg)
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Invalid DecisionReport JSON schema: {exc}. Fix and retry.",
                        }
                    )

        # Si la boucle a epuise ses pas d'outils sans DecisionReport, tentative de synthese finale
        if draft_report is None:
            final_turn = self.llm_client.chat(messages=messages, tools=[])
            total_tokens += int(getattr(final_turn, "tokens_used", 0))
            content_val = getattr(final_turn, "final_content", None) or getattr(
                final_turn, "content", None
            )
            if content_val is not None:
                try:
                    raw_payload = (
                        json.loads(content_val) if isinstance(content_val, str) else content_val
                    )
                    draft_report = DecisionReport.model_validate(raw_payload)
                except (json.JSONDecodeError, ValidationError) as exc:
                    schema_errors.append(f"DecisionReportValidationError: {exc}")
                    draft_report = None

        # Si toujours aucun DecisionReport valide -> Abstention deterministe GR-00-MAX-STEPS
        if draft_report is None:
            extra_guardrails.append("GR-00-MAX-STEPS")
            draft_report = DecisionReport(
                sku_id=trigger.sku_id,
                status=ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA,
                order_qty=0,
                expedited=False,
                expected_total_cost_eur=0.0,
                confidence_level=0.0,
                evidence_tool_calls=[
                    f"{tc.tool_name}: {json.dumps(tc.raw_output)}"
                    for tc in tool_calls
                    if tc.success
                ],
                reasoning_summary=(
                    "Abstention forcee : nombre maximal d'etapes d'outils atteint sans convergence."
                ),
                abstention_reason="max_steps_exceeded",
                guardrails_triggered=list(extra_guardrails),
            )

        # Application obligatoire et non contournable des garde-fous post-generation S8
        outcome = self.guardrail_engine.evaluate(
            draft_report=draft_report,
            tool_calls=tool_calls,
        )
        all_guardrails = list(dict.fromkeys([*extra_guardrails, *outcome.guardrails_triggered]))

        final_report = DecisionReport(
            sku_id=outcome.final_report.sku_id,
            status=outcome.final_report.status,
            order_qty=outcome.final_report.order_qty,
            expedited=outcome.final_report.expedited,
            expected_total_cost_eur=outcome.final_report.expected_total_cost_eur,
            confidence_level=outcome.final_report.confidence_level,
            evidence_tool_calls=outcome.final_report.evidence_tool_calls,
            reasoning_summary=outcome.final_report.reasoning_summary,
            abstention_reason=outcome.final_report.abstention_reason,
            guardrails_triggered=all_guardrails,
        )

        # Journalisation atomique dans logs/agent_traces.jsonl
        record_execution_trace(
            trace_id=trace_id,
            sku_id=trigger.sku_id,
            trigger_event=trigger.trigger_event,
            tool_calls=tool_calls,
            tokens_consumed=total_tokens,
            schema_validation_errors=schema_errors,
            decision_final_status=final_report.status,
            guardrail_applied=all_guardrails,
            log_path=self.traces_path,
        )

        return OrchestratorRunResult(
            trace_id=trace_id,
            report=final_report,
            tool_calls=tool_calls,
            tokens_consumed=total_tokens,
            schema_validation_errors=schema_errors,
            guardrails_triggered=all_guardrails,
        )
