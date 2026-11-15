"""Schemas Pydantic v2 de l'Agent Supply Chain, des Garde-Fous Deterministes et du Workflow HITL."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionStatus(StrEnum):
    """Statuts finaux possibles d'une investigation agentique ou d'une decision HITL."""

    APPROVED_AUTO = "APPROVED_AUTO"
    PENDING_HUMAN_APPROVAL = "PENDING_HUMAN_APPROVAL"
    ABSTAIN_INSUFFICIENT_DATA = "ABSTAIN_INSUFFICIENT_DATA"
    APPROVED_HUMAN = "APPROVED_HUMAN"
    REJECTED_HUMAN = "REJECTED_HUMAN"


class GuardrailViolationType(StrEnum):
    """Taxonomie deterministe des garde-fous post-generation."""

    DATA_CORRUPTION = "data_corruption"
    HIGH_UNCERTAINTY = "high_uncertainty"
    UNANCHORED_NUMERIC_VALUE = "unanchored_numeric_value"
    HITL_AMOUNT_EXCEEDED = "hitl_amount_exceeded"
    HITL_EXPEDITED_SHIPPING = "hitl_expedited_shipping"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"


class GuardrailViolation(BaseModel):
    """Enregistrement structure d'un garde-fou declenche apres la generation LLM."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., description="Identifiant de la regle (ex: GR-01-DATA-CORRUPTION)")
    violation_type: GuardrailViolationType
    severity: Literal["block_abstain", "escalate_hitl"]
    message: str = Field(..., min_length=1, description="Description factuelle et citee")
    observed_value: str | None = Field(default=None)
    threshold_value: str | None = Field(default=None)


class IncidentTrigger(BaseModel):
    """Evenement declencheur d'une investigation par l'agent."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=1, description="Identifiant du SKU concerne")
    trigger_event: str = Field(
        default="criticality_alert",
        min_length=1,
        description="Type d'alerte (ex: stockout_risk_high, anomaly_detected)",
    )
    incident_id: str | None = Field(default=None, description="Identifiant unique de l'alerte")
    trigger_type: str | None = Field(
        default=None, description="Alias de compatibilite pour trigger_event"
    )
    horizon_days: int = Field(default=14, ge=1, le=90)
    anomaly_window_days: int = Field(default=14, ge=1, le=90)
    window_days: int = Field(default=14, ge=1, le=90)
    obs_date: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_trigger_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "incident_id" not in data:
                data["incident_id"] = f"INC-{uuid4().hex[:8].upper()}"
            if "trigger_event" not in data and "trigger_type" in data:
                data["trigger_event"] = str(data["trigger_type"]).lower()
            elif "trigger_event" in data and "trigger_type" not in data:
                data["trigger_type"] = str(data["trigger_event"]).upper()
            if "window_days" in data and "anomaly_window_days" not in data:
                data["anomaly_window_days"] = data["window_days"]
            elif "anomaly_window_days" in data and "window_days" not in data:
                data["window_days"] = data["anomaly_window_days"]
        return data


class ToolCallRequest(BaseModel):
    """Appel d'outil emis par le modele de langage lors d'un pas de raisonnement."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool_name


class LLMMessage(BaseModel):
    """Message normalise retourne par l'interface abstraite LLMClient."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tokens_used: int = 0

    @property
    def final_content(self) -> str | None:
        return self.content


class ToolCallRecord(BaseModel):
    """Trace unitaire d'un appel d'outil execute durant la session."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=1)
    tool_name: str = Field(..., min_length=1)
    validated_args: dict[str, Any] = Field(default_factory=dict)
    raw_output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(..., ge=0.0)
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)


class DecisionProposal(BaseModel):
    """Proposition d'action soumise a l'outil submit_decision_proposal."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=1)
    horizon_days: int = Field(default=7, ge=1, le=56)
    order_qty: int = Field(..., ge=0, description="Quantite recommandee (0 si abstention)")
    expedited: bool = Field(default=False, description="True si transport express demande")
    order_cost_eur: float = Field(default=0.0, ge=0.0)
    expected_total_cost_eur: float = Field(..., ge=0.0)
    unit_cost_eur: float = Field(default=0.0, ge=0.0)
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] | float = Field(...)
    reasoning_summary: str = Field(..., min_length=1)
    evidence_tool_calls: list[str] = Field(
        default_factory=list,
        description="Citations textuelles exactes des champs retournes par les outils",
    )
    abstain: bool = Field(default=False)
    abstention_reason: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_proposal_consistency(self) -> DecisionProposal:
        if self.abstain:
            if not self.abstention_reason or not self.abstention_reason.strip():
                raise ValueError("abstention_reason est obligatoire lorsque abstain=True.")
            if self.order_qty != 0:
                raise ValueError("order_qty doit etre egal a 0 lorsque abstain=True.")
        else:
            if not self.evidence_tool_calls:
                raise ValueError(
                    "evidence_tool_calls ne peut pas etre vide pour une proposition d'action."
                )
        return self


class SubmissionResult(BaseModel):
    """Sortie de l'outil submit_decision_proposal."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    status: ExecutionStatus
    sku_id: str = Field(default="")
    order_qty: int = Field(default=0, ge=0)
    expedited: bool = Field(default=False)
    order_cost_eur: float = Field(default=0.0, ge=0.0)
    expected_total_cost_eur: float = Field(default=0.0, ge=0.0)
    requires_human_approval: bool = Field(default=False)
    policy_reason: str = Field(default="")
    reason: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _sync_reason(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "reason" not in data and "policy_reason" in data:
                data["reason"] = data["policy_reason"]
            elif "policy_reason" not in data and "reason" in data:
                data["policy_reason"] = data["reason"]
        return data


class DecisionReport(BaseModel):
    """Rapport final structure produit par l'agent et valide par les garde-fous."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=1)
    status: ExecutionStatus
    order_qty: int = Field(..., ge=0)
    expedited: bool = Field(default=False)
    expected_total_cost_eur: float = Field(..., ge=0.0)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    evidence_tool_calls: list[str] = Field(
        default_factory=list,
        description="Citations textuelles exactes des preuves issues des outils",
    )
    reasoning_summary: str = Field(..., min_length=1)
    abstention_reason: str | None = Field(default=None)
    guardrails_triggered: list[str] = Field(
        default_factory=list,
        description="Liste des identifiants de garde-fous appliques en post-generation",
    )
    incident_id: str = Field(default_factory=lambda: f"INC-{uuid4().hex[:8].upper()}")
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    steps_executed: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _fill_report_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "incident_id" not in data:
                data["incident_id"] = f"INC-{uuid4().hex[:8].upper()}"
            if "steps_executed" not in data:
                data["steps_executed"] = len(data.get("tool_trace", []))
            if "total_latency_ms" not in data:
                data["total_latency_ms"] = 0.0
            if "confidence_level" in data and isinstance(data["confidence_level"], str):
                c_str = data["confidence_level"].upper()
                if c_str == "HIGH":
                    data["confidence_level"] = 0.90
                elif c_str == "MEDIUM":
                    data["confidence_level"] = 0.60
                elif c_str == "LOW":
                    data["confidence_level"] = 0.30
        return data

    @model_validator(mode="after")
    def _validate_consistency(self) -> DecisionReport:
        if self.status == ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA:
            if not self.abstention_reason or not self.abstention_reason.strip():
                raise ValueError(
                    "abstention_reason est obligatoire lorsque status == ABSTAIN_INSUFFICIENT_DATA."
                )
            if self.order_qty != 0:
                raise ValueError("order_qty doit valoir 0 en cas d'abstention.")
        return self


class AgentTrace(BaseModel):
    """Schema Pydantic obligatoire d'une ligne du journal logs/agent_traces.jsonl."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Horodatage ISO-8601 UTC",
    )
    sku_id: str = Field(..., min_length=1)
    trigger_event: str = Field(..., min_length=1)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tokens_consumed: int = Field(default=0, ge=0)
    schema_validation_errors: list[str] = Field(default_factory=list)
    decision_final_status: ExecutionStatus
    guardrail_applied: list[str] = Field(default_factory=list)
    operator_event: dict[str, Any] | None = Field(
        default=None,
        description="Metadonnees d'audit lorsqu'un operateur humain approuve ou rejette",
    )


class DecisionRecord(BaseModel):
    """Representation typee d'une ligne de la table SQL decisions."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    sku_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    status: ExecutionStatus
    report_json: DecisionReport
    created_at: str
    decided_at: str | None = None
    operator_id: str | None = None
    operator_justification: str | None = None


# --- Contrats REST pour les endpoints HITL de src/supply_chain/api.py ---


class InvestigateRequest(BaseModel):
    """Payload entrant de POST /v1/agent/investigate."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=1, max_length=64)
    trigger_event: str = Field(default="criticality_alert", min_length=1, max_length=128)
    horizon_days: int = Field(default=14, ge=1, le=90)


class InvestigateResponse(BaseModel):
    """Reponse de POST /v1/agent/investigate."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    trace_id: str
    report: DecisionReport


class OperatorDecisionRequest(BaseModel):
    """Payload entrant de POST /v1/decisions/{id}/approve_or_reject."""

    model_config = ConfigDict(extra="forbid")

    operator_id: str = Field(..., min_length=1, max_length=64)
    decision: Literal["approve", "reject"]
    justification: str = Field(..., min_length=3, max_length=1000)


class OperatorDecisionResponse(BaseModel):
    """Reponse de POST /v1/decisions/{id}/approve_or_reject."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    trace_id: str
    previous_status: ExecutionStatus
    new_status: ExecutionStatus
    operator_id: str
    decided_at: str
