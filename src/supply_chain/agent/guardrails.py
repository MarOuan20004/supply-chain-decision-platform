"""Moteur de garde-fous post-generation deterministes et non contournables.

Principes fondamentaux :
1. Le LLM ne decide jamais seul de s'abstenir ou d'escalader a l'humain.
2. Les regles s'executent APRES la generation du DecisionReport par le LLM, en comparant
   le rapport aux sorties brutes (raw_output) des outils executes durant la session.
3. Priorite stricte des verrous :
   Abstention (Corruption / Incertitude / Hallucination numerique) > Escalade HITL > Auto.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from supply_chain.agent.schemas import (
    DecisionReport,
    ExecutionStatus,
    GuardrailViolation,
    GuardrailViolationType,
    ToolCallRecord,
)
from supply_chain.config import SupplyChainSettings, settings

# Capture les nombres autonomes (entiers ou flottants), sans capturer les chiffres colles
# a des lettres/tirets comme 'SKU-001' ou 'H14'.
_STANDALONE_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_-])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_-])")


@dataclass(frozen=True)
class GuardrailEvaluationOutcome:
    """Resultat complet de l'application des garde-fous sur un DecisionReport."""

    final_report: DecisionReport
    violations: list[GuardrailViolation]
    guardrails_triggered: list[str]


def _extract_numbers_recursive(payload: Any, acc: set[float]) -> None:
    """Extrait recursivement toutes les valeurs numeriques d'une structure JSON/Python."""
    if isinstance(payload, bool) or payload is None:
        return
    if isinstance(payload, (int, float)):
        val = float(payload)
        if math.isfinite(val):
            acc.add(val)
        return
    if isinstance(payload, str):
        for match in _STANDALONE_NUMBER_RE.findall(payload):
            try:
                val = float(match)
                if math.isfinite(val):
                    acc.add(val)
            except ValueError:
                continue
        return
    if isinstance(payload, dict):
        for v in payload.values():
            _extract_numbers_recursive(v, acc)
        return
    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            _extract_numbers_recursive(item, acc)


def extract_report_numbers(report: DecisionReport) -> list[float]:
    """Extrait tous les nombres cites dans les champs et textes du DecisionReport."""
    numbers: list[float] = [
        float(report.order_qty),
        float(report.expected_total_cost_eur),
    ]
    text_blobs = [report.reasoning_summary, *report.evidence_tool_calls]
    for blob in text_blobs:
        for match in _STANDALONE_NUMBER_RE.findall(blob):
            try:
                val = float(match)
                if math.isfinite(val):
                    numbers.append(val)
            except ValueError:
                continue
    return numbers


class DeterministicGuardrailEngine:
    """Evaluateur deterministe post-generation des verrous de securite operationnels."""

    def __init__(self, cfg: SupplyChainSettings = settings) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        draft_report: DecisionReport,
        tool_calls: Sequence[ToolCallRecord],
    ) -> GuardrailEvaluationOutcome:
        """Applique les garde-fous sur draft_report a partir des sorties d'outils de la session."""
        violations: list[GuardrailViolation] = []
        triggered_ids: list[str] = []

        # Indexation des sorties brutes des outils reussis par nom d'outil
        outputs_by_tool: dict[str, list[dict[str, Any]]] = {}
        for tc in tool_calls:
            if tc.success:
                outputs_by_tool.setdefault(tc.tool_name, []).append(tc.raw_output)

        # --- VERROU 1A : Detection de corruption de donnees dans les outils ---
        corruption_violation = self._check_data_corruption(outputs_by_tool)
        if corruption_violation is not None:
            violations.append(corruption_violation)
            triggered_ids.append(corruption_violation.rule_id)

        # --- VERROU 1B : Detection d'incertitude relative excessive (q90 - q10)/q50 > 0.80 ---
        uncertainty_violation = self._check_high_uncertainty(outputs_by_tool)
        if uncertainty_violation is not None:
            violations.append(uncertainty_violation)
            triggered_ids.append(uncertainty_violation.rule_id)

        # Si corruption ou incertitude elevee -> Abstention forcee immediate sans achat
        if violations:
            primary = violations[0]
            abstained_report = DecisionReport(
                sku_id=draft_report.sku_id,
                status=ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA,
                order_qty=0,
                expedited=False,
                expected_total_cost_eur=0.0,
                confidence_level=0.0,
                evidence_tool_calls=draft_report.evidence_tool_calls,
                reasoning_summary=(
                    f"Abstention deterministe imposee par {primary.rule_id} : {primary.message}"
                ),
                abstention_reason=primary.message,
                guardrails_triggered=triggered_ids,
            )
            return GuardrailEvaluationOutcome(
                final_report=abstained_report,
                violations=violations,
                guardrails_triggered=triggered_ids,
            )

        # --- VERROU 2 : Validateur d'ancrage numerique (tolerance 1e-2 EUR) ---
        unanchored_violation = self._check_numeric_grounding(draft_report, tool_calls)
        if unanchored_violation is not None:
            violations.append(unanchored_violation)
            triggered_ids.append(unanchored_violation.rule_id)
            rejected_report = DecisionReport(
                sku_id=draft_report.sku_id,
                status=ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA,
                order_qty=0,
                expedited=False,
                expected_total_cost_eur=0.0,
                confidence_level=0.0,
                evidence_tool_calls=draft_report.evidence_tool_calls,
                reasoning_summary=(
                    f"Rapport rejete par le validateur d'ancrage numerique : "
                    f"{unanchored_violation.message}"
                ),
                abstention_reason="unanchored_numeric_value",
                guardrails_triggered=triggered_ids,
            )
            return GuardrailEvaluationOutcome(
                final_report=rejected_report,
                violations=violations,
                guardrails_triggered=triggered_ids,
            )

        # --- VERROU 3 : Escalade HITL obligatoire (> 5000 EUR ou transport express) ---
        final_status = draft_report.status
        hitl_violations = self._check_hitl_escalation(draft_report, outputs_by_tool)
        if hitl_violations:
            for hv in hitl_violations:
                violations.append(hv)
                triggered_ids.append(hv.rule_id)
            if final_status != ExecutionStatus.ABSTAIN_INSUFFICIENT_DATA:
                final_status = ExecutionStatus.PENDING_HUMAN_APPROVAL

        final_report = DecisionReport(
            sku_id=draft_report.sku_id,
            status=final_status,
            order_qty=draft_report.order_qty,
            expedited=draft_report.expedited,
            expected_total_cost_eur=draft_report.expected_total_cost_eur,
            confidence_level=draft_report.confidence_level,
            evidence_tool_calls=draft_report.evidence_tool_calls,
            reasoning_summary=draft_report.reasoning_summary,
            abstention_reason=draft_report.abstention_reason,
            guardrails_triggered=triggered_ids,
        )
        return GuardrailEvaluationOutcome(
            final_report=final_report,
            violations=violations,
            guardrails_triggered=triggered_ids,
        )

    def _check_data_corruption(
        self, outputs_by_tool: dict[str, list[dict[str, Any]]]
    ) -> GuardrailViolation | None:
        """Verifie l'integrite des sorties d'anomalies et d'inventaire SQL."""
        # 1. Inspection des diagnostics d'anomalies
        for anom in outputs_by_tool.get("inspect_anomaly_diagnostics", []):
            contribs = anom.get("feature_contributions")
            if contribs is None or (isinstance(contribs, dict) and len(contribs) == 0):
                return GuardrailViolation(
                    rule_id="GR-01-DATA-CORRUPTION",
                    violation_type=GuardrailViolationType.DATA_CORRUPTION,
                    severity="block_abstain",
                    message=(
                        "Donnees de diagnostic manquantes : feature_contributions vide dans "
                        "inspect_anomaly_diagnostics."
                    ),
                    observed_value="feature_contributions={}",
                    threshold_value="non_empty_dict",
                )
            if anom.get("data_corrupted") is True or anom.get("anomaly_type") == "data_corruption":
                detail = str(anom.get("summary") or anom.get("details") or "corruption_detectee")
                return GuardrailViolation(
                    rule_id="GR-01-DATA-CORRUPTION",
                    violation_type=GuardrailViolationType.DATA_CORRUPTION,
                    severity="block_abstain",
                    message=f"Corruption des donnees signalee par IsolationForest : {detail}",
                    observed_value=detail,
                )

        # 2. Inspection de l'enregistrement SQL d'inventaire et fournisseur
        for inv in outputs_by_tool.get("query_inventory_and_supplier_sql", []):
            stock = inv.get("current_stock")
            if stock is not None and float(stock) < 0.0:
                return GuardrailViolation(
                    rule_id="GR-01-DATA-CORRUPTION",
                    violation_type=GuardrailViolationType.DATA_CORRUPTION,
                    severity="block_abstain",
                    message=f"Stock physique negatif incoherent detecte : current_stock={stock}.",
                    observed_value=str(stock),
                    threshold_value=">= 0",
                )

            unit_cost = inv.get("unit_cost_eur")
            if unit_cost is not None and float(unit_cost) <= 0.0:
                return GuardrailViolation(
                    rule_id="GR-01-DATA-CORRUPTION",
                    violation_type=GuardrailViolationType.DATA_CORRUPTION,
                    severity="block_abstain",
                    message=f"Prix unitaire nul ou negatif detecte : unit_cost_eur={unit_cost}.",
                    observed_value=str(unit_cost),
                    threshold_value="> 0.0",
                )

            lt_mean = inv.get("lead_time_mean_days")
            lt_std = inv.get("lead_time_std_days")
            if lt_mean is not None and float(lt_mean) <= 0.0:
                return GuardrailViolation(
                    rule_id="GR-01-DATA-CORRUPTION",
                    violation_type=GuardrailViolationType.DATA_CORRUPTION,
                    severity="block_abstain",
                    message=f"Delai fournisseur moyen invalide : lead_time_mean_days={lt_mean}.",
                    observed_value=str(lt_mean),
                    threshold_value="> 0.0",
                )
            if lt_mean is not None and lt_std is not None and float(lt_std) > float(lt_mean):
                return GuardrailViolation(
                    rule_id="GR-01-DATA-CORRUPTION",
                    violation_type=GuardrailViolationType.DATA_CORRUPTION,
                    severity="block_abstain",
                    message=(
                        f"Delai fournisseur incoherent (corrupted_data) : "
                        f"lead_time_std_days={lt_std} > lead_time_mean_days={lt_mean}."
                    ),
                    observed_value=f"std={lt_std}, mean={lt_mean}",
                    threshold_value="std <= mean",
                )

        return None

    def _check_high_uncertainty(
        self, outputs_by_tool: dict[str, list[dict[str, Any]]]
    ) -> GuardrailViolation | None:
        """Verifie que la largeur relative (q90 - q10) / q50 ne depasse pas tau_uncert = 0.80."""
        tau = self.cfg.agent_uncertainty_threshold
        for fc in outputs_by_tool.get("get_sku_forecast_and_uncertainty", []):
            horizons = fc.get("forecasts") or fc.get("quantiles") or []
            if isinstance(horizons, dict):
                horizons = [horizons]
            for h_item in horizons:
                if not isinstance(h_item, dict):
                    continue
                q10 = float(h_item.get("q10", 0.0))
                q50 = float(h_item.get("q50", 0.0))
                q90 = float(h_item.get("q90", 0.0))
                if q50 <= 0.0:
                    return GuardrailViolation(
                        rule_id="GR-02-HIGH-UNCERTAINTY",
                        violation_type=GuardrailViolationType.HIGH_UNCERTAINTY,
                        severity="block_abstain",
                        message=(
                            f"Mediane de prevision q50={q50} <= 0 rendant l'incertitude infinie."
                        ),
                        observed_value=f"q50={q50}",
                        threshold_value="q50 > 0",
                    )
                rel_width = (q90 - q10) / q50
                if rel_width > tau:
                    return GuardrailViolation(
                        rule_id="GR-02-HIGH-UNCERTAINTY",
                        violation_type=GuardrailViolationType.HIGH_UNCERTAINTY,
                        severity="block_abstain",
                        message=(
                            f"Incertitude relative excessive : (q90 - q10)/q50 = "
                            f"({q90} - {q10})/{q50} = {rel_width:.3f} > {tau:.2f}."
                        ),
                        observed_value=f"{rel_width:.4f}",
                        threshold_value=f"<= {tau:.2f}",
                    )
        return None

    def _check_numeric_grounding(
        self,
        report: DecisionReport,
        tool_calls: Sequence[ToolCallRecord],
    ) -> GuardrailViolation | None:
        """Verifie que chaque nombre du DecisionReport est present dans les sorties d'outils."""
        successful_calls = [tc for tc in tool_calls if tc.success]
        if not successful_calls:
            return None

        grounded_numbers: set[float] = {0.0}
        for tc in successful_calls:
            _extract_numbers_recursive(tc.raw_output, grounded_numbers)
            _extract_numbers_recursive(tc.validated_args, grounded_numbers)

        # Ajout des montants d'engagement exacts calculables depuis inventaire x quantite
        for inv_tc in tool_calls:
            if inv_tc.success and inv_tc.tool_name == "query_inventory_and_supplier_sql":
                u_cost = inv_tc.raw_output.get("unit_cost_eur")
                if u_cost is not None:
                    grounded_numbers.add(round(float(report.order_qty) * float(u_cost), 4))

        tol = self.cfg.numeric_grounding_tolerance
        cited_numbers = extract_report_numbers(report)

        for val in cited_numbers:
            is_anchored = any(abs(val - ref) <= tol for ref in grounded_numbers)
            if not is_anchored:
                return GuardrailViolation(
                    rule_id="GR-03-NUMERIC-GROUNDING",
                    violation_type=GuardrailViolationType.UNANCHORED_NUMERIC_VALUE,
                    severity="block_abstain",
                    message=(
                        f"unanchored_numeric_value: la valeur {val} citee dans le DecisionReport "
                        f"est absente des sorties d'outils de la session (tolerance {tol})."
                    ),
                    observed_value=str(val),
                    threshold_value=f"tol <= {tol}",
                )
        return None

    def _check_hitl_escalation(
        self,
        report: DecisionReport,
        outputs_by_tool: dict[str, list[dict[str, Any]]],
    ) -> list[GuardrailViolation]:
        """Verifie les seuils financiers (> 5000 EUR) et le transport express (expedited)."""
        hitl_list: list[GuardrailViolation] = []
        threshold = self.cfg.hitl_approval_threshold_eur

        unit_cost = 0.0
        inv_records = outputs_by_tool.get("query_inventory_and_supplier_sql", [])
        if inv_records:
            unit_cost = float(inv_records[-1].get("unit_cost_eur", 0.0))

        purchase_commitment = float(report.order_qty) * unit_cost
        effective_exposure = max(purchase_commitment, float(report.expected_total_cost_eur))

        if effective_exposure > threshold:
            hitl_list.append(
                GuardrailViolation(
                    rule_id="GR-04-HITL-AMOUNT",
                    violation_type=GuardrailViolationType.HITL_AMOUNT_EXCEEDED,
                    severity="escalate_hitl",
                    message=(
                        f"Engagement financier {effective_exposure:.2f} EUR > seuil HITL "
                        f"{threshold:.2f} EUR : approbation humaine obligatoire."
                    ),
                    observed_value=f"{effective_exposure:.2f}",
                    threshold_value=f"<= {threshold:.2f}",
                )
            )

        if report.expedited:
            hitl_list.append(
                GuardrailViolation(
                    rule_id="GR-05-HITL-EXPEDITED",
                    violation_type=GuardrailViolationType.HITL_EXPEDITED_SHIPPING,
                    severity="escalate_hitl",
                    message=(
                        "Transport express (expedited=True) demande : "
                        "approbation humaine obligatoire quel que soit le montant."
                    ),
                    observed_value="True",
                    threshold_value="False",
                )
            )

        return hitl_list
