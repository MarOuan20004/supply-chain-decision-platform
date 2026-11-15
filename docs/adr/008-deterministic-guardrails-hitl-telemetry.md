# ADR-008: Deterministic Post-Generation Guardrails, Human-in-the-Loop & Audit Telemetry

## Status
Accepted

## Author
Marwane Hajjy

## Components
`src/supply_chain/agent/guardrails.py`, `src/supply_chain/agent/telemetry.py`, `src/supply_chain/agent/decision_store.py`

## 1. Context
The decision agent queries ML models and the SQL database to recommend inventory replenishment orders. Delegating self-evaluation of uncertainty, numerical sanity, or financial limits to a generative LLM exposes the platform to arithmetic hallucinations, prompt injection, and unauthorized capital expenditure.

## 2. Architectural Decisions
1. **Non-Bypassable Post-Generation Verification (`guardrails.py`):**
   A deterministic Python engine evaluates the LLM's draft `DecisionReport` directly against raw outputs (`raw_output`) from all tools executed during the session.
2. **Strict Abstention Policy (`ABSTAIN_INSUFFICIENT_DATA`):**
   Abstention is unconditionally forced when:
   - Relative forecast uncertainty $(q_{90} - q_{10}) / q_{50} > 0.80$.
   - Sensor or ERP corruption is detected ($\text{on\_hand} < 0$, $\text{unit\_cost} \le 0$, $\sigma_{\text{lead}} > \mu_{\text{lead}}$).
   - Any numerical value in the reasoning text is unanchored in tool outputs (tolerance: $10^{-2}$ EUR).
3. **Mandatory Human-in-the-Loop Escalation (`PENDING_HUMAN_APPROVAL`):**
   Any purchase order exceeding 5,000 EUR or requesting expedited transport is escalated to human review, persisted in the SQL `decisions` table, and managed through idempotent REST endpoints (`409 Conflict` on duplicate reviews).
4. **Append-Only JSONL Audit Telemetry (`logs/agent_traces.jsonl`):**
   All execution steps, tool records, token usage, and operator decisions are logged atomically using a thread-safe mutex lock.

## 3. Consequences
- Zero false auto-approvals on corrupted sensor inputs or volatile demand forecasts.
- Total traceability and auditability for financial operations.
