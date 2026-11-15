# ADR-007: Bounded Multi-Step Autonomous Decision Agent Architecture

## Status
Accepted

## Author
Marwane Hajjy

## Components
`src/supply_chain/agent/`

## 1. Context
The agentic AI subsystem investigates SKU criticality alerts and anomaly events by synthesizing quantile demand forecasts, Isolation Forest diagnostics, relational inventory state, and cost simulations. An unbounded ReAct architecture poses severe operational risks, including infinite tool loops, unpredictable latency spikes, and unverified action execution.

## 2. Architectural Decisions
1. **Bounded State Machine in Pure Python:**
   The orchestrator enforces a hard step budget `MAX_STEPS = 5` and a strict timeout (`timeout_sec = 15.0s`).
2. **Direct In-Memory Typed Tool Dispatch:**
   The 5 operational tools (`get_sku_forecast_and_uncertainty`, `inspect_anomaly_diagnostics`, `query_inventory_and_supplier_sql`, `simulate_reorder_policy`, `submit_decision_proposal`) invoke internal Python services directly without network loopback. Tool arguments and responses are strictly validated via Pydantic v2 with `extra="forbid"`.
3. **Pluggable LLMClient Abstraction:**
   An abstract interface decouples orchestration logic from the model provider. Implementations include `MockLLMClient` (deterministic CI/testing), `OllamaClient` (local quantized models), and `OpenAIClient` (cloud APIs like Mistral/OpenAI).

## 3. Consequences
- Completely deterministic test execution with zero token costs in CI.
- Strong protection against tool call validation errors and runaway step loops.
- Seamless portability between local private LLM runtimes and enterprise cloud endpoints.
