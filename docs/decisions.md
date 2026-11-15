# Architecture Decision Records (ADRs)

This document indexes all foundational architectural decisions for **SupplyChain-Decision-Platform**.

Each record follows the standard ADR structure (Context, Decision, Consequences, Alternatives).

---

### [ADR-001: Temporal Data Leakage Prevention](adr/001-temporal-data-leakage-prevention.md)
- **Status:** Accepted
- **Summary:** Strict $\mathcal{F}_t$-measurable causal feature engineering, `ExpandingWindowTimeSplit` cross-validation with a purge gap $H$, and synthetic future mutation unit testing to guarantee zero lookahead bias.

### [ADR-002: Probabilistic Calibration and Quantile Monotonicity](adr/002-probabilistic-calibration-and-quantile-monotonicity.md)
- **Status:** Accepted
- **Summary:** Multi-horizon CatBoost quantile regression with post-inference isotonic sorting ($q_{10} \le q_{50} \le q_{90}$), Platt/Isotonic conformal probability calibration for stockout risk, and Isolation Forest for anomaly identification.

### [ADR-006: High-Throughput Serving & Vectorized Inventory Optimization](adr/006-mlops-serving-and-deterministic-optimizer.md)
- **Status:** Accepted
- **Summary:** Asynchronous FastAPI microservice with `@champion` model preloading during application lifespan, offloading CPU-bound tasks via Starlette's `run_in_threadpool`, and pure NumPy replenishment optimization (`InventoryOptimizer`).

### [ADR-007: Bounded Multi-Step Autonomous Decision Agent](adr/007-bounded-supply-chain-decision-agent.md)
- **Status:** Accepted
- **Summary:** Bounded state machine architecture (`MAX_STEPS = 5`, `timeout = 15.0s`), typed Pydantic v2 tool contracts without HTTP loopback overhead, and provider-agnostic `LLMClient` abstraction supporting mock, local Ollama, and OpenAI/Mistral endpoints.

### [ADR-008: Deterministic Guardrails, Human-in-the-Loop & Audit Telemetry](adr/008-deterministic-guardrails-hitl-telemetry.md)
- **Status:** Accepted
- **Summary:** Non-bypassable post-generation verification (`DeterministicGuardrailEngine`) with forced abstention on high uncertainty ($> 0.80$) or corruptions, numeric grounding verification (tolerance $10^{-2}$ EUR), mandatory HITL escalation ($> 5\,000$ EUR or expedited freight), and thread-safe append-only JSONL telemetry.

### [ADR-009: Deterministic Benchmark Suite & CI Reproducibility](adr/009-deterministic-benchmark-and-ci-reproducibility.md)
- **Status:** Accepted
- **Summary:** 40-scenario stratified empirical evaluation suite (`benchmarks/supply_chain_incidents_40.json`), 3-way evaluation harness comparing classical $(s, S)$ rules, static CatBoost ML, and Governed Decision Agent, with automated smoke testing in GitHub Actions CI.
