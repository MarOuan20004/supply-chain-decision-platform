# SupplyChain-Decision-Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.5-ffcc00.svg?logo=catboost&logoColor=black)](https://catboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/Type%20Checked-Mypy-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-End Predictive MLOps & Governed Autonomous Decision Engine for Multi-Horizon Supply Chain Planning, Conformal Risk Calibration, and Deterministic Guardrails.**

**Author:** [Marwane Hajjy](https://github.com/MarOuan20004) (Grenoble INP - Ensimag & M2 AI UGA)  
**Domain:** Forward Deployed AI Engineering & Autonomous Decision Systems

---

## 🚀 Overview & Key Highlights

Supply chain planners face critical operational challenges when managing hundreds of stock-keeping units (SKUs):
1. **False Alarms & Sensor Corruption:** Triggering expensive replenishment orders on corrupted inventory readings (negative physical stock, zero unit price, supply shocks).
2. **Uncalibrated AI Risk:** Unbounded LLMs hallucinate numbers and auto-commit large capital budgets without formal financial bounds or human oversight.

**SupplyChain-Decision-Platform** resolves this by fusing **calibrated predictive ML** with a **bounded autonomous agent** governed by non-bypassable post-generation guardrails.

```text
                                SUPPLY CHAIN DECISION PLATFORM ARCHITECTURE

  +-------------------------------------------------------------------------------------------------+
  | RELATIONAL STORAGE & CAUSAL FEATURE ENGINEERING (data/supply_chain.db)                          |
  |                                                                                                 |
  |   • SQLite Relational Schema: skus, daily_demand, inventory_snapshots, supplier_deliveries      |
  |   • Causal Feature Pipeline: strictly Ft-measurable lags & rolling windows (Zero Future Leakage)|
  |   • ExpandingWindowTimeSplit: Purge gap H in {7, 14, 28} days (Zero Train/Val overlap)         |
  +-------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
  +-------------------------------------------------------------------------------------------------+
  | PREDICTIVE MLOPS ENGINE & MLFLOW MODEL REGISTRY                                                 |
  |                                                                                                 |
  |   • CatBoost MultiQuantile: simultaneous q10, q50, q90 forecasting with isotonic sorting        |
  |   • Probability Calibration: Platt Sigmoid & Isotonic regression (Brier / ECE minimization)     |
  |   • Anomaly Detection: Isolation Forest with TreeSHAP local attribution drivers                 |
  |   • MLflow Registry: automated promotion of @champion models with typed Pydantic signatures     |
  +-------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
  +-------------------------------------------------------------------------------------------------+
  | PRODUCTION REST API MICROSERVICE (FastAPI on CPU)                                              |
  |                                                                                                 |
  |   • Lifespan Preloading: in-memory model instantiation at startup (sub-5ms inference)           |
  |   • Non-Blocking CPU Offloading: starlette.concurrency.run_in_threadpool worker delegation      |
  |   • Deterministic Inventory Optimizer: pure NumPy vectorized replenishment simulator            |
  +-------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
  +-------------------------------------------------------------------------------------------------+
  | BOUNDED AUTONOMOUS DECISION AGENT (MAX_STEPS = 5, timeout = 15s)                                |
  |                                                                                                 |
  |   1. get_sku_forecast_and_uncertainty   --> CatBoost calibrated quantiles                     |
  |   2. inspect_anomaly_diagnostics         --> IsolationForest + SHAP attribution                 |
  |   3. query_inventory_and_supplier_sql    --> Read-only SQLite query (?mode=ro)                  |
  |   4. simulate_reorder_policy             --> Pure NumPy cost simulator (InventoryOptimizer)    |
  |   5. submit_decision_proposal            --> Formal proposal with tool evidence citations       |
  +-------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
  +-------------------------------------------------------------------------------------------------+
  | DETERMINISTIC POST-GENERATION GUARDRAILS (DeterministicGuardrailEngine)                         |
  |                                                                                                 |
  |   • Gate 1: Forced Abstention (ABSTAIN_INSUFFICIENT_DATA) if uncertainty > 0.80 or data corrupt |
  |   • Gate 2: Numeric Grounding Validator: zero unanchored numbers tolerated (tolerance 0.01 EUR) |
  |   • Gate 3: Mandatory HITL Escalation (PENDING_HUMAN_APPROVAL) if cost > 5,000 EUR or expedited  |
  +----------------------------------------+------------------------------------+-------------------+
                                           |                                    |
                                           v                                    v
                                   [APPROVED_AUTO]                  [PENDING_HUMAN_APPROVAL]
                                 (Automated execution)                 (SQL decisions table)
                                           |                                    |
                                           +-----------------+------------------+
                                                             |
                                                             v
                                            [logs/agent_traces.jsonl]
                                            (Append-only thread-safe audit log)
```

---

## 📊 Empirical Benchmark Results (40 Stratified Scenarios)

The platform includes an automated 3-way benchmark suite comparing operational paradigms on 40 realistic, stratified incident scenarios (`benchmarks/supply_chain_incidents_40.json`):

```bash
# Execute the full benchmark suite
python3 benchmarks/evaluate_agent_vs_baselines.py
```

### Overall Comparative Metrics

| Approach | Decision Accuracy | False Action Prevention (FAPR) | Numeric Hallucination (NHR) | Invalid Tool Call (ITCR) | Total Logistics Cost | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: (s, S) Rule** | 57.5% | 0.0% | N/A | N/A | 315,907.50 EUR | **0.03 ms** |
| **Baseline 2: Static ML (CatBoost)** | 70.0% | 0.0% | N/A | N/A | 298,410.82 EUR | 0.28 ms |
| **Governed Decision Agent** | **100.0%** | **100.0%** | **0.0%** | **0.0%** | **290,087.15 EUR** | 4.57 ms |
| **Delta (Agent vs Baseline 1)** | **+42.5%** | **+100.0%** | — | — | **-25,820.35 EUR** | +4.54 ms |
| **Delta (Agent vs Baseline 2)** | **+30.0%** | **+100.0%** | — | — | **-8,323.67 EUR** | +4.29 ms |

### Accuracy Breakdown by Incident Category

| Incident Category | Classical (s, S) Rule | Static CatBoost ML | Governed Decision Agent |
| :--- | :---: | :---: | :---: |
| **1. Foreseeable Imminent Stockouts** | 100.0% | 100.0% | **100.0%** |
| **2. Demand Shocks with MOQ & Lead-Time Delays** | 70.0% | 100.0% | **100.0%** |
| **3. False Alarms (Corrupted Sensor Readings)** | 60.0% | 40.0% | **100.0%** (Blocked) |
| **4. Extreme Uncertainty Scenarios ($(q_{90}-q_{10})/q_{50} > 0.80$)** | 0.0% | 40.0% | **100.0%** (Abstained) |

---

## 🛠️ Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/MarOuan20004/supply-chain-decision-platform.git
cd supply-chain-decision-platform

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,agent]"
```

### 2. Run Quality Checks & Test Suite

```bash
# Static linting and type checking
ruff check src/ scripts/ benchmarks/ tests/
ruff format --check src/ scripts/ benchmarks/ tests/
mypy src/

# Run complete automated test suite (51 tests)
pytest tests/ -v
```

### 3. Launch Interactive Incident Demo

```bash
# Run the dual contrasting case study demo
python3 scripts/demo_incident_response.py
```

### 4. Run API Microservice with Docker

```bash
docker compose up --build -d
curl -s http://localhost:8000/health | jq .
```

---

## 📁 Repository Structure

```text
supply-chain-decision-platform/
├── src/supply_chain/           # Core library
│   ├── config.py               # Centralized Pydantic v2 settings
│   ├── data_pipeline.py        # SQLite schema & causal feature engineering
│   ├── validation.py           # ExpandingWindowTimeSplit & cross-validation
│   ├── models.py               # MultiQuantile CatBoost, Calibration & IsolationForest
│   ├── optimizer.py            # Vectorized pure NumPy inventory optimizer
│   ├── mlflow_loader.py        # @champion model registry loader
│   ├── api.py                  # High-performance FastAPI ASGI microservice
│   ├── schemas.py              # Strict Pydantic contracts (extra='forbid')
│   └── agent/                  # Governed Agentic AI subsystem
│       ├── orchestrator.py     # Bounded state machine (MAX_STEPS=5, timeout=15s)
│       ├── tools.py            # 5 typed in-memory tool dispatchers
│       ├── guardrails.py       # Deterministic post-generation verification
│       ├── telemetry.py        # Append-only thread-safe JSONL logger
│       ├── decision_store.py   # SQL decisions table persistence & HITL state
│       └── llm_client.py       # Pluggable LLMClient (Mock, Ollama, OpenAI)
├── benchmarks/                 # Empirical evaluation harness
│   ├── supply_chain_incidents_40.json  # 40 stratified deterministic scenarios
│   ├── evaluate_agent_vs_baselines.py  # 3-way comparative evaluation harness
│   └── results/                        # Versioned benchmark reports & figures
├── scripts/                    # Operational automation scripts
│   ├── generate_incidents_40.py        # Generates deterministic scenarios & oracles
│   ├── demo_incident_response.py       # Interactive live demo (2 contrasting cases)
│   ├── train_and_register_mlflow.py    # Training & MLflow registry promotion
│   └── run_baselines_mlflow.py         # Baseline causal benchmarks
├── tests/                      # Automated test suite (51 passing unit & integration tests)
│   ├── test_no_leakage.py              # Zero temporal leakage mutation proofs
│   ├── test_calibration.py            # Monotonicity & probability calibration
│   ├── test_optimizer.py              # NumPy cost additive identity tests
│   ├── test_api.py                    # FastAPI endpoints & MLflow mocking
│   ├── test_agent_tools.py            # Typed tool contracts
│   ├── test_agent_orchestrator.py     # State machine loop bounds & timeouts
│   ├── test_agent_guardrails.py       # Adversarial guardrail tests
│   ├── test_hitl_api.py               # HITL endpoints & idempotency tests
│   └── test_benchmark.py              # 40-scenario benchmark & demo tests
├── docs/                       # Architecture Decision Records (ADRs)
│   ├── decisions.md            # ADR index and architectural roadmap
│   └── adr/                    # Individual ADR documents (001 to 009)
├── .github/workflows/          # Continuous Integration (GitHub Actions)
│   └── ci.yml                  # Lint, type-check, unit tests & smoke benchmark
├── docker-compose.yml          # Containerized deployment specification
├── Dockerfile                  # Multi-stage production container build
└── pyproject.toml              # Build system, dependencies, Ruff & Mypy config
```

---

## 🏛️ Architecture Decision Records (ADRs)

Key architectural decisions are documented under [`docs/adr/`](docs/decisions.md):
- **[ADR-001](docs/adr/001-temporal-data-leakage-prevention.md)**: Temporal Data Leakage Prevention in Causal Feature Pipelines
- **[ADR-002](docs/adr/002-probabilistic-calibration-and-quantile-monotonicity.md)**: Quantile Monotonicity Enforcement & Conformal Probability Calibration
- **[ADR-006](docs/adr/006-mlops-serving-and-deterministic-optimizer.md)**: High-Throughput Serving via Lifespan Preloading & Vectorized Inventory Optimization
- **[ADR-007](docs/adr/007-bounded-supply-chain-decision-agent.md)**: Bounded Multi-Step Autonomous Decision Agent Architecture
- **[ADR-008](docs/adr/008-deterministic-guardrails-hitl-telemetry.md)**: Deterministic Post-Generation Guardrails, HITL & Audit Telemetry
- **[ADR-009](docs/adr/009-deterministic-benchmark-and-ci-reproducibility.md)**: Deterministic 40-Scenario Benchmark Suite & CI Reproducibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
