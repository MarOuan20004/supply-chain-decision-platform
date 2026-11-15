# ADR-006: High-Throughput Model Serving via Lifespan Preloading and Vectorized Inventory Optimization

## Status
Accepted

## Author
Marwane Hajjy

## Components
`src/supply_chain/api.py`, `src/supply_chain/mlflow_loader.py`, `src/supply_chain/optimizer.py`

## 1. Context
The MLOps platform must serve multi-quantile forecasts, calibrated stockout risk probabilities, and anomaly diagnostics with sub-50 ms latency. Furthermore, it must provide the autonomous decision agent with an exact, deterministic replenishment policy optimizer that is 100% immune to arithmetic hallucinations or stochastic float variations.

## 2. Architectural Decisions
1. **MLflow `@champion` Preloading in FastAPI Lifespan:**
   All three production models (`forecaster`, `classifier`, `anomaly_detector`) are resolved from the MLflow Model Registry once at application startup during the async lifespan event. If `@champion` models are absent, the service starts in a graceful degraded state (returning HTTP 503 on model endpoints, while keeping health and database endpoints active).
2. **Asynchronous Non-Blocking Execution:**
   FastAPI endpoints delegate CPU-intensive CatBoost quantile and SHAP inferences to bounded background worker pools via Starlette's `run_in_threadpool`, ensuring the async event loop is never blocked.
3. **Pure NumPy Vectorized Optimizer (`InventoryOptimizer`):**
   Replenishment evaluation uses a closed-form, deterministic inventory cost simulator implemented in pure NumPy. Costs follow the additive identity:
   $$\text{Total Cost} = \text{Order Cost} + \text{Holding Cost} + \text{Stockout Penalty}$$
   This simulation runs in $< 0.1\text{ ms}$ and eliminates any reliance on LLM arithmetic.

## 3. Consequences
- Zero runtime model loading overhead on API requests.
- Consistent sub-5 ms latency for quantile and policy evaluation.
- Fully deterministic cost computation guaranteeing strict auditability.
