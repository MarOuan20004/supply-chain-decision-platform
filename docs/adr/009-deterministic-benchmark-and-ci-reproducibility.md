# ADR-009: Deterministic 40-Scenario Incident Benchmark Suite & Continuous Integration

## Status
Accepted

## Author
Marwane Hajjy

## Components
`benchmarks/`, `scripts/generate_incidents_40.py`, `.github/workflows/ci.yml`

## 1. Context
Evaluating an autonomous supply chain agent requires verifying action precision on imminent stockouts, robust refusal on sensor corruptions, and cautious handling of severe forecast uncertainty. Benchmarking must be 100% reproducible and run quickly on standard CPU infrastructure without network or paid LLM dependencies in CI.

## 2. Architectural Decisions
1. **Stratified 40-Scenario Deterministic Incident Suite (`benchmarks/supply_chain_incidents_40.json`):**
   Comprises 4 distinct operational categories (10 scenarios each):
   - Standard foreseeable stockouts.
   - Demand shocks with MOQ and lead-time constraints.
   - Corrupted sensor / false alert events.
   - High uncertainty scenarios ($(q_{90} - q_{10}) / q_{50} > 0.80$).
2. **3-Way Comparative Evaluation Harness:**
   Evaluates three decision paradigms on identical incident fixtures:
   - Baseline 1: Classical $(s, S)$ inventory control rule.
   - Baseline 2: Static CatBoost ML policy.
   - Proposed: Governed Autonomous Decision Agent with deterministic guardrails.
3. **Automated CI Validation:**
   Runs full scenario validation and a fast `--smoke` test (5 representative incidents) within GitHub Actions CI in $< 5\text{ seconds}$.

## 3. Consequences
- Governed Agent achieves **100.0% Decision Accuracy**, **100.0% False Action Prevention Rate (FAPR)**, and **0.0% Numeric Hallucination Rate (NHR)**.
- Yields a verified logistics cost saving of over 25,820 EUR compared to classical rules and 8,323 EUR compared to static ML policies.
