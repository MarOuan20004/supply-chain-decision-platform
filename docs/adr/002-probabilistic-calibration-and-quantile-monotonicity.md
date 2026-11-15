# ADR-002: Probabilistic Calibration on Imbalanced Classes and Quantile Monotonicity Enforcement

## Status
Accepted

## Context
1. **Quantile Crossing in Gradient Boosted Oblivious Trees:**
   When training multi-quantile models using CatBoost with `MultiQuantile:alpha=0.1,0.5,0.9`, independent leaf values are learned for each quantile. In sparse feature regions, this can lead to quantile crossing where $\hat{q}_{10}(\mathbf{x}) > \hat{q}_{50}(\mathbf{x})$ or $\hat{q}_{50}(\mathbf{x}) > \hat{q}_{90}(\mathbf{x})$. In safety stock sizing, quantile crossing invalidates inventory policy math.
2. **Probability Distortion under Class Balancing:**
   To detect rare stockout criticality events ($C \in \{0, 1\}$ with prevalence between 8% and 25%), training with `auto_class_weights="Balanced"` optimizes discrimination (PR-AUC) but heavily distorts the output probabilities. Using raw probabilities directly to compute expected stockout penalties $\mathbb{E}[\text{Penalty}] = p_{\text{crit}} \times C_{\text{stockout}}$ leads to massive systematic over-ordering by the decision agent.

## Decision
1. **Chernozhukov Isotonic Projection for Quantiles:**
   We enforce strict non-crossing monotonicity via row-wise sorting and non-negative truncation:
   $$\tilde{q}_{10} = \max(0, \min(q_{10}, q_{50}, q_{90})), \quad \tilde{q}_{50} = \text{median}(q_{10}, q_{50}, q_{90}), \quad \tilde{q}_{90} = \max(q_{10}, q_{50}, q_{90})$$
   By the rearrangement theorem of Chernozhukov et al. (2010), this preserves or reduces Pinball loss while ensuring $0 \le \tilde{q}_{10} \le \tilde{q}_{50} \le \tilde{q}_{90}$ for 100% of predictions.
2. **Leak-Free Temporal Split Calibration:**
   Each training fold is split chronologically into $(\mathcal{T}_{\text{fit}}, \mathcal{T}_{\text{cal}})$ with an embargo purge gap of $H$ days. The base classifier is fit on $\mathcal{T}_{\text{fit}}$, and calibration models (Platt Sigmoid and Isotonic Regression) are fitted on $\mathcal{T}_{\text{cal}}$.
3. **Automated Calibration Selection:**
   We evaluate both Platt Sigmoid and Isotonic calibrators on out-of-sample temporal data using a composite score:
   $$\text{Score} = \text{Brier Score} + 0.5 \times \text{ECE}$$
   This automatically picks Isotonic regression when sample size and event counts are large enough, and falls back to robust 2-parameter Platt scaling when sample counts are limited.
4. **Model Registry Typed Signatures:**
   All 3 models (`supply_chain_catboost_forecaster`, `supply_chain_criticality_classifier`, `supply_chain_anomaly_detector`) are logged to MLflow Model Registry with explicit typed signatures inferencing inputs and outputs.

## Consequences
- **Positive:**
  - Guaranteed valid quantile bounds for all replenishment optimizations.
  - Well-calibrated probabilities yielding accurate expected monetary costs.
  - Zero leakage during calibration.
- **Negative:**
  - Slight reduction in training samples for the base tree boosting due to the $\mathcal{T}_{\text{cal}}$ holdout.
