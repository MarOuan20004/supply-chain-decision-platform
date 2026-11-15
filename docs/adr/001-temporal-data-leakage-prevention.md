# ADR-001: Prevention of Temporal Data Leakage in Relational Time-Series & Cross-Validation

## Status
Accepted

## Context
In supply chain demand forecasting and stockout criticality estimation, time-series data exhibits strong autocorrelation, supplier lead-time stochasticity, and inventory replenishment dynamics. A common catastrophic defect in predictive ML platforms is **temporal data leakage**, occurring through:
1. **Unshifted or Centered Rolling Windows:** Computing rolling features at date $t$ that include observations from $t$ or $t+1 \dots t+k$ when making predictions for horizon $H$.
2. **Supplier Lead-Time Lookahead:** Calculating supplier delivery delays using orders where the arrival date $\text{actual\_delivery\_date} > t$ was not yet observable at date $t$.
3. **Horizon Overlap in Cross-Validation:** Performing train/validation splits where the multi-step target $Y_t^{(H)} = \frac{1}{H}\sum_{h=1}^H y_{t+h}$ of training observations overlaps with the validation window.
4. **Global Normalization Leakage:** Fitting scalers or encoders across the entire dataset before temporal splitting.

## Decision
1. **$\mathcal{F}_t$-Measurable Feature Pipeline:**
   - All time-series features are strictly computed from past or current end-of-day observations ($s \le t$).
   - Supplier delivery metrics (`causal_lead_time_mean`, `causal_lead_time_std`, `causal_fill_rate_mean`) are conditioned strictly on $\text{actual\_delivery\_date} \le \text{obs\_date}$ using binary search (`np.searchsorted(side='right')`).
2. **ExpandingWindowTimeSplit with Embargo Purge Gap:**
   - Train and validation splits are strictly ordered chronologically.
   - An embargo purge gap equal to the forecast horizon $H$ ($\text{purge\_gap\_days} = H$) separates the last training date $t_{\text{train\_end}}$ and the first validation date $t_{\text{val\_start}}$, guaranteeing $t_{\text{train\_end}} + H \le t_{\text{val\_start}}$.
3. **Automated Adversarial Mutation Proof:**
   - A dedicated unit test suite (`tests/test_no_leakage.py`) perturbs future demands, stocks, and deliveries for $s > t_{\text{cutoff}}$ by $+10^6$ and verifies with machine precision ($10^{-12}$) that past features $X_{s \le t_{\text{cutoff}}}$ remain strictly invariant.
4. **Isolated Preprocessing:**
   - Scaling (`StandardScaler`) is encapsulated within `scikit-learn.pipeline.Pipeline`, fit only on training folds.

## Consequences
- **Positive:**
  - Guaranteed zero lookahead bias, ensuring offline validation metrics directly match real-world production performance.
  - Provable correctness under synthetic adversarial perturbation.
  - Full auditability in MLflow with clean fold-by-fold logging.
- **Negative:**
  - Slight reduction in training samples due to the $H$-day embargo gap between folds.
