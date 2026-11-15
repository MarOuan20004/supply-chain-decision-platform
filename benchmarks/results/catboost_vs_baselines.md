# SupplyChain-Decision-Platform - Rapport Experimental CatBoost MLOps
## Comparaison Baselines  vs CatBoost MultiQuantile & Calibration 

### 1. Synthese Multi-Horizon sur 4 Plis Temporels

| Horizon | Modele | MASE (Mean +/- Std) | WAPE | Pinball Moy. | PR-AUC (Mean +/- Std) | Brier Score | ECE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **7j** | `baseline_seasonal_naive_lag7` | **5.5790** +/- 0.5982 | 0.4570 | 5.7520 | **0.8517** +/- 0.0232 | **0.0760** | **0.0780** |
| **7j** | `baseline_moving_average_14d` | **2.2875** +/- 0.3496 | 0.1865 | 3.0761 | **0.8515** +/- 0.0232 | **0.0761** | **0.0782** |
| **7j** | `catboost_multiquantile_calibrated_isotonic` | **1.9943** +/- 0.2808 | 0.1628 | 2.0594 | **0.8190** +/- 0.0200 | **0.0650** | **0.0325** |
| **14j** | `baseline_seasonal_naive_lag7` | **10.6333** +/- 1.1180 | 0.4405 | 5.4697 | **0.9163** +/- 0.0064 | **0.0522** | **0.0517** |
| **14j** | `baseline_moving_average_14d` | **3.6676** +/- 0.5523 | 0.1512 | 2.7747 | **0.9163** +/- 0.0065 | **0.0522** | **0.0520** |
| **14j** | `catboost_multiquantile_calibrated_isotonic` | **2.9646** +/- 0.4446 | 0.1223 | 1.5605 | **0.8839** +/- 0.0103 | **0.0491** | **0.0292** |
| **28j** | `baseline_seasonal_naive_lag7` | **19.9680** +/- 1.9270 | 0.4293 | 5.1851 | **0.9543** +/- 0.0067 | **0.0353** | **0.0342** |
| **28j** | `baseline_moving_average_14d` | **6.2017** +/- 0.9545 | 0.1326 | 2.5616 | **0.9541** +/- 0.0066 | **0.0356** | **0.0352** |
| **28j** | `catboost_multiquantile_calibrated_isotonic` | **4.7335** +/- 0.7051 | 0.1026 | 1.2748 | **0.9190** +/- 0.0090 | **0.0425** | **0.0408** |

### 2. Impact de la Calibration Probabiliste Hors-Echantillon (H = 7 jours)

| Methode | PR-AUC | Brier Score | ECE (10 bacs) | Prob. Moyenne Predite | Taux Empirique Reel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `uncalibrated` | 0.8165 | **0.1056** | **0.0757** | 0.3171 | 0.2414 |
| `sigmoid` | 0.8165 | **0.0979** | **0.0522** | 0.2287 | 0.2414 |
| `isotonic` | 0.7942 | **0.0945** | **0.0384** | 0.2339 | 0.2414 |

### 3. Importance Globale et Locale TreeSHAP (H = 7 jours, Top 8 Variables)

| Rang | Variable Causale | Importance Moyenne Abs(SHAP) |
| :---: | :--- | :---: |
| 1 | `demand_roll_mean_28` | 8.1223 |
| 2 | `doy_sin` | 2.0507 |
| 3 | `days_of_supply_est` | 1.4247 |
| 4 | `causal_lead_time_mean` | 1.3788 |
| 5 | `demand_roll_mean_14` | 1.3075 |
| 6 | `on_hand_qty` | 1.2417 |
| 7 | `stock_to_reorder_ratio` | 0.7523 |
| 8 | `doy_cos` | 0.7363 |

### 4. Diagnostic du Detecteur d'Anomalies IsolationForest

- **Observations analysees** : `200`
- **Anomalies detectees** : `6` (`3.00%`)
- **Modeles enregistres dans MLflow Model Registry** : `supply_chain_catboost_forecaster`, `supply_chain_criticality_classifier`, `supply_chain_anomaly_detector`
