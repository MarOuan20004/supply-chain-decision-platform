# Benchmark Agent Supply Chain vs Baselines (20261115)

- **Scénarios évalués** : `40` (mode LLM : `mock`)
- **Horodatage UTC** : `2026-11-15T03:21:26.721825+00:00`

## 1. Synthèse Globale

| Approche | Decision Accuracy | False Action Prevention | Numeric Hallucination | Invalid Tool Rate | Coût Logistique Total (EUR) | Latence Moy. (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 : Règle (s, S) | 0.0% | 0.0% | 0.0% | N/A | 312201.87 | 0.01 |
| Baseline 2 : ML Statique (CatBoost) | 25.0% | 0.0% | 0.0% | N/A | 228742.41 | 0.01 |
| **Agent Gouverné** | **100.0%** | **100.0%** | **0.0%** | **0.0%** | **290087.15** | **1.88** |

## 2. Decision Accuracy par Catégorie d'Incident

| Catégorie | Règle (s, S) | ML Statique | Agent Gouverné |
| :--- | :---: | :---: | :---: |
| 1. Ruptures standards prévisibles | 0.0% | 100.0% | **100.0%** |
| 2. Chocs de demande (MOQ / délai) | 0.0% | 0.0% | **100.0%** |
| 3. Fausses alertes (capteurs corrompus) | 0.0% | 0.0% | **100.0%** |
| 4. Très haute incertitude (> 0.80) | 0.0% | 0.0% | **100.0%** |
