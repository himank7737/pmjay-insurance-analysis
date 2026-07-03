# Project Plan: PM-JAY's Impact on Health Insurance Coverage in India

## Core Question
Did PM-JAY increase health insurance coverage, and which households are still left behind?

---

## Phase 1 — Data Preparation (`phase1_build_combined.py`)
Load household recodes from NFHS-4 and NFHS-5, extract insurance and demographic
variables, and combine into one clean dataset.

**Output:** `nfhs_combined.parquet` — 1,238,208 rows, one per household, both waves

---

## Phase 2 — Feature Engineering (`phase2_feature_engineering.py`)
Create analysis-ready variables for DiD and ML.

Key variables created:
- `insured` = 1 if covered by any health scheme
- `pmjay` = 1 if covered by RSBY/PM-JAY specifically
- `bpl` = 1 if household has BPL card
- `treated` = 1 if state adopted PM-JAY
- `did_term` = treated × wave (DiD interaction)
- `wealth_cat`, `urban_rural`, `caste_type`, `region`

**Output:** `nfhs_analysis.parquet` — 1,236,390 rows, 21 columns

---

## Phase 3 — Descriptive Analysis (`phase3_4_descriptive_did.py`)
Coverage rates by wave, state, wealth, caste, urban/rural.

**Key finding:** National coverage rose from 26.0% (NFHS-4) to 42.8% (NFHS-5) — +16.8 pp

**Output:** `descriptive_tables.xlsx` (6 sheets)

---

## Phase 4 — DiD Econometrics (`phase3_4_descriptive_did.py`)
Difference-in-Differences with state fixed effects.

```
insured = β0 + β1(wave) + β2(treated×wave) + β3(controls) + state_FE + ε
```

| Model | DiD Coef | N |
|---|---|---|
| M1 Simple DiD | −1.26 pp | 1,229,590 |
| M2 + Controls | −2.30 pp | 1,170,914 |
| M3 + State FE (preferred) | +12.86 pp*** | 1,170,914 |
| M4 Poor HH sub-group | +2.02 pp*** | 521,877 |
| M5 Rural HH sub-group | −1.64 pp*** | 854,912 |

**Output:** `did_results.txt`, `did_results.csv`

---

## Phase 5 — XGBoost + SHAP (`phase5_xgboost_shap.py`)
Predict uninsured households and identify key drivers.

**Model performance:** AUC = 0.81, Accuracy = 74.2%, F1 = 0.66

**SHAP ranking:**
1. State (0.724) — geography dominates
2. Survey Wave (0.500) — PM-JAY era effect
3. BPL Card (0.333) — gateway to coverage
4. Wealth Quintile (0.166)
5. Urban/Rural (0.091)
6. Religion (0.076)
7. Household Size (0.073)
8. Caste (0.053)
9. PM-JAY State (0.030)

**Output:** SHAP plots, `vulnerable_groups.csv`, `xgb_metrics.txt`

---

## Phase 6 — Policy Gap Analysis (`phase6_policy_gap.py`)
Combine DiD and SHAP to identify who falls behind.

**Most at-risk groups (% uninsured, NFHS-5):**
- East region: 73.6%
- General caste: 72.1%
- Poorest quintile: 68.3%
- Northeast region: 67.7%
- Urban households: 67.4%

**Output:** `policy_gap_table.xlsx`, `vulnerable_groups_chart.png`

---

## Phase 7 — Visualizations (`phase7_visualizations.py`)
Final report-quality charts:
1. Coverage rate NFHS-4 vs NFHS-5 (bar chart)
2. DiD coefficient plot across models
3. Parallel trends visualization
4. State-level coverage ranking chart
5. SHAP beeswarm (from Phase 5)
6. SHAP dependence plots (from Phase 5)
7. Vulnerable group breakdown (from Phase 6)
