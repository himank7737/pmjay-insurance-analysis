# PM-JAY's Impact on Health Insurance Coverage in India
### A Difference-in-Differences + XGBoost/SHAP Analysis using NFHS-4 & NFHS-5

---

## Research Question
> **Did PM-JAY increase health insurance coverage, and which households are still left behind?**

This project uses household-level data from India's National Family Health Survey (NFHS-4: 2015–16 and NFHS-5: 2019–21) to estimate the causal impact of the Pradhan Mantri Jan Arogya Yojana (PM-JAY) on health insurance coverage, and to identify the most vulnerable uninsured groups using machine learning.

---

## Key Results

**XGBoost Model:** AUC = 0.83 , Accuracy = 84.2% , F1 = 0.76

**Top SHAP drivers of coverage:** State (0.72) > Wave/Time (0.50) > BPL Card (0.33) > Wealth (0.17)

**Most at-risk groups (% uninsured):**
- East region: 73.6%
- General caste: 72.1%
- Poorest quintile: 68.3%
- Northeast region: 67.7%
- Urban households: 67.4%

---
## Setup

### 1. Clone the repo
```bash
git clone https://github.com/himank7737/pmjay-insurance-analysis.git
cd pmjay-insurance-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the data
NFHS data is publicly available from the DHS Program after free registration:
- https://dhsprogram.com/data/dataset/India_Standard-DHS_2015.cfm (NFHS-4)
- https://dhsprogram.com/data/dataset/India_Standard-DHS_2020.cfm (NFHS-5)

Download the **Household Recode** `.DTA` files for both rounds.
Place them anywhere and update the paths at the top of `scripts/phase1_build_combined.py`.

See `data/README_data.md` for detailed instructions.

### 4. Run the pipeline
```bash
python scripts/phase1_build_combined.py
python scripts/phase2_feature_engineering.py
python scripts/phase3_4_descriptive_did.py
python scripts/phase5_xgboost_shap.py
python scripts/phase6_policy_gap.py
python scripts/phase7_visualizations.py
```

---

## Methods

### Difference-in-Differences (DiD)
```
insured = β0 + β1(wave) + β2(treated × wave)
        + β3(controls) + state_FE + ε

β2 = DiD coefficient = causal effect of PM-JAY
```
- **Treated states:** PM-JAY adopters (32 states/UTs)
- **Control states:** Opt-outs with own schemes (Delhi, Odisha, Telangana, West Bengal)
- **Controls:** Urban/rural, wealth quintile, household size, SC/ST status
- **Standard errors:** HC3 heteroscedasticity-robust

### XGBoost + SHAP
- **Target:** `insured` (binary: 1 = any health insurance)
- **Features:** State, wave, BPL card, wealth, urban/rural, caste, religion, household size, PM-JAY state
- **Train/test split:** 80/20, stratified
- **Class imbalance:** Handled via `scale_pos_weight`
---

## Citation
If you use this code, please cite:
```
himank(2026). PM-JAY's Impact on Health Insurance Coverage in India:
A DiD and XGBoost/SHAP Analysis. GitHub Repository.
https://github.com/himank7737/pmjay-insurance-analysis
