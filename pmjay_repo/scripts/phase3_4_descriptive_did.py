"""
phase3_4_descriptive_did.py
---------------------------
Phase 3: Descriptive statistics and coverage tables
Phase 4: Difference-in-Differences regression with state fixed effects

Outputs:
  - descriptive_tables.xlsx   (coverage summaries)
  - did_results.txt           (regression output)
  - did_results.csv           (coefficient table for plotting)

Usage:
    pip install pandas pyarrow statsmodels openpyxl
    python phase3_4_descriptive_did.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH   = r"C:\path\to\nfhs_analysis.parquet"
TABLE_OUTPUT = r"C:\path\to\descriptive_tables.xlsx"
DID_TXT      = r"C:\path\to\did_results.txt"
DID_CSV      = r"C:\path\to\did_results.csv"
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_parquet(INPUT_PATH)
df_ins = df.dropna(subset=["insured"]).copy()

# Convert nullable Int8 to plain float for statsmodels
for col in ["insured","pmjay","bpl","treated","did_term","wave",
            "urban","wealth_quintile","hh_size","sc_st"]:
    df_ins[col] = df_ins[col].astype(float)

print(f"Working dataset: {len(df_ins):,} rows")

# ── Helper: extract key rows from summary2, using actual column names ─────────
STAT_COL = None   # will be set after first model fit

def key_coefs(model, drop_state_fe=True):
    """Return summary2 table, optionally dropping state FE rows."""
    global STAT_COL
    t = model.summary2().tables[1]
    if STAT_COL is None:
        # Detect whether statsmodels used 't' or 'z'
        STAT_COL = "t" if "t" in t.columns else "z"
        PVAL_COL = "P>|t|" if STAT_COL == "t" else "P>|z|"
    else:
        PVAL_COL = "P>|t|" if STAT_COL == "t" else "P>|z|"
    cols = ["Coef.", "Std.Err.", STAT_COL, PVAL_COL, "[0.025", "0.975]"]
    cols = [c for c in cols if c in t.columns]
    if drop_state_fe:
        t = t.loc[[i for i in t.index if not i.startswith("C(state_code)")]]
    return t[cols].round(4)

def did_row(name, model):
    """Extract the did_term row for the summary table."""
    global STAT_COL
    t = model.summary2().tables[1]
    if STAT_COL is None:
        STAT_COL = "t" if "t" in t.columns else "z"
    row = t.loc["did_term"]
    return {
        "Model":        name,
        "DiD Coef (pp)": round(row["Coef."] * 100, 2),
        "SE":            round(row["Std.Err."] * 100, 2),
        "p-value":       round(row["P>|t|"] if "P>|t|" in t.columns else row["P>|z|"], 4),
        "95% CI":        f"[{row['[0.025']*100:.2f}, {row['0.975]']*100:.2f}]",
        "N":             int(model.nobs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PHASE 3: DESCRIPTIVE ANALYSIS")
print("="*60)

# Table 1: Overall coverage by wave
t1 = (df_ins.groupby("wave")["insured"]
      .agg(["mean","count"])
      .rename(columns={"mean":"coverage_rate","count":"n_households"})
      .reset_index())
t1["wave_label"]   = t1["wave"].map({0:"NFHS-4 (Pre PM-JAY)", 1:"NFHS-5 (Post PM-JAY)"})
t1["coverage_pct"] = (t1["coverage_rate"] * 100).round(1)
print("\nTable 1: Overall Insurance Coverage")
print(t1[["wave_label","n_households","coverage_pct"]].to_string(index=False))

# Table 2: Coverage by wave × urban/rural
t2 = (df_ins.groupby(["wave","urban_rural"])["insured"]
      .agg(["mean","count"]).reset_index()
      .rename(columns={"mean":"coverage_rate","count":"n"}))
t2["coverage_pct"] = (t2["coverage_rate"] * 100).round(1)
t2_pivot = t2.pivot(index="urban_rural", columns="wave", values="coverage_pct")
t2_pivot.columns = ["NFHS-4","NFHS-5"]
t2_pivot["Change (pp)"] = (t2_pivot["NFHS-5"] - t2_pivot["NFHS-4"]).round(1)
print("\nTable 2: Coverage by Urban/Rural")
print(t2_pivot.to_string())

# Table 3: Coverage by wave × wealth quintile
wealth_order = ["Poorest","Poorer","Middle","Richer","Richest"]
t3 = (df_ins.groupby(["wave","wealth_cat"])["insured"]
      .mean().reset_index().rename(columns={"insured":"coverage_rate"}))
t3["coverage_pct"] = (t3["coverage_rate"] * 100).round(1)
t3_pivot = t3.pivot(index="wealth_cat", columns="wave", values="coverage_pct").reindex(wealth_order)
t3_pivot.columns = ["NFHS-4","NFHS-5"]
t3_pivot["Change (pp)"] = (t3_pivot["NFHS-5"] - t3_pivot["NFHS-4"]).round(1)
print("\nTable 3: Coverage by Wealth Quintile")
print(t3_pivot.to_string())

# Table 4: Coverage by wave × caste
t4 = (df_ins.dropna(subset=["caste_type"])
      .groupby(["wave","caste_type"])["insured"]
      .mean().reset_index().rename(columns={"insured":"coverage_rate"}))
t4["coverage_pct"] = (t4["coverage_rate"] * 100).round(1)
t4_pivot = t4.pivot(index="caste_type", columns="wave", values="coverage_pct")
t4_pivot.columns = ["NFHS-4","NFHS-5"]
t4_pivot["Change (pp)"] = (t4_pivot["NFHS-5"] - t4_pivot["NFHS-4"]).round(1)
print("\nTable 4: Coverage by Caste")
print(t4_pivot.to_string())

# Table 5: State-level coverage
t5 = (df_ins.groupby(["state_name","treated","wave"])["insured"]
      .mean().reset_index().rename(columns={"insured":"coverage_rate"}))
t5_pivot = t5.pivot(index=["state_name","treated"], columns="wave", values="coverage_rate")
t5_pivot.columns = ["NFHS-4","NFHS-5"]
t5_pivot["Change (pp)"] = ((t5_pivot["NFHS-5"] - t5_pivot["NFHS-4"]) * 100).round(1)
t5_pivot = t5_pivot.reset_index()
t5_pivot[["NFHS-4","NFHS-5"]] = (t5_pivot[["NFHS-4","NFHS-5"]] * 100).round(1)
print("\nTable 5: State-level Coverage (sorted by change)")
print(t5_pivot.sort_values("Change (pp)", ascending=False).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — DiD REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("PHASE 4: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*60)

print("\n--- Parallel Trends Context ---")
pre  = df_ins[df_ins["wave"]==0]
post = df_ins[df_ins["wave"]==1]
pre_gap  = pre[pre["treated"]==1]["insured"].mean()  - pre[pre["treated"]==0]["insured"].mean()
post_gap = post[post["treated"]==1]["insured"].mean() - post[post["treated"]==0]["insured"].mean()
print(f"Pre-period  gap (T - C): {pre_gap:.4f}")
print(f"Post-period gap (T - C): {post_gap:.4f}")
print(f"Raw DiD: {post_gap - pre_gap:.4f}  ({(post_gap-pre_gap)*100:.2f} pp)")
print("NOTE: With 2 waves, parallel trends is a theoretical assumption,")
print("      not empirically testable. Justify via state selection logic.")

# Model 1: Simple DiD
print("\n--- Model 1: Simple DiD ---")
m1 = smf.ols("insured ~ treated + wave + did_term", data=df_ins).fit(cov_type="HC3")
print(key_coefs(m1, drop_state_fe=False))

# Model 2: DiD + controls
print("\n--- Model 2: DiD + Controls ---")
df2 = df_ins.dropna(subset=["sc_st"])
m2 = smf.ols(
    "insured ~ treated + wave + did_term + urban + wealth_quintile + hh_size + sc_st",
    data=df2
).fit(cov_type="HC3")
print(key_coefs(m2, drop_state_fe=False))

# Model 3: DiD + controls + State FE  (preferred specification)
print("\n--- Model 3: DiD + Controls + State FE  [PREFERRED] ---")
df3 = df_ins.dropna(subset=["sc_st"])
m3 = smf.ols(
    "insured ~ wave + did_term + urban + wealth_quintile + hh_size + sc_st + C(state_code)",
    data=df3
).fit(cov_type="HC3")
print(key_coefs(m3))   # state FE rows suppressed
print(f"\nR-squared:     {m3.rsquared:.4f}")
print(f"Adj R-squared: {m3.rsquared_adj:.4f}")
print(f"N:             {int(m3.nobs):,}")

# Model 4: Sub-group — Poor households (wealth quintile 1-2)
print("\n--- Model 4: Sub-group DiD — Poor Households ---")
df4 = df_ins[df_ins["wealth_quintile"] <= 2].dropna(subset=["sc_st"])
m4 = smf.ols(
    "insured ~ wave + did_term + urban + hh_size + sc_st + C(state_code)",
    data=df4
).fit(cov_type="HC3")
print(key_coefs(m4))
print(f"N (poor HH): {int(m4.nobs):,}")

# Model 5: Sub-group — Rural households
print("\n--- Model 5: Sub-group DiD — Rural Households ---")
df5 = df_ins[df_ins["urban"]==0].dropna(subset=["sc_st"])
m5 = smf.ols(
    "insured ~ wave + did_term + wealth_quintile + hh_size + sc_st + C(state_code)",
    data=df5
).fit(cov_type="HC3")
print(key_coefs(m5))
print(f"N (rural HH): {int(m5.nobs):,}")

# Summary coefficient table
print("\n" + "="*60)
print("DiD COEFFICIENT SUMMARY (did_term across all models)")
print("="*60)
models = {"M1 Simple": m1, "M2 +Controls": m2,
          "M3 +State FE": m3, "M4 Poor HH": m4, "M5 Rural HH": m5}
coef_rows = [did_row(name, model) for name, model in models.items()]
coef_df = pd.DataFrame(coef_rows)
print(coef_df.to_string(index=False))


# ── Save outputs ──────────────────────────────────────────────────────────────

with pd.ExcelWriter(TABLE_OUTPUT, engine="openpyxl") as writer:
    t1.to_excel(writer, sheet_name="Overall Coverage", index=False)
    t2_pivot.reset_index().to_excel(writer, sheet_name="Urban-Rural", index=False)
    t3_pivot.reset_index().to_excel(writer, sheet_name="Wealth Quintile", index=False)
    t4_pivot.reset_index().to_excel(writer, sheet_name="Caste", index=False)
    t5_pivot.to_excel(writer, sheet_name="State Level", index=False)
    coef_df.to_excel(writer, sheet_name="DiD Coefficients", index=False)

with open(DID_TXT, "w", encoding="utf-8") as f:
    f.write("MODEL 3 — PREFERRED SPECIFICATION\n")
    f.write("DiD + Controls + State Fixed Effects\n")
    f.write("="*60 + "\n")
    f.write(str(m3.summary()))
    f.write("\n\nDiD COEFFICIENT SUMMARY\n" + "="*60 + "\n")
    f.write(coef_df.to_string(index=False))

coef_df.to_csv(DID_CSV, index=False)

print(f"\nSaved: {TABLE_OUTPUT}")
print(f"Saved: {DID_TXT}")
print(f"Saved: {DID_CSV}")
print("\nPhase 3 + 4 complete.")
