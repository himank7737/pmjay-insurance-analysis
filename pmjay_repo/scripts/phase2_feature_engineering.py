"""
phase2_feature_engineering.py
------------------------------
Phase 2: Take nfhs_combined.parquet from Phase 1 and produce
nfhs_analysis.parquet — the clean, analysis-ready dataset for
DiD regression and XGBoost/SHAP.

Fixes three issues from Phase 1 output:
  1. pmjay NaN → 0  (NaN means "not asked because not insured")
  2. caste_type float codes → labelled string categories
  3. Drop 1,818 rows with missing state_name

Usage:
    python phase2_feature_engineering.py
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = r"C:\path\to\nfhs_combined.parquet"
OUTPUT_PATH = r"C:\path\to\nfhs_analysis.parquet"
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_parquet(INPUT_PATH)
print(f"Loaded: {df.shape}")


# ── Fix 1: pmjay NaN → 0 ──────────────────────────────────────────────────────
# NaN means the household was not insured, so RSBY/PM-JAY sub-question
# was never asked. Correctly treat as "not covered by PM-JAY".
df["pmjay"] = df["pmjay"].fillna(0).astype("Int8")
print(f"pmjay NaN after fix: {df['pmjay'].isna().sum()}")


# ── Fix 2: caste_type codes → labels ─────────────────────────────────────────
# DHS encoding (both NFHS-4 sh36 and NFHS-5 sh49):
#   1 = Scheduled Caste (SC)
#   2 = Scheduled Tribe (ST)
#   3 = Other Backward Class (OBC)
#   4 = None / General
#   8 = Don't know → NaN
CASTE_MAP = {1.0: "SC", 2.0: "ST", 3.0: "OBC", 4.0: "General", 8.0: pd.NA}
df["caste_type"] = df["caste_type"].map(CASTE_MAP)

# Consolidated SC/ST flag — useful for DiD sub-group analysis
df["sc_st"] = df["caste_type"].isin(["SC", "ST"]).astype("Int8")
df.loc[df["caste_type"].isna(), "sc_st"] = pd.NA

print(f"\ncaste_type distribution:")
print(df["caste_type"].value_counts(dropna=False))


# ── Fix 3: Drop rows with missing state_name ──────────────────────────────────
n_before = len(df)
df = df.dropna(subset=["state_name"]).copy()
print(f"\nDropped {n_before - len(df)} rows with missing state_name")
print(f"Remaining rows: {len(df):,}")


# ── Verify treatment assignment ───────────────────────────────────────────────
# Confirm no treated NaN remain after dropping state-less rows
assert df["treated"].isna().sum() == 0, "Still have NaN in treated!"
assert df["did_term"].isna().sum() == 0, "Still have NaN in did_term!"


# ── Religion labels ───────────────────────────────────────────────────────────
# DHS India religion codes (sh34 / sh47)
RELIGION_MAP = {
    1: "Hindu",
    2: "Muslim",
    3: "Christian",
    4: "Sikh",
    5: "Buddhist/Neo-Buddhist",
    6: "Jain",
    7: "Jewish",
    8: "Parsi/Zoroastrian",
    9: "No religion",
    96: "Other",
}
df["religion_label"] = df["religion"].map(RELIGION_MAP)


# ── XGBoost feature matrix ────────────────────────────────────────────────────
# Encode categorical variables as integers for XGBoost.
# We create a separate _enc version to keep original labels intact.

# State as integer (state_code already exists)
# Urban/rural: Urban=1, Rural=0
df["urban"] = (df["urban_rural"] == "Urban").astype(int)

# Caste as integer (SC=0, ST=1, OBC=2, General=3, NaN=-1 then impute)
CASTE_INT = {"SC": 0, "ST": 1, "OBC": 2, "General": 3}
df["caste_int"] = df["caste_type"].map(CASTE_INT)
# Impute missing caste with mode per state (most common category)
caste_mode = df.groupby("state_name")["caste_int"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else 3
)
df["caste_int"] = df.apply(
    lambda r: caste_mode[r["state_name"]] if pd.isna(r["caste_int"]) else r["caste_int"],
    axis=1
).astype(int)

# Religion as integer (already numeric in religion column)
df["religion_int"] = df["religion"].fillna(96).astype(int)


# ── Final column selection and ordering ───────────────────────────────────────

FINAL_COLS = [
    # identifiers
    "cluster", "hh_num", "wave",

    # outcome variables
    "insured",    # 1 = any health insurance, 0 = none  ← primary outcome
    "pmjay",      # 1 = covered by RSBY/PM-JAY specifically
    "bpl",        # 1 = has BPL card

    # DiD variables
    "treated",    # 1 = PM-JAY adopting state
    "did_term",   # treated × wave  ← DiD coefficient variable

    # geography
    "state_code", "state_name", "region",

    # demographics — labelled (for reporting)
    "urban_rural", "wealth_cat", "wealth_3cat",
    "caste_type", "religion_label", "sc_st",

    # demographics — numeric (for XGBoost)
    "urban", "wealth_quintile", "caste_int", "religion_int",
    "hh_size",

    # weight
    "hh_weight",
]

df = df[FINAL_COLS]


# ── Sanity check printout ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("PHASE 2 SANITY CHECK")
print("="*60)

print(f"\nFinal shape: {df.shape}")

print("\n--- Outcome variable: insured ---")
print(df.groupby(["wave", "treated"])["insured"].mean().round(3).rename("coverage_rate"))

print("\n--- DiD setup check ---")
did_table = df.groupby(["wave", "treated"])["insured"].mean().unstack()
did_table.columns = ["Control (treated=0)", "Treated (treated=1)"]
did_table.index = ["Pre (wave=0)", "Post (wave=1)"]
did_table["Diff (T-C)"] = did_table["Treated (treated=1)"] - did_table["Control (treated=0)"]
print(did_table.round(3))
pre_diff  = did_table.loc["Pre (wave=0)",  "Diff (T-C)"]
post_diff = did_table.loc["Post (wave=1)", "Diff (T-C)"]
print(f"\n  DiD estimate (raw): {post_diff - pre_diff:.3f}")
print(f"  Interpretation: PM-JAY raised coverage by ~{(post_diff-pre_diff)*100:.1f} pp")
print(f"  (before controlling for state FE and demographics)")

print("\n--- Missing values in key columns ---")
print(df[["insured","pmjay","bpl","treated","caste_type","religion_label"]].isna().sum())

print("\n--- caste_int distribution (for XGBoost) ---")
print(df["caste_int"].value_counts().sort_index())

print("\n--- Sample counts by state (treated vs control) ---")
sc = df.groupby(["state_name","treated"]).size().reset_index(name="n")
ctrl  = sc[sc["treated"]==0][["state_name","n"]].rename(columns={"n":"n_hh"})
treat = sc[sc["treated"]==1][["state_name","n"]].rename(columns={"n":"n_hh"})
print("Control states:")
print(ctrl.to_string(index=False))
print("\nTreated states (top 10 by size):")
print(treat.nlargest(10,"n_hh").to_string(index=False))


# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
print(f"\nSaved: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB")
print("\nPhase 2 complete. Ready for DiD regression and XGBoost.")
