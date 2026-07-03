"""
phase1_build_combined.py
------------------------
Phase 1: Load NFHS-4 and NFHS-5 household recode .dta files,
extract insurance and demographic variables, and combine into
one clean dataset saved as nfhs_combined.parquet.

Variable names confirmed from official DHS recode documentation.

Usage:
    pip install pandas pyarrow pyreadstat
    python phase1_build_combined.py

Edit the two paths at the top before running.
"""

import pandas as pd
import numpy as np
import os

# ── Edit these paths ──────────────────────────────────────────────────────────
NFHS4_PATH = r"C:\path\to\IAHR74DT\IAHR74FL.DTA"   # NFHS-4 HH recode
NFHS5_PATH = r"C:\path\to\IAHR7EDT\IAHR7EFL.DTA"   # NFHS-5 HH recode
OUTPUT_PATH = r"C:\path\to\nfhs_combined.parquet"   # output file
# ─────────────────────────────────────────────────────────────────────────────

# ── Variable lists ────────────────────────────────────────────────────────────

# NFHS-4 variables to load (lowercase as stored in DTA)
NFHS4_VARS = [
    # merge keys
    "hv001",    # cluster number
    "hv002",    # household number
    # insurance
    "sh54",     # any health scheme/insurance
    "sh55d",    # RSBY specifically
    "sh58",     # BPL card
    # demographics
    "hv024",    # state
    "hv025",    # urban/rural
    "hv270",    # wealth index (5 categories)
    "sh34",     # religion of household head
    "sh35",     # caste/tribe of household head
    "sh36",     # type of caste (SC/ST/OBC/none)
    "hv009",    # household size
    # weight
    "shv005",   # state household weight
]

# NFHS-5 variables to load
NFHS5_VARS = [
    # merge keys
    "hv001",
    "hv002",
    # insurance — renumbered in NFHS-5
    "sh71",     # any health scheme/insurance (= sh54 in NFHS-4)
    "sh72d",    # RSBY specifically (= sh55d in NFHS-4)
    "sh75",     # BPL card (= sh58 in NFHS-4)
    # demographics
    "hv024",
    "hv025",
    "hv270",
    "sh47",     # religion of household head (= sh34 in NFHS-4)
    "sh48",     # caste/tribe name (= sh35 in NFHS-4)
    "sh49",     # SC/ST/OBC/none classification (new in NFHS-5)
    "hv009",
    # weight
    "shweight",  # state household weight (renamed in NFHS-5)
]


# ── Helper: load one wave ─────────────────────────────────────────────────────

def load_wave(path: str, vars_to_load: list, wave: int) -> pd.DataFrame:
    """Load selected columns from a DTA file; add wave indicator."""
    print(f"\nLoading NFHS-{3 + wave} from {path} ...")
    print(f"  Loading {len(vars_to_load)} variables (full file, chunked read)")

    # Lower-case the variable list — DTA variables are case-insensitive
    vars_lower = [v.lower() for v in vars_to_load]

    df = pd.read_stata(
        path,
        columns=vars_lower,
        convert_categoricals=False,   # keep raw numeric codes
    )
    df.columns = [c.lower() for c in df.columns]  # normalise column case

    missing = [v for v in vars_lower if v not in df.columns]
    if missing:
        print(f"  WARNING: these variables not found and will be skipped: {missing}")
        vars_lower = [v for v in vars_lower if v in df.columns]
        df = df[vars_lower]

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    df["wave"] = wave
    return df


# ── Load both waves ───────────────────────────────────────────────────────────

df4 = load_wave(NFHS4_PATH, NFHS4_VARS, wave=0)
df5 = load_wave(NFHS5_PATH, NFHS5_VARS, wave=1)


# ── Harmonise variable names to a common schema ───────────────────────────────
# Rename NFHS-4 → canonical names

df4 = df4.rename(columns={
    "sh54":   "insured_raw",
    "sh55d":  "rsby_raw",
    "sh58":   "bpl_raw",
    "sh34":   "religion",
    "sh35":   "caste_name",
    "sh36":   "caste_type",   # SC/ST/OBC/none — available in NFHS-4
    "shv005": "hh_weight",
})

# Rename NFHS-5 → canonical names
df5 = df5.rename(columns={
    "sh71":    "insured_raw",
    "sh72d":   "rsby_raw",
    "sh75":    "bpl_raw",
    "sh47":    "religion",
    "sh48":    "caste_name",
    "sh49":    "caste_type",   # SC/ST/OBC/none — cleaner in NFHS-5
    "shweight": "hh_weight",
})

# Add missing columns so both DFs have identical schema before concat
for col in ["caste_type"]:
    if col not in df4.columns:
        df4[col] = np.nan
    if col not in df5.columns:
        df5[col] = np.nan


# ── Stack the two waves ───────────────────────────────────────────────────────

df = pd.concat([df4, df5], ignore_index=True, sort=False)
print(f"\nCombined shape: {df.shape}")  # expect ~1.2M rows


# ── Binary outcome variables ──────────────────────────────────────────────────
# DHS codes: 1 = yes, 0 = no, 8/9 = don't know / missing

def dhs_to_binary(series: pd.Series) -> pd.Series:
    """Map DHS yes/no codes to 1/0; everything else → NaN."""
    return series.map({1: 1, 0: 0}).astype("Int8")   # nullable integer

df["insured"] = dhs_to_binary(df["insured_raw"])
df["pmjay"]   = dhs_to_binary(df["rsby_raw"])     # RSBY = PM-JAY predecessor
df["bpl"]     = dhs_to_binary(df["bpl_raw"])

# Drop raw columns
df.drop(columns=["insured_raw", "rsby_raw", "bpl_raw"], inplace=True)


# ── Urban/rural label ─────────────────────────────────────────────────────────
# hv025: 1 = urban, 2 = rural
df["urban_rural"] = df["hv025"].map({1: "Urban", 2: "Rural"})


# ── Wealth quintile label ─────────────────────────────────────────────────────
# hv270: 1=Poorest 2=Poorer 3=Middle 4=Richer 5=Richest
wealth_map = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
df["wealth_cat"] = df["hv270"].map(wealth_map)

# Coarser 3-way grouping for some analyses
df["wealth_3cat"] = df["hv270"].map({
    1: "Poor", 2: "Poor",
    3: "Middle",
    4: "Rich", 5: "Rich"
})


# ── State name lookup (hv024 codes for India DHS) ─────────────────────────────
# Standard DHS state codes for India (same across NFHS-4 and NFHS-5)
STATE_CODES = {
    1: "Jammu & Kashmir", 2: "Himachal Pradesh", 3: "Punjab",
    4: "Chandigarh", 5: "Uttarakhand", 6: "Haryana",
    7: "Delhi", 8: "Rajasthan", 9: "Uttar Pradesh",
    10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh",
    13: "Nagaland", 14: "Manipur", 15: "Mizoram",
    16: "Tripura", 17: "Meghalaya", 18: "Assam",
    19: "West Bengal", 20: "Jharkhand", 21: "Odisha",
    22: "Chhattisgarh", 23: "Madhya Pradesh", 24: "Gujarat",
    25: "Dadra & Nagar Haveli", 26: "Daman & Diu", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa",
    31: "Lakshadweep", 32: "Kerala", 33: "Tamil Nadu",
    34: "Puducherry", 35: "Andaman & Nicobar", 36: "Telangana",
    # NOTE: verify these codes against your actual hv024 values;
    # DHS sometimes uses different ordering.
}
df["state_name"] = df["hv024"].map(STATE_CODES)


# ── Region grouping ───────────────────────────────────────────────────────────
REGION_MAP = {
    "North":     ["Jammu & Kashmir", "Himachal Pradesh", "Punjab", "Chandigarh",
                  "Uttarakhand", "Haryana", "Delhi", "Rajasthan"],
    "Central":   ["Uttar Pradesh", "Chhattisgarh", "Madhya Pradesh"],
    "East":      ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "Northeast": ["Sikkim", "Arunachal Pradesh", "Nagaland", "Manipur",
                  "Mizoram", "Tripura", "Meghalaya", "Assam"],
    "West":      ["Gujarat", "Dadra & Nagar Haveli", "Daman & Diu",
                  "Maharashtra", "Goa"],
    "South":     ["Andhra Pradesh", "Telangana", "Karnataka", "Kerala",
                  "Tamil Nadu", "Puducherry", "Lakshadweep",
                  "Andaman & Nicobar"],
}
state_to_region = {state: region
                   for region, states in REGION_MAP.items()
                   for state in states}
df["region"] = df["state_name"].map(state_to_region)


# ── PM-JAY treatment indicator ────────────────────────────────────────────────
# PM-JAY launched September 2018.
# States that adopted PM-JAY before NFHS-5 fieldwork (2019-21):
# Most states adopted; a few opted out or had delays.
# Key non-adopters: Delhi, Telangana, Odisha, West Bengal (had own schemes).
# NOTE: Refine this list based on your literature review.
CONTROL_STATES = {
    "Delhi",        # opted out — runs its own scheme
    "Telangana",    # opted out initially
    "Odisha",       # launched own BSKY scheme instead
    "West Bengal",  # opted out
}
df["treated"] = (~df["state_name"].isin(CONTROL_STATES)).astype("Int8")
df.loc[df["state_name"].isna(), "treated"] = pd.NA

# DiD interaction term
df["did_term"] = (df["treated"] * df["wave"]).astype("Int8")


# ── Final clean-up ────────────────────────────────────────────────────────────

# Rename standard DHS columns to readable names
df = df.rename(columns={
    "hv001": "cluster",
    "hv002": "hh_num",
    "hv024": "state_code",
    "hv025": "urban_rural_code",
    "hv270": "wealth_quintile",
    "hv009": "hh_size",
})

# Column order
COL_ORDER = [
    "cluster", "hh_num", "wave",
    # outcomes
    "insured", "pmjay", "bpl",
    # treatment / DiD
    "treated", "did_term",
    # demographics
    "state_code", "state_name", "region",
    "urban_rural_code", "urban_rural",
    "wealth_quintile", "wealth_cat", "wealth_3cat",
    "religion", "caste_name", "caste_type",
    "hh_size",
    # weight
    "hh_weight",
]
COL_ORDER = [c for c in COL_ORDER if c in df.columns]
extra = [c for c in df.columns if c not in COL_ORDER]
df = df[COL_ORDER + extra]

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nInsurance coverage by wave:")
print(df.groupby("wave")["insured"].value_counts(normalize=True).round(3))

print(f"\nMissing values in key columns:")
key_cols = ["insured", "pmjay", "bpl", "treated", "state_name",
            "urban_rural", "wealth_cat", "caste_type"]
print(df[key_cols].isna().sum())


# ── Save ──────────────────────────────────────────────────────────────────────

df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
print(f"\nSaved → {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")
print("\nPhase 1 complete.")
