"""
phase6_policy_gap.py
--------------------
Phase 6: Combine DiD and SHAP results into a policy gap table
and generate the final vulnerable group breakdown chart.

Outputs (saved to OUTPUT_DIR):
  - policy_gap_table.csv
  - policy_gap_table.xlsx
  - vulnerable_groups_chart.png

Usage:
    python phase6_policy_gap.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SHAP_CSV       = r"E:\PPOC\myenv\phase5_outputs\shap_summary_data.csv"
VULN_CSV       = r"E:\PPOC\myenv\phase5_outputs\vulnerable_groups.csv"
DESC_XLSX      = r"E:\PPOC\myenv\descriptive_tables.xlsx"
DID_CSV        = r"E:\PPOC\myenv\did_results.csv"
OUTPUT_DIR     = r"E:\PPOC\myenv\phase6_outputs"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
def out(f): return os.path.join(OUTPUT_DIR, f)

# ── Load inputs ───────────────────────────────────────────────────────────────
shap_df  = pd.read_csv(SHAP_CSV)
vuln_df  = pd.read_csv(VULN_CSV)
did_df   = pd.read_csv(DID_CSV)
wealth   = pd.read_excel(DESC_XLSX, sheet_name="Wealth Quintile")
caste    = pd.read_excel(DESC_XLSX, sheet_name="Caste")
state_df = pd.read_excel(DESC_XLSX, sheet_name="State Level")
urban_df = pd.read_excel(DESC_XLSX, sheet_name="Urban-Rural")

print("All inputs loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6A — POLICY GAP TABLE
# Combines: DiD result + SHAP rank + descriptive coverage + vulnerability score
# ══════════════════════════════════════════════════════════════════════════════

# ── DiD results for key sub-groups ────────────────────────────────────────────
# From did_results.csv (M3=overall, M4=poor, M5=rural)
did_lookup = {row["Model"]: row for _, row in did_df.iterrows()}

# ── Build policy gap table manually from all results ─────────────────────────
# Columns: Group | Dimension | NFHS4_cov | NFHS5_cov | Change_pp |
#          DiD_effect | SHAP_rank | pred_uninsured_pct | Risk_Level | Policy_Note

rows = [
    # ── Wealth groups ──────────────────────────────────────────────────────
    {
        "Group":               "Poorest (Q1)",
        "Dimension":           "Wealth",
        "NFHS4_coverage_%":    22.0,
        "NFHS5_coverage_%":    40.2,
        "Change_pp":           18.2,
        "DiD_effect":          "+2.02 pp (M4)",
        "SHAP_rank":           4,
        "Pred_uninsured_%":    62.7,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Largest absolute gain but still >60% uninsured; "
                               "BPL card is strongest predictor — enrollment drives access",
    },
    {
        "Group":               "Poorer (Q2)",
        "Dimension":           "Wealth",
        "NFHS4_coverage_%":    25.4,
        "NFHS5_coverage_%":    44.4,
        "Change_pp":           19.0,
        "DiD_effect":          "+2.02 pp (M4)",
        "SHAP_rank":           4,
        "Pred_uninsured_%":    60.1,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Strong gains; still majority uninsured — "
                               "gap between BPL eligibility and actual enrollment",
    },
    {
        "Group":               "Richest (Q5)",
        "Dimension":           "Wealth",
        "NFHS4_coverage_%":    27.5,
        "NFHS5_coverage_%":    41.2,
        "Change_pp":           13.7,
        "DiD_effect":          "N/A",
        "SHAP_rank":           4,
        "Pred_uninsured_%":    61.8,
        "Risk_Level":          "MEDIUM",
        "Policy_Note":         "High predicted uninsured despite wealth — "
                               "suggests avoidance of public schemes; "
                               "not a PM-JAY target group",
    },

    # ── Urban/Rural ────────────────────────────────────────────────────────
    {
        "Group":               "Rural",
        "Dimension":           "Urban/Rural",
        "NFHS4_coverage_%":    26.1,
        "NFHS5_coverage_%":    43.8,
        "Change_ppp":          17.7,
        "Change_pp":           17.7,
        "DiD_effect":          "-1.64 pp (M5)",
        "SHAP_rank":           5,
        "Pred_uninsured_%":    57.8,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Higher raw gains than urban but DiD negative — "
                               "control states (Odisha BSKY) grew faster; "
                               "PM-JAY rural outreach underperformed",
    },
    {
        "Group":               "Urban",
        "Dimension":           "Urban/Rural",
        "NFHS4_coverage_%":    25.8,
        "NFHS5_coverage_%":    40.0,
        "Change_pp":           14.2,
        "DiD_effect":          "N/A",
        "SHAP_rank":           5,
        "Pred_uninsured_%":    61.5,
        "Risk_Level":          "MEDIUM",
        "Policy_Note":         "Lower gains than rural; urban informally employed "
                               "workers lack ESIS/CGHS access and are PM-JAY eligible "
                               "but under-enrolled",
    },

    # ── Caste ──────────────────────────────────────────────────────────────
    {
        "Group":               "General (non-reserved)",
        "Dimension":           "Caste",
        "NFHS4_coverage_%":    20.4,
        "NFHS5_coverage_%":    35.6,
        "Change_pp":           15.2,
        "DiD_effect":          "N/A",
        "SHAP_rank":           8,
        "Pred_uninsured_%":    70.9,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Highest predicted uninsured rate of any caste group; "
                               "lowest baseline coverage; not targeted by caste-based "
                               "schemes and often above BPL threshold",
    },
    {
        "Group":               "SC (Scheduled Caste)",
        "Dimension":           "Caste",
        "NFHS4_coverage_%":    27.0,
        "NFHS5_coverage_%":    43.7,
        "Change_pp":           16.7,
        "DiD_effect":          "N/A",
        "SHAP_rank":           8,
        "Pred_uninsured_%":    59.5,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Gains above national average; still majority uninsured; "
                               "SHAP shows caste is a weaker predictor than wealth — "
                               "wealth channels most of the SC disadvantage",
    },
    {
        "Group":               "ST (Scheduled Tribe)",
        "Dimension":           "Caste",
        "NFHS4_coverage_%":    33.7,
        "NFHS5_coverage_%":    47.2,
        "Change_pp":           13.5,
        "DiD_effect":          "N/A",
        "SHAP_rank":           8,
        "Pred_uninsured_%":    46.7,
        "Risk_Level":          "MEDIUM",
        "Policy_Note":         "Highest baseline coverage of all caste groups "
                               "(state tribal schemes pre-PM-JAY); "
                               "lowest predicted uninsured — relatively better off",
    },
    {
        "Group":               "OBC",
        "Dimension":           "Caste",
        "NFHS4_coverage_%":    26.0,
        "NFHS5_coverage_%":    44.4,
        "Change_pp":           18.4,
        "DiD_effect":          "N/A",
        "SHAP_rank":           8,
        "Pred_uninsured_%":    57.8,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Largest caste-group gains; still majority uninsured; "
                               "large and heterogeneous group — sub-group targeting needed",
    },

    # ── Region ─────────────────────────────────────────────────────────────
    {
        "Group":               "East",
        "Dimension":           "Region",
        "NFHS4_coverage_%":    None,
        "NFHS5_coverage_%":    None,
        "Change_pp":           None,
        "DiD_effect":          "N/A",
        "SHAP_rank":           1,
        "Pred_uninsured_%":    69.9,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "Highest regional uninsured rate; Bihar, Jharkhand, "
                               "West Bengal — low BPL card penetration and "
                               "awareness barriers dominate",
    },
    {
        "Group":               "Northeast",
        "Dimension":           "Region",
        "NFHS4_coverage_%":    None,
        "NFHS5_coverage_%":    None,
        "Change_pp":           None,
        "DiD_effect":          "N/A",
        "SHAP_rank":           1,
        "Pred_uninsured_%":    63.5,
        "Risk_Level":          "HIGH",
        "Policy_Note":         "High uninsured rate despite relatively high ST "
                               "population (which has better coverage); "
                               "state-level heterogeneity is high",
    },
]

gap_df = pd.DataFrame(rows)

# Drop internal duplicate column if present
if "Change_ppp" in gap_df.columns:
    gap_df = gap_df.drop(columns=["Change_ppp"])

# Sort by predicted uninsured rate descending
gap_df = gap_df.sort_values("Pred_uninsured_%", ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("POLICY GAP TABLE")
print("="*70)
print(gap_df[["Group","Dimension","Change_pp","DiD_effect",
              "Pred_uninsured_%","Risk_Level"]].to_string(index=False))


# ── State-level gap analysis ──────────────────────────────────────────────────
print("\n" + "="*70)
print("STATE-LEVEL COVERAGE GAPS (Bottom 10 in NFHS-5)")
print("="*70)
state_bottom = (state_df[state_df["NFHS-5"].notna()]
                .nsmallest(10, "NFHS-5")
                [["state_name","treated","NFHS-4","NFHS-5","Change (pp)"]])
print(state_bottom.to_string(index=False))

print("\nTop 10 gainers:")
state_top = (state_df[state_df["Change (pp)"].notna()]
             .nlargest(10, "Change (pp)")
             [["state_name","treated","NFHS-4","NFHS-5","Change (pp)"]])
print(state_top.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6B — VULNERABLE GROUP BREAKDOWN CHART
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating vulnerable groups chart...")

# Use actual_insured_pct to show uninsured = 100 - insured
vuln_plot = vuln_df.copy()
vuln_plot["uninsured_pct"] = 100 - vuln_plot["actual_insured_pct"]
vuln_plot = vuln_plot.sort_values("uninsured_pct", ascending=True)

# Colour by dimension
dim_colors = {
    "wealth_cat":  "#E07B54",
    "urban_rural": "#5B8DB8",
    "caste_type":  "#6BAF6B",
    "region":      "#9B72B0",
}
colors = [dim_colors.get(d, "#888888") for d in vuln_plot["Dimension"]]

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(
    vuln_plot["Group"],
    vuln_plot["uninsured_pct"],
    color=colors, edgecolor="white", height=0.65
)

# Value labels
for bar, val in zip(bars, vuln_plot["uninsured_pct"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)

# Legend
legend_patches = [
    mpatches.Patch(color="#E07B54", label="Wealth Quintile"),
    mpatches.Patch(color="#5B8DB8", label="Urban/Rural"),
    mpatches.Patch(color="#6BAF6B", label="Caste"),
    mpatches.Patch(color="#9B72B0", label="Region"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
          framealpha=0.9)

ax.set_xlabel("Uninsured Households (%)", fontsize=11)
ax.set_title("Who Falls Behind? Uninsured Rate by Sub-Group\n"
             "NFHS-5 (2019–21), India", fontsize=13, pad=12)
ax.set_xlim(0, 85)
ax.axvline(x=57.1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(57.5, 0.4, "National avg\n(57.1%)", fontsize=8, color="black", alpha=0.7)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(out("vulnerable_groups_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: vulnerable_groups_chart.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6C — DiD + SHAP SYNTHESIS TABLE  (for paper)
# ══════════════════════════════════════════════════════════════════════════════

synthesis = pd.DataFrame([
    {
        "Group":          "All households",
        "DiD Result":     "+12.86 pp***",
        "SHAP Finding":   "Wave = #2 driver (0.50)",
        "Verdict":        "PM-JAY increased coverage significantly after state FE",
    },
    {
        "Group":          "Poor households (Q1-Q2)",
        "DiD Result":     "+2.02 pp***",
        "SHAP Finding":   "BPL card = #3 driver (0.33); wealth = #4 (0.17)",
        "Verdict":        "Targeted gains but weak — enrollment barriers persist",
    },
    {
        "Group":          "Rural households",
        "DiD Result":     "-1.64 pp***",
        "SHAP Finding":   "Urban flag = #5 driver (0.09)",
        "Verdict":        "Rural PM-JAY outreach underperformed vs control states",
    },
    {
        "Group":          "General caste",
        "DiD Result":     "Not tested",
        "SHAP Finding":   "Caste = #8 driver (0.05); highest pred. uninsured (70.9%)",
        "Verdict":        "Largest coverage gap — not targeted by any scheme",
    },
    {
        "Group":          "East region",
        "DiD Result":     "Captured in state FE",
        "SHAP Finding":   "State = #1 driver (0.72); East states bottom-ranked",
        "Verdict":        "Geographic inequality dominates; Bihar/Jharkhand critical",
    },
    {
        "Group":          "Northeast region",
        "DiD Result":     "Captured in state FE",
        "SHAP Finding":   "State = #1 driver (0.72); high within-region variance",
        "Verdict":        "High uninsured despite tribal schemes; state-specific gaps",
    },
])

print("\n" + "="*70)
print("DiD + SHAP SYNTHESIS (for paper Section 6)")
print("="*70)
print(synthesis.to_string(index=False))


# ── Save all outputs ──────────────────────────────────────────────────────────
gap_df.to_csv(out("policy_gap_table.csv"), index=False, encoding="utf-8")
synthesis.to_csv(out("did_shap_synthesis.csv"), index=False, encoding="utf-8")

with pd.ExcelWriter(out("policy_gap_table.xlsx"), engine="openpyxl") as writer:
    gap_df.to_excel(writer, sheet_name="Policy Gap Table", index=False)
    synthesis.to_excel(writer, sheet_name="DiD-SHAP Synthesis", index=False)
    state_bottom.to_excel(writer, sheet_name="Bottom 10 States", index=False)
    state_top.to_excel(writer, sheet_name="Top 10 Gainers", index=False)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nPhase 6 complete. Ready for Phase 7 (Final Visualizations).")
