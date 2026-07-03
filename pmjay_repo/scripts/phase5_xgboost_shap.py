"""
phase5_xgboost_shap.py
-----------------------
Phase 5: Train XGBoost classifier to predict uninsured households,
then use SHAP to identify which features drive insurance coverage gaps.

Outputs (all in OUTPUT_DIR):
  - xgb_model.json              XGBoost model
  - xgb_metrics.txt             AUC, accuracy, F1, classification report
  - shap_values.npy             Raw SHAP values array
  - shap_summary_data.csv       Mean |SHAP| per feature (for beeswarm recreation)
  - vulnerable_groups.csv       Top uninsured sub-groups
  - feature_importance.csv      XGBoost native feature importance

Usage:
    pip install xgboost shap scikit-learn pandas pyarrow matplotlib
    python phase5_xgboost_shap.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on Windows
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, classification_report)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = r"E:\PPOC\myenv\nfhs_analysis.parquet"
OUTPUT_DIR = r"E:\PPOC\myenv\phase5_outputs"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_parquet(INPUT_PATH)
df = df.dropna(subset=["insured"]).copy()

# Convert nullable integers to plain numpy types
for col in ["insured","pmjay","bpl","treated","did_term","wave",
            "urban","wealth_quintile","hh_size","sc_st","caste_int"]:
    df[col] = df[col].astype(float)

# Feature set for XGBoost
FEATURES = [
    "wave",            # temporal (pre/post PM-JAY)
    "state_code",      # state identifier
    "urban",           # 1=urban, 0=rural
    "wealth_quintile", # 1–5
    "bpl",             # BPL card holder
    "caste_int",       # 0=SC, 1=ST, 2=OBC, 3=General
    "religion_int",    # numeric religion code
    "hh_size",         # household size
    "treated",         # PM-JAY adopting state
]
TARGET = "insured"

# Feature labels for SHAP plots (human-readable)
FEATURE_LABELS = {
    "wave":            "Survey Wave (Pre/Post PM-JAY)",
    "state_code":      "State",
    "urban":           "Urban (vs Rural)",
    "wealth_quintile": "Wealth Quintile",
    "bpl":             "BPL Card",
    "caste_int":       "Caste (SC/ST/OBC/Gen)",
    "religion_int":    "Religion",
    "hh_size":         "Household Size",
    "treated":         "PM-JAY State",
}

X = df[FEATURES].fillna(-1).astype(float)
y = df[TARGET].astype(int)

print(f"Dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
print(f"Class balance: {y.mean():.3f} insured, {1-y.mean():.3f} uninsured")


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST TRAINING
# ══════════════════════════════════════════════════════════════════════════════

print("\nTraining XGBoost...")

model = xgb.XGBClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 10,
    scale_pos_weight  = (y == 0).sum() / (y == 1).sum(),  # handle class imbalance
    eval_metric       = "auc",
    early_stopping_rounds = 30,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print(f"Best iteration: {model.best_iteration}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred      = model.predict(X_test)

auc      = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)
report   = classification_report(y_test, y_pred,
                                  target_names=["Uninsured","Insured"])

print(f"\n--- Evaluation Metrics ---")
print(f"AUC:      {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1:       {f1:.4f}")
print(report)

metrics_text = (
    f"XGBoost Evaluation\n{'='*40}\n"
    f"AUC:      {auc:.4f}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"F1:       {f1:.4f}\n\n"
    f"Classification Report:\n{report}\n"
    f"Best iteration: {model.best_iteration}\n"
)
with open(out("xgb_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(metrics_text)


# ══════════════════════════════════════════════════════════════════════════════
# SHAP VALUES
# ══════════════════════════════════════════════════════════════════════════════

print("\nCalculating SHAP values (this may take 1-2 minutes)...")

# Use a sample for SHAP to keep memory manageable (~50k rows is plenty)
SHAP_SAMPLE = 50_000
idx = np.random.default_rng(42).choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)),
                                        replace=False)
X_shap = X_test.iloc[idx].copy()

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)   # shape: (n_samples, n_features)

print(f"SHAP values shape: {shap_values.shape}")

# Save raw SHAP values
np.save(out("shap_values.npy"), shap_values)

# Mean absolute SHAP per feature
mean_shap = pd.DataFrame({
    "feature":       FEATURES,
    "feature_label": [FEATURE_LABELS[f] for f in FEATURES],
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

mean_shap.to_csv(out("shap_summary_data.csv"), index=False)

print("\n--- SHAP Feature Importance (mean |SHAP|) ---")
print(mean_shap.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SHAP PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# Rename columns for readable plot labels
X_shap_labeled = X_shap.rename(columns=FEATURE_LABELS)
shap_labeled   = shap_values.copy()   # values same, labels applied via X_shap_labeled

# Plot 1: Beeswarm summary plot
print("\nGenerating SHAP beeswarm plot...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_shap_labeled,
    plot_type="dot",
    max_display=9,
    show=False,
)
plt.title("SHAP Summary — Drivers of Health Insurance Coverage\n(NFHS-4 + NFHS-5, India)",
          fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(out("shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_beeswarm.png")

# Plot 2: Bar chart of mean |SHAP|
print("Generating SHAP bar chart...")
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(
    mean_shap["feature_label"][::-1],
    mean_shap["mean_abs_shap"][::-1],
    color="#4C72B0", edgecolor="white", height=0.6
)
ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=11)
ax.set_title("Feature Importance — XGBoost + SHAP\nPredicting Health Insurance Coverage",
             fontsize=12)
ax.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, mean_shap["mean_abs_shap"][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(out("shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_bar.png")

# Plot 3: SHAP dependence plots for top 3 features
top3 = mean_shap["feature"].head(3).tolist()
for feat in top3:
    label = FEATURE_LABELS[feat]
    print(f"Generating SHAP dependence plot: {feat}...")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat, shap_values, X_shap,
        feature_names=[FEATURE_LABELS[f] for f in FEATURES],
        ax=ax, show=False,
    )
    ax.set_title(f"SHAP Dependence: {label}", fontsize=12)
    plt.tight_layout()
    safe_name = feat.replace("/", "_").replace(" ", "_")
    plt.savefig(out(f"shap_dep_{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_dep_{safe_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# VULNERABLE GROUP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nIdentifying vulnerable (uninsured) groups...")

# Use the full test set for group analysis
df_test = df.iloc[X_test.index].copy()
df_test["pred_prob_insured"] = y_pred_prob
df_test["pred_uninsured"]    = (y_pred_prob < 0.5).astype(int)

def group_stats(df, group_col, label_col=None):
    """Coverage rate and predicted uninsured rate by group."""
    lc = label_col if label_col else group_col
    stats = df.groupby(lc).agg(
        n            = ("insured", "count"),
        actual_insured_pct   = ("insured",  lambda x: x.mean() * 100),
        pred_uninsured_pct   = ("pred_uninsured", lambda x: x.mean() * 100),
    ).round(1).reset_index()
    stats = stats.rename(columns={lc: "Group"})
    stats["Dimension"] = group_col
    return stats

groups = pd.concat([
    group_stats(df_test, "wealth_cat",  "wealth_cat"),
    group_stats(df_test, "urban_rural", "urban_rural"),
    group_stats(df_test, "caste_type",  "caste_type"),
    group_stats(df_test, "region",      "region"),
], ignore_index=True)

groups = groups[["Dimension","Group","n","actual_insured_pct","pred_uninsured_pct"]]
groups = groups.sort_values("pred_uninsured_pct", ascending=False)

print("\n--- Vulnerable Groups (sorted by predicted uninsured rate) ---")
print(groups.to_string(index=False))

groups.to_csv(out("vulnerable_groups.csv"), index=False)

# Native XGBoost feature importance
fi = pd.DataFrame({
    "feature": FEATURES,
    "feature_label": [FEATURE_LABELS[f] for f in FEATURES],
    "importance_gain": model.feature_importances_,
}).sort_values("importance_gain", ascending=False)
fi.to_csv(out("feature_importance.csv"), index=False)

# Save model
model.save_model(out("xgb_model.json"))

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nPhase 5 complete. Ready for Phase 6 (Policy Gap Table) and Phase 7 (Visualizations).")
