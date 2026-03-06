# =====================================================
# ZERVE BLOCK: SHAP Explainability (Temporal Model)
# Part of ChurnShield — Zerve Hackathon 2026
# =====================================================
# Loads the temporal model and computes SHAP values
# to explain which first-week behaviors drive retention.
# =====================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Zerve dark theme
plt.rcParams.update({
    "figure.facecolor": "#1D1D20", "axes.facecolor": "#1D1D20",
    "axes.edgecolor": "#909094", "axes.labelcolor": "#fbfbff",
    "text.color": "#fbfbff", "xtick.color": "#909094",
    "ytick.color": "#909094", "grid.color": "#2a2a2e",
    "legend.facecolor": "#1D1D20", "legend.edgecolor": "#909094",
})
ACCENT = "#ffd400"

# ── Load model and features ────────────────────────────────────────────────
print("=" * 60)
print("SHAP ANALYSIS: Loading model and features...")

bundle = joblib.load("models/temporal_best_model.joblib")
model = bundle["model"]
feat_cols = bundle["feature_cols"]
model_name = bundle["model_name"]

features_df = pd.read_parquet("data/day1_features.parquet")
targets_df = pd.read_parquet("data/temporal_targets.parquet")
model_df = features_df.merge(targets_df, on="user_id")

# Prep features (same transforms as training)
for c in feat_cols:
    if "time_to_first" in c:
        valid = model_df.loc[model_df[c] >= 0, c]
        fill = valid.max() * 1.5 if len(valid) > 0 else 9999
        model_df[c] = model_df[c].replace(-1, fill)
model_df[feat_cols] = model_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

X = model_df[feat_cols].values
y = model_df["retained_week2"].values

print(f"  Model: {model_name}")
print(f"  Users: {len(X)}, Features: {len(feat_cols)}")

# ── Compute SHAP values ────────────────────────────────────────────────────
print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(model)
shap_raw = explainer.shap_values(X)

# Handle RF binary classification (returns [neg, pos])
if isinstance(shap_raw, list):
    shap_vals = shap_raw[1]
elif shap_raw.ndim == 3:
    shap_vals = shap_raw[:, :, 1]
else:
    shap_vals = shap_raw

exp_val = explainer.expected_value
if isinstance(exp_val, (list, np.ndarray)):
    exp_val = exp_val[1] if len(exp_val) > 1 else exp_val[0]

# ── SHAP Summary (beeswarm) ────────────────────────────────────────────────
print("Generating SHAP plots...")

fig = plt.figure(figsize=(12, 10))
shap.summary_plot(shap_vals, X, feature_names=feat_cols, show=False, max_display=20)
plt.title(f"SHAP Summary — {model_name} (retained_week2)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/temporal_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── SHAP Bar ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X, feature_names=feat_cols, plot_type="bar", show=False, max_display=20)
plt.title("Feature Importance (mean |SHAP|)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/temporal_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── Feature ranking ────────────────────────────────────────────────────────
mean_shap = np.abs(shap_vals).mean(axis=0)
importance = pd.DataFrame({
    "feature": feat_cols,
    "mean_abs_shap": mean_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 15 features by SHAP importance:")
for i, row in importance.head(15).iterrows():
    print(f"  {i+1:2d}. {row['feature']:40s}  |SHAP| = {row['mean_abs_shap']:.4f}")

importance.to_csv("models/shap_feature_importance.csv", index=False)

# ── Top 3 dependence plots ────────────────────────────────────────────────
for feat in importance.head(3)["feature"]:
    idx = feat_cols.index(feat)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(idx, shap_vals, X, feature_names=feat_cols, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence — {feat}")
    plt.tight_layout()
    plt.savefig(f"figures/temporal_shap_dep_{feat.replace('/', '_')}.png", dpi=150)
    plt.close()

# ── Waterfall: 1 high-success, 1 low-success ──────────────────────────────
y_prob = model.predict_proba(X)[:, 1]
high_idx = int(np.argmax(y_prob))
low_idx = int(np.argmin(y_prob))

for label, idx in [("high_success", high_idx), ("low_success", low_idx)]:
    fig = plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(values=shap_vals[idx], base_values=exp_val,
                         data=X[idx], feature_names=feat_cols),
        max_display=15, show=False,
    )
    plt.title(f"SHAP Waterfall — {label} user")
    plt.tight_layout()
    plt.savefig(f"figures/temporal_shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

# ── Save enriched model bundle ─────────────────────────────────────────────
bundle["shap_values"] = shap_vals
bundle["expected_value"] = exp_val
bundle["shap_importance"] = importance
joblib.dump(bundle, "models/temporal_best_model.joblib")

print("\nSaved: shap_feature_importance.csv")
print("Saved: 7 SHAP figures to figures/")
print("Updated: temporal_best_model.joblib with SHAP values")

print("\nBlock complete: SHAP Analysis")
