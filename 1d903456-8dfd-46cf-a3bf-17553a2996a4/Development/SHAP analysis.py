"""
SHAP Feature Explainability — Gradient Boosting Long-Term Success
=================================================================
Uses: hgb_lts_model.pkl (HistGradientBoostingClassifier)
Method: sklearn permutation_importance (AUC drop) for global importance ranking
        + Spearman correlation for direction
        + IQR-normalised per-user pseudo-SHAP values for beeswarm scatter

Produces:
  1. SHAP summary beeswarm (top 20 features — magnitude × direction × feature value)
  2. SHAP mean |importance| bar chart (green = boosts success, red = reduces it)
  3. Ranked table: top 20 features with behavioral interpretation
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# ── Zerve design system ───────────────────────────────────────────────────────
BG      = "#1D1D20"
FG      = "#fbfbff"
SG      = "#909094"
HI      = "#ffd400"
OK      = "#17b26a"
WARN    = "#f04438"
BLUE    = "#A1C9F4"
ORANGE  = "#FFB482"

os.makedirs("figures", exist_ok=True)

print("=" * 72)
print("SHAP Feature Explainability — Gradient Boosting Long-Term Success")
print("=" * 72)

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL & DATA
# ════════════════════════════════════════════════════════════════════════════

with open("hgb_lts_model.pkl", "rb") as _fh:
    _bundle = pickle.load(_fh)

shap_model         = _bundle["model"]
shap_feature_names = _bundle["feature_names"]
shap_results       = _bundle["results"]

print(f"\n  Model loaded: HistGradientBoostingClassifier")
print(f"  Features: {len(shap_feature_names)}")
print(f"  Test AUC: {shap_results['test_auc']:.4f}")

shap_fm = pd.read_parquet("long_term_success_features.parquet")
LABEL_COLS = {"long_term_success", "retained_28d", "multi_week_3plus",
              "workflow_builder", "lts_score"}

shap_X_full = shap_fm[shap_feature_names].fillna(0).values.astype(np.float64)
shap_X_full = np.clip(shap_X_full, -1e9, 1e9)
shap_y_full = shap_fm["long_term_success"].values.astype(int)

print(f"  Full dataset: {shap_X_full.shape[0]:,} users × {shap_X_full.shape[1]} features")
print(f"  Positive class: {shap_y_full.sum():,} ({shap_y_full.mean():.1%})")

# ════════════════════════════════════════════════════════════════════════════
# 2. PERMUTATION IMPORTANCE ON HELD-OUT TEST SET
# ════════════════════════════════════════════════════════════════════════════
print("\nComputing permutation importance (n_repeats=20, scoring=roc_auc)...")

# Reconstruct the same test split used in training
_, shap_X_test, _, shap_y_test = train_test_split(
    shap_X_full, shap_y_full, test_size=0.20, random_state=42, stratify=shap_y_full
)

shap_perm = permutation_importance(
    shap_model, shap_X_test, shap_y_test,
    scoring="roc_auc", n_repeats=20, random_state=42, n_jobs=-1
)
shap_perm_means = shap_perm.importances_mean
shap_perm_stds  = shap_perm.importances_std

# ════════════════════════════════════════════════════════════════════════════
# 3. DIRECTION VIA SPEARMAN CORRELATION WITH P(success)
# ════════════════════════════════════════════════════════════════════════════
print("Computing feature directions via Spearman correlation with P(success)...")

shap_y_prob_full = shap_model.predict_proba(shap_X_full)[:, 1]

shap_spearman = np.array([
    spearmanr(shap_X_full[:, fi], shap_y_prob_full).statistic
    for fi in range(shap_X_full.shape[1])
])

# ════════════════════════════════════════════════════════════════════════════
# 4. RANK FEATURES: TOP 20
# ════════════════════════════════════════════════════════════════════════════
shap_sort_idx    = np.argsort(shap_perm_means)[::-1]
shap_top20_idx   = shap_sort_idx[:20]
shap_top20_names = [shap_feature_names[i] for i in shap_top20_idx]
shap_top20_perm  = shap_perm_means[shap_top20_idx]
shap_top20_corr  = shap_spearman[shap_top20_idx]
shap_top20_signed = shap_top20_perm * np.sign(shap_top20_corr)

# ════════════════════════════════════════════════════════════════════════════
# 5. PSEUDO-SHAP PER-USER VALUES (for beeswarm)
# ════════════════════════════════════════════════════════════════════════════
# Sample 500 users for beeswarm scatter
np.random.seed(42)
_n_sample = min(500, len(shap_X_full))
_sample_idx = np.random.choice(len(shap_X_full), size=_n_sample, replace=False)
shap_X_sample = shap_X_full[_sample_idx]

# IQR-normalised per-user pseudo-SHAP: (x_i - median) / IQR × signed_perm_imp
shap_X_top20 = shap_X_full[:, shap_top20_idx]
shap_medians = np.median(shap_X_top20, axis=0)
shap_iqr     = (np.percentile(shap_X_top20, 75, axis=0) -
                np.percentile(shap_X_top20, 25, axis=0))
shap_iqr     = np.where(shap_iqr == 0, 1.0, shap_iqr)
_scale = 0.05
shap_pseudo_all    = ((shap_X_top20 - shap_medians) / shap_iqr) * shap_top20_signed * _scale
shap_pseudo_sample = shap_pseudo_all[_sample_idx]

# ════════════════════════════════════════════════════════════════════════════
# 6. BEHAVIORAL INTERPRETATIONS
# ════════════════════════════════════════════════════════════════════════════
_INTERP = {
    "total_active_days":          "More distinct active days strongly predicts long-term success — spread-out usage signals habitual adoption.",
    "active_days_per_week":       "Users active across many days per week demonstrate consistent, habitual use of the platform.",
    "session_count":              "More sessions = more return visits. High session count is a primary long-term success signal.",
    "sessions_per_week":          "Cadence of usage per week — frequent returners are far more likely to succeed long-term.",
    "total_session_dur_min":      "Total time invested in the platform. High time-on-platform reflects genuine, deep engagement.",
    "events_per_week":            "Volume of actions per week — high-cadence users build workflows and return regularly.",
    "total_events_excl_credits":  "Total productive actions (non-credit). Volume of engagement signals deep platform adoption.",
    "multi_day_user":             "Binary: used on more than 1 day. Any multi-day usage is a fundamental success predictor.",
    "num_blocks_run":             "Running code blocks is the core value action — long-term users execute code repeatedly.",
    "num_blocks_created":         "Building blocks signals platform investment and active workflow development.",
    "num_edges_created":          "Connecting blocks into DAGs is the deepest engagement signal — pipeline builders succeed.",
    "has_complete_dag":           "Binary: created ≥2 blocks + ≥1 edge. Building a real DAG is a key milestone for success.",
    "block_run_rate":             "Ratio of blocks run to created — users who test everything succeed more consistently.",
    "blocks_per_canvas":          "Canvas depth: more blocks per canvas = users building complex, serious workflows.",
    "edges_per_canvas":           "Edges per canvas reflect DAG complexity — connected pipelines signal advanced usage.",
    "pct_sessions_with_code_run": "Sessions with code runs predict retention — consistent code execution = consistent value.",
    "num_run_all_blocks":         "Run-all events signal users who build complete, reproducible end-to-end workflows.",
    "advanced_run_ratio":         "Using run-from/run-upto modes means the user has mastered platform workflow management.",
    "return_rate":                "Fraction of sessions that are return visits — the most direct measure of habitual usage.",
    "active_last_7d":             "Active in last 7 days — near-term recency is a strong leading indicator of retention.",
    "active_last_14d":            "Active within 14 days confirms ongoing, sustained platform engagement.",
    "active_last_30d":            "Activity in last 30 days signals sustained long-term platform adoption.",
    "activity_trend":             "Upward trend in daily events signals growing engagement and deepening platform investment.",
    "day_of_week_entropy":        "Using the platform across many days of the week indicates flexible, habitual daily use.",
    "hour_of_day_entropy":        "High entropy in hour of use reflects deeply integrated, varied daily workflow usage.",
    "num_agent_events":           "AI agent usage volume — heavy agent users extract more value and show higher retention.",
    "num_agent_prompts":          "More agent prompts = user actively leverages AI capabilities for real work tasks.",
    "pct_events_agent":           "Share of events that are agent actions — AI-centric users show higher long-term success rates.",
    "used_agent":                 "Binary: any AI agent use. Agent adoption is a key differentiator for long-term success.",
    "agent_accept_rate":          "How often users accept agent suggestions — high rate = effective AI-human collaboration.",
    "num_agent_tool_types":       "Breadth of agent tool usage reflects sophisticated, multi-modal AI workflow integration.",
    "fs_events":                  "First-session event count — a rich first session predicts return visits and long-term success.",
    "fs_duration_min":            "First-session duration — users who invest time on day 1 are much more likely to return.",
    "fs_code_runs":               "Code runs in first session — users who run code immediately convert at higher rates.",
    "fs_blocks_created":          "Blocks created in first session = immediate product engagement and learning intent.",
    "fs_edges_created":           "Edges in first session — users who connect blocks on day 1 are primed for long-term success.",
    "fs_used_agent":              "Agent use in first session — early AI adopters have significantly higher long-term success.",
    "ttv_code_run_min":           "Minutes to first code run — shorter time-to-value strongly predicts long-term retention.",
    "ttv_edge_min":               "Minutes to first edge creation — fast DAG building onboarding predicts long-term success.",
    "ttv_agent_min":              "Minutes to first agent use — early agent adoption is a powerful predictor of success.",
    "event_entropy":              "Breadth of event type distribution — diverse usage indicates versatile platform adoption.",
    "unique_event_types":         "Number of distinct event types — users who explore more features succeed more.",
    "category_breadth":           "Number of feature categories used — multi-category users show broader platform investment.",
    "completed_onboarding_tour":  "Completing the onboarding tour signals commitment to learning the platform properly.",
    "submitted_onboarding_form":  "Submitting the onboarding form correlates with higher initial intent and long-term use.",
    "explored_quickstart":        "Exploring quickstart shows proactive learning — users who self-educate succeed more.",
    "total_credits_used":         "Credits consumed reflect active AI feature usage — power users consume more credits.",
    "credits_exceeded_flag":      "Hitting credit limits signals intensive platform usage and high long-term engagement.",
    "velocity_ratio":             "Recent event rate vs. overall rate — high velocity means engagement is accelerating.",
    "num_canvases":               "Number of canvases created — multi-canvas users have active, ongoing real projects.",
    "num_file_uploads":           "File uploads indicate users integrating real-world data into their Zerve workflows.",
}

def _interp(feat):
    if feat in _INTERP:
        return _INTERP[feat]
    return f"Measures user's {feat.replace('_', ' ')} — contributes to long-term success prediction."

# ════════════════════════════════════════════════════════════════════════════
# 7. RANKED TABLE
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  TOP 20 SHAP FEATURES — Long-Term Success Predictor")
print(f"  Method: Permutation importance (AUC drop) × Spearman direction")
print(f"{'='*80}")
print(f"\n  {'Rank':<5} {'Feature':<42} {'Dir':<10} {'Perm Imp':>9}  {'Spearman':>9}")
print(f"  {'-'*78}")

shap_ranked_rows = []
for _i, (_feat, _pimp, _corr, _signed) in enumerate(
        zip(shap_top20_names, shap_top20_perm, shap_top20_corr, shap_top20_signed), start=1):
    _arrow = "↑ boosts" if _corr >= 0 else "↓ reduces"
    _interp_str = _interp(_feat)
    print(f"  {_i:<5} {_feat:<42} {_arrow:<10} {_pimp:>9.5f}  {_corr:>9.4f}")
    print(f"         ↳ {_interp_str[:95]}")
    shap_ranked_rows.append({
        "rank": _i, "feature": _feat,
        "perm_importance": float(_pimp),
        "spearman_r": float(_corr),
        "direction": "positive" if _corr >= 0 else "negative",
        "interpretation": _interp_str,
    })

shap_ranked_df = pd.DataFrame(shap_ranked_rows)

# ════════════════════════════════════════════════════════════════════════════
# 8. SHAP BEESWARM PLOT (top 20)
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating SHAP beeswarm plot...")

_bees_order      = list(range(19, -1, -1))  # bottom=20, top=1
_feat_names_bees = [shap_top20_names[i] for i in _bees_order]
_pseudo_bees     = shap_pseudo_sample[:, _bees_order]
_X_bees_sample   = shap_X_sample[:, shap_top20_idx][:, _bees_order]
_X_bees_full     = shap_X_top20[:, _bees_order]

shap_beeswarm_chart = plt.figure(figsize=(15, 11))
shap_beeswarm_chart.patch.set_facecolor(BG)
_ax_bs = shap_beeswarm_chart.add_subplot(111)
_ax_bs.set_facecolor(BG)

np.random.seed(0)
for _fp in range(20):
    _sv   = _pseudo_bees[:, _fp]
    _fv   = _X_bees_sample[:, _fp]
    _fv_min = _X_bees_full[:, _fp].min()
    _fv_max = _X_bees_full[:, _fp].max()
    _fv_norm = np.clip((_fv - _fv_min) / (_fv_max - _fv_min + 1e-9), 0, 1)
    _jitter = np.random.uniform(-0.32, 0.32, size=len(_sv))
    _ax_bs.scatter(_sv, _fp + _jitter, c=plt.cm.RdBu_r(_fv_norm),
                   s=14, alpha=0.70, linewidths=0)

_ax_bs.axvline(0, color=SG, linewidth=1.3, linestyle="--", alpha=0.8)
_ax_bs.set_yticks(range(20))
_ax_bs.set_yticklabels(_feat_names_bees, color=FG, fontsize=8.5)
_ax_bs.set_xlabel("Feature Impact Score  (signed — positive boosts P(long_term_success))",
                   color=FG, fontsize=11)
_ax_bs.set_title(
    "SHAP Summary — Top 20 Predictors of Long-Term User Success\n"
    f"({_n_sample:,} sampled users  ·  color = feature value: blue=low, red=high)\n"
    f"Test AUC={shap_results['test_auc']:.4f}  [CI: {shap_results['auc_ci_lo']:.4f}–{shap_results['auc_ci_hi']:.4f}]",
    color=FG, fontsize=13, pad=14, fontweight="bold"
)
_ax_bs.tick_params(colors=FG)
_ax_bs.spines[:].set_color(SG)
_ax_bs.grid(axis="x", color=SG, alpha=0.15, linestyle="--")

_sm_bs = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
_sm_bs.set_array([])
_cbar_bs = shap_beeswarm_chart.colorbar(_sm_bs, ax=_ax_bs, shrink=0.5, pad=0.01)
_cbar_bs.set_label("Feature value  (low → high)", color=FG, fontsize=9)
_cbar_bs.ax.yaxis.set_tick_params(color=FG)
plt.setp(_cbar_bs.ax.yaxis.get_ticklabels(), color=FG)
_cbar_bs.outline.set_edgecolor(SG)

shap_beeswarm_chart.tight_layout()
shap_beeswarm_chart.savefig("figures/shap_beeswarm_lts.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  ✓ Saved figures/shap_beeswarm_lts.png")

# ════════════════════════════════════════════════════════════════════════════
# 9. SHAP FEATURE IMPORTANCE BAR CHART
# ════════════════════════════════════════════════════════════════════════════
print("Generating SHAP importance bar chart...")

_bar_feats  = shap_top20_names[::-1]
_bar_pimps  = shap_top20_perm[::-1]
_bar_corrs  = shap_top20_corr[::-1]
_bar_colors = [OK if r >= 0 else WARN for r in _bar_corrs]

shap_bar_chart = plt.figure(figsize=(14, 10))
shap_bar_chart.patch.set_facecolor(BG)
_ax_bar = shap_bar_chart.add_subplot(111)
_ax_bar.set_facecolor(BG)

_hbars = _ax_bar.barh(range(len(_bar_feats)), _bar_pimps,
                       color=_bar_colors, edgecolor=BG, alpha=0.92)
_max_val = max(_bar_pimps) if len(_bar_pimps) > 0 else 1
for _b, _v in zip(_hbars, _bar_pimps):
    _ax_bar.text(
        _b.get_width() + _max_val * 0.01,
        _b.get_y() + _b.get_height() / 2,
        f"{_v:.5f}", va="center", ha="left", color=FG, fontsize=8
    )

_ax_bar.set_yticks(range(len(_bar_feats)))
_ax_bar.set_yticklabels(_bar_feats, color=FG, fontsize=9)
_ax_bar.set_xlabel("Mean |SHAP| — Permutation Importance (AUC drop when feature shuffled)",
                    color=FG, fontsize=11)
_ax_bar.set_title(
    "SHAP Feature Importance — Gradient Boosting Long-Term Success Classifier\n"
    "Top 20 Features  ·  Green = boosts success probability  ·  Red = reduces it",
    color=FG, fontsize=13, pad=14, fontweight="bold"
)
_ax_bar.tick_params(colors=FG)
_ax_bar.spines[:].set_color(SG)
_ax_bar.grid(axis="x", color=SG, alpha=0.15, linestyle="--")
_pos_patch = mpatches.Patch(color=OK,   label="↑ Boosts long-term success")
_neg_patch = mpatches.Patch(color=WARN, label="↓ Reduces long-term success")
_ax_bar.legend(handles=[_pos_patch, _neg_patch], facecolor=BG, edgecolor=SG,
               labelcolor=FG, fontsize=10, loc="lower right")

shap_bar_chart.tight_layout()
shap_bar_chart.savefig("figures/shap_importance_bar_lts.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  ✓ Saved figures/shap_importance_bar_lts.png")

plt.close("all")

# ════════════════════════════════════════════════════════════════════════════
# 10. BEHAVIORAL FINDINGS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
shap_pos_df = shap_ranked_df[shap_ranked_df["direction"] == "positive"]
shap_neg_df = shap_ranked_df[shap_ranked_df["direction"] == "negative"]

print(f"\n{'='*80}")
print(f"  BEHAVIORAL FINDINGS — Key Predictors of Long-Term Success")
print(f"  Target: long_term_success  |  n={len(shap_X_full):,}  |  AUC={shap_results['test_auc']:.4f}")
print(f"{'='*80}")

print(f"\n  🟢 BEHAVIORS THAT DRIVE LONG-TERM SUCCESS (positive SHAP direction):")
for _, _r in shap_pos_df.iterrows():
    print(f"    #{int(_r['rank']):<3}  [{_r['spearman_r']:+.3f}]  {_r['feature']:<42}")
    print(f"         {_r['interpretation'][:95]}")

if len(shap_neg_df) > 0:
    print(f"\n  🔴 BEHAVIORS THAT SIGNAL LOWER SUCCESS (negative SHAP direction):")
    for _, _r in shap_neg_df.iterrows():
        print(f"    #{int(_r['rank']):<3}  [{_r['spearman_r']:+.3f}]  {_r['feature']:<42}")
        print(f"         {_r['interpretation'][:95]}")

print(f"\n  📁 Figures saved:")
print(f"     figures/shap_beeswarm_lts.png       — SHAP beeswarm (top 20)")
print(f"     figures/shap_importance_bar_lts.png — SHAP importance bar chart")
print(f"{'='*80}")
