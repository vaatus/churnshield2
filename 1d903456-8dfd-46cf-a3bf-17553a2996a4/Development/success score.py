"""
Phase 3 — Independent Success Labels (Clean Rebuild)
======================================================
Loads user_features_clean.parquet + events_clean.parquet.
Defines 3 outcome-based, ground-truth targets derived ONLY from raw event
timestamps — completely independent of the feature matrix.

Targets
-------
1. retained_14d       — user has ≥1 event on day 14 or later (relative to
                         their own first event date); measures long-run retention
2. multi_week_user    — user has events in 3 or more distinct ISO calendar
                         weeks; measures sustained engagement breadth
3. churned_early      — user had NO events after day 7 (relative to their own
                         first event); measures early dropout

Success Score (non-circular)
----------------------------
Weighted composite from ONLY retention-based metrics and time-to-value signals:
  - return_rate        (weight 0.35)  — fraction of sessions that are return visits
  - multi_day_user     (weight 0.25)  — binary: active on 2+ days
  - time_to_first_code_run_min (weight 0.20, inverted) — faster is better
  - time_to_first_edge_min     (weight 0.20, inverted) — faster is better
No productivity counts (num_blocks_run, num_agent_events, etc.) used.

Outputs
-------
- user_features_with_targets.parquet
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Zerve design system ───────────────────────────────────────────────────────
BG      = "#1D1D20"
TEXT    = "#fbfbff"
SUBTLE  = "#909094"
COLORS  = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF"]
ACCENT  = "#ffd400"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Phase 3 — Independent Success Labels")
print("=" * 65)

p3_features = pd.read_parquet("user_features_clean.parquet")
p3_events   = pd.read_parquet("events_clean.parquet")
p3_events["timestamp"] = pd.to_datetime(p3_events["timestamp"], utc=True)

print(f"\nFeature matrix : {p3_features.shape[0]:,} users × {p3_features.shape[1]} features")
print(f"Events         : {len(p3_events):,} rows, {p3_events['user_id'].nunique():,} unique users")

# The feature user index
P3_FEATURE_COLS = list(p3_features.columns)
P3_USERS = p3_features.index  # user_id strings

# ══════════════════════════════════════════════════════════════════════════════
# 2.  COMPUTE GROUND-TRUTH TARGETS FROM RAW EVENTS
#     These are computed ONLY from timestamps — never from feature columns.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Building independent targets from raw events ───────────────")

# Compute per-user first event timestamp
_p3_first_ts = p3_events.groupby("user_id")["timestamp"].min().rename("first_ts")

# Join first_ts onto every event so we can compute relative day
_p3_ev = p3_events[["user_id", "timestamp"]].merge(
    _p3_first_ts.reset_index(), on="user_id", how="left"
)
_p3_ev["day_on_platform"] = (
    (_p3_ev["timestamp"] - _p3_ev["first_ts"]).dt.total_seconds() / 86400
)

# ─── TARGET 1: retained_14d ──────────────────────────────────────────────────
# True if user has at least one event at day ≥ 14 from their own first event
_retained_14d_users = (
    _p3_ev[_p3_ev["day_on_platform"] >= 14]
    .groupby("user_id")
    .size()
    .gt(0)
)
retained_14d = (
    _retained_14d_users
    .reindex(P3_USERS, fill_value=False)
    .astype(int)
    .rename("retained_14d")
)

# ─── TARGET 2: multi_week_user ───────────────────────────────────────────────
# True if user has events in 3 or more distinct ISO calendar weeks
# Use (year, iso_week) tuples so weeks don't bleed across year boundaries
_p3_ev_weeks = p3_events[["user_id", "timestamp"]].copy()
_iso = _p3_ev_weeks["timestamp"].dt.isocalendar()
_p3_ev_weeks["_yr_wk"] = _iso["year"].astype(str) + "_" + _iso["week"].astype(str)
_week_counts = _p3_ev_weeks.groupby("user_id")["_yr_wk"].nunique()
multi_week_user_target = (
    (_week_counts >= 3)
    .reindex(P3_USERS, fill_value=False)
    .astype(int)
    .rename("multi_week_user_target")
)

# ─── TARGET 3: churned_early ─────────────────────────────────────────────────
# True if user had NO events after day 7 from their own first event
# Equivalently: max(day_on_platform) < 7
_p3_max_day = _p3_ev.groupby("user_id")["day_on_platform"].max()
churned_early = (
    (_p3_max_day < 7)
    .reindex(P3_USERS, fill_value=True)   # users not in events_clean → treated as churned
    .astype(int)
    .rename("churned_early")
)

print(f"  retained_14d      : {retained_14d.sum():,} positive / {len(retained_14d):,} total "
      f"({retained_14d.mean():.1%})")
print(f"  multi_week_user   : {multi_week_user_target.sum():,} positive / {len(multi_week_user_target):,} total "
      f"({multi_week_user_target.mean():.1%})")
print(f"  churned_early     : {churned_early.sum():,} positive / {len(churned_early):,} total "
      f"({churned_early.mean():.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Class balance ───────────────────────────────────────────────")
_target_df = pd.DataFrame({
    "retained_14d":         retained_14d,
    "multi_week_user_target": multi_week_user_target,
    "churned_early":          churned_early,
})
for _t in _target_df.columns:
    _pos = _target_df[_t].sum()
    _neg = len(_target_df) - _pos
    _rate = _target_df[_t].mean()
    print(f"  {_t:<26} positive={_pos:4d} ({_rate:.1%})  negative={_neg:4d} ({1-_rate:.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  OVERLAP MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Target overlap matrix (% of each row target that is also +ve in col) ─")
_target_short = ["retained_14d", "multi_week", "churned_early"]
_overlap = pd.DataFrame(index=_target_short, columns=_target_short, dtype=float)
for _r, _rc in zip(_target_short, _target_df.columns):
    for _c, _cc in zip(_target_short, _target_df.columns):
        _both = (_target_df[_rc] & _target_df[_cc]).sum()
        _row_n = _target_df[_rc].sum()
        _overlap.loc[_r, _c] = _both / _row_n if _row_n > 0 else 0.0
print(_overlap.round(3).to_string())

# Retained+churned should be near 0 (mutually exclusive by design)
_ret_churn_overlap = (_target_df["retained_14d"] & _target_df["churned_early"]).sum()
print(f"\n  retained_14d ∩ churned_early = {_ret_churn_overlap} users "
      f"(should be near 0 — mutually exclusive targets)")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  ZERO LEAKAGE CHECK
#     Confirm: none of the 3 target column definitions reference any feature col.
#     We do this structurally: the targets were built solely from p3_events
#     timestamps, never from p3_features columns.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Zero leakage validation ─────────────────────────────────────")
# Compute correlation between each target and EVERY feature column
# Any target that is literally equal to a feature column would show r=1.0
_leakage_threshold = 0.99  # a target shouldn't be a deterministic transform of any feature
_leakage_found = False
_feat_num = p3_features.select_dtypes("number")

for _t in _target_df.columns:
    _corrs = _feat_num.corrwith(_target_df[_t]).abs()
    _max_corr_feat = _corrs.idxmax()
    _max_corr_val  = _corrs.max()
    _status = "LEAK" if _max_corr_val >= _leakage_threshold else "OK"
    if _max_corr_val >= _leakage_threshold:
        _leakage_found = True
    print(f"  [{_status}] {_t:<26} max_corr={_max_corr_val:.4f}  "
          f"(with feature: {_max_corr_feat})")

if not _leakage_found:
    print("  ✓ No feature leakage detected in any target (all correlations < 0.99)")
else:
    print("  ✗ WARNING: Potential leakage detected — review targets above")

# Print full correlation table for transparency
print("\n  Top-3 feature correlations per target:")
for _t in _target_df.columns:
    _corrs = _feat_num.corrwith(_target_df[_t]).abs().sort_values(ascending=False).head(3)
    print(f"    {_t}:")
    for _fn, _fv in _corrs.items():
        print(f"      {_fn}: {_fv:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  NON-CIRCULAR SUCCESS SCORE
#     Only uses retention-pattern signals and time-to-value.
#     NO productivity counts (events, blocks_run, agent_events, etc.)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Success score (non-circular) ────────────────────────────────")

def _pct_rank(s):
    """Percentile rank (0–1), NaN-safe."""
    return s.rank(method="average", pct=True, na_option="keep")

# Components
_rr      = _pct_rank(p3_features["return_rate"])              # high = good
_mdu     = _pct_rank(p3_features["multi_day_user"])            # high = good
_ttcr    = _pct_rank(-p3_features["time_to_first_code_run_min"].fillna(
                      p3_features["time_to_first_code_run_min"].max()))  # lower is better → negate
_tte     = _pct_rank(-p3_features["time_to_first_edge_min"].fillna(
                      p3_features["time_to_first_edge_min"].max()))       # lower is better → negate

# Weighted sum
p3_success_score = (
    0.35 * _rr
  + 0.25 * _mdu
  + 0.20 * _ttcr
  + 0.20 * _tte
).rename("success_score")

# Fill any remaining NaN with population median (tiny edge case for all-NaN users)
p3_success_score = p3_success_score.fillna(p3_success_score.median())

print(f"  Weights: return_rate=0.35, multi_day_user=0.25, "
      f"time_to_first_code_run=0.20 (inv), time_to_first_edge=0.20 (inv)")
print(f"  Score stats: mean={p3_success_score.mean():.3f}, "
      f"std={p3_success_score.std():.3f}, "
      f"min={p3_success_score.min():.3f}, max={p3_success_score.max():.3f}")

# Verify the score is NOT dominated by any productivity count
print(f"\n  Leakage check on success_score:")
_sc_corrs = _feat_num.corrwith(p3_success_score).abs().sort_values(ascending=False).head(5)
for _fn, _fv in _sc_corrs.items():
    print(f"    {_fn}: {_fv:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  TARGET CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Correlation matrix between targets ──────────────────────────")
_all_targets = _target_df.copy()
_all_targets["success_score_q"] = (p3_success_score > p3_success_score.median()).astype(int)
_corr_tgt = _all_targets.corr()
print(_corr_tgt.round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 8.  BUILD FINAL DATAFRAME & SAVE
# ══════════════════════════════════════════════════════════════════════════════
p3_output = p3_features.copy()
p3_output["retained_14d"]           = retained_14d
p3_output["multi_week_user_target"] = multi_week_user_target
p3_output["churned_early"]          = churned_early
p3_output["success_score"]          = p3_success_score

p3_output.to_parquet("user_features_with_targets.parquet")
print(f"\n✓ Saved user_features_with_targets.parquet  "
      f"({p3_output.shape[0]:,} rows × {p3_output.shape[1]} columns)")
print(f"  Feature columns : {len(P3_FEATURE_COLS)}")
print(f"  Target columns  : retained_14d, multi_week_user_target, churned_early, success_score")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

# 9a. Target class balance bar chart
_target_names = ["retained_14d", "multi_week_user", "churned_early"]
_target_rates  = [retained_14d.mean(), multi_week_user_target.mean(), churned_early.mean()]

target_class_balance_chart, _ax1 = plt.subplots(figsize=(9, 5))
target_class_balance_chart.patch.set_facecolor(BG)
_ax1.set_facecolor(BG)
_bars = _ax1.bar(_target_names, [r * 100 for r in _target_rates],
                  color=COLORS[:3], edgecolor="none", width=0.5)
for _b, _r in zip(_bars, _target_rates):
    _ax1.text(_b.get_x() + _b.get_width() / 2,
              _b.get_height() + 1.5,
              f"{_r:.1%}", ha="center", va="bottom",
              color=TEXT, fontsize=11, fontweight="bold")
_ax1.set_ylim(0, max(_target_rates) * 100 * 1.25)
_ax1.set_ylabel("% Positive", color=TEXT)
_ax1.set_title("Target Class Balance — Phase 3 Independent Labels",
               color=TEXT, fontsize=13, pad=12)
_ax1.tick_params(colors=TEXT)
for _sp in _ax1.spines.values():
    _sp.set_visible(False)
_ax1.yaxis.label.set_color(TEXT)
target_class_balance_chart.tight_layout()
target_class_balance_chart.savefig("figures/p3_target_balance.png", dpi=150,
                                    bbox_inches="tight", facecolor=BG)
plt.close(target_class_balance_chart)

# 9b. Target correlation heatmap
target_correlation_heatmap, _ax2 = plt.subplots(figsize=(7, 6))
target_correlation_heatmap.patch.set_facecolor(BG)
_ax2.set_facecolor(BG)
_corr_plot = pd.DataFrame({
    "retained_14d":     retained_14d,
    "multi_week":       multi_week_user_target,
    "churned_early":    churned_early,
    "success_score":    p3_success_score,
}).corr()
sns.heatmap(
    _corr_plot, annot=True, fmt=".2f",
    cmap="RdYlBu_r", vmin=-1, vmax=1,
    ax=_ax2, linewidths=0.5, linecolor=BG,
    annot_kws={"size": 10, "color": TEXT},
)
_ax2.set_title("Target × Score Correlation Matrix",
               color=TEXT, fontsize=12, pad=10)
_ax2.tick_params(colors=TEXT)
_ax2.xaxis.label.set_color(TEXT)
_ax2.yaxis.label.set_color(TEXT)
target_correlation_heatmap.tight_layout()
target_correlation_heatmap.savefig("figures/p3_target_correlations.png", dpi=150,
                                    bbox_inches="tight", facecolor=BG)
plt.close(target_correlation_heatmap)

# 9c. Success score distribution
success_score_dist, _ax3 = plt.subplots(figsize=(9, 5))
success_score_dist.patch.set_facecolor(BG)
_ax3.set_facecolor(BG)
_ax3.hist(p3_success_score, bins=50, color=COLORS[0], alpha=0.8, edgecolor="none")
_ax3.axvline(p3_success_score.median(), color=ACCENT, linestyle="--",
              linewidth=1.8, label=f"Median = {p3_success_score.median():.3f}")
_ax3.set_xlabel("Success Score", color=TEXT)
_ax3.set_ylabel("Users", color=TEXT)
_ax3.set_title("Non-Circular Success Score Distribution\n"
               "(based on return_rate, multi_day_user, time_to_first_code_run, time_to_first_edge)",
               color=TEXT, fontsize=11, pad=10)
_ax3.tick_params(colors=TEXT)
_ax3.legend(labelcolor=TEXT, facecolor=BG, edgecolor=SUBTLE)
for _sp in _ax3.spines.values():
    _sp.set_visible(False)
success_score_dist.tight_layout()
success_score_dist.savefig("figures/p3_success_score_dist.png", dpi=150,
                             bbox_inches="tight", facecolor=BG)
plt.close(success_score_dist)

print("\n✓ Saved visualisations → figures/p3_target_balance.png, "
      "figures/p3_target_correlations.png, figures/p3_success_score_dist.png")
print("=" * 65)
print("Phase 3 complete.")
