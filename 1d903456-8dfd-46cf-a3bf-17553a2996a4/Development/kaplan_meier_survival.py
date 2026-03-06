
"""
Kaplan-Meier Survival Analysis — Time to Churn (pure numpy/scipy/statsmodels)
==============================================================================
Churn = user's last event was >30 days before the data cutoff.
Duration = calendar days from first to last observed event.
Censored = user was still active within last 30 days.

KM curves segmented by:
  1. Used Agent vs Not
  2. Built Complete DAG vs Not
  3. Completed Onboarding Tour vs Not
  4. Early Code Runner (ran code ≤60 min) vs Not
  5. Session depth tiers (1, 2-4, 5+)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Zerve palette ─────────────────────────────────────────────────────────────
BG, FG, FG2 = "#1D1D20", "#fbfbff", "#909094"
C1, C2, C3, C4, C5 = "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF"
GOLD, SUCCESS_C, WARN_C = "#ffd400", "#17b26a", "#f04438"

# ── Kaplan-Meier estimator ────────────────────────────────────────────────────
def km_estimate(durations, events):
    """
    Returns (times, survival_prob, lower_95, upper_95) using Greenwood's formula.
    durations: array-like of observation times
    events: 1=event occurred (churn), 0=censored
    """
    durations = np.asarray(durations, dtype=float)
    events    = np.asarray(events, dtype=int)
    order     = np.argsort(durations)
    durations = durations[order]
    events    = events[order]

    unique_times = np.unique(durations[events == 1])
    n    = len(durations)
    S    = 1.0
    surv, t_out, lower, upper = [], [], [], []
    greenwood_sum = 0.0

    surv.append(1.0); t_out.append(0.0); lower.append(1.0); upper.append(1.0)

    for t in unique_times:
        at_risk = np.sum(durations >= t)
        died    = np.sum((durations == t) & (events == 1))
        if at_risk == 0:
            continue
        S = S * (1 - died / at_risk)
        if died > 0 and at_risk > died:
            greenwood_sum += died / (at_risk * (at_risk - died))
        se = S * np.sqrt(greenwood_sum)
        z = 1.96
        lo = max(0, S - z * se)
        hi = min(1, S + z * se)
        surv.append(S); t_out.append(t); lower.append(lo); upper.append(hi)

    return np.array(t_out), np.array(surv), np.array(lower), np.array(upper)

def km_median(times, surv):
    """Median survival time (first time S ≤ 0.5)."""
    idx = np.searchsorted(-surv, -0.5)
    if idx >= len(times):
        return np.inf
    return times[idx]

def logrank_test(t1, e1, t2, e2):
    """Log-rank test, returns (test_stat, p_value)."""
    t1, e1, t2, e2 = map(np.asarray, [t1, e1, t2, e2])
    all_times = np.unique(np.concatenate([t1[e1==1], t2[e2==1]]))
    O1_sum, E1_sum, V_sum = 0.0, 0.0, 0.0
    for t in all_times:
        n1 = np.sum(t1 >= t)
        n2 = np.sum(t2 >= t)
        d1 = np.sum((t1 == t) & (e1 == 1))
        d2 = np.sum((t2 == t) & (e2 == 1))
        n  = n1 + n2
        d  = d1 + d2
        if n < 2:
            continue
        E1 = n1 * d / n
        V  = n1 * n2 * d * (n - d) / (n**2 * (n - 1)) if n > 1 else 0
        O1_sum += d1; E1_sum += E1; V_sum += V
    if V_sum == 0:
        return 0.0, 1.0
    chi2 = (O1_sum - E1_sum)**2 / V_sum
    pval = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, pval

def plot_km(ax, groups, title):
    """Plot KM curves for a list of (label, times, survs, lowers, uppers, color, n)."""
    _style_ax(ax, title)
    for label, t, s, lo, hi, color, n in groups:
        ax.step(t, s, where="post", color=color, linewidth=2.5, label=f"{label} (n={n:,})")
        ax.fill_between(t, lo, hi, step="post", color=color, alpha=0.15)
    ax.legend(loc="upper right", framealpha=0.2, labelcolor=FG,
              facecolor=BG, edgecolor="#555", fontsize=8.5)

def _style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Days Since First Event", color=FG2, fontsize=9)
    ax.set_ylabel("Survival Probability", color=FG2, fontsize=9)
    ax.tick_params(colors=FG2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(axis="y", color="#333", linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, 1.05)

# ── Build survival dataset ────────────────────────────────────────────────────
print("=" * 65)
print("KAPLAN-MEIER SURVIVAL ANALYSIS — TIME TO CHURN")
print("=" * 65)

_first_ts = raw.groupby("user_id")["timestamp"].min()
_last_ts  = raw.groupby("user_id")["timestamp"].max()
_REF      = raw["timestamp"].max()

_tenure_days = ((_last_ts - _first_ts).dt.total_seconds() / 86400).clip(lower=0.5)
_days_since  = ((_REF - _last_ts).dt.total_seconds() / 86400)
_churn       = (_days_since > 30).astype(int)

km_survival_df = pd.DataFrame({
    "duration_days": _tenure_days.values,
    "churn":         _churn.values,
}, index=_tenure_days.index)

_fm_cols = ["used_agent", "has_complete_dag", "completed_onboarding_tour",
            "multi_day_user", "ttv_code_run_min", "long_term_success",
            "pct_events_agent", "num_blocks_created", "session_count",
            "fs_used_agent", "sessions_per_week", "event_entropy",
            "total_active_days", "num_edges_created", "advanced_run_ratio",
            "pct_sessions_with_code_run"]

km_survival_df = km_survival_df.join(feature_matrix[_fm_cols], how="inner")
km_survival_df = km_survival_df.dropna(subset=["duration_days", "churn"])
km_survival_df["early_code_runner"] = (km_survival_df["ttv_code_run_min"].fillna(9999) <= 60).astype(int)

print(f"\nSurvival dataset: {len(km_survival_df):,} users")
print(f"Churned (event=1): {km_survival_df['churn'].sum():,} ({km_survival_df['churn'].mean():.1%})")
print(f"Censored (event=0): {(km_survival_df['churn']==0).sum():,} ({(km_survival_df['churn']==0).mean():.1%})")
print(f"Median tenure (days): {km_survival_df['duration_days'].median():.1f}")

# ── Figure 1: 4-panel behavioral KM curves ──────────────────────────────────
segments = [
    ("used_agent",               "Agent Usage",        [(0, "No Agent",          C2),  (1, "Used Agent",          C1)]),
    ("has_complete_dag",         "Workflow Builder",   [(0, "No DAG",            C4),  (1, "Built DAG",           C3)]),
    ("completed_onboarding_tour","Onboarding",         [(0, "Skipped Tour",      GOLD),(1, "Completed Tour",      SUCCESS_C)]),
    ("early_code_runner",        "Early Code Runner",  [(0, "Late/No Code Run",  C5),  (1, "Ran Code ≤60 min",   C2)]),
]

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Kaplan-Meier Survival Curves by Behavioral Segment",
              color=FG, fontsize=16, fontweight="bold", y=0.98)

km_segment_summary = {}
for ax, (col, label, groups) in zip(axes1.flatten(), segments):
    _style_ax(ax, f"Survival by: {label}")
    patches = []
    km_data = {}
    for gval, gname, color in groups:
        _mask = km_survival_df[col] == gval
        _sub  = km_survival_df[_mask]
        if len(_sub) < 5:
            continue
        t, s, lo, hi = km_estimate(_sub["duration_days"].values, _sub["churn"].values)
        ax.step(t, s, where="post", color=color, linewidth=2.5)
        ax.fill_between(t, lo, hi, step="post", color=color, alpha=0.15)
        _med = km_median(t, s)
        patches.append(mpatches.Patch(color=color, label=f"{gname} (n={len(_sub):,}, med={_med:.1f}d)"))
        km_data[gval] = (_sub["duration_days"].values, _sub["churn"].values)

    # Log-rank test
    if 0 in km_data and 1 in km_data:
        chi2, pval = logrank_test(km_data[0][0], km_data[0][1], km_data[1][0], km_data[1][1])
        pstr = f"p={pval:.4f}" if pval >= 0.0001 else "p<0.0001"
        ax.annotate(f"Log-rank {pstr}", xy=(0.98, 0.08), xycoords="axes fraction",
                    ha="right", fontsize=8.5, color=FG2)
        km_segment_summary[label] = {"chi2": chi2, "pval": pval}

    ax.legend(handles=patches, loc="upper right", framealpha=0.2,
              labelcolor=FG, facecolor=BG, edgecolor="#555", fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
km_segment_curves = fig1
print("\n✅ Figure 1 (4-panel KM by behavioral segment) ready.")

# ── Figure 2: Overall + session depth ───────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Overall Survival & Retention by Session Depth",
              color=FG, fontsize=15, fontweight="bold")

# Panel A: overall KM
ax_ov = axes2[0]
_style_ax(ax_ov, "Overall Survival Curve (All Users)")
t_all, s_all, lo_all, hi_all = km_estimate(km_survival_df["duration_days"].values, km_survival_df["churn"].values)
ax_ov.step(t_all, s_all, where="post", color=C1, linewidth=2.5, label="All Users")
ax_ov.fill_between(t_all, lo_all, hi_all, step="post", color=C1, alpha=0.15)
_med_all = km_median(t_all, s_all)
ax_ov.axhline(0.5, color=GOLD, linestyle="--", linewidth=1.3, alpha=0.8)
ax_ov.annotate(f"Median tenure: {_med_all:.1f}d", xy=(0.52, 0.53), xycoords="axes fraction",
               color=GOLD, fontsize=10, fontweight="bold")
ax_ov.legend(loc="upper right", framealpha=0.2, labelcolor=FG,
             facecolor=BG, edgecolor="#555", fontsize=9)

# Panel B: session depth tiers
ax_sd = axes2[1]
_style_ax(ax_sd, "Survival by Session Depth")
session_tiers = [
    ("Single Session", km_survival_df["session_count"] == 1, C4),
    ("2–4 Sessions",  (km_survival_df["session_count"] >= 2) & (km_survival_df["session_count"] <= 4), C2),
    ("5+ Sessions",    km_survival_df["session_count"] >= 5, C3),
]
patches2 = []
for tname, tmask, tcolor in session_tiers:
    _sub = km_survival_df[tmask]
    if len(_sub) < 5:
        continue
    t, s, lo, hi = km_estimate(_sub["duration_days"].values, _sub["churn"].values)
    ax_sd.step(t, s, where="post", color=tcolor, linewidth=2.5)
    ax_sd.fill_between(t, lo, hi, step="post", color=tcolor, alpha=0.12)
    _med = km_median(t, s)
    patches2.append(mpatches.Patch(color=tcolor,
                                   label=f"{tname} (n={tmask.sum():,}, med={_med:.1f}d)"))
ax_sd.legend(handles=patches2, loc="upper right", framealpha=0.2,
             labelcolor=FG, facecolor=BG, edgecolor="#555", fontsize=8.5)

plt.tight_layout()
km_overall_curves = fig2
print("✅ Figure 2 (overall + session depth) ready.")

# ── Summary Table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KM SUMMARY: MEDIAN SURVIVAL TIME BY SEGMENT")
print("=" * 65)
for col, label, groups in segments:
    print(f"\n  {label}:")
    _g_data = {}
    for gval, gname, _ in groups:
        _sub = km_survival_df[km_survival_df[col] == gval]
        if len(_sub) < 5:
            continue
        t, s, _, _ = km_estimate(_sub["duration_days"].values, _sub["churn"].values)
        _med = km_median(t, s)
        _med_str = f"{_med:.1f}d" if np.isfinite(_med) else "not reached"
        print(f"    {gname:<30} n={len(_sub):>5,}  median={_med_str}")
        _g_data[gval] = (_sub["duration_days"].values, _sub["churn"].values)
    if 0 in _g_data and 1 in _g_data:
        chi2, pval = logrank_test(_g_data[0][0], _g_data[0][1], _g_data[1][0], _g_data[1][1])
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        print(f"    Log-rank: χ²={chi2:.2f}  p={pval:.4f}  {stars}")

print(f"\n  Overall median survival: {_med_all:.1f}d")
