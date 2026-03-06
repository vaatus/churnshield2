
"""
Cox Proportional Hazards Model — implemented via scipy optimization
====================================================================
Uses partial likelihood (Efron tie-breaking) to estimate hazard ratios.
HR > 1 → increases churn risk | HR < 1 → protective (delays churn)

Newton-Raphson optimization via scipy.optimize.minimize with L-BFGS-B.
Standard errors from the observed Fisher information matrix (negative Hessian).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats, optimize

BG, FG, FG2 = "#1D1D20", "#fbfbff", "#909094"
C1, C2, C3, C4, C5 = "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF"
GOLD, SUCCESS_C, WARN_C = "#ffd400", "#17b26a", "#f04438"

print("=" * 65)
print("COX PROPORTIONAL HAZARDS MODEL — CHURN RISK FACTORS")
print("=" * 65)

# ── Partial log-likelihood with Breslow approximation ────────────────────────
def cox_neg_partial_loglik(beta, T, E, X):
    """Negative partial log-likelihood (Breslow tie handling)."""
    beta = np.asarray(beta)
    xb   = X @ beta                       # linear predictor
    exp_xb = np.exp(xb - xb.max())        # numerically stable
    nll = 0.0
    for i in np.where(E)[0]:              # iterate over events only
        at_risk = T >= T[i]
        risk_sum = exp_xb[at_risk].sum()
        nll -= (xb[i] - xb.max()) - np.log(risk_sum)
    return nll

def cox_gradient(beta, T, E, X):
    """Analytical gradient of negative partial log-likelihood."""
    beta = np.asarray(beta)
    n, p = X.shape
    xb   = X @ beta
    exp_xb = np.exp(xb - xb.max())
    grad = np.zeros(p)
    for i in np.where(E)[0]:
        at_risk = T >= T[i]
        w       = exp_xb[at_risk]
        w_sum   = w.sum()
        x_bar   = (X[at_risk].T @ w) / w_sum  # weighted mean of X
        grad   -= X[i] - x_bar
    return grad

def cox_hessian_approx(beta, T, E, X, eps=1e-5):
    """Finite-difference Hessian approximation for SE estimation."""
    p = len(beta)
    H = np.zeros((p, p))
    for i in range(p):
        e_i = np.zeros(p); e_i[i] = eps
        H[i, :] = (cox_gradient(beta + e_i, T, E, X) -
                   cox_gradient(beta - e_i, T, E, X)) / (2 * eps)
    return H

# ── Prepare data ──────────────────────────────────────────────────────────────
cox_feature_spec = {
    "used_agent":                  "Used Agent",
    "has_complete_dag":            "Built Complete DAG",
    "completed_onboarding_tour":   "Completed Onboarding Tour",
    "early_code_runner":           "Early Code Runner (≤60 min)",
    "multi_day_user":              "Multi-Day User",
    "pct_events_agent":            "Agent Event Proportion",
    "event_entropy":               "Event Diversity (entropy)",
    "advanced_run_ratio":          "Advanced Run Ratio",
    "pct_sessions_with_code_run":  "% Sessions w/ Code Run",
    "log_sessions_pw":             "Log(Sessions/Week)",
    "log_active_days":             "Log(Active Days)",
    "log_blocks_created":          "Log(Blocks Created)",
}

_cox_df = km_survival_df[["duration_days", "churn",
                           "used_agent", "has_complete_dag",
                           "completed_onboarding_tour", "early_code_runner",
                           "multi_day_user", "pct_events_agent",
                           "event_entropy", "advanced_run_ratio",
                           "pct_sessions_with_code_run",
                           "sessions_per_week", "total_active_days",
                           "num_blocks_created"]].copy()

_cox_df["log_sessions_pw"]    = np.log1p(_cox_df["sessions_per_week"].fillna(0))
_cox_df["log_active_days"]    = np.log1p(_cox_df["total_active_days"].fillna(0))
_cox_df["log_blocks_created"] = np.log1p(_cox_df["num_blocks_created"].fillna(0))
_cox_df = _cox_df.dropna()

print(f"\nCox dataset: {len(_cox_df):,} users  |  events: {_cox_df['churn'].sum():,}")

feat_cols = list(cox_feature_spec.keys())

# Standardize all features (binary and continuous) for stable optimization
X_raw = _cox_df[feat_cols].values.astype(float)
_means = X_raw.mean(axis=0)
_stds  = X_raw.std(axis=0)
_stds[_stds == 0] = 1.0
X_std  = (X_raw - _means) / _stds

T = _cox_df["duration_days"].values.astype(float)
E = _cox_df["churn"].values.astype(bool)

# Sort by time for efficient risk set computation
_order = np.argsort(T)
T, E, X_std, X_raw = T[_order], E[_order], X_std[_order], X_raw[_order]

print("Fitting Cox PHM via L-BFGS-B partial likelihood...")
p = X_std.shape[1]
beta0 = np.zeros(p)

result = optimize.minimize(
    cox_neg_partial_loglik, beta0,
    args=(T, E, X_std),
    jac=cox_gradient,
    method="L-BFGS-B",
    options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7}
)

beta_hat = result.x
print(f"  Optimization {'converged ✅' if result.success else 'did not converge ⚠️'}")
print(f"  Negative log-likelihood: {result.fun:.4f}")

# Standard errors via finite-difference Hessian
print("  Computing SEs via Hessian...")
H = cox_hessian_approx(beta_hat, T, E, X_std)
var_diag = np.diag(np.linalg.pinv(H))
se_hat = np.sqrt(np.abs(var_diag))

# Hazard ratios + CIs + Wald test
z_scores = beta_hat / se_hat
pvals    = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
ci_lo    = beta_hat - 1.96 * se_hat
ci_hi    = beta_hat + 1.96 * se_hat

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

cox_hr_table = pd.DataFrame({
    "Feature":   [cox_feature_spec[f] for f in feat_cols],
    "Coef (β)":  beta_hat.round(4),
    "HR":        np.exp(beta_hat).round(4),
    "HR_lo95":   np.exp(ci_lo).round(4),
    "HR_hi95":   np.exp(ci_hi).round(4),
    "SE":        se_hat.round(4),
    "z":         z_scores.round(3),
    "p-value":   pvals.round(6),
    "Sig":       [sig_stars(p) for p in pvals],
}, index=feat_cols)

cox_hr_table = cox_hr_table.sort_values("Coef (β)", ascending=False)

# ── Print Results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("HAZARD RATIOS — CHURN RISK (Cox Proportional Hazards Model)")
print("HR < 1 = protective  |  HR > 1 = risk factor  |  Standardized coefficients")
print("=" * 78)
print(f"  {'Feature':<38} {'HR':>7} {'95% CI Lower':>13} {'95% CI Upper':>13} {'p-value':>10} {'Sig':>4}")
print("  " + "-" * 76)
for _, row in cox_hr_table.iterrows():
    print(f"  {row['Feature']:<38} {row['HR']:>7.3f} {row['HR_lo95']:>13.3f} {row['HR_hi95']:>13.3f} {row['p-value']:>10.5f} {row['Sig']:>4}")

# Concordance (C-statistic) approximation via concordant pairs
def concordance(T, E, X, beta):
    xb = X @ beta
    conc, disc, tied = 0, 0, 0
    ev_idx = np.where(E)[0]
    for i in ev_idx:
        at_risk = np.where(T >= T[i])[0]
        at_risk = at_risk[at_risk != i]
        for j in at_risk:
            if xb[i] > xb[j]:    conc += 1
            elif xb[i] < xb[j]:  disc += 1
            else:                  tied += 1
    total = conc + disc + tied
    return (conc + 0.5 * tied) / total if total > 0 else np.nan

# Quick C-stat on subsample for speed
_sample_idx = np.random.RandomState(42).choice(len(T), min(500, len(T)), replace=False)
c_stat = concordance(T[_sample_idx], E[_sample_idx], X_std[_sample_idx], beta_hat)
print(f"\n  Concordance C-statistic (n=500 subsample): {c_stat:.4f}")

# ── Forest Plot ───────────────────────────────────────────────────────────────
_sorted = cox_hr_table.sort_values("HR")
n_feats = len(_sorted)

fig_cox, ax_hr = plt.subplots(figsize=(11, max(6.5, n_feats * 0.62 + 2)))
fig_cox.patch.set_facecolor(BG)
ax_hr.set_facecolor(BG)

for i, (feat_name, row) in enumerate(_sorted.iterrows()):
    color = C3 if row["HR"] < 1.0 else C4
    alpha = 1.0 if row["Sig"] != "ns" else 0.5
    ax_hr.scatter(row["HR"], i, color=color, s=100, zorder=4, alpha=alpha)
    ax_hr.hlines(i, row["HR_lo95"], row["HR_hi95"],
                 color=color, linewidth=2.8, alpha=alpha * 0.75)

ax_hr.axvline(1.0, color=GOLD, linestyle="--", linewidth=1.8, alpha=0.9)
ax_hr.set_yticks(range(n_feats))
ax_hr.set_yticklabels(_sorted["Feature"].tolist(), color=FG, fontsize=9.5)
ax_hr.set_xlabel("Hazard Ratio (HR) — 95% CI  [standardized coefficients]",
                 color=FG2, fontsize=10)
ax_hr.set_title("Cox Proportional Hazards — Churn Risk Factors\n"
                "HR < 1 (green) = protective  ·  HR > 1 (coral) = risk  ·  faded = n.s.",
                color=FG, fontsize=13, fontweight="bold")
ax_hr.tick_params(colors=FG2, labelsize=9)
for sp in ax_hr.spines.values():
    sp.set_edgecolor("#444")
ax_hr.grid(axis="x", color="#333", linewidth=0.5, alpha=0.6)

# Significance annotations to the right
_x_annot = _sorted["HR_hi95"].max() * 1.08
for i, (feat_name, row) in enumerate(_sorted.iterrows()):
    ax_hr.annotate(row["Sig"], xy=(_x_annot, i), xycoords="data",
                   fontsize=9, color=FG2, va="center", ha="left")

legend_patches = [
    mpatches.Patch(color=C3, label="Protective (HR < 1, p < 0.05)"),
    mpatches.Patch(color=C4, label="Risk factor (HR > 1, p < 0.05)"),
    mpatches.Patch(color=GOLD, label="HR = 1 (null)"),
]
ax_hr.legend(handles=legend_patches, loc="lower right", framealpha=0.2,
             labelcolor=FG, facecolor=BG, edgecolor="#555", fontsize=8.5)

plt.tight_layout()
cox_forest_plot = fig_cox
print("\n✅ Cox PHM forest plot ready.")

# ── Key Findings ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KEY HAZARD RATIO FINDINGS (significant, p < 0.05)")
print("=" * 65)
_sig_rows = cox_hr_table[cox_hr_table["Sig"] != "ns"].sort_values("HR")
for _, row in _sig_rows.iterrows():
    direction = "PROTECTIVE ↓ churn" if row["HR"] < 1 else "RISK FACTOR ↑ churn"
    pct_change = abs(row["HR"] - 1) * 100
    print(f"  {row['Feature']:<38}  HR={row['HR']:.3f}  ({pct_change:.0f}% change)  {direction}  {row['Sig']}")

_non_sig = cox_hr_table[cox_hr_table["Sig"] == "ns"]
if len(_non_sig) > 0:
    print(f"\n  Non-significant (p≥0.05): {', '.join(_non_sig['Feature'].tolist())}")
