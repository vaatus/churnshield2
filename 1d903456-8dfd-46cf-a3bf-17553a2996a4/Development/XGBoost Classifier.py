"""
Gradient Boosting — Long-Term Success Prediction
=================================================
Algorithm: HistGradientBoostingClassifier (sklearn 1.2+)
           — equivalent performance to LightGBM/XGBoost, fastest sklearn estimator
Target   : long_term_success (composite label: ≥2 of 3 criteria)
Features : 74 engineered features (no leakage from label components)

Protocol
--------
1. Load feature_matrix from long_term_success_features.parquet
2. Drop label-component columns to avoid leakage
3. class_weight='balanced' to handle 3.2% positive class
4. 5-fold stratified CV — AUC-ROC, PR-AUC, F1
5. 80/20 holdout test set evaluation with bootstrap 95% CI on AUC
6. F1-optimal threshold from PR curve
7. ROC curve, Precision-Recall curve, metrics comparison chart
8. Save hgb_lts_model.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
)

os.makedirs("figures", exist_ok=True)

# ── Zerve design system ───────────────────────────────────────────────────────
BG      = "#1D1D20"
FG      = "#fbfbff"
SG      = "#909094"
HI      = "#ffd400"
OK      = "#17b26a"
WARN    = "#f04438"
BLUE    = "#A1C9F4"
GREEN   = "#8DE5A1"

print("=" * 70)
print("HistGradientBoosting — Long-Term Success Prediction")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════════════════════

hgb_fm = pd.read_parquet("long_term_success_features.parquet")
print(f"\nLoaded feature_matrix: {hgb_fm.shape[0]:,} users × {hgb_fm.shape[1]} columns")

hgb_y = hgb_fm["long_term_success"].values.astype(int)

LABEL_COLS = {"long_term_success", "retained_28d", "multi_week_3plus",
              "workflow_builder", "lts_score"}
hgb_feature_names = [c for c in hgb_fm.columns
                     if c not in LABEL_COLS
                     and pd.api.types.is_numeric_dtype(hgb_fm[c])]

hgb_X = hgb_fm[hgb_feature_names].fillna(0).values.astype(np.float64)
hgb_X = np.clip(hgb_X, -1e9, 1e9)

n_pos = int(hgb_y.sum())
n_neg = len(hgb_y) - n_pos
pos_rate = hgb_y.mean()

print(f"\nFeatures: {len(hgb_feature_names)}")
print(f"Class balance: {n_pos:,} success ({pos_rate:.1%}) / {n_neg:,} non-success")
print(f"Using class_weight='balanced' (auto scale_pos_weight)")

# ════════════════════════════════════════════════════════════════════════════
# 2. TRAIN/TEST SPLIT
# ════════════════════════════════════════════════════════════════════════════

hgb_X_train, hgb_X_test, hgb_y_train, hgb_y_test = train_test_split(
    hgb_X, hgb_y, test_size=0.20, random_state=42, stratify=hgb_y
)
print(f"\nTrain: {len(hgb_y_train):,}  |  Test: {len(hgb_y_test):,}")

# ════════════════════════════════════════════════════════════════════════════
# 3. HELPER: compute class weights for sample_weight
# ════════════════════════════════════════════════════════════════════════════
def _make_weights(y):
    """Balanced class weights as sample_weight array."""
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[c] for c in y])

# ════════════════════════════════════════════════════════════════════════════
# 4. 5-FOLD STRATIFIED CV
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("5-Fold Stratified Cross-Validation (on train set)")
print("─" * 70)

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs, cv_aps, cv_f1s, cv_precs, cv_recs = [], [], [], [], []

for _fold, (_tr_idx, _val_idx) in enumerate(cv_splitter.split(hgb_X_train, hgb_y_train), 1):
    _Xtr, _Xval = hgb_X_train[_tr_idx], hgb_X_train[_val_idx]
    _ytr, _yval = hgb_y_train[_tr_idx], hgb_y_train[_val_idx]
    _sw = _make_weights(_ytr)

    _fold_model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )
    _fold_model.fit(_Xtr, _ytr, sample_weight=_sw)
    _yval_prob = _fold_model.predict_proba(_Xval)[:, 1]

    _fold_auc = roc_auc_score(_yval, _yval_prob)
    _fold_ap  = average_precision_score(_yval, _yval_prob)
    _prec_c, _rec_c, _thr_c = precision_recall_curve(_yval, _yval_prob)
    _f1_c = 2 * _prec_c * _rec_c / (_prec_c + _rec_c + 1e-9)
    _best_thr_c = float(_thr_c[np.argmax(_f1_c[:-1])]) if len(_thr_c) > 0 else 0.5
    _yval_opt = (_yval_prob >= _best_thr_c).astype(int)
    _fold_f1  = f1_score(_yval, _yval_opt, zero_division=0)
    _fold_p   = precision_score(_yval, _yval_opt, zero_division=0)
    _fold_r   = recall_score(_yval, _yval_opt, zero_division=0)

    cv_aucs.append(_fold_auc)
    cv_aps.append(_fold_ap)
    cv_f1s.append(_fold_f1)
    cv_precs.append(_fold_p)
    cv_recs.append(_fold_r)
    print(f"  Fold {_fold}: AUC={_fold_auc:.4f}  AP={_fold_ap:.4f}  F1={_fold_f1:.4f}  P={_fold_p:.4f}  R={_fold_r:.4f}")

cv_auc_mean  = np.mean(cv_aucs);  cv_auc_std  = np.std(cv_aucs)
cv_ap_mean   = np.mean(cv_aps);   cv_f1_mean  = np.mean(cv_f1s)
cv_prec_mean = np.mean(cv_precs); cv_rec_mean = np.mean(cv_recs)

print(f"\n  CV Summary:")
print(f"    AUC-ROC:        {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
print(f"    Avg Precision:  {cv_ap_mean:.4f} ± {np.std(cv_aps):.4f}")
print(f"    F1 (opt thr):   {cv_f1_mean:.4f} ± {np.std(cv_f1s):.4f}")
print(f"    Precision:      {cv_prec_mean:.4f} ± {np.std(cv_precs):.4f}")
print(f"    Recall:         {cv_rec_mean:.4f} ± {np.std(cv_recs):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 5. FINAL MODEL + TEST SET EVALUATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Training final model on full train set...")

_sw_train = _make_weights(hgb_y_train)
hgb_model = HistGradientBoostingClassifier(
    max_iter=300, max_depth=6, learning_rate=0.05,
    min_samples_leaf=10, l2_regularization=0.1, random_state=42,
)
hgb_model.fit(hgb_X_train, hgb_y_train, sample_weight=_sw_train)
hgb_y_prob = hgb_model.predict_proba(hgb_X_test)[:, 1]

_prec_t, _rec_t, _thr_t = precision_recall_curve(hgb_y_test, hgb_y_prob)
_f1_t = 2 * _prec_t * _rec_t / (_prec_t + _rec_t + 1e-9)
hgb_best_thr = float(_thr_t[np.argmax(_f1_t[:-1])]) if len(_thr_t) > 0 else 0.5
hgb_y_pred   = (hgb_y_prob >= hgb_best_thr).astype(int)

def _bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

test_auc  = roc_auc_score(hgb_y_test, hgb_y_prob)
test_ap   = average_precision_score(hgb_y_test, hgb_y_prob)
test_f1   = f1_score(hgb_y_test, hgb_y_pred, zero_division=0)
test_prec = precision_score(hgb_y_test, hgb_y_pred, zero_division=0)
test_rec  = recall_score(hgb_y_test, hgb_y_pred, zero_division=0)
auc_lo, auc_hi = _bootstrap_auc_ci(hgb_y_test, hgb_y_prob)
hgb_cm = confusion_matrix(hgb_y_test, hgb_y_pred)
cv_test_gap = cv_auc_mean - test_auc

print(f"\nTest Set Results (20% holdout, threshold={hgb_best_thr:.3f}):")
print(f"  AUC-ROC:       {test_auc:.4f}  [95% CI: {auc_lo:.4f}–{auc_hi:.4f}]")
print(f"  Avg Precision: {test_ap:.4f}")
print(f"  F1:            {test_f1:.4f}")
print(f"  Precision:     {test_prec:.4f}")
print(f"  Recall:        {test_rec:.4f}")
print(f"  CV→Test gap:   {cv_test_gap:+.4f}")
print(f"\nConfusion Matrix:\n{hgb_cm}")

if test_auc > 0.95:
    print(f"\n  ⚠️  AUC > 0.95 — possible leakage. Check top features.")

# ════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODEL
# ════════════════════════════════════════════════════════════════════════════
hgb_results = {
    "n_pos": n_pos, "n_neg": n_neg, "pos_rate": float(pos_rate),
    "cv_auc": cv_auc_mean, "cv_auc_std": cv_auc_std,
    "cv_ap": cv_ap_mean, "cv_f1": cv_f1_mean, "cv_prec": cv_prec_mean, "cv_rec": cv_rec_mean,
    "test_auc": test_auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
    "test_ap": test_ap, "test_f1": test_f1, "test_prec": test_prec, "test_rec": test_rec,
    "threshold": hgb_best_thr, "cv_test_gap": cv_test_gap,
}
with open("hgb_lts_model.pkl", "wb") as _fh:
    pickle.dump({
        "model": hgb_model,
        "feature_names": hgb_feature_names,
        "results": hgb_results,
        "y_test": hgb_y_test,
        "y_prob": hgb_y_prob,
    }, _fh)
print("\n✓ Saved hgb_lts_model.pkl")

# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

# ── ROC Curve ─────────────────────────────────────────────────────────────
hgb_roc_fig = plt.figure(figsize=(9, 7))
hgb_roc_fig.patch.set_facecolor(BG)
_ax_roc = hgb_roc_fig.add_subplot(111)
_ax_roc.set_facecolor(BG)
_fpr, _tpr, _ = roc_curve(hgb_y_test, hgb_y_prob)
_ax_roc.plot([0, 1], [0, 1], color=SG, linestyle="--", linewidth=1.5, label="Random (AUC=0.50)")
_ax_roc.plot(_fpr, _tpr, color=BLUE, linewidth=2.8,
             label=f"HistGradientBoosting\nAUC={test_auc:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]")
_ax_roc.set_xlabel("False Positive Rate", color=FG, fontsize=12)
_ax_roc.set_ylabel("True Positive Rate", color=FG, fontsize=12)
_ax_roc.set_title(
    "ROC Curve — Gradient Boosting Long-Term Success Classifier\n"
    f"5-Fold CV AUC={cv_auc_mean:.4f}±{cv_auc_std:.4f}  |  Test AUC={test_auc:.4f}  [95% CI]",
    color=FG, fontsize=12, pad=12, fontweight="bold"
)
_ax_roc.tick_params(colors=FG)
_ax_roc.spines[:].set_color(SG)
_ax_roc.legend(facecolor=BG, edgecolor=SG, labelcolor=FG, fontsize=10, loc="lower right")
_ax_roc.grid(color=SG, alpha=0.15, linestyle="--")
hgb_roc_fig.tight_layout()
hgb_roc_fig.savefig("figures/hgb_roc_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✓ Saved figures/hgb_roc_curve.png")

# ── Precision-Recall Curve ────────────────────────────────────────────────
hgb_pr_fig = plt.figure(figsize=(9, 7))
hgb_pr_fig.patch.set_facecolor(BG)
_ax_pr = hgb_pr_fig.add_subplot(111)
_ax_pr.set_facecolor(BG)
_ax_pr.plot(_rec_t, _prec_t, color=GREEN, linewidth=2.8,
            label=f"HistGradientBoosting (AP={test_ap:.4f})")
_ax_pr.axhline(hgb_y_test.mean(), color=SG, linestyle=":", linewidth=1.5,
               label=f"Random baseline ({hgb_y_test.mean():.1%})")
_ax_pr.axvline(test_rec, color=HI, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Opt threshold ({hgb_best_thr:.3f}): P={test_prec:.3f}, R={test_rec:.3f}")
_ax_pr.set_xlabel("Recall", color=FG, fontsize=12)
_ax_pr.set_ylabel("Precision", color=FG, fontsize=12)
_ax_pr.set_title(
    "Precision-Recall Curve — Gradient Boosting Long-Term Success\n"
    f"Avg Precision={test_ap:.4f}  |  F1={test_f1:.4f} at threshold={hgb_best_thr:.3f}",
    color=FG, fontsize=12, pad=12, fontweight="bold"
)
_ax_pr.tick_params(colors=FG)
_ax_pr.spines[:].set_color(SG)
_ax_pr.legend(facecolor=BG, edgecolor=SG, labelcolor=FG, fontsize=9, loc="upper right")
_ax_pr.grid(color=SG, alpha=0.15, linestyle="--")
hgb_pr_fig.tight_layout()
hgb_pr_fig.savefig("figures/hgb_pr_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✓ Saved figures/hgb_pr_curve.png")

# ── CV vs Test Metrics Bar ────────────────────────────────────────────────
hgb_metrics_fig = plt.figure(figsize=(11, 6))
hgb_metrics_fig.patch.set_facecolor(BG)
_ax_m = hgb_metrics_fig.add_subplot(111)
_ax_m.set_facecolor(BG)
_metrics_nm = ["AUC-ROC", "Avg Prec", "F1", "Precision", "Recall"]
_cv_vals   = [cv_auc_mean, cv_ap_mean, cv_f1_mean, cv_prec_mean, cv_rec_mean]
_test_vals = [test_auc, test_ap, test_f1, test_prec, test_rec]
_x_pos = np.arange(len(_metrics_nm));  _w = 0.35
_bars_cv   = _ax_m.bar(_x_pos - _w/2, _cv_vals,   _w, label="5-Fold CV", color=BLUE,  alpha=0.88, edgecolor=BG)
_bars_test = _ax_m.bar(_x_pos + _w/2, _test_vals, _w, label="Test (20%)", color=GREEN, alpha=0.88, edgecolor=BG)
for _b, _v in list(zip(_bars_cv, _cv_vals)) + list(zip(_bars_test, _test_vals)):
    if _v > 0.02:
        _ax_m.text(_b.get_x() + _b.get_width()/2, _v + 0.01, f"{_v:.3f}",
                   ha="center", va="bottom", fontsize=8, color=FG)
_ax_m.axhline(0.5, color=HI, linestyle="--", linewidth=1.2, alpha=0.6, label="AUC baseline (0.5)")
_ax_m.set_xticks(_x_pos)
_ax_m.set_xticklabels(_metrics_nm, color=FG, fontsize=11)
_ax_m.set_ylim(0, 1.15)
_ax_m.set_ylabel("Score", color=FG, fontsize=11)
_ax_m.set_title(
    "Gradient Boosting Long-Term Success — CV vs Test Performance\n"
    f"n={len(hgb_y):,} users  |  {n_pos:,} positive ({pos_rate:.1%})  |  class_weight=balanced",
    color=FG, fontsize=12, pad=12, fontweight="bold"
)
_ax_m.tick_params(colors=FG)
_ax_m.spines[:].set_color(SG)
_ax_m.legend(facecolor=BG, edgecolor=SG, labelcolor=FG, fontsize=10)
_ax_m.grid(axis="y", color=SG, alpha=0.15, linestyle="--")
hgb_metrics_fig.tight_layout()
hgb_metrics_fig.savefig("figures/hgb_metrics_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✓ Saved figures/hgb_metrics_comparison.png")

# ── Confusion Matrix ──────────────────────────────────────────────────────
hgb_cm_fig = plt.figure(figsize=(6, 5))
hgb_cm_fig.patch.set_facecolor(BG)
_ax_cm = hgb_cm_fig.add_subplot(111)
_ax_cm.set_facecolor(BG)
_disp = ConfusionMatrixDisplay(confusion_matrix=hgb_cm, display_labels=["Not Success", "Success"])
_disp.plot(ax=_ax_cm, colorbar=False, cmap="Blues")
_ax_cm.set_facecolor(BG);  _ax_cm.xaxis.label.set_color(FG);  _ax_cm.yaxis.label.set_color(FG)
_ax_cm.tick_params(colors=FG)
for _txt in _ax_cm.texts:
    _txt.set_color(FG)
_ax_cm.set_title(f"Confusion Matrix — Gradient Boosting LTS\nThreshold = {hgb_best_thr:.3f}",
                  color=FG, fontsize=11, pad=10)
hgb_cm_fig.tight_layout()
hgb_cm_fig.savefig("figures/hgb_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("✓ Saved figures/hgb_confusion_matrix.png")
plt.close("all")

# ════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GRADIENT BOOSTING — FINAL RESULTS")
print("=" * 70)
print(f"  Algorithm:     HistGradientBoostingClassifier (sklearn {HistGradientBoostingClassifier.__module__})")
print(f"  Target:        long_term_success (≥2/3: 28d retention + multi-week + DAG)")
print(f"  Users:         {len(hgb_y):,}  |  Positive: {n_pos:,} ({pos_rate:.1%})")
print(f"  Features:      {len(hgb_feature_names)}")
print(f"\n  ── Cross-Validation (5-fold, stratified) ──")
print(f"  AUC-ROC:       {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
print(f"  Avg Precision: {cv_ap_mean:.4f}")
print(f"  F1:            {cv_f1_mean:.4f}")
print(f"  Precision:     {cv_prec_mean:.4f}")
print(f"  Recall:        {cv_rec_mean:.4f}")
print(f"\n  ── Test Set (20% holdout) ──")
print(f"  AUC-ROC:       {test_auc:.4f}  [95% CI: {auc_lo:.4f}–{auc_hi:.4f}]")
print(f"  Avg Precision: {test_ap:.4f}")
print(f"  F1:            {test_f1:.4f}  (threshold={hgb_best_thr:.3f})")
print(f"  Precision:     {test_prec:.4f}")
print(f"  Recall:        {test_rec:.4f}")
print(f"  CV→Test gap:   {cv_test_gap:+.4f}")
print("=" * 70)
