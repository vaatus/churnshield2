# =====================================================
# ZERVE BLOCK: Temporal Prediction Model
# Part of ChurnShield — Zerve Hackathon 2026
# =====================================================
# Trains on first-week features, predicts week 2-3 retention.
# This is the honest temporal split (CV AUC ~0.82), replacing
# the leaky AUC=1.0 model.
# =====================================================

import warnings
warnings.filterwarnings("ignore")

import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.stats import entropy as shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Zerve dark theme
plt.rcParams.update({
    "figure.facecolor": "#1D1D20", "axes.facecolor": "#1D1D20",
    "axes.edgecolor": "#909094", "axes.labelcolor": "#fbfbff",
    "text.color": "#fbfbff", "xtick.color": "#909094",
    "ytick.color": "#909094", "grid.color": "#2a2a2e",
    "legend.facecolor": "#1D1D20", "legend.edgecolor": "#909094",
})
PALETTE = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF6B6B", "#C4A1F4"]
ACCENT = "#ffd400"

# ── 0. Load cleaned events ──────────────────────────────────────────────────
print("=" * 60)
print("TEMPORAL MODEL: Loading data...")

# Try parquet first, fall back to CSV
if os.path.exists("events_clean.parquet"):
    df = pd.read_parquet("events_clean.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
else:
    print("  events_clean.parquet not found, loading from raw CSV...")
    RAW_CSV = "zerve_hackathon_for_reviewc8fa7c7.csv"
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    COLUMN_MAP = {
        "distinct_id": "user_id", "event": "event",
        "timestamp": "timestamp", "prop_surface": "surface",
    }
    keep = [c for c in COLUMN_MAP if c in raw.columns]
    df = raw[keep].rename(columns=COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["user_id"])

    # Event categories
    AGENT_EVENTS = {
        "agent_block_created", "agent_worker_created",
        "agent_tool_call_run_block_tool", "agent_tool_call_create_block_tool",
        "agent_tool_call_get_block_tool", "agent_tool_call_refactor_block_tool",
        "agent_tool_call_finish_ticket_tool", "agent_tool_call_get_canvas_summary_tool",
        "agent_tool_call_get_variable_preview_tool", "agent_tool_call_delete_block_tool",
        "agent_tool_call_create_edges_tool", "agent_message",
        "agent_start_from_prompt", "agent_accept_suggestion",
        "agent_new_chat", "agent_open", "agent_suprise_me",
        "agent_open_error_assist", "agent_upload_files",
    }
    EVENT_CATEGORIES = {
        "agent": list(AGENT_EVENTS),
        "code_execution": ["run_block", "run_all_blocks", "run_upto_block", "run_from_block", "stop_block"],
        "creation": ["block_create", "canvas_create", "edge_create", "layer_create"],
        "modification": ["block_delete", "block_resize", "edge_delete", "block_rename", "block_copy"],
        "navigation": ["canvas_open", "fullscreen_open", "fullscreen_close"],
        "auth": ["sign_in", "sign_up"],
        "onboarding": ["skip_onboarding_form"],
        "file_ops": ["files_upload"],
        "credits": ["credits_used", "addon_credits_used", "credits_below_1",
                    "credits_below_2", "credits_below_3", "credits_below_4", "credits_exceeded"],
    }
    e2c = {}
    for cat, evts in EVENT_CATEGORIES.items():
        for e in evts:
            e2c[e] = cat
    df["event_category"] = df["event"].map(e2c).fillna("other")

df["date"] = df["timestamp"].dt.date
print(f"  {len(df):,} events, {df['user_id'].nunique()} users, {df['date'].min()} to {df['date'].max()}")

# ── 1. Per-user temporal split ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Per-user temporal split (first 7 days -> predict days 8-21)")

user_first = df.groupby("user_id")["timestamp"].min().reset_index()
user_first.columns = ["user_id", "first_seen"]
df = df.merge(user_first, on="user_id")
df["days_since_first"] = (df["timestamp"] - df["first_seen"]).dt.total_seconds() / 86400

OBS_DAYS = 7
TARGET_START = 7
TARGET_END = 21

max_ts = df["timestamp"].max()
user_first_ts = df.groupby("user_id")["timestamp"].min()
eligible = user_first_ts[(max_ts - user_first_ts).dt.total_seconds() / 86400 >= TARGET_END].index

obs_events = df[(df["user_id"].isin(eligible)) & (df["days_since_first"] < OBS_DAYS)]
user_obs_counts = obs_events.groupby("user_id").size()
qualified = set(user_obs_counts[user_obs_counts >= 3].index)

obs_df = df[(df["user_id"].isin(qualified)) & (df["days_since_first"] < OBS_DAYS)].copy()
target_df = df[(df["user_id"].isin(qualified)) &
               (df["days_since_first"] >= TARGET_START) &
               (df["days_since_first"] < TARGET_END)].copy()

print(f"  Qualified users: {len(qualified)}")
print(f"  Observation events: {len(obs_df):,}")
print(f"  Target window events: {len(target_df):,}")

# ── 2. Sessionize & build features ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature engineering (observation window only)")

obs_df = obs_df.sort_values(["user_id", "timestamp"])
obs_df["time_diff"] = obs_df.groupby("user_id")["timestamp"].diff()
obs_df["new_session"] = (
    obs_df["time_diff"].isna() |
    (obs_df["time_diff"] > pd.Timedelta(minutes=30))
).astype(int)
obs_df["session_seq"] = obs_df.groupby("user_id")["new_session"].cumsum()


def compute_features(ue):
    uid = ue["user_id"].iloc[0]
    n = len(ue)
    events = ue["event"]
    cats = ue["event_category"]
    ts = ue["timestamp"]

    cc = cats.value_counts()
    ec = events.value_counts()
    n_agent = cc.get("agent", cc.get("agent_use", 0))
    n_code = cc.get("code_execution", 0)
    n_creation = cc.get("creation", 0)
    n_nav = cc.get("navigation", 0)

    n_blocks_run = ec.get("run_block", 0)
    n_blocks_created = ec.get("block_create", 0)
    n_canvases = ec.get("canvas_create", 0)
    n_edges = ec.get("edge_create", 0)
    n_files = ec.get("files_upload", 0)
    n_agent_prompts = ec.get("agent_start_from_prompt", 0)
    n_agent_msg = ec.get("agent_message", 0)
    n_agent_acc = ec.get("agent_accept_suggestion", 0)
    n_agent_workers = ec.get("agent_worker_created", 0)

    ep = events.value_counts(normalize=True).values
    entropy = float(shannon_entropy(ep, base=2))
    tenure_h = (ts.max() - ts.min()).total_seconds() / 3600
    active_days = ue["date"].nunique()
    first_ts = ts.min()

    # time-to-first
    def ttf(mask):
        sub = ue[mask]["timestamp"]
        return (sub.min() - first_ts).total_seconds() / 60 if len(sub) > 0 else -1

    ttf_code = ttf(ue["event"] == "run_block")
    ttf_agent = ttf(ue["event_category"].isin(["agent", "agent_use"]))
    ttf_edge = ttf(ue["event"] == "edge_create")

    # Sessions
    sessions = ue.groupby("session_seq")
    n_sess = sessions.ngroups
    s_durs = np.array([(s["timestamp"].max() - s["timestamp"].min()).total_seconds() / 60 for _, s in sessions])

    fs = ue[ue["session_seq"] == 1]
    fs_dur = (fs["timestamp"].max() - fs["timestamp"].min()).total_seconds() / 60
    fs_n = len(fs)
    fs_unique = fs["event"].nunique()
    fs_ran = int("run_block" in fs["event"].values)
    fs_agent = int(fs["event_category"].isin(["agent", "agent_use"]).any())
    fs_canvas = int("canvas_create" in fs["event"].values)
    fs_edge = int("edge_create" in fs["event"].values)
    fs_intensity = fs_n / max(fs_dur, 0.1)

    td = ts.diff().dropna().dt.total_seconds() / 60
    mean_ie = td.mean() if len(td) > 0 else 0
    longest_gap = td.max() if len(td) > 0 else 0

    hc = ue.set_index("timestamp").resample("h").size()
    hc = hc[hc > 0]
    burst = hc.max() / hc.mean() if len(hc) > 1 else 1.0

    hours = ts.dt.hour
    peak_h = hours.mode().iloc[0] if len(hours) > 0 else 12
    pct_day = ((hours >= 8) & (hours < 18)).mean()

    # Activity trend
    if active_days >= 2:
        dc = ue.groupby(ue["timestamp"].dt.date).size()
        if len(dc) >= 2:
            x = np.arange(len(dc))
            trend = np.polyfit(x, dc.values, 1)[0]
        else:
            trend = 0
    else:
        trend = 0

    return {
        "user_id": uid,
        "total_events": n, "total_sessions": n_sess,
        "unique_event_types": events.nunique(), "unique_categories": cats.nunique(),
        "active_days": active_days, "multi_day_user": int(active_days >= 2),
        "used_agent": int(n_agent > 0), "num_agent_events": n_agent,
        "pct_agent_events": n_agent / n,
        "num_agent_prompts": n_agent_prompts, "num_agent_accepted": n_agent_acc,
        "agent_acceptance_rate": n_agent_acc / max(n_agent_prompts + n_agent_msg, 1),
        "num_agent_tool_types": ue[ue["event"].str.startswith("agent_tool_call_")]["event"].nunique(),
        "num_agent_workers": n_agent_workers,
        "ran_code": int(n_blocks_run > 0), "num_blocks_run": n_blocks_run,
        "pct_code_events": n_code / n, "block_run_rate": n_blocks_run / max(n_blocks_created, 1),
        "code_run_per_session": n_blocks_run / max(n_sess, 1),
        "num_blocks_created": n_blocks_created, "num_canvases_created": n_canvases,
        "created_canvas": int(n_canvases > 0), "num_edges_created": n_edges,
        "has_edges": int(n_edges > 0), "edge_per_canvas": n_edges / max(n_canvases, 1),
        "num_files_uploaded": n_files, "has_file_uploads": int(n_files > 0),
        "first_session_duration_min": fs_dur, "first_session_events": fs_n,
        "first_session_unique_types": fs_unique, "first_session_ran_code": fs_ran,
        "first_session_used_agent": fs_agent, "first_session_created_canvas": fs_canvas,
        "first_session_created_edge": fs_edge, "first_session_intensity": fs_intensity,
        "tenure_hours": tenure_h,
        "time_to_first_code_run_min": ttf_code, "time_to_first_agent_use_min": ttf_agent,
        "time_to_first_edge_min": ttf_edge,
        "mean_inter_event_min": mean_ie, "longest_gap_min": longest_gap,
        "burst_score": burst, "peak_hour": peak_h, "pct_daytime": pct_day,
        "event_entropy": entropy,
        "skipped_onboarding": int("skip_onboarding_form" in events.values),
        "avg_session_duration_min": s_durs.mean(), "max_session_duration_min": s_durs.max(),
        "std_session_duration_min": s_durs.std() if len(s_durs) > 1 else 0,
        "events_per_session": n / max(n_sess, 1),
        "pct_creation_events": n_creation / n, "pct_navigation_events": n_nav / n,
        "activity_trend": trend,
    }


features_list = [compute_features(grp) for _, grp in obs_df.groupby("user_id")]
features_df = pd.DataFrame(features_list)
print(f"  Feature matrix: {features_df.shape[0]} users x {features_df.shape[1]} cols")

# ── 3. Build targets ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building retention targets (days 8-21)")

tgt_by_user = target_df.groupby("user_id")
targets = []
for uid in features_df["user_id"]:
    if uid in tgt_by_user.groups:
        uf = tgt_by_user.get_group(uid)
        targets.append({"user_id": uid, "retained_week2": 1,
                        "future_events": len(uf), "future_active_days": uf["date"].nunique()})
    else:
        targets.append({"user_id": uid, "retained_week2": 0,
                        "future_events": 0, "future_active_days": 0})

targets_df = pd.DataFrame(targets)
med = targets_df.loc[targets_df["future_events"] > 0, "future_events"].median()
if pd.isna(med):
    med = 0
targets_df["is_power_user_future"] = (
    (targets_df["future_events"] > med) & (targets_df["retained_week2"] == 1)
).astype(int)

ret_rate = targets_df["retained_week2"].mean()
print(f"  Retention rate: {ret_rate:.1%} ({targets_df['retained_week2'].sum()}/{len(targets_df)})")

features_df.to_parquet("data/day1_features.parquet", index=False)
targets_df.to_parquet("data/temporal_targets.parquet", index=False)

# ── 4. Train models ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training models")

model_df = features_df.merge(targets_df, on="user_id")
feat_cols = [c for c in features_df.columns if c != "user_id"]

for c in feat_cols:
    if "time_to_first" in c:
        valid = model_df.loc[model_df[c] >= 0, c]
        fill = valid.max() * 1.5 if len(valid) > 0 else 9999
        model_df[c] = model_df[c].replace(-1, fill)

model_df[feat_cols] = model_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

TARGET = "retained_week2"
X = model_df[feat_cols].values
y = model_df[TARGET].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_pos = int(y.sum())
n_neg = len(y) - n_pos
spw = n_neg / max(n_pos, 1)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED, C=0.1),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=spw, eval_metric="logloss", random_state=SEED,
                                  reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    scale_pos_weight=spw, random_state=SEED,
                                    reg_alpha=1.0, reg_lambda=2.0, min_child_samples=5, verbose=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = {}

for name, mdl in models.items():
    Xu = X_scaled if name == "LogisticRegression" else X
    auc_cv = cross_val_score(mdl, Xu, y, cv=cv, scoring="roc_auc")
    f1_cv = cross_val_score(mdl, Xu, y, cv=cv, scoring="f1")
    mdl.fit(Xu, y)
    yp = mdl.predict_proba(Xu)[:, 1]
    t_auc = roc_auc_score(y, yp)
    scores[name] = {"cv_auc": auc_cv.mean(), "cv_std": auc_cv.std(),
                    "cv_f1": f1_cv.mean(), "train_auc": t_auc,
                    "model": mdl, "y_prob": yp, "X": Xu}
    gap = t_auc - auc_cv.mean()
    print(f"  {name:20s}: CV AUC = {auc_cv.mean():.3f} +/- {auc_cv.std():.3f} | Train = {t_auc:.3f} | Gap = {gap:.3f}")

# Ensemble
top3 = sorted(scores, key=lambda n: scores[n]["cv_auc"], reverse=True)[:3]
ens_p = np.mean([scores[n]["y_prob"] for n in top3], axis=0)
ens_auc = roc_auc_score(y, ens_p)
print(f"\n  Ensemble ({', '.join(top3)}): Train AUC = {ens_auc:.3f}")

best_name = top3[0]
best_model = scores[best_name]["model"]
best_cv = scores[best_name]["cv_auc"]

# ROC curve plot
fig, ax = plt.subplots(figsize=(8, 6))
for name in scores:
    fpr, tpr, _ = roc_curve(y, scores[name]["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={scores[name]['cv_auc']:.3f})", linewidth=2)
fpr_e, tpr_e, _ = roc_curve(y, ens_p)
ax.plot(fpr_e, tpr_e, label=f"Ensemble ({ens_auc:.3f})", linewidth=2.5, linestyle="--", color=ACCENT)
ax.plot([0, 1], [0, 1], ":", color="#666")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC — Temporal Model")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("figures/temporal_roc.png", dpi=150); plt.close()

# Confusion matrix
yp_best = best_model.predict(scores[best_name]["X"])
cm = confusion_matrix(y, yp_best)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Churned", "Retained"]); ax.set_yticklabels(["Churned", "Retained"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout(); plt.savefig("figures/temporal_cm.png", dpi=150); plt.close()

# Save results
results = {
    "target": TARGET, "best_model": best_name, "best_cv_auc": float(best_cv),
    "ensemble_auc": float(ens_auc), "n_users": len(y),
    "n_retained": int(n_pos), "retention_rate": float(ret_rate),
    "model_scores": {n: {"cv_auc": float(s["cv_auc"]), "cv_std": float(s["cv_std"]),
                         "train_auc": float(s["train_auc"]), "cv_f1": float(s["cv_f1"])}
                     for n, s in scores.items()},
}
with open("models/temporal_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save model bundle
joblib.dump({
    "model": best_model, "model_name": best_name,
    "feature_cols": feat_cols, "scaler": scaler,
}, "models/temporal_best_model.joblib")

# ── 5. Lift analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Lift analysis")

base = model_df[TARGET].mean()
print(f"  Baseline: {base:.1%}")

behaviors = {
    "Ran code": "ran_code", "Used agent": "used_agent",
    "Created canvas": "created_canvas", "Created edge (DAG)": "has_edges",
    "Uploaded files": "has_file_uploads",
    "Ran code 1st session": "first_session_ran_code",
    "Agent in 1st session": "first_session_used_agent",
    "Canvas in 1st session": "first_session_created_canvas",
    "Edge in 1st session": "first_session_created_edge",
    "Active 2+ days": "multi_day_user",
    "Multiple sessions": ("total_sessions", 1),
    "High diversity (>10)": ("unique_event_types", 10),
    "Skipped onboarding": "skipped_onboarding",
}

lifts = []
for bname, spec in behaviors.items():
    if isinstance(spec, str):
        mask = model_df[spec] == 1
    else:
        mask = model_df[spec[0]] > spec[1]
    nw = int(mask.sum())
    rw = model_df.loc[mask, TARGET].mean() if nw > 0 else 0
    rwo = model_df.loc[~mask, TARGET].mean() if (~mask).sum() > 0 else 0
    lift = rw / max(rwo, 0.001)
    lifts.append({"behavior": bname, "users": nw, "rate_with": rw, "rate_without": rwo, "lift": lift})

lifts_df = pd.DataFrame(lifts).sort_values("lift", ascending=False)
for _, r in lifts_df.iterrows():
    print(f"  {r['behavior']:30s} | {r['users']:4d} users | {r['rate_with']:.1%} vs {r['rate_without']:.1%} | {r['lift']:.1f}x")

# Lift bar chart
fig, ax = plt.subplots(figsize=(12, 7))
sl = lifts_df.sort_values("lift", ascending=True)
colors = [ACCENT if v > 1.5 else PALETTE[0] for v in sl["lift"]]
ax.barh(range(len(sl)), sl["lift"], color=colors)
ax.set_yticks(range(len(sl))); ax.set_yticklabels(sl["behavior"], fontsize=10)
ax.axvline(x=1, color="#666", linestyle=":")
ax.set_xlabel("Lift"); ax.set_title(f"First-Week Behavior Lift (baseline: {base:.1%})")
for i, v in enumerate(sl["lift"]):
    ax.text(v + 0.02, i, f"{v:.1f}x", va="center", fontsize=9, color="#fbfbff")
plt.tight_layout(); plt.savefig("figures/temporal_lift.png", dpi=150); plt.close()

lifts_df.to_csv("models/lift_analysis.csv", index=False)

print("\n" + "=" * 60)
print(f"DONE! Best model: {best_name}, CV AUC = {best_cv:.3f}")
print(f"Saved: models/temporal_best_model.joblib, models/temporal_results.json")
print(f"Saved: figures/temporal_roc.png, temporal_cm.png, temporal_lift.png")
print("=" * 60)

print("\nBlock complete: Temporal Prediction Model")
