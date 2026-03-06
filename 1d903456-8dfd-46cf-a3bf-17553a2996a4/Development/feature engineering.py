"""
Phase 2 — Feature Engineering (Clean Rebuild)
==============================================
Builds a clean, auditable user-level feature matrix from events_clean.parquet
and sessions_clean.parquet. Exactly 8 feature groups, no interaction terms,
no percentile ranks. Saves user_features_clean.parquet.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

# ── 0. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
p2_events = pd.read_parquet("events_clean.parquet")
p2_sessions = pd.read_parquet("sessions_clean.parquet")

# Ensure timestamp is UTC datetime
p2_events["timestamp"] = pd.to_datetime(p2_events["timestamp"], utc=True)
p2_sessions["session_start"] = pd.to_datetime(p2_sessions["session_start"], utc=True)

# Reference date = max timestamp in dataset (used for recency)
P2_REFERENCE_DATE = p2_events["timestamp"].max()

# Universe of users
p2_users = p2_events["user_id"].unique()
n_users = len(p2_users)

print(f"  Events:    {len(p2_events):,} rows, {p2_events['user_id'].nunique():,} users")
print(f"  Sessions:  {len(p2_sessions):,} rows")
print(f"  Reference date: {P2_REFERENCE_DATE}")

# ── Credit event names (system/billing noise to exclude from engagement) ──────
P2_CREDIT_EVENTS = {
    "credits_used", "addon_credits_used",
    "credits_below_1", "credits_below_2",
    "credits_below_3", "credits_below_4",
    "credits_exceeded",
}
P2_CREDIT_CATEGORY = "credits"

# ── Agent event names ─────────────────────────────────────────────────────────
P2_AGENT_EVENTS = {
    "agent_block_created", "agent_worker_created",
    "agent_tool_call_run_block_tool", "agent_tool_call_create_block_tool",
    "agent_tool_call_get_block_tool", "agent_tool_call_refactor_block_tool",
    "agent_tool_call_finish_ticket_tool", "agent_tool_call_get_canvas_summary_tool",
    "agent_tool_call_get_variable_preview_tool", "agent_tool_call_delete_block_tool",
    "agent_tool_call_create_edges_tool",
    "agent_message", "agent_start_from_prompt", "agent_accept_suggestion",
    "agent_new_chat", "agent_open", "agent_suprise_me",
    "agent_open_error_assist", "agent_upload_files",
}

# ── Helper: entropy of a value series ────────────────────────────────────────
def _col_entropy(x):
    counts = x.value_counts().values
    return float(entropy(counts, base=2)) if len(counts) > 1 else 0.0

# Pre-compute first timestamp per user (needed for time-to-value and tenure)
p2_first_ts = p2_events.groupby("user_id")["timestamp"].min()

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — ENGAGEMENT VOLUME
# Features: total_events_excl_credits, events_per_week, sessions_per_week,
#           total_active_days
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Engagement volume...")

# Total events excluding system credit events (billing noise)
p2_non_credit = p2_events[~p2_events["event"].isin(P2_CREDIT_EVENTS)]
total_events_excl_credits = (
    p2_non_credit.groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
    .rename("total_events_excl_credits")
)

# Unique active days (date-level, not session-level)
total_active_days = (
    p2_events.groupby("user_id")["timestamp"]
    .apply(lambda x: x.dt.date.nunique())
    .reindex(p2_users, fill_value=0)
    .rename("total_active_days")
)

# Tenure in weeks — for rate normalization; floor at 1/7 to avoid div-by-zero
p2_last_ts = p2_events.groupby("user_id")["timestamp"].max()
_tenure_weeks = (
    (p2_last_ts - p2_first_ts).dt.total_seconds() / 604800
).clip(lower=1.0 / 7)

# Events per week (excluding credits), sessions per week
total_sessions_raw = (
    p2_sessions.groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
)
events_per_week = (total_events_excl_credits / _tenure_weeks).rename("events_per_week")
sessions_per_week = (total_sessions_raw / _tenure_weeks).reindex(p2_users, fill_value=0).rename("sessions_per_week")

p2_engagement = pd.DataFrame({
    "total_events_excl_credits": total_events_excl_credits,
    "events_per_week": events_per_week,
    "sessions_per_week": sessions_per_week,
    "total_active_days": total_active_days,
})
print(f"  -> {len(p2_engagement.columns)} features: {list(p2_engagement.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — WORKFLOW DEPTH
# Features: num_blocks_run, num_edges_created, edges_per_canvas,
#           max_blocks_per_canvas, has_complete_pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("[2/8] Workflow depth...")

# Count events for specific workflow actions
num_blocks_run = (
    p2_events[p2_events["event"] == "run_block"]
    .groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
    .rename("num_blocks_run")
)
num_blocks_created = (
    p2_events[p2_events["event"] == "block_create"]
    .groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
)
num_edges_created = (
    p2_events[p2_events["event"] == "edge_create"]
    .groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
    .rename("num_edges_created")
)
num_canvases_created = (
    p2_events[p2_events["event"] == "canvas_create"]
    .groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
)

# edges_per_canvas: edges / max(1, canvases created)
edges_per_canvas = (
    num_edges_created / num_canvases_created.replace(0, np.nan)
).fillna(0).rename("edges_per_canvas")

# max_blocks_per_canvas: proxy = blocks_created / max(1, canvases_created)
# (we don't have per-canvas block data, so this is an approximation)
max_blocks_per_canvas = (
    num_blocks_created / num_canvases_created.replace(0, np.nan)
).fillna(0).rename("max_blocks_per_canvas")

# has_complete_pipeline: user has ≥2 blocks created, ≥1 edge, ≥1 block run
# (meaning they built a connected pipeline AND executed it)
has_complete_pipeline = (
    (num_blocks_created >= 2)
    & (num_edges_created >= 1)
    & (num_blocks_run >= 1)
).astype(int).rename("has_complete_pipeline")

p2_workflow = pd.DataFrame({
    "num_blocks_run": num_blocks_run,
    "num_edges_created": num_edges_created,
    "edges_per_canvas": edges_per_canvas,
    "max_blocks_per_canvas": max_blocks_per_canvas,
    "has_complete_pipeline": has_complete_pipeline,
})
print(f"  -> {len(p2_workflow.columns)} features: {list(p2_workflow.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — TIME-TO-VALUE
# Features: time_to_first_code_run_min, time_to_first_edge_min,
#           time_to_first_canvas_min, time_to_first_agent_min
# null (NaN) if user never performed the action
# ══════════════════════════════════════════════════════════════════════════════
print("[3/8] Time-to-value...")

def _time_to_first(event_mask, feat_name):
    """Minutes from user's first event to first occurrence of masked event.
    NaN if the user never triggered the event."""
    first_occurrence = (
        p2_events[event_mask]
        .groupby("user_id")["timestamp"].min()
    )
    delta_min = (first_occurrence - p2_first_ts).dt.total_seconds() / 60
    # Only reindex to users who triggered the event — NaN for others
    return delta_min.reindex(p2_users).rename(feat_name)

time_to_first_code_run_min = _time_to_first(
    p2_events["event"] == "run_block",
    "time_to_first_code_run_min"
)
time_to_first_edge_min = _time_to_first(
    p2_events["event"] == "edge_create",
    "time_to_first_edge_min"
)
time_to_first_canvas_min = _time_to_first(
    p2_events["event"] == "canvas_create",
    "time_to_first_canvas_min"
)
# Agent: any event in agent category
_agent_mask = p2_events["event"].isin(P2_AGENT_EVENTS)
if "event_category" in p2_events.columns:
    _agent_mask = _agent_mask | (p2_events["event_category"] == "agent")
time_to_first_agent_min = _time_to_first(
    _agent_mask,
    "time_to_first_agent_min"
)

p2_ttv = pd.DataFrame({
    "time_to_first_code_run_min": time_to_first_code_run_min,
    "time_to_first_edge_min": time_to_first_edge_min,
    "time_to_first_canvas_min": time_to_first_canvas_min,
    "time_to_first_agent_min": time_to_first_agent_min,
})
print(f"  -> {len(p2_ttv.columns)} features: {list(p2_ttv.columns)}")
print(f"     null rates: code_run={p2_ttv['time_to_first_code_run_min'].isna().mean():.1%}, "
      f"edge={p2_ttv['time_to_first_edge_min'].isna().mean():.1%}, "
      f"canvas={p2_ttv['time_to_first_canvas_min'].isna().mean():.1%}, "
      f"agent={p2_ttv['time_to_first_agent_min'].isna().mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — SESSION QUALITY
# Features: median_session_duration_min, session_duration_cv,
#           first_session_events, first_session_duration_min
# ══════════════════════════════════════════════════════════════════════════════
print("[4/8] Session quality...")

# Sort sessions by seq to identify "first session" cleanly
_p2_sess_sorted = p2_sessions.sort_values(["user_id", "session_start"])
_p2_first_sess = _p2_sess_sorted.groupby("user_id").first()

# Median session duration
median_session_duration_min = (
    p2_sessions.groupby("user_id")["duration_minutes"].median()
    .reindex(p2_users, fill_value=0)
    .rename("median_session_duration_min")
)

# Coefficient of variation of session duration (std/mean); 0 for single-session users
_sdur_grp = p2_sessions.groupby("user_id")["duration_minutes"]
session_duration_cv = (
    (_sdur_grp.std() / _sdur_grp.mean())
    .fillna(0)  # single-session users → std is NaN → fill 0
    .reindex(p2_users, fill_value=0)
    .rename("session_duration_cv")
)

# First session: event count and duration
first_session_events = (
    _p2_first_sess["num_events"]
    .reindex(p2_users, fill_value=0)
    .rename("first_session_events")
)
_fs_dur_col = "duration_minutes" if "duration_minutes" in _p2_first_sess.columns else None
if _fs_dur_col:
    first_session_duration_min = (
        _p2_first_sess[_fs_dur_col]
        .reindex(p2_users, fill_value=0)
        .rename("first_session_duration_min")
    )
else:
    first_session_duration_min = pd.Series(0.0, index=p2_users, name="first_session_duration_min")

p2_session_quality = pd.DataFrame({
    "median_session_duration_min": median_session_duration_min,
    "session_duration_cv": session_duration_cv,
    "first_session_events": first_session_events,
    "first_session_duration_min": first_session_duration_min,
})
print(f"  -> {len(p2_session_quality.columns)} features: {list(p2_session_quality.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — AGENT USAGE
# Features: num_agent_events, agent_adoption_flag, pct_events_agent,
#           num_agent_tool_types
# ══════════════════════════════════════════════════════════════════════════════
print("[5/8] Agent usage...")

# All agent events
_p2_agent_events = p2_events[p2_events["event"].isin(P2_AGENT_EVENTS)]
if "event_category" in p2_events.columns:
    _p2_agent_events = p2_events[
        p2_events["event"].isin(P2_AGENT_EVENTS) |
        (p2_events["event_category"] == "agent")
    ]

num_agent_events = (
    _p2_agent_events.groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
    .rename("num_agent_events")
)

# Binary flag: did user ever use agent?
agent_adoption_flag = (num_agent_events > 0).astype(int).rename("agent_adoption_flag")

# Percentage of all events (incl. credits) that are agent events
_total_events_all = p2_events.groupby("user_id").size().reindex(p2_users, fill_value=1)
pct_events_agent = (
    (num_agent_events / _total_events_all)
    .rename("pct_events_agent")
)

# Number of unique agent tool types used (tool call events only)
_p2_tool_calls = p2_events[p2_events["event"].str.startswith("agent_tool_call_")]
num_agent_tool_types = (
    _p2_tool_calls.groupby("user_id")["event"].nunique()
    .reindex(p2_users, fill_value=0)
    .rename("num_agent_tool_types")
)

p2_agent = pd.DataFrame({
    "num_agent_events": num_agent_events,
    "agent_adoption_flag": agent_adoption_flag,
    "pct_events_agent": pct_events_agent,
    "num_agent_tool_types": num_agent_tool_types,
})
print(f"  -> {len(p2_agent.columns)} features: {list(p2_agent.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — RECENCY
# Features: days_since_last_event, days_since_last_code_run,
#           active_last_7d, active_last_14d
# ══════════════════════════════════════════════════════════════════════════════
print("[6/8] Recency...")

_last_event_ts = p2_events.groupby("user_id")["timestamp"].max()
days_since_last_event = (
    (P2_REFERENCE_DATE - _last_event_ts).dt.total_seconds() / 86400
).reindex(p2_users, fill_value=np.nan).rename("days_since_last_event")

_p2_code_events = p2_events[p2_events["event"] == "run_block"]
_last_code_ts = _p2_code_events.groupby("user_id")["timestamp"].max()
days_since_last_code_run = (
    (P2_REFERENCE_DATE - _last_code_ts).dt.total_seconds() / 86400
).reindex(p2_users, fill_value=np.nan).rename("days_since_last_code_run")
# NaN = user never ran code (not an error — intentional null)

# Activity flags for 7d and 14d windows
_cutoff_7d = P2_REFERENCE_DATE - pd.Timedelta(days=7)
_cutoff_14d = P2_REFERENCE_DATE - pd.Timedelta(days=14)
active_last_7d = (
    p2_events[p2_events["timestamp"] >= _cutoff_7d]
    .groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
    .gt(0).astype(int)
    .rename("active_last_7d")
)
active_last_14d = (
    p2_events[p2_events["timestamp"] >= _cutoff_14d]
    .groupby("user_id").size()
    .reindex(p2_users, fill_value=0)
    .gt(0).astype(int)
    .rename("active_last_14d")
)

p2_recency = pd.DataFrame({
    "days_since_last_event": days_since_last_event,
    "days_since_last_code_run": days_since_last_code_run,
    "active_last_7d": active_last_7d,
    "active_last_14d": active_last_14d,
})
print(f"  -> {len(p2_recency.columns)} features: {list(p2_recency.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 7 — TEMPORAL PATTERNS
# Features: tenure_hours, return_rate, multi_day_user,
#           day_of_week_entropy, hour_of_day_entropy
# ══════════════════════════════════════════════════════════════════════════════
print("[7/8] Temporal patterns...")

# Tenure: hours from first to last event
tenure_hours = (
    (p2_last_ts - p2_first_ts).dt.total_seconds() / 3600
).reindex(p2_users, fill_value=0).rename("tenure_hours")

# Return rate: fraction of sessions that are NOT the first
# (sessions beyond first / total sessions)
_sess_counts = p2_sessions.groupby("user_id").size().reindex(p2_users, fill_value=0)
# Sessions after first = total - 1, floored at 0
_return_numerator = (_sess_counts - 1).clip(lower=0)
return_rate = (
    (_return_numerator / _sess_counts.replace(0, np.nan))
    .fillna(0)
    .rename("return_rate")
)

# Multi-day user: binary flag, active on 2+ distinct calendar days
multi_day_user = (total_active_days > 1).astype(int).rename("multi_day_user")

# Day-of-week entropy: how uniformly spread is the user's activity across 0–6?
day_of_week_entropy = (
    p2_events.assign(_dow=p2_events["timestamp"].dt.dayofweek)
    .groupby("user_id")["_dow"]
    .apply(_col_entropy)
    .reindex(p2_users, fill_value=0)
    .rename("day_of_week_entropy")
)

# Hour-of-day entropy: how uniformly spread across 0–23?
hour_of_day_entropy = (
    p2_events.assign(_hod=p2_events["timestamp"].dt.hour)
    .groupby("user_id")["_hod"]
    .apply(_col_entropy)
    .reindex(p2_users, fill_value=0)
    .rename("hour_of_day_entropy")
)

p2_temporal = pd.DataFrame({
    "tenure_hours": tenure_hours,
    "return_rate": return_rate,
    "multi_day_user": multi_day_user,
    "day_of_week_entropy": day_of_week_entropy,
    "hour_of_day_entropy": hour_of_day_entropy,
})
print(f"  -> {len(p2_temporal.columns)} features: {list(p2_temporal.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 8 — CREDITS
# Features: total_credits_used, credits_per_code_run, credits_exceeded_flag
# ══════════════════════════════════════════════════════════════════════════════
print("[8/8] Credits...")

# total_credits_used: sum of credit_amount (numeric parse)
_credit_col = None
for _col_candidate in ["credit_amount", "credits_used"]:
    if _col_candidate in p2_events.columns:
        _credit_col = _col_candidate
        break

if _credit_col:
    _credit_num = pd.to_numeric(p2_events[_credit_col], errors="coerce").fillna(0)
    total_credits_used = (
        p2_events.assign(_cn=_credit_num)
        .groupby("user_id")["_cn"].sum()
        .reindex(p2_users, fill_value=0)
        .rename("total_credits_used")
    )
else:
    # Fallback: count credit events as proxy
    total_credits_used = (
        p2_events[p2_events["event"].isin(P2_CREDIT_EVENTS)]
        .groupby("user_id").size()
        .reindex(p2_users, fill_value=0)
        .rename("total_credits_used")
    )

# credits_per_code_run: total_credits / num_blocks_run (NaN if no code runs)
credits_per_code_run = (
    (total_credits_used / num_blocks_run.replace(0, np.nan))
    .reindex(p2_users, fill_value=np.nan)
    .rename("credits_per_code_run")
)

# credits_exceeded_flag: did the user ever hit credits_exceeded?
credits_exceeded_flag = (
    (p2_events[p2_events["event"] == "credits_exceeded"]
     .groupby("user_id").size()
     .reindex(p2_users, fill_value=0) > 0)
    .astype(int)
    .rename("credits_exceeded_flag")
)

p2_credits = pd.DataFrame({
    "total_credits_used": total_credits_used,
    "credits_per_code_run": credits_per_code_run,
    "credits_exceeded_flag": credits_exceeded_flag,
})
print(f"  -> {len(p2_credits.columns)} features: {list(p2_credits.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE ALL GROUPS
# ══════════════════════════════════════════════════════════════════════════════
print("\nMerging feature groups...")

# Concatenate all groups — index is user_id across all DataFrames
user_features_clean = pd.concat([
    p2_engagement,     # (1) Engagement volume
    p2_workflow,       # (2) Workflow depth
    p2_ttv,            # (3) Time-to-value  [intentional NaN for non-adopters]
    p2_session_quality,# (4) Session quality
    p2_agent,          # (5) Agent usage
    p2_recency,        # (6) Recency         [intentional NaN for never-ran-code]
    p2_temporal,       # (7) Temporal patterns
    p2_credits,        # (8) Credits         [intentional NaN for no-code-run]
], axis=1)

# Deduplicate columns if any slipped through (e.g., total_active_days re-used in temporal)
user_features_clean = user_features_clean.loc[:, ~user_features_clean.columns.duplicated()]

# Confirm no infinite values; replace only inf (not NaN — NaN is intentional)
n_inf = np.isinf(user_features_clean.select_dtypes("number")).sum().sum()
if n_inf > 0:
    print(f"  WARNING: Replacing {n_inf} infinite values with NaN")
    user_features_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

n_features = len(user_features_clean.columns)
n_rows = len(user_features_clean)

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS: Feature count, null rates, distributions
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"FEATURE MATRIX: {n_rows:,} users × {n_features} features")
print("=" * 60)

group_sizes = {
    "(1) Engagement volume":    len(p2_engagement.columns),
    "(2) Workflow depth":       len(p2_workflow.columns),
    "(3) Time-to-value":        len(p2_ttv.columns),
    "(4) Session quality":      len(p2_session_quality.columns),
    "(5) Agent usage":          len(p2_agent.columns),
    "(6) Recency":              len(p2_recency.columns),
    "(7) Temporal patterns":    len(p2_temporal.columns),
    "(8) Credits":              len(p2_credits.columns),
}
print("\nFeature group breakdown:")
for gname, gsize in group_sizes.items():
    print(f"  {gname}: {gsize}")
print(f"  TOTAL: {n_features}")

# ── Null rates ────────────────────────────────────────────────────────────────
print("\n── Null rates per feature ─────────────────────────────────────────")
null_rates = user_features_clean.isna().mean().sort_values(ascending=False)
# Print ALL features with their null rate
null_df = null_rates.reset_index()
null_df.columns = ["feature", "null_rate"]
null_df["null_pct"] = (null_df["null_rate"] * 100).round(2)
# Features with nulls (expected: time-to-value and days_since_last_code_run)
has_nulls = null_df[null_df["null_rate"] > 0]
print(f"  Features with nulls: {len(has_nulls)}/{n_features}")
for _, row in has_nulls.iterrows():
    print(f"  {row['feature']}: {row['null_pct']:.1f}% null  "
          f"({'intentional — user never triggered this action' if row['null_rate'] < 1.0 else 'WARNING'})")
no_null_count = (null_df["null_rate"] == 0).sum()
print(f"  Features with zero nulls: {no_null_count}/{n_features}")

# ── Distributions for key features ───────────────────────────────────────────
print("\n── Key feature distributions (non-null) ──────────────────────────")
key_features = [
    "total_events_excl_credits", "events_per_week", "sessions_per_week",
    "total_active_days", "num_blocks_run", "num_edges_created",
    "has_complete_pipeline", "time_to_first_code_run_min",
    "time_to_first_agent_min", "median_session_duration_min",
    "session_duration_cv", "num_agent_events", "agent_adoption_flag",
    "pct_events_agent", "num_agent_tool_types",
    "days_since_last_event", "days_since_last_code_run",
    "tenure_hours", "return_rate", "multi_day_user",
    "total_credits_used", "credits_exceeded_flag",
]
# Only print features that exist
key_features = [f for f in key_features if f in user_features_clean.columns]

_stats = user_features_clean[key_features].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
print(_stats.T[["mean", "std", "50%", "95%", "max"]].to_string())

# ── Binary flag summaries ─────────────────────────────────────────────────────
print("\n── Binary flag summaries ──────────────────────────────────────────")
binary_flags = ["has_complete_pipeline", "agent_adoption_flag",
                "active_last_7d", "active_last_14d",
                "multi_day_user", "credits_exceeded_flag"]
for flag in binary_flags:
    if flag in user_features_clean.columns:
        rate = user_features_clean[flag].mean()
        count = user_features_clean[flag].sum()
        print(f"  {flag}: {count:,} users ({rate:.1%})")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "user_features_clean.parquet"
user_features_clean.to_parquet(out_path)
print(f"\n✓ Saved {out_path}  ({n_rows:,} rows × {n_features} columns)")
print("=" * 60)
