"""
Long-Term Success Feature Matrix + Success Label Definition
===========================================================
Builds a rich, multi-dimensional feature matrix directly from raw events.
Defines 'long-term success' with a clear, justified label.

Success Definition:
  A user is "long-term successful" if they meet ≥2 of these 3 criteria:
  1. retained_28d       — active on day ≥ 28 from first event
  2. multi_week_3plus   — events spanning ≥ 3 distinct ISO weeks
  3. workflow_builder   — ever created ≥2 blocks + ≥1 edge (built a real DAG)
  
  This composite is intentionally harder than simple day-14 retention:
  it rewards depth (DAG building), breadth (multi-week) and longevity (28d).
  Binary label: long_term_success (1 = successful, 0 = not)

Feature Groups (10 total, 60+ features):
  [A] Session Depth
  [B] Feature Usage Breadth
  [C] Workflow Complexity Index
  [D] Reproducibility Signals
  [E] Time-Based Engagement Patterns
  [F] Agent Usage Depth
  [G] Onboarding Completion
  [H] Recency & Velocity
  [I] First-Session Power
  [J] Credit & Resource Signals
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

# ── Config ────────────────────────────────────────────────────────────────────

SESSION_GAP_MIN = 30          # 30-minute inactivity → new session
BOT_MIN_EVENTS  = 3           # fewer than 3 events AND < 60s max session = bot
BOT_MIN_SESSION = 60          # seconds

P0_COLUMN_MAP = {
    "distinct_id":              "user_id",
    "person_id":                "person_id",
    "event":                    "event",
    "timestamp":                "timestamp",
    "uuid":                     "event_uuid",
    "prop_$session_id":         "session_id_raw",
    "prop_$browser":            "browser",
    "prop_$os":                 "os",
    "prop_$device_type":        "device_type",
    "prop_$geoip_country_name": "country",
    "prop_$referring_domain":   "referring_domain",
    "prop_surface":             "surface",
    "prop_tool_name":           "tool_name",
    "prop_credit_amount":       "credit_amount",
    "prop_credits_used":        "credits_used",
}

P0_DROP_PREFIXES = ("prop_$set.", "prop_$set_once.")

AGENT_EVENTS = {
    "agent_block_created", "agent_worker_created",
    "agent_tool_call_run_block_tool", "agent_tool_call_create_block_tool",
    "agent_tool_call_get_block_tool", "agent_tool_call_refactor_block_tool",
    "agent_tool_call_finish_ticket_tool", "agent_tool_call_get_canvas_summary_tool",
    "agent_tool_call_get_variable_preview_tool", "agent_tool_call_delete_block_tool",
    "agent_tool_call_create_edges_tool",
    "agent_message", "agent_start_from_prompt", "agent_accept_suggestion",
    "agent_new_chat", "agent_open", "agent_suprise_me",
    "agent_open_error_assist", "agent_upload_files",
    "agent_tool_call_run_block_tool",
}

CREDIT_EVENTS = {
    "credits_used", "addon_credits_used",
    "credits_below_1", "credits_below_2", "credits_below_3", "credits_below_4",
    "credits_exceeded",
}

EVENT_TO_CAT = {}
for _cat, _evts in {
    "agent_use":       list(AGENT_EVENTS),
    "code_execution":  ["run_block", "run_all_blocks", "run_upto_block", "run_from_block", "stop_block"],
    "workflow_build":  ["block_create", "canvas_create", "edge_create", "layer_create",
                        "block_delete", "block_resize", "edge_delete", "block_rename",
                        "block_copy", "block_open_compute_settings", "block_output_copy",
                        "app_publish", "scheduled_job_start", "scheduled_job_stop"],
    "navigation":      ["canvas_open", "fullscreen_open", "fullscreen_close",
                        "folder_open", "link_clicked", "button_clicked"],
    "credits":         list(CREDIT_EVENTS),
    "onboarding":      ["sign_in", "sign_up", "new_user_created", "promo_code_redeemed",
                        "skip_onboarding_form", "submit_onboarding_form",
                        "canvas_onboarding_tour_started", "canvas_onboarding_tour_finished",
                        "quickstart_explore_playground", "quickstart_add_dataset"],
    "file_ops":        ["files_upload", "files_download", "files_delete"],
    "sharing":         ["app_publish", "share_canvas", "invite_user", "canvas_shared", "app_view"],
}.items():
    for _e in _evts:
        EVENT_TO_CAT[_e] = _cat

# ── Helper functions ──────────────────────────────────────────────────────────

def _col_entropy(x):
    counts = x.value_counts().values
    return float(shannon_entropy(counts, base=2)) if len(counts) > 1 else 0.0

def _reindex(s, users, fill=0):
    return s.reindex(users, fill_value=fill)

def _rate(num, den, fill=0.0):
    return (num / den.replace(0, np.nan)).fillna(fill)

# ── 0. LOAD & CLEAN ───────────────────────────────────────────────────────────

print("=" * 70)
print("LONG-TERM SUCCESS FEATURE ENGINEERING")
print("=" * 70)
print("\n[LOAD] Reading user_retention.parquet ...")

raw = pd.read_parquet("user_retention.parquet")
print(f"  Raw: {len(raw):,} rows × {len(raw.columns)} columns")

# Drop prop_$set.* columns
_drop = [c for c in raw.columns if c.startswith(P0_DROP_PREFIXES)]
raw = raw.drop(columns=_drop)

# Rename columns
_remap = {k: v for k, v in P0_COLUMN_MAP.items() if k in raw.columns}
raw = raw.rename(columns=_remap)

# Timestamp normalisation
raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
raw = raw.dropna(subset=["timestamp"])

# user_id fill
if "user_id" not in raw.columns and "distinct_id" in raw.columns:
    raw = raw.rename(columns={"distinct_id": "user_id"})
if "person_id" in raw.columns:
    _null = raw["user_id"].isna()
    raw.loc[_null, "user_id"] = raw.loc[_null, "person_id"]
raw = raw.dropna(subset=["user_id"])

# Dedup
if "event_uuid" in raw.columns:
    raw = raw.drop_duplicates(subset=["event_uuid"], keep="first")

# Event taxonomy
raw["event_category"] = raw["event"].map(EVENT_TO_CAT).fillna("other")
raw = raw.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

print(f"  After clean: {len(raw):,} rows, {raw['user_id'].nunique():,} users")
print(f"  Date range: {raw['timestamp'].min().date()} → {raw['timestamp'].max().date()}")

# ── Sessionize ───────────────────────────────────────────────────────────────

_tdiff = raw.groupby("user_id")["timestamp"].diff()
_new_sess = _tdiff.isna() | (_tdiff > pd.Timedelta(minutes=SESSION_GAP_MIN))
raw["session_seq"] = _new_sess.groupby(raw["user_id"]).cumsum().astype(int)
raw["session_id_derived"] = raw["user_id"] + "_S" + raw["session_seq"].astype(str)

# Session table
_sess_agg = raw.groupby("session_id_derived").agg(
    user_id         = ("user_id",   "first"),
    session_start   = ("timestamp", "min"),
    session_end     = ("timestamp", "max"),
    num_events      = ("event",     "count"),
    num_event_types = ("event",     "nunique"),
    num_cat_types   = ("event_category", "nunique"),
    agent_events    = ("event_category", lambda x: (x == "agent_use").sum()),
    code_runs       = ("event",     lambda x: (x == "run_block").sum()),
    blocks_created  = ("event",     lambda x: (x == "block_create").sum()),
    edges_created   = ("event",     lambda x: (x == "edge_create").sum()),
    session_seq     = ("session_seq", "first"),
).reset_index()

_sess_agg["duration_sec"] = (
    (_sess_agg["session_end"] - _sess_agg["session_start"]).dt.total_seconds()
)
_sess_agg["duration_min"] = _sess_agg["duration_sec"] / 60.0

# Bot filter
_user_stats = _sess_agg.groupby("user_id").agg(
    _te  = ("num_events", "sum"),
    _msd = ("duration_sec", "max"),
).reset_index()
_bots = set(_user_stats.loc[
    (_user_stats["_te"] < BOT_MIN_EVENTS) & (_user_stats["_msd"] < BOT_MIN_SESSION),
    "user_id"
])
raw   = raw[~raw["user_id"].isin(_bots)].reset_index(drop=True)
_sess = _sess_agg[~_sess_agg["user_id"].isin(_bots)].reset_index(drop=True)

_REF_DATE = raw["timestamp"].max()
_ALL_USERS = raw["user_id"].unique()
_N_USERS   = len(_ALL_USERS)
print(f"\n  Clean users: {_N_USERS:,}  (removed {len(_bots):,} bots)")
print(f"  Clean events: {len(raw):,}  |  Sessions: {len(_sess):,}")

# Per-user first/last timestamps
_first_ts = raw.groupby("user_id")["timestamp"].min()
_last_ts  = raw.groupby("user_id")["timestamp"].max()
_tenure_sec  = (_last_ts - _first_ts).dt.total_seconds()
_tenure_hours = (_tenure_sec / 3600).reindex(_ALL_USERS, fill_value=0)
_tenure_weeks = (_tenure_sec / 604800).clip(lower=1.0/7).reindex(_ALL_USERS)

# Helper: time-to-first for a boolean mask → minutes
def _ttf_minutes(mask, name):
    _first_occ = raw[mask].groupby("user_id")["timestamp"].min()
    _delta = (_first_occ - _first_ts).dt.total_seconds() / 60
    return _delta.reindex(_ALL_USERS).rename(name)

# ══════════════════════════════════════════════════════════════════════════════
# [A] SESSION DEPTH  (8 features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Session depth ...")

_sess_per_user = _sess.groupby("user_id")
_sp = _sess_per_user

session_count          = _sp["session_id_derived"].count()
total_session_duration = _sp["duration_min"].sum()
median_session_dur     = _sp["duration_min"].median()
max_session_dur        = _sp["duration_min"].max()
session_dur_cv         = (_sp["duration_min"].std() / _sp["duration_min"].mean()).fillna(0)
events_per_session     = _sp["num_events"].mean()
pct_multi_event_sess   = (_sp["num_events"].apply(lambda x: (x > 3).mean()))
sessions_per_week      = session_count / _tenure_weeks

A = pd.DataFrame({
    "session_count":          _reindex(session_count, _ALL_USERS),
    "total_session_dur_min":  _reindex(total_session_duration, _ALL_USERS),
    "median_session_dur_min": _reindex(median_session_dur, _ALL_USERS),
    "max_session_dur_min":    _reindex(max_session_dur, _ALL_USERS),
    "session_dur_cv":         _reindex(session_dur_cv, _ALL_USERS),
    "events_per_session":     _reindex(events_per_session, _ALL_USERS),
    "pct_multi_event_sessions": _reindex(pct_multi_event_sess, _ALL_USERS),
    "sessions_per_week":      _reindex(sessions_per_week, _ALL_USERS),
}, index=_ALL_USERS)
print(f"  -> {A.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [B] FEATURE USAGE BREADTH  (8 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[B] Feature usage breadth ...")

_non_credit = raw[~raw["event"].isin(CREDIT_EVENTS)]

total_events_excl_credits = _reindex(
    _non_credit.groupby("user_id").size(), _ALL_USERS)
unique_event_types = _reindex(
    raw.groupby("user_id")["event"].nunique(), _ALL_USERS)
category_breadth = _reindex(
    raw.groupby("user_id")["event_category"].nunique(), _ALL_USERS)
event_entropy = (
    raw.groupby("user_id")["event"].apply(_col_entropy)
    .reindex(_ALL_USERS, fill_value=0)
)
used_agent          = (_reindex(raw[raw["event_category"] == "agent_use"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
used_file_ops       = (_reindex(raw[raw["event_category"] == "file_ops"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
used_sharing        = (_reindex(raw[raw["event_category"] == "sharing"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
events_per_week     = total_events_excl_credits / _tenure_weeks

B = pd.DataFrame({
    "total_events_excl_credits": total_events_excl_credits,
    "unique_event_types":        unique_event_types,
    "category_breadth":          category_breadth,
    "event_entropy":             event_entropy,
    "used_agent":                used_agent,
    "used_file_ops":             used_file_ops,
    "used_sharing":              used_sharing,
    "events_per_week":           events_per_week,
}, index=_ALL_USERS)
print(f"  -> {B.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [C] WORKFLOW COMPLEXITY INDEX  (9 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[C] Workflow complexity ...")

num_blocks_created  = _reindex(raw[raw["event"] == "block_create"].groupby("user_id").size(), _ALL_USERS)
num_blocks_run      = _reindex(raw[raw["event"] == "run_block"].groupby("user_id").size(), _ALL_USERS)
num_edges_created   = _reindex(raw[raw["event"] == "edge_create"].groupby("user_id").size(), _ALL_USERS)
num_canvases        = _reindex(raw[raw["event"] == "canvas_create"].groupby("user_id").size(), _ALL_USERS)
num_layers          = _reindex(raw[raw["event"] == "layer_create"].groupby("user_id").size(), _ALL_USERS)
block_run_rate      = _rate(num_blocks_run, num_blocks_created, fill=0.0)
edges_per_canvas    = _rate(num_edges_created, num_canvases, fill=0.0)
blocks_per_canvas   = _rate(num_blocks_created, num_canvases, fill=0.0)
has_complete_dag    = ((num_blocks_created >= 2) & (num_edges_created >= 1) & (num_blocks_run >= 1)).astype(int)

C = pd.DataFrame({
    "num_blocks_created":  num_blocks_created,
    "num_blocks_run":      num_blocks_run,
    "num_edges_created":   num_edges_created,
    "num_canvases":        num_canvases,
    "num_layers":          num_layers,
    "block_run_rate":      block_run_rate,
    "edges_per_canvas":    edges_per_canvas,
    "blocks_per_canvas":   blocks_per_canvas,
    "has_complete_dag":    has_complete_dag,
}, index=_ALL_USERS)
print(f"  -> {C.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [D] REPRODUCIBILITY SIGNALS  (6 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[D] Reproducibility signals ...")

# Reproducibility = user consistently runs blocks, re-executes workflows,
# and shows multi-run depth (not just one-off experimentation)
num_run_all      = _reindex(raw[raw["event"] == "run_all_blocks"].groupby("user_id").size(), _ALL_USERS)
num_run_from     = _reindex(raw[raw["event"] == "run_from_block"].groupby("user_id").size(), _ALL_USERS)
num_run_upto     = _reindex(raw[raw["event"] == "run_upto_block"].groupby("user_id").size(), _ALL_USERS)
num_block_stop   = _reindex(raw[raw["event"] == "stop_block"].groupby("user_id").size(), _ALL_USERS)
# Ratio of advanced execution modes (not just run_block)
advanced_run_ratio = _rate(
    num_run_all + num_run_from + num_run_upto,
    num_blocks_run + num_run_all + num_run_from + num_run_upto,
    fill=0.0,
)
# sessions with code runs / total sessions — consistent execution rate
_code_run_sessions = raw[raw["event"] == "run_block"].groupby("user_id")["session_id_derived"].nunique()
pct_sessions_with_code = _rate(
    _reindex(_code_run_sessions, _ALL_USERS),
    A["session_count"],
    fill=0.0,
)

D = pd.DataFrame({
    "num_run_all_blocks":   num_run_all,
    "num_run_from_block":   num_run_from,
    "num_run_upto_block":   num_run_upto,
    "num_block_stops":      num_block_stop,
    "advanced_run_ratio":   advanced_run_ratio,
    "pct_sessions_with_code_run": pct_sessions_with_code,
}, index=_ALL_USERS)
print(f"  -> {D.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [E] TIME-BASED ENGAGEMENT PATTERNS  (11 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[E] Time-based engagement patterns ...")

total_active_days  = raw.groupby("user_id")["timestamp"].apply(lambda x: x.dt.date.nunique()).reindex(_ALL_USERS, fill_value=0)
multi_day_user     = (total_active_days > 1).astype(int)
active_days_per_week = total_active_days / _tenure_weeks

day_of_week_entropy = (
    raw.assign(_dow=raw["timestamp"].dt.dayofweek)
    .groupby("user_id")["_dow"].apply(_col_entropy)
    .reindex(_ALL_USERS, fill_value=0)
)
hour_of_day_entropy = (
    raw.assign(_hod=raw["timestamp"].dt.hour)
    .groupby("user_id")["_hod"].apply(_col_entropy)
    .reindex(_ALL_USERS, fill_value=0)
)

# Activity trend: slope of daily event counts over first 14 active days
def _activity_trend(grp):
    """Slope of daily event count for first 14 days."""
    dc = grp.dt.date.value_counts().sort_index()
    if len(dc) < 2:
        return 0.0
    x = np.arange(min(len(dc), 14))
    y = dc.iloc[:14].values
    return float(np.polyfit(x, y, 1)[0])

activity_trend = (
    raw.groupby("user_id")["timestamp"]
    .apply(_activity_trend)
    .reindex(_ALL_USERS, fill_value=0)
)

# Peak hour (most active hour of day)
peak_hour = (
    raw.assign(_hod=raw["timestamp"].dt.hour)
    .groupby("user_id")["_hod"]
    .apply(lambda x: x.mode().iloc[0] if len(x) > 0 else 12)
    .reindex(_ALL_USERS, fill_value=12)
)
pct_daytime = (
    raw.assign(_daytime=((raw["timestamp"].dt.hour >= 8) & (raw["timestamp"].dt.hour < 18)))
    .groupby("user_id")["_daytime"].mean()
    .reindex(_ALL_USERS, fill_value=0)
)

# Time-to-value signals
ttv_code_run  = _ttf_minutes(raw["event"] == "run_block",       "ttv_code_run_min")
ttv_edge      = _ttf_minutes(raw["event"] == "edge_create",     "ttv_edge_min")
ttv_agent     = _ttf_minutes(raw["event_category"] == "agent_use", "ttv_agent_min")

E = pd.DataFrame({
    "total_active_days":       total_active_days,
    "multi_day_user":          multi_day_user,
    "active_days_per_week":    active_days_per_week,
    "day_of_week_entropy":     day_of_week_entropy,
    "hour_of_day_entropy":     hour_of_day_entropy,
    "activity_trend":          activity_trend,
    "peak_hour":               peak_hour,
    "pct_daytime":             pct_daytime,
    "ttv_code_run_min":        ttv_code_run,
    "ttv_edge_min":            ttv_edge,
    "ttv_agent_min":           ttv_agent,
}, index=_ALL_USERS)
print(f"  -> {E.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [F] AGENT USAGE DEPTH  (8 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[F] Agent usage depth ...")

_ag_events = raw[raw["event_category"] == "agent_use"]
num_agent_events  = _reindex(_ag_events.groupby("user_id").size(), _ALL_USERS)
num_agent_prompts = _reindex(raw[raw["event"] == "agent_start_from_prompt"].groupby("user_id").size(), _ALL_USERS)
num_agent_msgs    = _reindex(raw[raw["event"] == "agent_message"].groupby("user_id").size(), _ALL_USERS)
num_agent_accepts = _reindex(raw[raw["event"] == "agent_accept_suggestion"].groupby("user_id").size(), _ALL_USERS)
num_agent_workers = _reindex(raw[raw["event"] == "agent_worker_created"].groupby("user_id").size(), _ALL_USERS)
_tool_calls       = raw[raw["event"].str.startswith("agent_tool_call_")]
num_agent_tool_types = _reindex(_tool_calls.groupby("user_id")["event"].nunique(), _ALL_USERS)
pct_events_agent  = _rate(num_agent_events, total_events_excl_credits.replace(0, np.nan))
agent_accept_rate = _rate(num_agent_accepts, (num_agent_prompts + num_agent_msgs).replace(0, np.nan))

F = pd.DataFrame({
    "num_agent_events":        num_agent_events,
    "num_agent_prompts":       num_agent_prompts,
    "num_agent_messages":      num_agent_msgs,
    "num_agent_accepts":       num_agent_accepts,
    "num_agent_workers":       num_agent_workers,
    "num_agent_tool_types":    num_agent_tool_types,
    "pct_events_agent":        pct_events_agent,
    "agent_accept_rate":       agent_accept_rate,
}, index=_ALL_USERS)
print(f"  -> {F.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [G] ONBOARDING COMPLETION  (5 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[G] Onboarding completion ...")

_onb = raw[raw["event_category"] == "onboarding"]
completed_tour  = (_reindex(raw[raw["event"] == "canvas_onboarding_tour_finished"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
submitted_form  = (_reindex(raw[raw["event"] == "submit_onboarding_form"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
skipped_onboard = (_reindex(raw[raw["event"] == "skip_onboarding_form"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
explored_qs     = (_reindex(raw[raw["event"] == "quickstart_explore_playground"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)
added_dataset   = (_reindex(raw[raw["event"] == "quickstart_add_dataset"].groupby("user_id").size(), _ALL_USERS) > 0).astype(int)

G = pd.DataFrame({
    "completed_onboarding_tour": completed_tour,
    "submitted_onboarding_form": submitted_form,
    "skipped_onboarding":        skipped_onboard,
    "explored_quickstart":       explored_qs,
    "added_dataset_quickstart":  added_dataset,
}, index=_ALL_USERS)
print(f"  -> {G.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [H] RECENCY & VELOCITY  (6 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[H] Recency & velocity ...")

days_since_last = (
    (_REF_DATE - _last_ts).dt.total_seconds() / 86400
).reindex(_ALL_USERS, fill_value=np.nan)

_cutoff7  = _REF_DATE - pd.Timedelta(days=7)
_cutoff14 = _REF_DATE - pd.Timedelta(days=14)
_cutoff30 = _REF_DATE - pd.Timedelta(days=30)
active_last_7d  = (raw[raw["timestamp"] >= _cutoff7 ].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0) > 0).astype(int)
active_last_14d = (raw[raw["timestamp"] >= _cutoff14].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0) > 0).astype(int)
active_last_30d = (raw[raw["timestamp"] >= _cutoff30].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0) > 0).astype(int)

# Return rate: sessions after first / total sessions
_sc = A["session_count"]
return_rate = _rate((_sc - 1).clip(lower=0), _sc)

# Events in most recent 7d vs overall events_per_week (velocity ratio)
_recent7_events = raw[raw["timestamp"] >= _cutoff7].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0)
velocity_ratio  = _rate(_recent7_events / 7, B["events_per_week"].replace(0, np.nan))

H = pd.DataFrame({
    "days_since_last_event":  days_since_last,
    "active_last_7d":         active_last_7d,
    "active_last_14d":        active_last_14d,
    "active_last_30d":        active_last_30d,
    "return_rate":            return_rate,
    "velocity_ratio":         velocity_ratio,
}, index=_ALL_USERS)
print(f"  -> {H.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [I] FIRST-SESSION POWER  (8 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[I] First-session power ...")

_first_sess = _sess[_sess["session_seq"] == 0].set_index("user_id")
_reindex_first = lambda col, fill=0: _first_sess[col].reindex(_ALL_USERS, fill_value=fill)

fs_events          = _reindex_first("num_events")
fs_dur_min         = _reindex_first("duration_min")
fs_event_types     = _reindex_first("num_event_types")
fs_cat_types       = _reindex_first("num_cat_types")
fs_agent           = (_reindex_first("agent_events") > 0).astype(int)
fs_code_runs       = _reindex_first("code_runs")
fs_blocks_created  = _reindex_first("blocks_created")
fs_edges_created   = _reindex_first("edges_created")

I = pd.DataFrame({
    "fs_events":          fs_events,
    "fs_duration_min":    fs_dur_min,
    "fs_event_types":     fs_event_types,
    "fs_category_types":  fs_cat_types,
    "fs_used_agent":      fs_agent,
    "fs_code_runs":       fs_code_runs,
    "fs_blocks_created":  fs_blocks_created,
    "fs_edges_created":   fs_edges_created,
}, index=_ALL_USERS)
print(f"  -> {I.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# [J] CREDIT & RESOURCE SIGNALS  (5 features)
# ══════════════════════════════════════════════════════════════════════════════
print("[J] Credit & resource signals ...")

# Parse credit_amount numerically
_has_credit = "credit_amount" in raw.columns
if _has_credit:
    _ca = pd.to_numeric(raw["credit_amount"], errors="coerce").fillna(0)
    total_credits_used = raw.assign(_ca=_ca).groupby("user_id")["_ca"].sum().reindex(_ALL_USERS, fill_value=0)
else:
    total_credits_used = raw[raw["event"].isin(CREDIT_EVENTS)].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0)

credits_per_run     = _rate(total_credits_used, num_blocks_run.replace(0, np.nan))
credits_exceeded    = (raw[raw["event"] == "credits_exceeded"].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0) > 0).astype(int)
addon_credits       = (raw[raw["event"] == "addon_credits_used"].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0) > 0).astype(int)
num_file_uploads    = raw[raw["event"] == "files_upload"].groupby("user_id").size().reindex(_ALL_USERS, fill_value=0)

J = pd.DataFrame({
    "total_credits_used":    total_credits_used,
    "credits_per_run":       credits_per_run,
    "credits_exceeded_flag": credits_exceeded,
    "used_addon_credits":    addon_credits,
    "num_file_uploads":      num_file_uploads,
}, index=_ALL_USERS)
print(f"  -> {J.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE ALL FEATURE GROUPS
# ══════════════════════════════════════════════════════════════════════════════
print("\nMerging feature groups ...")

feature_matrix = pd.concat([A, B, C, D, E, F, G, H, I, J], axis=1)
# Remove accidental duplicate columns
feature_matrix = feature_matrix.loc[:, ~feature_matrix.columns.duplicated()]
# Replace inf
feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"  Feature matrix: {feature_matrix.shape[0]:,} users × {feature_matrix.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# SUCCESS LABEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUCCESS LABEL: Long-Term Success")
print("=" * 70)

# ─── Component 1: retained_28d ───────────────────────────────────────────────
# User has ≥1 event at day ≥ 28 from their own first event
_ev_with_first = raw[["user_id", "timestamp"]].merge(
    _first_ts.rename("first_ts").reset_index(), on="user_id"
)
_ev_with_first["day_on_platform"] = (
    (_ev_with_first["timestamp"] - _ev_with_first["first_ts"]).dt.total_seconds() / 86400
)
retained_28d = (
    _ev_with_first[_ev_with_first["day_on_platform"] >= 28]
    .groupby("user_id").size().gt(0)
    .reindex(_ALL_USERS, fill_value=False).astype(int)
)

# ─── Component 2: multi_week_3plus ───────────────────────────────────────────
# User active in ≥3 distinct ISO calendar weeks
_iso_wks = raw[["user_id", "timestamp"]].copy()
_iw = _iso_wks["timestamp"].dt.isocalendar()
_iso_wks["_yr_wk"] = _iw["year"].astype(str) + "_" + _iw["week"].astype(str)
multi_week_3plus = (
    _iso_wks.groupby("user_id")["_yr_wk"].nunique()
    .ge(3).reindex(_ALL_USERS, fill_value=False).astype(int)
)

# ─── Component 3: workflow_builder ───────────────────────────────────────────
# User created ≥2 blocks + ≥1 edge (built a real connected DAG)
workflow_builder = (
    (num_blocks_created >= 2) & (num_edges_created >= 1)
).astype(int)

# ─── Composite: long_term_success (≥2 of 3 criteria) ─────────────────────────
_composite_score = retained_28d + multi_week_3plus + workflow_builder
long_term_success = (_composite_score >= 2).astype(int).rename("long_term_success")

# Add components and label to feature matrix
feature_matrix["retained_28d"]      = retained_28d
feature_matrix["multi_week_3plus"]  = multi_week_3plus
feature_matrix["workflow_builder"]  = workflow_builder
feature_matrix["lts_score"]         = _composite_score   # 0/1/2/3
feature_matrix["long_term_success"] = long_term_success

n_success   = long_term_success.sum()
n_total     = len(long_term_success)
success_rate = n_success / n_total

print(f"""
  Definition: User meets ≥2 of 3 criteria:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. retained_28d     — active ≥ day 28 from their first event│
  │ 2. multi_week_3plus — events in ≥ 3 distinct ISO weeks      │
  │ 3. workflow_builder — created ≥2 blocks AND ≥1 edge         │
  └─────────────────────────────────────────────────────────────┘
  
  Rationale:
  • 28-day retention is industry-standard for 'sticky' product engagement
  • Multi-week breadth confirms habitual use, not a single exploration spike
  • Workflow building signals investment: user has learned the core product
  • Composite ≥2/3 avoids label noise (a user might just have a long account
    but never build; or build once but never return)
  
  Class balance:
    Successful users   : {n_success:>6,}  ({success_rate:.1%})
    Unsuccessful users : {n_total - n_success:>6,}  ({1-success_rate:.1%})
    Total              : {n_total:>6,}
    
  Component rates:
    retained_28d      : {retained_28d.mean():.1%}
    multi_week_3plus  : {multi_week_3plus.mean():.1%}
    workflow_builder  : {workflow_builder.mean():.1%}
""")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("FEATURE SUMMARY STATISTICS")
print("=" * 70)

_feature_cols = [c for c in feature_matrix.columns
                 if c not in {"retained_28d", "multi_week_3plus",
                               "workflow_builder", "lts_score", "long_term_success"}]
_feat_df = feature_matrix[_feature_cols]

_stats = _feat_df.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
_stats["null_pct"] = (_feat_df.isna().mean() * 100).round(1)
_stats["group"]    = [c.split("_")[0] if "_" in c else c for c in _feature_cols]

print(f"\nTotal features: {len(_feature_cols)}")
print("\nGroup breakdown:")
_grp_map = {"session": "[A]", "total": "[B]", "unique": "[B]",
            "category": "[B]", "event": "[B/E]", "used": "[B/F]",
            "events": "[B/A]", "num": "[C/F]", "has": "[C]",
            "block": "[C]", "edges": "[C]", "blocks": "[C]",
            "pct": "[D/F]", "advanced": "[D]", "active": "[E/H]",
            "multi": "[E]", "day": "[E]", "hour": "[E]",
            "activity": "[E]", "peak": "[E]", "ttv": "[E]",
            "agent": "[F]", "completed": "[G]", "submitted": "[G]",
            "skipped": "[G]", "explored": "[G]", "added": "[G]",
            "days": "[H]", "return": "[H]", "velocity": "[H]",
            "fs": "[I]", "total": "[J]", "credits": "[J]"}

_feature_groups = {
    "[A] Session Depth":                    [c for c in _feature_cols if c.startswith("session_") or c in ("events_per_session", "pct_multi_event_sessions", "sessions_per_week", "total_session_dur_min", "median_session_dur_min", "max_session_dur_min", "session_dur_cv", "session_count")],
    "[B] Feature Usage Breadth":            [c for c in _feature_cols if c in ("total_events_excl_credits", "unique_event_types", "category_breadth", "event_entropy", "used_agent", "used_file_ops", "used_sharing", "events_per_week")],
    "[C] Workflow Complexity":              [c for c in _feature_cols if c in ("num_blocks_created", "num_blocks_run", "num_edges_created", "num_canvases", "num_layers", "block_run_rate", "edges_per_canvas", "blocks_per_canvas", "has_complete_dag")],
    "[D] Reproducibility Signals":          [c for c in _feature_cols if c in ("num_run_all_blocks", "num_run_from_block", "num_run_upto_block", "num_block_stops", "advanced_run_ratio", "pct_sessions_with_code_run")],
    "[E] Time-Based Patterns":              [c for c in _feature_cols if c in ("total_active_days", "multi_day_user", "active_days_per_week", "day_of_week_entropy", "hour_of_day_entropy", "activity_trend", "peak_hour", "pct_daytime", "ttv_code_run_min", "ttv_edge_min", "ttv_agent_min")],
    "[F] Agent Usage Depth":               [c for c in _feature_cols if c.startswith("num_agent") or c in ("pct_events_agent", "agent_accept_rate")],
    "[G] Onboarding Completion":            [c for c in _feature_cols if c in ("completed_onboarding_tour", "submitted_onboarding_form", "skipped_onboarding", "explored_quickstart", "added_dataset_quickstart")],
    "[H] Recency & Velocity":               [c for c in _feature_cols if c in ("days_since_last_event", "active_last_7d", "active_last_14d", "active_last_30d", "return_rate", "velocity_ratio")],
    "[I] First-Session Power":              [c for c in _feature_cols if c.startswith("fs_")],
    "[J] Credit & Resource Signals":        [c for c in _feature_cols if c in ("total_credits_used", "credits_per_run", "credits_exceeded_flag", "used_addon_credits", "num_file_uploads")],
}

for _gname, _gcols in _feature_groups.items():
    _gcols_present = [c for c in _gcols if c in _feature_cols]
    print(f"  {_gname}: {len(_gcols_present)} features")

print("\n── Per-group mean comparison (success=1 vs success=0) ──────────────")
_success_mask = feature_matrix["long_term_success"] == 1
for _gname, _gcols in _feature_groups.items():
    _gcols_present = [c for c in _gcols if c in feature_matrix.columns]
    if not _gcols_present:
        continue
    _s1 = feature_matrix.loc[_success_mask, _gcols_present].mean()
    _s0 = feature_matrix.loc[~_success_mask, _gcols_present].mean()
    print(f"\n  {_gname}")
    _comp = pd.DataFrame({"success=1": _s1, "success=0": _s0})
    _comp["ratio (1/0)"] = (_s1 / _s0.replace(0, np.nan)).fillna(np.inf).round(2)
    _comp["success=1"] = _comp["success=1"].round(3)
    _comp["success=0"] = _comp["success=0"].round(3)
    print(_comp.to_string())

# Full numeric summary
print("\n── Full feature matrix statistics ──────────────────────────────────")
print(_stats[["mean", "std", "50%", "95%", "max", "null_pct"]].to_string())

# Save
feature_matrix.to_parquet("long_term_success_features.parquet")
print(f"\n✅ Saved long_term_success_features.parquet")
print(f"   {feature_matrix.shape[0]:,} users × {feature_matrix.shape[1]} columns")
print(f"   ({len(_feature_cols)} features + 5 label columns)")
print("=" * 70)
print("DONE — long_term_success label defined and feature matrix built.")
print("=" * 70)
