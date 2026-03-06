
# Phase 0 — Hardened Data Pipeline (Rebuilt from Scratch)
# Loads user_retention DataFrame → cleans → sessionizes → filters bots → saves

import numpy as np
import pandas as pd

# ─── Config ────────────────────────────────────────────────────────────────────

SESSION_GAP_MINUTES = 30
BOT_MIN_EVENTS = 3        # users with < 3 total events
BOT_MIN_SESSION_SEC = 60  # AND all sessions shorter than 60s → spam

# Comprehensive column rename map (raw PostHog → clean names)
P0_COLUMN_MAP = {
    "distinct_id":                  "user_id",
    "person_id":                    "person_id",
    "event":                        "event",
    "timestamp":                    "timestamp",
    "created_at":                   "created_at",
    "uuid":                         "event_uuid",
    "prop_$session_id":             "session_id_raw",
    "prop_$browser":                "browser",
    "prop_$os":                     "os",
    "prop_$device_type":            "device_type",
    "prop_$geoip_country_name":     "country",
    "prop_$geoip_country_code":     "country_code",
    "prop_$timezone":               "timezone",
    "prop_$referrer":               "referrer",
    "prop_$referring_domain":       "referring_domain",
    "prop_$pathname":               "pathname",
    "prop_$current_url":            "current_url",
    "prop_$screen_width":           "screen_width",
    "prop_$screen_height":          "screen_height",
    "prop_surface":                 "surface",
    "prop_tool_name":               "tool_name",
    "prop_credit_amount":           "credit_amount",
    "prop_credits_used":            "credits_used",
    "prop_$lib":                    "sdk_lib",
}

# Columns to keep (clean names, in order)
P0_KEEP_COLS = list(P0_COLUMN_MAP.values())

# Drop columns matching these prefixes
P0_DROP_PREFIXES = ("prop_$set.", "prop_$set_once.")

# Clean event taxonomy — 9 categories
# Format: event_name → category
P0_EVENT_CATEGORIES = {
    # code_execution: running/stopping blocks
    "code_execution": [
        "run_block", "run_all_blocks", "run_upto_block", "run_from_block", "stop_block",
    ],
    # agent_use: all agent interactions
    "agent_use": [
        "agent_block_created", "agent_worker_created",
        "agent_tool_call_run_block_tool", "agent_tool_call_create_block_tool",
        "agent_tool_call_get_block_tool", "agent_tool_call_refactor_block_tool",
        "agent_tool_call_finish_ticket_tool", "agent_tool_call_get_canvas_summary_tool",
        "agent_tool_call_get_variable_preview_tool", "agent_tool_call_delete_block_tool",
        "agent_tool_call_create_edges_tool", "agent_tool_call_run_block_tool",
        "agent_message", "agent_start_from_prompt", "agent_accept_suggestion",
        "agent_new_chat", "agent_open", "agent_suprise_me",
        "agent_open_error_assist", "agent_upload_files",
    ],
    # workflow_build: creating/modifying canvas objects
    "workflow_build": [
        "block_create", "canvas_create", "edge_create", "layer_create",
        "block_delete", "block_resize", "edge_delete", "block_rename",
        "block_copy", "block_open_compute_settings", "block_output_copy",
        "app_publish", "scheduled_job_start", "scheduled_job_stop",
    ],
    # navigation: browsing, fullscreen, UI interactions
    "navigation": [
        "canvas_open", "fullscreen_open", "fullscreen_close",
        "fullscreen_preview_output", "fullscreen_preview_input",
        "folder_open", "link_clicked", "button_clicked",
        "canvas_onboarding_tour_running_blocks_step",
        "canvas_onboarding_tour_code_and_variables_step",
        "canvas_onboarding_tour_compute_step",
        "canvas_onboarding_tour_add_block_step",
        "canvas_onboarding_tour_ai_assistant_step",
        "canvas_onboarding_tour_you_are_ready_step",
    ],
    # credits: credit consumption events
    "credits": [
        "credits_used", "addon_credits_used",
        "credits_below_1", "credits_below_2", "credits_below_3", "credits_below_4",
        "credits_exceeded",
    ],
    # onboarding: sign-up flow and first-time guidance
    "onboarding": [
        "sign_in", "sign_up", "new_user_created",
        "promo_code_redeemed", "referral_modal_open",
        "skip_onboarding_form", "submit_onboarding_form",
        "canvas_onboarding_tour_started", "canvas_onboarding_tour_finished",
        "quickstart_explore_playground", "quickstart_add_dataset",
    ],
    # file_ops: file system operations
    "file_ops": [
        "files_upload", "files_download", "files_delete",
    ],
    # sharing: publishing and collaboration
    "sharing": [
        "app_publish", "share_canvas", "invite_user",
        "canvas_shared", "app_view",
    ],
}

# Build reverse lookup: event_name → category
_P0_EVENT_TO_CAT = {}
for _cat_name, _ev_list in P0_EVENT_CATEGORIES.items():
    for _ev_name in _ev_list:
        _P0_EVENT_TO_CAT[_ev_name] = _cat_name

# ─── Step 1: Load raw data ─────────────────────────────────────────────────────

print("=" * 65)
print("PHASE 0 — HARDENED DATA PIPELINE")
print("=" * 65)

# Use the user_retention DataFrame variable already in memory
p0_raw = user_retention.parquest.copy()
print(f"\n[LOAD] Raw rows: {len(p0_raw):,}  |  Columns: {len(p0_raw.columns)}")

# ─── Step 2: Column standardization ───────────────────────────────────────────

# Drop prop_$set.* and prop_$set_once.* columns
p0_drop_cols = [c for c in p0_raw.columns if c.startswith(P0_DROP_PREFIXES)]
p0_raw = p0_raw.drop(columns=p0_drop_cols)
print(f"[COLS] Dropped {len(p0_drop_cols)} prop_set columns")

# Rename mapped columns
p0_rename_map = {k: v for k, v in P0_COLUMN_MAP.items() if k in p0_raw.columns}
p0_raw = p0_raw.rename(columns=p0_rename_map)
print(f"[COLS] Renamed {len(p0_rename_map)} columns using mapping")

# Keep only clean columns that are present
p0_keep = [c for c in P0_KEEP_COLS if c in p0_raw.columns]
p0_raw = p0_raw[p0_keep].copy()
print(f"[COLS] Final schema ({len(p0_keep)} columns): {p0_keep}")

# ─── Step 3: Timestamp normalization ──────────────────────────────────────────

for _ts_col in ["timestamp", "created_at"]:
    if _ts_col in p0_raw.columns:
        if not pd.api.types.is_datetime64_any_dtype(p0_raw[_ts_col]):
            p0_raw[_ts_col] = pd.to_datetime(p0_raw[_ts_col], utc=True, errors="coerce")
        elif p0_raw[_ts_col].dt.tz is None:
            p0_raw[_ts_col] = p0_raw[_ts_col].dt.tz_localize("UTC")
        else:
            p0_raw[_ts_col] = p0_raw[_ts_col].dt.tz_convert("UTC")

n_null_ts = p0_raw["timestamp"].isna().sum()
if n_null_ts > 0:
    print(f"[TS]   WARNING: {n_null_ts:,} null timestamps — dropping")
    p0_raw = p0_raw.dropna(subset=["timestamp"])

print(f"[TS]   Timestamps normalized to UTC. Range: "
      f"{p0_raw['timestamp'].min().date()} → {p0_raw['timestamp'].max().date()}")

# ─── Step 4: user_id fill & dedup ─────────────────────────────────────────────

_null_uid = p0_raw["user_id"].isna()
if _null_uid.any():
    p0_raw.loc[_null_uid, "user_id"] = p0_raw.loc[_null_uid, "person_id"]
    print(f"[USER] Filled {_null_uid.sum():,} null user_ids from person_id")

_before_dedup = len(p0_raw)
p0_raw = p0_raw.dropna(subset=["user_id"])
print(f"[USER] Dropped {_before_dedup - len(p0_raw):,} rows with still-null user_id")

if "event_uuid" in p0_raw.columns:
    _before_dedup = len(p0_raw)
    p0_raw = p0_raw.drop_duplicates(subset=["event_uuid"], keep="first")
    print(f"[DEDUP] Dropped {_before_dedup - len(p0_raw):,} duplicate event_uuids")

# ─── Step 5: Event taxonomy ────────────────────────────────────────────────────

p0_raw = p0_raw.sort_values("timestamp").reset_index(drop=True)
p0_raw["event_category"] = p0_raw["event"].map(_P0_EVENT_TO_CAT).fillna("other")

print(f"\n[TAXONOMY] Event category distribution:")
_cat_counts = p0_raw["event_category"].value_counts()
for _cat_name, _cnt in _cat_counts.items():
    _pct = 100.0 * _cnt / len(p0_raw)
    print(f"  {_cat_name:<20} {_cnt:>8,}  ({_pct:.1f}%)")

# ─── Step 6: Sessionize (30-min inactivity gap per user) ──────────────────────

p0_raw = p0_raw.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

_time_diff = p0_raw.groupby("user_id")["timestamp"].diff()
_gap_threshold = pd.Timedelta(minutes=SESSION_GAP_MINUTES)
_new_session_flag = _time_diff.isna() | (_time_diff > _gap_threshold)

p0_raw["session_seq"] = _new_session_flag.groupby(p0_raw["user_id"]).cumsum().astype(int)
p0_raw["session_id_derived"] = p0_raw["user_id"] + "_S" + p0_raw["session_seq"].astype(str)
p0_raw["event_seq_in_session"] = p0_raw.groupby("session_id_derived").cumcount() + 1

_sess_starts = p0_raw.groupby("session_id_derived")["timestamp"].transform("min")
p0_raw["time_since_session_start_s"] = (
    (p0_raw["timestamp"] - _sess_starts).dt.total_seconds()
)

print(f"\n[SESSION] Sessions created: {p0_raw['session_id_derived'].nunique():,}")
print(f"[SESSION] Users: {p0_raw['user_id'].nunique():,}")

# ─── Step 7: Build sessions table ─────────────────────────────────────────────

p0_sessions_agg = p0_raw.groupby("session_id_derived").agg(
    user_id=("user_id", "first"),
    session_start=("timestamp", "min"),
    session_end=("timestamp", "max"),
    num_events=("event", "count"),
    num_unique_events=("event", "nunique"),
    num_agent_events=("event_category", lambda x: (x == "agent_use").sum()),
    num_code_runs=("event", lambda x: (x == "run_block").sum()),
    num_blocks_created=("event", lambda x: (x == "block_create").sum()),
    num_canvases_opened=("event", lambda x: (x == "canvas_open").sum()),
    had_sign_up=("event", lambda x: (x == "sign_up").any()),
    session_seq=("session_seq", "first"),
).reset_index()

p0_sessions_agg["duration_seconds"] = (
    (p0_sessions_agg["session_end"] - p0_sessions_agg["session_start"])
    .dt.total_seconds()
)
p0_sessions_agg["duration_minutes"] = p0_sessions_agg["duration_seconds"] / 60.0

_cat_str = (
    p0_raw.groupby("session_id_derived")["event_category"]
    .apply(lambda x: ",".join(sorted(x.unique())))
    .reset_index()
    .rename(columns={"event_category": "event_categories_used"})
)
p0_sessions_agg = p0_sessions_agg.merge(_cat_str, on="session_id_derived", how="left")

p0_sessions_agg["used_agent"] = p0_sessions_agg["num_agent_events"] > 0
p0_sessions_agg["date"] = p0_sessions_agg["session_start"].dt.date
p0_sessions_agg["hour_of_day"] = p0_sessions_agg["session_start"].dt.hour
p0_sessions_agg["day_of_week"] = p0_sessions_agg["session_start"].dt.dayofweek

print(f"[SESSION] Sessions table built: {len(p0_sessions_agg):,} rows x {len(p0_sessions_agg.columns)} cols")

# ─── Step 8: Filter bot/spam users ────────────────────────────────────────────
# Criterion: total_events < 3 AND max_session_duration < 60s

_user_stats = p0_sessions_agg.groupby("user_id").agg(
    _total_events=("num_events", "sum"),
    _max_session_sec=("duration_seconds", "max"),
).reset_index()

_bot_mask = (
    (_user_stats["_total_events"] < BOT_MIN_EVENTS) &
    (_user_stats["_max_session_sec"] < BOT_MIN_SESSION_SEC)
)
_bot_users = set(_user_stats.loc[_bot_mask, "user_id"])

n_users_before = p0_raw["user_id"].nunique()
n_bots = len(_bot_users)

p0_events_clean = p0_raw[~p0_raw["user_id"].isin(_bot_users)].reset_index(drop=True)
p0_sessions_clean = p0_sessions_agg[~p0_sessions_agg["user_id"].isin(_bot_users)].reset_index(drop=True)

print(f"\n[BOT FILTER] Users before: {n_users_before:,}")
print(f"[BOT FILTER] Bot/spam users removed: {n_bots:,}  (< {BOT_MIN_EVENTS} events AND < {BOT_MIN_SESSION_SEC}s max session)")
print(f"[BOT FILTER] Users after:  {p0_events_clean['user_id'].nunique():,}")
print(f"[BOT FILTER] Events before: {len(p0_raw):,}  |  Events after: {len(p0_events_clean):,}")

# ─── Step 9: Save ─────────────────────────────────────────────────────────────

p0_events_clean.to_parquet("events_clean.parquet", index=False)
p0_sessions_clean.to_parquet("sessions_clean.parquet", index=False)
print(f"\n[SAVE] events_clean.parquet   → {len(p0_events_clean):,} rows")
print(f"[SAVE] sessions_clean.parquet → {len(p0_sessions_clean):,} rows")

# ─── Step 10: Final summary ────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PHASE 0 SUMMARY")
print("=" * 65)
print(f"  Raw events loaded:        {len(p0_raw):,}")
print(f"  Clean events:             {len(p0_events_clean):,}")
print(f"  Clean sessions:           {len(p0_sessions_clean):,}")
print(f"  Clean users:              {p0_events_clean['user_id'].nunique():,}")
print(f"  Bot users filtered:       {n_bots:,}")
print(f"  Date range:               {p0_events_clean['timestamp'].min().date()} → {p0_events_clean['timestamp'].max().date()}")
print(f"  Unique event types:       {p0_events_clean['event'].nunique()}")
print(f"  Columns in events_clean:  {len(p0_events_clean.columns)}")
print(f"  Columns in sessions_clean:{len(p0_sessions_clean.columns)}")

print("\n[SCHEMA] events_clean columns:")
for _c in p0_events_clean.columns:
    print(f"  {_c:<35} {str(p0_events_clean[_c].dtype)}")

print("\n[CATEGORY DISTRIBUTION] (clean events only):")
_clean_cats = p0_events_clean["event_category"].value_counts()
for _cat_name, _cnt in _clean_cats.items():
    _pct = 100.0 * _cnt / len(p0_events_clean)
    print(f"  {_cat_name:<20} {_cnt:>8,}  ({_pct:.1f}%)")

print("\n✅ Phase 0 complete — events_clean.parquet + sessions_clean.parquet saved")
