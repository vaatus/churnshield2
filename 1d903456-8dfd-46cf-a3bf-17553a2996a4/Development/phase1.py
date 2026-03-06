# ZERVE BLOCK: Exploratory Data Analysis
# Phase 1 — event landscape, user population, engagement, agent usage, sessions, journeys

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PLOT_DPI = 150
FIGURES_DIR = "figures"
REPORTS_DIR = "reports"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")

EVENT_CATEGORIES = {
    "agent": [
        "agent_block_created", "agent_worker_created",
        "agent_tool_call_run_block_tool", "agent_tool_call_create_block_tool",
        "agent_tool_call_get_block_tool", "agent_tool_call_refactor_block_tool",
        "agent_tool_call_finish_ticket_tool", "agent_tool_call_get_canvas_summary_tool",
        "agent_tool_call_get_variable_preview_tool", "agent_tool_call_delete_block_tool",
        "agent_tool_call_create_edges_tool",
        "agent_message", "agent_start_from_prompt", "agent_accept_suggestion",
        "agent_new_chat", "agent_open", "agent_suprise_me",
        "agent_open_error_assist", "agent_upload_files",
    ],
    "code_execution": [
        "run_block", "run_all_blocks", "run_upto_block",
        "run_from_block", "stop_block",
    ],
    "creation": [
        "block_create", "canvas_create", "edge_create",
        "layer_create", "app_publish", "scheduled_job_start",
    ],
    "modification": [
        "block_delete", "block_resize", "edge_delete",
        "block_rename", "block_copy", "block_open_compute_settings",
        "block_output_copy", "scheduled_job_stop",
    ],
    "navigation": [
        "canvas_open", "fullscreen_open", "fullscreen_close",
        "fullscreen_preview_output", "fullscreen_preview_input",
        "folder_open", "link_clicked", "button_clicked",
    ],
    "auth": [
        "sign_in", "sign_up", "new_user_created",
        "promo_code_redeemed", "referral_modal_open",
    ],
    "onboarding": [
        "skip_onboarding_form", "submit_onboarding_form",
        "canvas_onboarding_tour_started", "canvas_onboarding_tour_finished",
        "canvas_onboarding_tour_running_blocks_step",
        "canvas_onboarding_tour_code_and_variables_step",
        "canvas_onboarding_tour_compute_step",
        "canvas_onboarding_tour_add_block_step",
        "canvas_onboarding_tour_ai_assistant_step",
        "canvas_onboarding_tour_you_are_ready_step",
        "quickstart_explore_playground", "quickstart_add_dataset",
    ],
    "file_ops": ["files_upload", "files_download", "files_delete"],
    "credits": [
        "credits_used", "addon_credits_used",
        "credits_below_1", "credits_below_2",
        "credits_below_3", "credits_below_4",
        "credits_exceeded",
    ],
}


def _savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── Load Phase 0 outputs ────────────────────────────────────────────────────

events_df = pd.read_parquet("events.parquet")
sessions_df = pd.read_parquet("sessions.parquet")

# Ensure timestamp is a proper datetime column
if "timestamp" not in events_df.columns:
    for col in events_df.columns:
        if "time" in col.lower() or "date" in col.lower():
            events_df = events_df.rename(columns={col: "timestamp"})
            break
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True)
logger.info("Loaded events (%d rows) and sessions (%d rows)", len(events_df), len(sessions_df))
logger.info("Columns: %s", list(events_df.columns))

# ── 1. Event Landscape ──────────────────────────────────────────────────────

event_counts = events_df["event"].value_counts()
cat_counts = events_df["event_category"].value_counts()

fig, ax = plt.subplots(figsize=(10, 8))
event_counts.head(30).iloc[::-1].plot.barh(ax=ax)
ax.set_title("Top 30 Event Types")
ax.set_xlabel("Count")
_savefig(fig, "event_frequency.png")

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%", pctdistance=0.8, startangle=90)
ax.add_patch(plt.Circle((0, 0), 0.5, fc="white"))
ax.set_title("Event Category Distribution")
_savefig(fig, "event_categories.png")

hourly = events_df.groupby(events_df["timestamp"].dt.floor("h")).size()
fig, ax = plt.subplots(figsize=(14, 5))
hourly.plot(ax=ax)
ax.set_title("Events Over Time (Hourly)")
ax.set_xlabel("Time")
ax.set_ylabel("Event Count")
_savefig(fig, "events_timeline.png")

agent_count = cat_counts.get("agent", 0)
total_events = cat_counts.sum()
agent_ratio = agent_count / total_events

# ── 2. User Population ──────────────────────────────────────────────────────

user_events = events_df.groupby("user_id").size()
user_sessions = sessions_df.groupby("user_id").size()
total_users = events_df["user_id"].nunique()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
user_events.clip(upper=user_events.quantile(0.95)).plot.hist(bins=50, ax=axes[0], edgecolor="black", alpha=0.7)
axes[0].set_title("Events per User (clipped 95th pct)")
axes[0].set_xlabel("Events")
user_events.clip(upper=user_events.quantile(0.95)).plot.box(ax=axes[1])
axes[1].set_title("Events per User Box Plot")
_savefig(fig, "events_per_user.png")

active_days = events_df.groupby("user_id")["timestamp"].apply(lambda x: x.dt.date.nunique())
user_tenure = events_df.groupby("user_id")["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds() / 3600)

if "country" in events_df.columns:
    country_users = events_df.dropna(subset=["country"]).groupby("country")["user_id"].nunique()
    top20 = country_users.nlargest(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    top20.iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Users by Country (Top 20)")
    ax.set_xlabel("Unique Users")
    _savefig(fig, "geo_distribution.png")

# ── 3. Engagement Depth ─────────────────────────────────────────────────────

user_event_types = events_df.groupby("user_id")["event"].nunique()

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(user_events, user_event_types, alpha=0.4, s=15)
ax.set_xlabel("Total Events")
ax.set_ylabel("Unique Event Types")
ax.set_title("User Engagement Spectrum")
ax.set_xscale("log")
_savefig(fig, "engagement_spectrum.png")

def users_with_event(ev):
    return set(events_df.loc[events_df["event"] == ev, "user_id"].unique())

def users_with_category(cat):
    return set(events_df.loc[events_df["event_category"] == cat, "user_id"].unique())

funnel = {
    "total_users": total_users,
    "signed_up": len(users_with_event("sign_up")),
    "created_canvas": len(users_with_event("canvas_create")),
    "created_block": len(users_with_event("block_create")),
    "ran_block": len(users_with_event("run_block")),
    "used_agent": len(users_with_category("agent")),
    "created_edges": len(users_with_event("edge_create")),
}

fig, ax = plt.subplots(figsize=(10, 6))
labels = list(funnel.keys())
values = list(funnel.values())
bars = ax.barh(labels[::-1], values[::-1])
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + total_users * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({val / total_users * 100:.1f}%)", va="center", fontsize=9)
ax.set_title("Feature Adoption Funnel")
ax.set_xlabel("Users")
_savefig(fig, "adoption_funnel.png")

# ── 4. Agent Usage ──────────────────────────────────────────────────────────

agent_events = events_df[events_df["event_category"] == "agent"]
agent_users = agent_events["user_id"].unique()
agent_per_user = agent_events.groupby("user_id").size()
agent_users_pct = len(agent_users) / total_users

total_per_user = events_df.groupby("user_id").size()
merged = pd.DataFrame({"total": total_per_user, "agent": agent_per_user}).fillna(0)
agent_corr = merged["total"].corr(merged["agent"])

fig, ax = plt.subplots(figsize=(8, 6))
plot_data = pd.DataFrame({
    "User Type": ["Agent User"] * len(agent_per_user) + ["Non-Agent"] * (total_users - len(agent_users)),
    "Total Events": list(total_per_user.reindex(agent_users).dropna().values) +
                   list(total_per_user.drop(agent_users, errors="ignore").values),
})
if len(plot_data) > 0:
    sns.violinplot(data=plot_data, x="User Type", y="Total Events", ax=ax, cut=0)
    ax.set_yscale("log")
ax.set_title("Engagement: Agent Users vs Non-Agent Users")
_savefig(fig, "agent_usage.png")

# ── 5. Session Patterns ─────────────────────────────────────────────────────

user_dates = sessions_df.groupby("user_id")["date"].nunique()
multi_day = (user_dates > 1).sum()
return_rate = multi_day / len(user_dates)

pos_dur = sessions_df["duration_minutes"][sessions_df["duration_minutes"] > 0]
fig, ax = plt.subplots(figsize=(10, 5))
if len(pos_dur) > 0:
    ax.hist(pos_dur.clip(upper=pos_dur.quantile(0.99)), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xscale("log")
ax.set_title("Session Duration Distribution")
ax.set_xlabel("Duration (minutes, log scale)")
ax.set_ylabel("Count")
_savefig(fig, "session_duration.png")

hourly_sessions = sessions_df["hour_of_day"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
hourly_sessions.plot.bar(ax=ax, edgecolor="black", alpha=0.7)
ax.set_title("Sessions by Hour of Day (UTC)")
ax.set_xlabel("Hour")
ax.set_ylabel("Session Count")
_savefig(fig, "hourly_activity.png")

# ── 6. User Journeys ────────────────────────────────────────────────────────

sorted_events = events_df.sort_values(["user_id", "timestamp"])
sorted_events["next_event"] = sorted_events.groupby("user_id")["event"].shift(-1)
transitions = sorted_events.dropna(subset=["next_event"])

top15_events = events_df["event"].value_counts().head(15).index.tolist()
trans_matrix = (
    transitions[transitions["event"].isin(top15_events) & transitions["next_event"].isin(top15_events)]
    .groupby(["event", "next_event"]).size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 10))
if trans_matrix.size > 0:
    sns.heatmap(trans_matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Event Transition Heatmap (Top 15)")
ax.set_xlabel("Next Event")
ax.set_ylabel("Current Event")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
_savefig(fig, "event_transitions.png")

user_first = events_df.groupby("user_id")["timestamp"].min()
first_code = events_df[events_df["event"] == "run_block"].groupby("user_id")["timestamp"].min()
time_to_code = (first_code - user_first).dropna().dt.total_seconds() / 60
first_agent = events_df[events_df["event_category"] == "agent"].groupby("user_id")["timestamp"].min()
time_to_agent = (first_agent - user_first).dropna().dt.total_seconds() / 60

all_users_set = set(events_df["user_id"].unique())
n_users = len(all_users_set)

user_dates = events_df.groupby("user_id")["timestamp"].apply(lambda x: x.dt.date.nunique())
retained_users = set(user_dates[user_dates > 1].index)
base_retention = len(retained_users) / n_users
min_users = max(20, int(n_users * 0.01))

user_first_day = events_df.groupby("user_id")["timestamp"].min().dt.date
events_copy = events_df.copy()
events_copy["_day"] = events_copy["timestamp"].dt.date
events_copy["_first_day"] = events_copy["user_id"].map(user_first_day)
first_day_events = events_copy[events_copy["_day"] == events_copy["_first_day"]]

aha_data = []
for ev in first_day_events["event"].unique():
    ev_users = set(first_day_events.loc[first_day_events["event"] == ev, "user_id"].unique())
    if len(ev_users) < min_users:
        continue
    non_ev_users = all_users_set - ev_users
    ret_with = len(ev_users & retained_users) / len(ev_users) if ev_users else 0
    ret_without = len(non_ev_users & retained_users) / len(non_ev_users) if non_ev_users else 0
    aha_data.append({
        "event": ev, "retention_with": ret_with, "retention_without": ret_without,
        "lift": ret_with / base_retention if base_retention > 0 else 0,
        "n_users": len(ev_users),
    })

aha_df = pd.DataFrame(aha_data).sort_values("retention_with", ascending=False)

fig, ax = plt.subplots(figsize=(11, 7))
top_aha = aha_df.head(20).iloc[::-1]
ax.barh(top_aha["event"], top_aha["retention_with"] * 100)
for i, (_, row) in enumerate(top_aha.iterrows()):
    ax.text(row["retention_with"] * 100 + 0.5, i,
            f"{row['lift']:.1f}x lift  (n={row['n_users']:,.0f})",
            va="center", fontsize=8)
ax.set_title("Aha Moment: First-Day Actions That Predict Retention")
ax.set_xlabel("Retention Rate (%)")
ax.axvline(x=base_retention * 100, color="red", linestyle="--", alpha=0.7,
           label=f"overall retention ({base_retention:.1%})")
ax.legend()
_savefig(fig, "aha_moment.png")

# ── Report ───────────────────────────────────────────────────────────────────

report_path = os.path.join(REPORTS_DIR, "eda_report.md")
lines = [
    "# ChurnShield EDA Report",
    "",
    "## Executive Summary",
    "",
    f"- **{total_users:,} users** generated **{total_events:,} events** across **{events_df['event'].nunique()} event types**",
    f"- Agent adoption: **{agent_users_pct * 100:.1f}%** of users have used the AI agent",
    f"- Return rate: **{return_rate * 100:.1f}%** of users active on 2+ days",
    f"- Median session duration: **{sessions_df['duration_minutes'].median():.1f} minutes**",
    f"- Agent usage correlates with engagement (r={agent_corr:.2f})",
    "",
    "## 1. Event Landscape",
    "",
    f"Total event types: {events_df['event'].nunique()}",
    f"Agent event ratio: {agent_ratio * 100:.1f}%",
    "",
    "![Event Frequency](../figures/event_frequency.png)",
    "![Event Categories](../figures/event_categories.png)",
    "![Events Timeline](../figures/events_timeline.png)",
    "",
    "## 2. User Population",
    "",
    f"- Total users: {total_users:,}",
    f"- Median events per user: {user_events.median():.0f}",
    f"- Median sessions per user: {user_sessions.median():.0f}",
    f"- Median active days: {active_days.median():.0f}",
    f"- Median tenure: {user_tenure.median():.1f} hours",
    "",
    "![Events per User](../figures/events_per_user.png)",
    "![Geographic Distribution](../figures/geo_distribution.png)",
    "",
    "## 3. Engagement Depth & Adoption Funnel",
    "",
]
for step, count in funnel.items():
    pct = count / total_users * 100
    lines.append(f"- {step}: {count:,} ({pct:.1f}%)")
lines += [
    "",
    "![Adoption Funnel](../figures/adoption_funnel.png)",
    "![Engagement Spectrum](../figures/engagement_spectrum.png)",
    "",
    "## 4. Agent Usage",
    "",
    f"- Agent users: {len(agent_users):,} ({agent_users_pct * 100:.1f}%)",
    f"- Median agent events per agent user: {agent_per_user.median():.0f}",
    f"- Agent-engagement correlation: {agent_corr:.2f}",
    "",
    "![Agent Usage](../figures/agent_usage.png)",
    "",
    "## 5. Session Patterns",
    "",
    f"- Total sessions: {len(sessions_df):,}",
    f"- Median duration: {sessions_df['duration_minutes'].median():.1f} min",
    f"- Return rate: {return_rate * 100:.1f}%",
    f"- Multi-day users: {multi_day:,}",
    "",
    "![Session Duration](../figures/session_duration.png)",
    "![Hourly Activity](../figures/hourly_activity.png)",
    "",
    "## 6. User Journeys & Aha Moments",
    "",
    f"- Median time to first code run: {time_to_code.median():.1f} min",
    f"- Median time to first agent use: {time_to_agent.median():.1f} min",
    "",
    "![Event Transitions](../figures/event_transitions.png)",
    "![Aha Moment](../figures/aha_moment.png)",
    "",
    "## Hypotheses for Success Definition",
    "",
    "1. Agent adoption is the strongest signal of platform engagement",
    "2. Users who run code in their first session are more likely to return",
    "3. DAG creation (edges) indicates advanced, sustained usage",
    "4. Multi-day users show qualitatively different behavior than single-day",
    "5. Early agent usage predicts long-term engagement depth",
]
with open(report_path, "w") as f:
    f.write("\n".join(lines))

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"EDA COMPLETE")
print(f"{'='*60}")
print(f"Users: {total_users:,}")
print(f"Agent adoption: {agent_users_pct*100:.1f}%")
print(f"Return rate: {return_rate*100:.1f}%")
print(f"Agent-engagement correlation: {agent_corr:.2f}")
print(f"\nAdoption funnel:")
for step, count in funnel.items():
    print(f"  {step}: {count:,} ({count/total_users*100:.1f}%)")
print(f"\nAha moments (first-day actions → retention, baseline {base_retention:.1%}):")
for _, row in aha_df.head(5).iterrows():
    print(f"  {row['event']}: {row['retention_with']:.1%} retention "
          f"({row['lift']:.1f}x lift, n={row['n_users']:,})")
print(f"\nSaved 12 figures to {FIGURES_DIR}/")
print(f"Report: {report_path}")
