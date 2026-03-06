"""
Behavioral Archetype Segmentation
===================================
1. Load feature matrix from long_term_success_features.parquet
2. Select behavioral features (exclude label columns)
3. Find optimal K via silhouette + elbow (K-Means, k=3-6)
4. Cluster users into 3-6 meaningful archetypes
5. Reduce to 2D via UMAP (fallback: PCA) for visualization
6. Profile each cluster: success rate, key behaviors
7. Assign meaningful archetype labels
8. Visualize with Zerve design system
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Zerve design system ─────────────────────────────────────────────────────
BG   = "#1D1D20"
FG   = "#fbfbff"
SG   = "#909094"
HL   = "#ffd400"
SUC  = "#17b26a"
WARN = "#f04438"
PAL  = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B",
        "#D0BBFF", "#1F77B4", "#9467BD", "#8C564B"]

print("=" * 70)
print("  BEHAVIORAL ARCHETYPE SEGMENTATION")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] Loading feature matrix...")

seg_raw = pd.read_parquet("long_term_success_features.parquet")
print(f"    Loaded: {len(seg_raw):,} users x {seg_raw.shape[1]} columns")

LABEL_COLS = {"retained_28d", "multi_week_3plus", "workflow_builder",
              "lts_score", "long_term_success"}

feat_cols = [c for c in seg_raw.columns if c not in LABEL_COLS]
print(f"    Behavioral features: {len(feat_cols)}")

X_raw = seg_raw[feat_cols].copy()
y_success = seg_raw["long_term_success"].values
success_rate_overall = float(y_success.mean())
print(f"    Overall long_term_success rate: {success_rate_overall:.1%}")

# ══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESS
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Preprocessing...")

X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

ttv_cols = [c for c in feat_cols if "ttv_" in c]
for c in ttv_cols:
    _p99 = X_raw[c].quantile(0.99)
    _fill = float(_p99 * 1.5) if _p99 > 0 else 9999.0
    X_raw[c] = X_raw[c].fillna(_fill)

X_raw.fillna(0, inplace=True)

_var = X_raw.var()
_zero_var = list(_var[_var == 0].index)
if _zero_var:
    print(f"    Dropping {len(_zero_var)} zero-variance features")
    X_raw = X_raw.drop(columns=_zero_var)
    feat_cols = [c for c in feat_cols if c not in _zero_var]

print(f"    Final feature count: {X_raw.shape[1]}")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"    Scaled matrix: {X_scaled.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 3. FIND OPTIMAL K (silhouette + inertia)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Finding optimal K (k=2..7)...")

np.random.seed(42)
k_range = list(range(2, 8))
inertias = []
silhouettes = []

for k in k_range:
    _km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=42, max_iter=500)
    _labels = _km.fit_predict(X_scaled)
    inertias.append(float(_km.inertia_))
    _sil = float(silhouette_score(X_scaled, _labels,
                                  sample_size=min(1000, len(X_scaled)),
                                  random_state=42))
    silhouettes.append(_sil)
    print(f"    k={k}: inertia={_km.inertia_:.0f}, silhouette={_sil:.4f}")

k_scores_36 = {k: silhouettes[k - 2] for k in range(3, 7)}
best_k = int(max(k_scores_36, key=k_scores_36.get))
print(f"\n    Best K = {best_k} (silhouette={k_scores_36[best_k]:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 4. FIT FINAL K-MEANS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[4] Fitting KMeans with k={best_k}...")

kmeans_final = KMeans(n_clusters=best_k, init="k-means++", n_init=30,
                      random_state=42, max_iter=1000)
archetype_labels = kmeans_final.fit_predict(X_scaled)
cluster_sizes = list(np.bincount(archetype_labels))
print(f"    Cluster sizes: {cluster_sizes}")

final_sil = float(silhouette_score(X_scaled, archetype_labels,
                                   sample_size=min(1000, len(X_scaled)),
                                   random_state=42))
print(f"    Final silhouette: {final_sil:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. DIMENSIONALITY REDUCTION (UMAP or PCA fallback)
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Dimensionality reduction for visualization...")

try:
    import umap as umap_module
    print("    Using UMAP...")
    reducer = umap_module.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                               metric="euclidean", random_state=42, verbose=False)
    X_2d = reducer.fit_transform(X_scaled)
    dim_method = "UMAP"
    print("    UMAP complete.")
except ImportError:
    print("    UMAP not available — using PCA instead.")
    pca_obj = PCA(n_components=2, random_state=42)
    X_2d = pca_obj.fit_transform(X_scaled)
    pca_var = float(pca_obj.explained_variance_ratio_.sum() * 100)
    dim_method = f"PCA ({pca_var:.1f}% var)"
    print(f"    PCA variance explained: {pca_var:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 6. PROFILE EACH CLUSTER
# ══════════════════════════════════════════════════════════════════════════
print("\n[6] Profiling clusters...")

seg_df = seg_raw.copy()
seg_df["cluster_id"] = archetype_labels
seg_df["dim1"] = X_2d[:, 0]
seg_df["dim2"] = X_2d[:, 1]

PROFILE_FEATURES = [
    "session_count", "total_session_dur_min", "median_session_dur_min",
    "events_per_session", "sessions_per_week", "pct_multi_event_sessions",
    "total_events_excl_credits", "unique_event_types", "category_breadth",
    "event_entropy", "used_agent", "num_agent_events", "pct_events_agent",
    "agent_accept_rate", "num_blocks_created", "num_blocks_run",
    "num_edges_created", "has_complete_dag", "block_run_rate",
    "edges_per_canvas", "blocks_per_canvas",
    "num_run_all_blocks", "advanced_run_ratio", "pct_sessions_with_code_run",
    "total_active_days", "multi_day_user", "active_days_per_week",
    "day_of_week_entropy", "activity_trend",
    "days_since_last_event", "active_last_7d", "active_last_14d", "return_rate",
    "velocity_ratio", "fs_events", "fs_duration_min", "fs_code_runs",
    "fs_blocks_created", "fs_edges_created", "fs_used_agent",
    "total_credits_used", "num_file_uploads",
    "completed_onboarding_tour", "submitted_onboarding_form",
]
PROFILE_FEATURES = [f for f in PROFILE_FEATURES if f in seg_df.columns]

profiles = []
for cid in range(best_k):
    mask = seg_df["cluster_id"] == cid
    cdf = seg_df[mask]
    p = {
        "cluster_id": int(cid),
        "n_users": int(mask.sum()),
        "pct_total": round(float(mask.mean()) * 100, 1),
        "success_rate": round(float(cdf["long_term_success"].mean()), 4),
    }
    for f in PROFILE_FEATURES:
        p[f"mean_{f}"] = round(float(cdf[f].mean()), 4)
    profiles.append(p)

profiles_df = pd.DataFrame(profiles).sort_values(
    "success_rate", ascending=False).reset_index(drop=True)

print(f"\n  {'Cluster':>8} {'N':>6} {'%Total':>7} {'Success%':>10}")
print(f"  {'-'*40}")
for i in range(len(profiles_df)):
    row = profiles_df.iloc[i]
    print(f"  {int(row['cluster_id']):>8} {int(row['n_users']):>6,} "
          f"{row['pct_total']:>6.1f}% {row['success_rate']*100:>9.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 7. ASSIGN MEANINGFUL ARCHETYPE LABELS
# ══════════════════════════════════════════════════════════════════════════
print("\n[7] Assigning archetype labels...")

def _gmean_row(prow, col, default=0.0):
    """Safely get mean_{col} from a profiles_df row."""
    key = f"mean_{col}"
    if key in prow.index:
        return float(prow[key])
    return float(default)

label_pool = [
    ("power",    "Power Users",       SUC),
    ("agent",    "AI Explorers",      "#A1C9F4"),
    ("workflow", "Workflow Builders", "#8DE5A1"),
    ("casual",   "Casual Explorers",  "#FFB482"),
    ("at_risk",  "At-Risk Users",     HL),
    ("dormant",  "Dormant",           WARN),
]

archetype_name_map = {}
archetype_color_map = {}
used_keys = set()

for i in range(len(profiles_df)):
    prow = profiles_df.iloc[i]
    cid = int(prow["cluster_id"])

    agent_pct  = _gmean_row(prow, "used_agent")
    agent_evts = _gmean_row(prow, "num_agent_events")
    dag_use    = _gmean_row(prow, "has_complete_dag")
    complexity = _gmean_row(prow, "edges_per_canvas")
    recency7   = _gmean_row(prow, "active_last_7d")
    multi_day  = _gmean_row(prow, "multi_day_user")

    if i == 0:
        chosen_key = "power"
    elif i == len(profiles_df) - 1:
        chosen_key = "dormant" if "dormant" not in used_keys else "at_risk"
    else:
        if agent_pct > 0.3 and agent_evts > 5 and "agent" not in used_keys:
            chosen_key = "agent"
        elif (dag_use > 0.2 or complexity > 0.5) and "workflow" not in used_keys:
            chosen_key = "workflow"
        elif recency7 < 0.1 and multi_day < 0.3 and "at_risk" not in used_keys:
            chosen_key = "at_risk"
        elif "casual" not in used_keys:
            chosen_key = "casual"
        else:
            chosen_key = "at_risk"

    # fallback: pick first unused from pool
    if chosen_key in used_keys:
        for lk, _ln, _lc in label_pool:
            if lk not in used_keys:
                chosen_key = lk
                break

    used_keys.add(chosen_key)
    lname = next(ln for lk, ln, lc in label_pool if lk == chosen_key)
    lcolor = next(lc for lk, ln, lc in label_pool if lk == chosen_key)
    archetype_name_map[cid] = lname
    archetype_color_map[cid] = lcolor

seg_df["archetype"] = seg_df["cluster_id"].map(archetype_name_map)
profiles_df["archetype"] = profiles_df["cluster_id"].map(archetype_name_map)
profiles_df["color"] = profiles_df["cluster_id"].map(archetype_color_map)
profiles_df_sorted = profiles_df.sort_values(
    "success_rate", ascending=False).reset_index(drop=True)

archetype_order = list(profiles_df_sorted["archetype"])

print("\n  ARCHETYPE SUMMARY")
print(f"  {'Archetype':<22} {'N':>5} {'%Total':>7} {'Success%':>10} "
      f"{'Agent%':>8} {'MultiDay%':>10}")
print(f"  {'-'*70}")
for i in range(len(profiles_df_sorted)):
    row = profiles_df_sorted.iloc[i]
    aname = str(row["archetype"])
    agent_pct = _gmean_row(row, "used_agent") * 100
    mday_pct  = _gmean_row(row, "multi_day_user") * 100
    print(f"  {aname:<22} {int(row['n_users']):>5,} {row['pct_total']:>6.1f}% "
          f"{row['success_rate']*100:>9.1f}%  {agent_pct:>6.1f}%   {mday_pct:>7.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 8. DETAILED BEHAVIORAL PROFILE TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n[8] Detailed behavioral profiles...")

key_feats = [
    ("session_count",               "Avg Sessions"),
    ("total_active_days",           "Avg Active Days"),
    ("sessions_per_week",           "Sessions/Week"),
    ("total_events_excl_credits",   "Avg Events"),
    ("used_agent",                  "Agent Adoption %"),
    ("num_agent_events",            "Avg Agent Events"),
    ("has_complete_dag",            "Has DAG %"),
    ("num_blocks_created",          "Avg Blocks Created"),
    ("num_edges_created",           "Avg Edges Created"),
    ("block_run_rate",              "Block Run Rate"),
    ("pct_sessions_with_code_run",  "% Sessions w/ Code"),
    ("active_last_7d",              "Active Last 7d %"),
    ("return_rate",                 "Return Rate"),
]
PCT_FEATS = {"used_agent", "has_complete_dag", "active_last_7d",
             "multi_day_user", "completed_onboarding_tour"}
RATE_FEATS = {"block_run_rate", "return_rate", "pct_sessions_with_code_run"}

print(f"\n  {'Feature':<35} " + "  ".join(f"{a:<22}" for a in archetype_order))
print(f"  {'-'*110}")

for feat_col, feat_label in key_feats:
    row_strs = []
    for i in range(len(profiles_df_sorted)):
        prow = profiles_df_sorted.iloc[i]
        val = _gmean_row(prow, feat_col)
        if feat_col in PCT_FEATS:
            row_strs.append(f"{val*100:>5.1f}%")
        elif feat_col in RATE_FEATS:
            row_strs.append(f"{val:>6.3f}")
        else:
            row_strs.append(f"{val:>7.1f}")
    print(f"  {feat_label:<35} " + "  ".join(f"{v:<22}" for v in row_strs))

# ══════════════════════════════════════════════════════════════════════════
# 9. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n[9] Generating visualizations...")

# ── 9a. 2D scatter (UMAP/PCA) ────────────────────────────────────────────
archetype_scatter_fig = plt.figure(figsize=(12, 8))
archetype_scatter_fig.patch.set_facecolor(BG)
ax_sc = archetype_scatter_fig.add_subplot(111)
ax_sc.set_facecolor(BG)

for cid in range(best_k):
    arch_name  = archetype_name_map[cid]
    arch_color = archetype_color_map[cid]
    mask       = seg_df["cluster_id"] == cid
    ax_sc.scatter(
        seg_df.loc[mask, "dim1"], seg_df.loc[mask, "dim2"],
        c=arch_color, label=f"{arch_name} (n={int(mask.sum()):,})",
        s=14, alpha=0.65, linewidths=0
    )

ax_sc.set_xlabel(f"{dim_method} Dimension 1", color=FG, fontsize=11)
ax_sc.set_ylabel(f"{dim_method} Dimension 2", color=FG, fontsize=11)
ax_sc.set_title(
    f"User Behavioral Archetypes — {dim_method} Projection\n"
    f"KMeans (k={best_k}) on {X_scaled.shape[1]} Behavioral Features",
    color=FG, fontsize=13, pad=14)
ax_sc.tick_params(colors=FG)
for spine in ax_sc.spines.values():
    spine.set_color(SG)
ax_sc.grid(color=SG, alpha=0.1, linestyle="--")
ax_sc.legend(facecolor=BG, edgecolor=SG, labelcolor=FG, fontsize=10,
             markerscale=2.5, loc="best")
archetype_scatter_fig.tight_layout()
print(f"  Scatter plot ({dim_method})")

# ── 9b. Success rate by archetype ────────────────────────────────────────
archetype_success_fig = plt.figure(figsize=(10, 6))
archetype_success_fig.patch.set_facecolor(BG)
ax_sr = archetype_success_fig.add_subplot(111)
ax_sr.set_facecolor(BG)

sr_vals, sr_colors, sr_ns = [], [], []
for a in archetype_order:
    mask_a = profiles_df_sorted["archetype"] == a
    sr_vals.append(float(profiles_df_sorted.loc[mask_a, "success_rate"].values[0]) * 100)
    sr_colors.append(str(profiles_df_sorted.loc[mask_a, "color"].values[0]))
    sr_ns.append(int(profiles_df_sorted.loc[mask_a, "n_users"].values[0]))

bars_sr = ax_sr.bar(archetype_order, sr_vals, color=sr_colors, edgecolor=BG, width=0.6)
ax_sr.axhline(success_rate_overall * 100, color=HL, linewidth=2,
              linestyle="--", label=f"Overall {success_rate_overall*100:.1f}%")

for bar, val, n_val in zip(bars_sr, sr_vals, sr_ns):
    ax_sr.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.3,
        f"{val:.1f}%\n(n={n_val:,})",
        ha="center", va="bottom", color=FG, fontsize=10, fontweight="bold"
    )

ax_sr.set_ylabel("Long-Term Success Rate (%)", color=FG, fontsize=11)
ax_sr.set_title("Long-Term Success Rate by Behavioral Archetype",
                color=FG, fontsize=13, pad=12)
ax_sr.tick_params(colors=FG)
for spine in ax_sr.spines.values():
    spine.set_color(SG)
ax_sr.grid(axis="y", color=SG, alpha=0.12, linestyle="--")
ax_sr.legend(facecolor=BG, edgecolor=SG, labelcolor=FG, fontsize=10)
archetype_success_fig.tight_layout()
print("  Success rate bar chart")

# ── 9c. Multi-metric behavioral profile ───────────────────────────────────
metric_defs = [
    ("session_count",             "Sessions"),
    ("active_days_per_week",      "Active Days/Week"),
    ("used_agent",                "Agent Adoption"),
    ("pct_sessions_with_code_run","Code Run Rate"),
    ("has_complete_dag",          "Has DAG"),
    ("block_run_rate",            "Block Run Rate"),
    ("return_rate",               "Return Rate"),
    ("active_last_7d",            "Active Last 7d"),
]
metric_defs = [(f, lb) for f, lb in metric_defs if f in seg_df.columns]
n_metrics = len(metric_defs)

archetype_profile_fig, axes_p = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 5))
archetype_profile_fig.patch.set_facecolor(BG)
archetype_profile_fig.suptitle(
    "Behavioral Profile by Archetype (Normalized)",
    color=FG, fontsize=13, y=1.03)

for mi in range(n_metrics):
    feat, label = metric_defs[mi]
    ax_p = axes_p[mi]
    ax_p.set_facecolor(BG)
    raw_vals = []
    for i in range(len(profiles_df_sorted)):
        prow = profiles_df_sorted.iloc[i]
        raw_vals.append(_gmean_row(prow, feat))
    raw_arr = np.array(raw_vals, dtype=float)
    mn, mx = raw_arr.min(), raw_arr.max()
    rng = mx - mn if mx > mn else 1.0
    norm_vals = (raw_arr - mn) / rng

    a_colors = []
    for i in range(len(profiles_df_sorted)):
        a_colors.append(str(profiles_df_sorted.iloc[i]["color"]))

    ax_p.bar(range(len(archetype_order)), norm_vals, color=a_colors,
             edgecolor=BG, width=0.7)
    ax_p.set_xticks(range(len(archetype_order)))
    ax_p.set_xticklabels(
        [a.replace(" ", "\n") for a in archetype_order],
        color=FG, fontsize=7)
    ax_p.set_title(label, color=FG, fontsize=9, pad=6)
    ax_p.tick_params(colors=FG, labelsize=7)
    for spine in ax_p.spines.values():
        spine.set_color(SG)
    ax_p.set_yticks([])

    for xi in range(len(raw_vals)):
        rv = float(raw_vals[xi])
        nv = float(norm_vals[xi])
        if feat in PCT_FEATS:
            display = f"{rv*100:.0f}%"
        elif feat in RATE_FEATS:
            display = f"{rv:.2f}"
        else:
            display = f"{rv:.1f}"
        ax_p.text(xi, nv + 0.03, display,
                  ha="center", va="bottom", color=FG, fontsize=7, fontweight="bold")

archetype_profile_fig.tight_layout()
print("  Multi-metric profile chart")

# ── 9d. K selection chart ─────────────────────────────────────────────────
archetype_k_selection_fig, (ax_sil, ax_in) = plt.subplots(1, 2, figsize=(12, 5))
archetype_k_selection_fig.patch.set_facecolor(BG)
archetype_k_selection_fig.suptitle(
    "K-Means: Choosing Optimal Number of Clusters",
    color=FG, fontsize=13)

sil_bar_colors = [SUC if k == best_k else PAL[0] for k in k_range]
ax_sil.set_facecolor(BG)
ax_sil.bar(k_range, silhouettes, color=sil_bar_colors, edgecolor=BG, width=0.6)
ax_sil.set_xlabel("Number of Clusters (K)", color=FG, fontsize=11)
ax_sil.set_ylabel("Silhouette Score", color=FG, fontsize=11)
ax_sil.set_title("Silhouette Score vs K\n(higher = better)", color=FG, fontsize=11)
ax_sil.tick_params(colors=FG)
for spine in ax_sil.spines.values():
    spine.set_color(SG)
ax_sil.grid(axis="y", color=SG, alpha=0.12, linestyle="--")
for k, sv in zip(k_range, silhouettes):
    ax_sil.text(k, sv + 0.002, f"{sv:.3f}", ha="center", va="bottom",
                color=HL if k == best_k else FG, fontsize=9,
                fontweight="bold" if k == best_k else "normal")

ax_in.set_facecolor(BG)
ax_in.plot(k_range, inertias, color=PAL[0], linewidth=2,
           marker="o", markersize=7, markerfacecolor=SUC)
ax_in.axvline(best_k, color=HL, linewidth=2, linestyle="--",
              label=f"Best K={best_k}")
ax_in.set_xlabel("Number of Clusters (K)", color=FG, fontsize=11)
ax_in.set_ylabel("Inertia (Within-cluster SSE)", color=FG, fontsize=11)
ax_in.set_title("Elbow Method — Inertia vs K", color=FG, fontsize=11)
ax_in.tick_params(colors=FG)
for spine in ax_in.spines.values():
    spine.set_color(SG)
ax_in.grid(color=SG, alpha=0.1, linestyle="--")
ax_in.legend(facecolor=BG, edgecolor=SG, labelcolor=FG, fontsize=10)
archetype_k_selection_fig.tight_layout()
print("  K selection chart")

plt.close("all")

# ══════════════════════════════════════════════════════════════════════════
# 10. FINAL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ARCHETYPE SUMMARY REPORT")
print("=" * 70)
print(f"\n  Algorithm    : KMeans (k={best_k}, k-means++ init, n_init=30)")
print(f"  Silhouette   : {final_sil:.4f}")
print(f"  Projection   : {dim_method}")
print(f"  Users        : {len(seg_df):,}")
print(f"  Overall LTS  : {success_rate_overall:.1%}\n")

for i in range(len(profiles_df_sorted)):
    row = profiles_df_sorted.iloc[i]
    aname      = str(row["archetype"])
    n_u        = int(row["n_users"])
    pct        = float(row["pct_total"])
    sr         = float(row["success_rate"]) * 100
    lift       = float(row["success_rate"]) / success_rate_overall if success_rate_overall > 0 else 0.0
    sessions   = _gmean_row(row, "session_count")
    active_d   = _gmean_row(row, "total_active_days")
    agent_p    = _gmean_row(row, "used_agent") * 100
    dag_p      = _gmean_row(row, "has_complete_dag") * 100
    code_r     = _gmean_row(row, "pct_sessions_with_code_run") * 100
    ret_rt     = _gmean_row(row, "return_rate")
    rec7       = _gmean_row(row, "active_last_7d") * 100
    agent_evts = _gmean_row(row, "num_agent_events")

    print(f"  >> {aname:<22} ({n_u:,} users, {pct:.1f}%)")
    print(f"     Long-term success: {sr:.1f}%  ({lift:.1f}x overall rate)")
    print(f"     Avg sessions     : {sessions:.1f}  |  Active days: {active_d:.1f}")
    print(f"     Agent adoption   : {agent_p:.1f}%  |  Agent events avg: {agent_evts:.1f}")
    print(f"     Has DAG          : {dag_p:.1f}%  |  Code run rate: {code_r:.1f}%")
    print(f"     Return rate      : {ret_rt:.3f}  |  Active last 7d: {rec7:.1f}%")
    print()

print("=" * 70)
print("  DONE")
print("=" * 70)

# ── Export for downstream ─────────────────────────────────────────────────
archetype_profiles_df = profiles_df_sorted.copy()
archetype_seg_df = seg_df[[
    "cluster_id", "archetype", "long_term_success",
    "dim1", "dim2",
    "session_count", "total_active_days", "used_agent",
    "num_agent_events", "has_complete_dag",
    "num_edges_created", "return_rate"
]].copy()
archetype_k = best_k
archetype_dim_method = dim_method
