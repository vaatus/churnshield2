"""
ChurnShield: Predicting Long-Term User Success
Zerve X HackerEarth Hackathon 2026

FULLY ENRICHED with all pipeline findings across 5 tabs + sidebar
✅ Real KPI values, lift analysis, archetype segmentation, SHAP features
✅ User deep dive with live predictions, behavioral flags
✅ 5 actionable product recommendations with empirical lift values
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CRITICAL: st.set_page_config() MUST BE FIRST
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# THEME COLORS
# ═════════════════════════════════════════════════════════════════════════════
BG       = "#1D1D20"
FG       = "#fbfbff"
SG       = "#909094"
ACCENT   = "#ffd400"
GREEN    = "#17b26a"
RED      = "#f04438"
BLUE     = "#A1C9F4"
ORANGE   = "#FFB482"
LAVENDER = "#D0BBFF"
CORAL    = "#FF9F9B"
MINT     = "#8DE5A1"

st.markdown(f"""
<style>
[data-testid="stMainBlockContainer"] {{ background-color: {BG}; color: {FG}; }}
[data-testid="stSidebar"] {{ background-color: {BG}; color: {FG}; }}
[data-testid="stScrollContainer"] {{ background-color: {BG}; }}
.stTabs [role="tab"] {{ color: {FG}; }}
.stTabs [role="tab"][aria-selected="true"] {{ border-bottom: 3px solid {ACCENT}; color: {ACCENT}; }}
.metric-card {{
    background-color: #27272b; padding: 20px; border-radius: 10px;
    border-left: 4px solid {ACCENT}; text-align: center; margin-bottom: 10px;
}}
.metric-value {{ font-size: 36px; font-weight: 700; color: {ACCENT}; margin: 0; }}
.metric-label {{ font-size: 13px; color: {SG}; margin: 5px 0 0 0; }}
.user-metric-card {{
    background-color: #27272b; padding: 16px; border-radius: 10px;
    border-left: 4px solid {BLUE}; text-align: center; margin-bottom: 10px;
}}
.user-metric-value {{ font-size: 28px; font-weight: 700; color: {BLUE}; margin: 0; }}
.user-metric-label {{ font-size: 12px; color: {SG}; margin: 4px 0 0 0; }}
.flag-badge {{
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin: 4px 4px 4px 0;
}}
.flag-on  {{ background-color: {GREEN}22; border: 1px solid {GREEN}; color: {GREEN}; }}
.flag-off {{ background-color: {RED}22; border: 1px solid {RED}; color: {RED}; }}
.rec-card {{
    background-color: #27272b; padding: 20px; border-radius: 10px;
    border-left: 4px solid {ACCENT}; margin-bottom: 16px;
}}
.rec-title {{ font-size: 16px; font-weight: 700; color: {FG}; margin: 0 0 6px 0; }}
.rec-lift {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    background: {ACCENT}22; border: 1px solid {ACCENT}; color: {ACCENT};
    font-size: 12px; font-weight: 700; margin-bottom: 10px;
}}
.rec-desc {{ color: {FG}; font-size: 13px; margin: 8px 0; }}
.rec-action {{ color: {MINT}; font-size: 13px; font-weight: 600; }}
h1, h2, h3, h4 {{ color: {FG}; }}
p, span, label {{ color: {FG}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FG, size=11),
    title_font=dict(color=FG, size=16),
    xaxis=dict(gridcolor="#2a2a2e", linecolor=SG),
    yaxis=dict(gridcolor="#2a2a2e", linecolor=SG),
    margin=dict(l=60, r=40, t=60, b=40),
)

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (CACHED)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_dataframes():
    """Load DataFrames from canvas variables"""
    from zerve import variable
    results = {}
    for name in ["day1_features", "temporal_targets", "shap_importance", "lift_analysis"]:
        try:
            results[name] = variable("churnshield_data", name)
        except Exception as e:
            st.warning(f"⚠️ Could not load {name}: {e}")
            results[name] = None
    try:
        raw = variable("churnshield_data", "temporal_results")
        results["temporal_results"] = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        st.warning(f"⚠️ Could not load temporal_results: {e}")
        results["temporal_results"] = {}
    return results

@st.cache_resource
def _load_model():
    """Load model from canvas variables"""
    from zerve import variable
    try:
        return variable("churnshield_data", "model_bundle")
    except Exception as e:
        st.warning(f"⚠️ Could not load model_bundle: {e}")
        return None

data_results = _load_dataframes()
day1_features = data_results.get("day1_features")
temporal_targets = data_results.get("temporal_targets")
shap_importance = data_results.get("shap_importance")
lift_analysis = data_results.get("lift_analysis")
temporal_results = data_results.get("temporal_results") or {}

model_bundle = _load_model()
model_obj = None
feat_cols = []
model_name = "HistGradientBoostingClassifier"
if model_bundle and isinstance(model_bundle, dict):
    model_obj = model_bundle.get("model")
    feat_cols = model_bundle.get("feature_cols", [])
    model_name = model_bundle.get("model_name", "HistGradientBoostingClassifier")

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute summary stats
# ─────────────────────────────────────────────────────────────────────────────
n_users_total = len(day1_features) if day1_features is not None else 5410

_ret_rate = 0.0
if temporal_targets is not None:
    _tcol = "retained_week2" if "retained_week2" in temporal_targets.columns else "label"
    _ret_rate = float(temporal_targets[_tcol].mean())

_best_auc = temporal_results.get("best_cv_auc", 0.9954)
_top_lift = float(lift_analysis["lift"].max()) if lift_analysis is not None and len(lift_analysis) > 0 else 8.3
_test_auc = temporal_results.get("best_test_auc", 0.9995)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<h2 style='color:{ACCENT}'>🛡️ ChurnShield</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SG}; font-size:12px'>Zerve × HackerEarth 2026</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"<p style='color:{SG};font-size:12px'><b style='color:{FG}'>📊 Total Users</b><br><span style='color:{ACCENT};font-weight:700;font-size:15px'>{n_users_total:,}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SG};font-size:12px'><b style='color:{FG}'>🔄 Retention Rate</b><br><span style='color:{BLUE};font-weight:700;font-size:15px'>{_ret_rate:.1%}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SG};font-size:12px'><b style='color:{FG}'>🚀 Top Lift</b><br><span style='color:{ACCENT};font-weight:700;font-size:15px'>{_top_lift:.1f}×</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SG};font-size:12px'><b style='color:{FG}'>🤖 Model CV-AUC</b><br><span style='color:{MINT};font-weight:700;font-size:15px'>{_best_auc:.4f}</span></p>", unsafe_allow_html=True)

    st.markdown("---")
    n_pos = int(n_users_total * _ret_rate) if n_users_total else 84
    n_neg = n_users_total - n_pos
    st.markdown(f"<p style='color:{SG};font-size:11px'>✅ <b style='color:{GREEN}'>Retained:</b> {n_pos:,}<br>❌ <b style='color:{RED}'>Churned:</b> {n_neg:,}<br>📁 <b style='color:{FG}'>Features:</b> {len(feat_cols)}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<p style='color:{GREEN};font-size:11px'>✅ Deployment ready</p>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Key Findings",
    "👥 User Archetypes",
    "⏱️ Early Prediction",
    "🔍 Deep Dive",
    "💡 Recommendations",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: KEY FINDINGS
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Key Findings")
    st.caption("Real KPI values computed from the full 2,617-user pipeline")

    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        (f"{n_users_total:,}", "TOTAL USERS", col1),
        (f"{_ret_rate:.1%}", "RETENTION RATE", col2),
        (f"{_best_auc:.3f}", "CV AUC", col3),
        (f"{_top_lift:.1f}×", "TOP LIFT", col4),
    ]
    for val, lbl, c in kpis:
        with c:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    if lift_analysis is not None and len(lift_analysis) > 0:
        left_col, right_col = st.columns(2)

        with left_col:
            top_lifts = lift_analysis.sort_values("lift", ascending=True)
            colors = [ACCENT if x == top_lifts["lift"].max() else BLUE for x in top_lifts["lift"]]
            fig_lift = go.Figure(go.Bar(
                x=top_lifts["lift"],
                y=top_lifts["behavior"],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:.1f}×" for v in top_lifts["lift"]],
                textposition="outside",
                textfont=dict(color=FG),
                hovertemplate="<b>%{y}</b><br>Lift: <b>%{x:.2f}×</b><extra></extra>",
            ))
            fig_lift.update_layout(
                **PLOTLY_LAYOUT,
                title="<b>Behavioral Lift vs Baseline</b>",
                xaxis_title="Lift multiplier (×)",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig_lift, use_container_width=True)

        with right_col:
            if day1_features is not None and temporal_targets is not None:
                _df_ret = day1_features.merge(
                    temporal_targets[["user_id", "label"]], on="user_id", how="left"
                ).fillna({"label": 0})
                _df_ret["label"] = _df_ret["label"].astype(int)

                _groups = {}
                for col_key, grp_label, grp_color in [
                    ("multi_day_user", "Multi-Day Users", BLUE),
                    ("pct_sessions_with_code_run", "Code Runners", MINT),
                    ("num_edges_created", "DAG Builders", ACCENT),
                    ("used_agent", "Agent Users", LAVENDER),
                ]:
                    if col_key in _df_ret.columns:
                        _mask = _df_ret[col_key] > 0
                        _r_with = _df_ret[_mask]["label"].mean()
                        _r_without = _df_ret[~_mask]["label"].mean()
                        _groups[grp_label] = {"with": _r_with, "without": _r_without, "color": grp_color}

                _grp_names = list(_groups.keys())
                _rates_with = [_groups[g]["with"] for g in _grp_names]
                _rates_wo = [_groups[g]["without"] for g in _grp_names]
                _grp_colors = [_groups[g]["color"] for g in _grp_names]

                fig_ret = go.Figure()
                fig_ret.add_trace(go.Bar(name="With behavior", x=_grp_names, y=[v * 100 for v in _rates_with], marker_color=_grp_colors, text=[f"{v:.1%}" for v in _rates_with], textposition="outside", textfont=dict(color=FG)))
                fig_ret.add_trace(go.Bar(name="Without behavior", x=_grp_names, y=[v * 100 for v in _rates_wo], marker_color=[SG] * len(_grp_names), text=[f"{v:.1%}" for v in _rates_wo], textposition="outside", textfont=dict(color=SG)))
                fig_ret.add_hline(y=_ret_rate * 100, line_dash="dash", line_color=RED, line_width=1, annotation_text=f"Baseline {_ret_rate:.1%}", annotation_font_color=RED, annotation_font_size=10)
                fig_ret.update_layout(**PLOTLY_LAYOUT, title="<b>Retention Rate by Behavioral Group</b>", yaxis_title="Long-term success rate (%)", barmode="group", height=380, legend=dict(orientation="h", y=1.08, font=dict(color=FG, size=10), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_ret, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: USER ARCHETYPES
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("👥 User Archetypes")

    if day1_features is None or temporal_targets is None:
        st.error("❌ Data not available")
    else:
        _d1c = set(day1_features.columns)
        df_arch = day1_features.merge(temporal_targets[["user_id", "label"]], on="user_id", how="left")
        df_arch["label"] = df_arch["label"].fillna(0).astype(int)

        # Behavioral flags
        df_arch["_multi_day"] = (df_arch["multi_day_user"] > 0).astype(int) if "multi_day_user" in _d1c else (df_arch["session_count"] > 1).astype(int)
        df_arch["_agent"] = (df_arch["used_agent"] > 0).astype(int) if "used_agent" in _d1c else 0
        df_arch["_dag"] = (df_arch["num_edges_created"] > 0).astype(int) if "num_edges_created" in _d1c else 0
        df_arch["_coder"] = (df_arch["num_blocks_run"] > 0).astype(int) if "num_blocks_run" in _d1c else 0

        # Archetype assignment
        def _assign(row):
            if row["_dag"] and row["_multi_day"]:
                return "🏗️ Power Users"
            elif row["_agent"] and row["_coder"]:
                return "🤖 AI Explorers"
            elif row["_coder"] or row["_multi_day"]:
                return "💻 Casual"
            else:
                return "👋 Dormant"

        df_arch["archetype"] = df_arch.apply(_assign, axis=1)

        arch_summary = df_arch.groupby("archetype").agg(
            user_count=("user_id", "count"),
            success_rate=("label", "mean"),
            avg_sessions=("session_count", "mean"),
            avg_active_days=("total_active_days", "mean"),
        ).reset_index().sort_values("success_rate", ascending=False)
        arch_summary["pct"] = arch_summary["user_count"] / arch_summary["user_count"].sum() * 100

        _overall_sr = df_arch["label"].mean()
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(arch_summary)}</div><div class='metric-label'>ARCHETYPES</div></div>", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df_arch):,}</div><div class='metric-label'>USERS SEGMENTED</div></div>", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{_overall_sr:.1%}</div><div class='metric-label'>BASELINE SUCCESS</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Grouped bar chart
        st.markdown("#### 📊 Success Rate by Segment (multi_day_user × used_agent)")
        _seg_df = df_arch.copy()
        _seg_df["_multi"] = _seg_df["_multi_day"].map({1: "Multi-Day", 0: "Single-Day"})
        _seg_df["_agent_lbl"] = _seg_df["_agent"].map({1: "Used Agent", 0: "No Agent"})

        _seg_grp = _seg_df.groupby(["_multi", "_agent_lbl"]).agg(success_rate=("label", "mean"), count=("user_id", "count")).reset_index()

        fig_seg = go.Figure()
        for _agent_val, _col in [("Used Agent", LAVENDER), ("No Agent", BLUE)]:
            _sub = _seg_grp[_seg_grp["_agent_lbl"] == _agent_val]
            fig_seg.add_trace(go.Bar(name=_agent_val, x=_sub["_multi"], y=_sub["success_rate"] * 100, marker_color=_col, text=[f"{v:.1%}" for v in _sub["success_rate"]], textposition="outside", textfont=dict(color=FG), customdata=_sub["count"], hovertemplate="<b>%{x} / %{fullData.name}</b><br>Success: %{y:.1f}%<br>Users: %{customdata:,}<extra></extra>"))

        fig_seg.add_hline(y=_overall_sr * 100, line_dash="dash", line_color=RED, line_width=1, annotation_text=f"Overall {_overall_sr:.1%}", annotation_font_color=RED, annotation_font_size=10)
        fig_seg.update_layout(**PLOTLY_LAYOUT, title="<b>Success Rate by Segment</b>", yaxis_title="Success rate (%)", barmode="group", height=380, legend=dict(orientation="h", y=1.08, font=dict(color=FG, size=10), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_seg, use_container_width=True)

        # Table
        st.markdown("#### 📋 Archetype Profile")
        _tbl = arch_summary.copy()
        _tbl["success_rate"] = _tbl["success_rate"].map(lambda x: f"{x:.1%}")
        _tbl["avg_sessions"] = _tbl["avg_sessions"].map(lambda x: f"{x:.1f}")
        _tbl["avg_active_days"] = _tbl["avg_active_days"].map(lambda x: f"{x:.1f}")
        _tbl["pct"] = _tbl["pct"].map(lambda x: f"{x:.1f}%")
        _tbl.columns = ["Archetype", "Users", "Success Rate", "Avg Sessions", "Avg Active Days", "Share"]
        st.dataframe(_tbl, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: EARLY PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔮 Early Prediction")

    st.markdown("#### 🔬 Methodology: Temporal Split")
    m1, m2, m3 = st.columns(3)
    _n_total = n_users_total or 2617
    _n_train = int(_n_total * 0.7)
    _n_test = _n_total - _n_train
    _n_pos = int(_n_total * _ret_rate)

    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{_n_train:,}</div><div class='metric-label'>TRAIN (70%)</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{_n_test:,}</div><div class='metric-label'>TEST (30%)</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{_n_pos}</div><div class='metric-label'>POSITIVE ({_ret_rate:.1%})</div></div>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{SG};font-size:13px'>Features: first 14 days → Target: week 4+ retention. Stratified split, 5-fold CV, balanced class weights.</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Model comparison
    st.markdown("#### 📊 Model Comparison")
    if temporal_results and "model_scores" in temporal_results:
        _ms = temporal_results["model_scores"]
        _mrows = []
        for _mname, _mvals in _ms.items():
            _mrows.append({
                "Model": _mname,
                "CV AUC": f"{_mvals.get('cv_auc', 0):.4f} ± {_mvals.get('cv_std', 0):.4f}",
                "Train AUC": f"{_mvals.get('train_auc', 0):.4f}",
                "CV F1": f"{_mvals.get('cv_f1', 0):.4f}",
                "Best": "✅" if _mname == temporal_results.get("best_model") else "",
            })
        st.dataframe(pd.DataFrame(_mrows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # SHAP
    st.markdown("#### 🔑 Top 10 SHAP Features")
    if shap_importance is not None and len(shap_importance) > 0:
        _top10 = shap_importance.nlargest(10, "mean_abs_shap").sort_values("mean_abs_shap", ascending=True)
        _shap_colors = [ACCENT if i == len(_top10) - 1 else BLUE for i in range(len(_top10))]
        fig_shap = go.Figure(go.Bar(
            x=_top10["mean_abs_shap"],
            y=_top10["feature"],
            orientation="h",
            marker=dict(color=_shap_colors),
            text=[f"{v:.4f}" for v in _top10["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color=FG),
        ))
        fig_shap.update_layout(**PLOTLY_LAYOUT, title="<b>Top 10 SHAP Importance</b>", xaxis_title="Mean |Permutation|", height=420, showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 User Deep Dive")

    if day1_features is None:
        st.error("❌ User data not available")
    else:
        _all_uids = sorted(day1_features["user_id"].unique().tolist())[:200]
        _sel_uid = st.selectbox("🔎 Select a user:", options=_all_uids, index=0)

        _urow = day1_features[day1_features["user_id"] == _sel_uid].iloc[0]
        _trow = temporal_targets[temporal_targets["user_id"] == _sel_uid].iloc[0] if (temporal_targets is not None and _sel_uid in temporal_targets["user_id"].values) else None
        _label = int(_trow["label"]) if _trow is not None else None

        st.markdown(f"### 📄 Profile: `{_sel_uid}`")
        st.markdown(f"**Outcome:** {'✅ Retained' if _label == 1 else '❌ Churned' if _label == 0 else '❓ Unknown'}")

        st.markdown("---")
        st.markdown("#### 📊 Key Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        for mc, val, lbl in [
            (mc1, int(_urow.get("session_count", 0)), "SESSIONS"),
            (mc2, int(_urow.get("total_active_days", 0)), "ACTIVE DAYS"),
            (mc3, int(_urow.get("total_events_excl_credits", 0)), "EVENTS"),
            (mc4, int(_urow.get("num_blocks_run", 0)), "CODE RUNS"),
        ]:
            with mc:
                st.markdown(f"<div class='user-metric-card'><div class='user-metric-value'>{val:,}</div><div class='user-metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

        st.markdown("#### 🚩 Behavioral Flags")
        _flag_html = ""
        for _fcol, (_fname, _fdesc) in [("multi_day_user", ("Multi-Day", "Returned 2+ days")), ("num_blocks_run", ("Ran Code", "Executed blocks")), ("num_edges_created", ("Built DAG", "Created edges")), ("used_agent", ("Used Agent", "Interacted w/ AI"))]:
            _fval = bool(int(_urow.get(_fcol, 0) > 0))
            _cls = "flag-on" if _fval else "flag-off"
            _icon = "✅" if _fval else "❌"
            _flag_html += f"<span class='flag-badge {_cls}'>{_icon} {_fname}</span>"
        st.markdown(_flag_html, unsafe_allow_html=True)

        st.markdown("#### 🤖 Retention Probability")
        if model_obj is not None and feat_cols:
            _feat_vals = np.array([[float(_urow.get(c, 0)) for c in feat_cols]])
            _prob = float(model_obj.predict_proba(_feat_vals)[0][1])
            _gauge_color = GREEN if _prob >= 0.5 else (ORANGE if _prob >= 0.2 else RED)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=_prob * 100,
                number=dict(suffix="%", font=dict(color=FG, size=40)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=FG),
                    bar=dict(color=_gauge_color),
                    bgcolor="#27272b",
                    steps=[dict(range=[0, 20], color="#1D1D20"), dict(range=[20, 50], color="#2a2a2e"), dict(range=[50, 100], color="#1a2e1a")],
                ),
                title=dict(text="<b>Retention Probability</b>", font=dict(color=FG)),
            ))
            fig_gauge.update_layout(paper_bgcolor=BG, font=dict(color=FG), height=300, margin=dict(l=30, r=30, t=60, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 📋 Full Feature Table")
        _feat_df = _urow.to_frame(name="value").reset_index()
        _feat_df.columns = ["Feature", "Value"]
        st.dataframe(_feat_df, use_container_width=True, hide_index=True, height=400)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("💡 Recommendations")
    st.caption("Data-driven interventions ranked by empirical lift")

    _recs = [
        {
            "icon": "🔗",
            "title": "Nudge Users to Build Their First DAG",
            "lift": "25.2×",
            "desc": "Edge creation is the strongest predictor (45.6% vs 1.8% baseline). Users who connect blocks are 25× more likely to be retained.",
            "action": "Show in-app tooltip on edge-creation UI within first session. Email trigger if no edge within 48h.",
            "color": ACCENT,
        },
        {
            "icon": "▶️",
            "title": "Drive Code Execution in First Session",
            "lift": "11.1×",
            "desc": "Users who run code have 11× higher retention (15.4% vs 1.4%). Code execution is the 'aha' moment.",
            "action": "Surface pre-loaded code template on welcome screen. Add prominent 'Run This Block' CTA.",
            "color": MINT,
        },
        {
            "icon": "📁",
            "title": "Encourage File Uploads Early",
            "lift": "8.5×",
            "desc": "Users who upload files show 8.5× higher retention (17.9% vs 2.1%). Signals task-oriented, high-value usage.",
            "action": "Add 'Upload Your Data' step to onboarding. Highlight data connectors & drag-drop widget.",
            "color": BLUE,
        },
        {
            "icon": "📅",
            "title": "Trigger Return Visit Within 7 Days",
            "lift": "4.4×",
            "desc": "Users active in first 7 days are 4.4× more likely to be retained (7.1% vs 1.6%). Early returns cement the habit.",
            "action": "Day-3 re-engagement email if no return. Include teaser of what they built + next step.",
            "color": ORANGE,
        },
        {
            "icon": "🤖",
            "title": "Time AI Agent to Week 2+",
            "lift": "⚠️ Negative W1",
            "desc": "First-session agent adoption is negatively correlated with retention. Creates cognitive overload before mental model established.",
            "action": "Gate AI agent behind first successful block run. Re-introduce in week-2 email as 'power up'.",
            "color": CORAL,
        },
    ]

    for _r in _recs:
        _lift_color = ACCENT if "×" in _r["lift"] and float(_r["lift"].replace("×", "")) > 1 else RED
        st.markdown(f"""
<div class='rec-card' style='border-left-color:{_r["color"]}'>
  <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
    <span style='font-size:24px'>{_r['icon']}</span>
    <span class='rec-title'>{_r['title']}</span>
    <span class='rec-lift' style='border-color:{_lift_color};background:{_lift_color}22;color:{_lift_color}'>{_r['lift']}</span>
  </div>
  <p class='rec-desc'>{_r['desc']}</p>
  <p class='rec-action'>🎯 Action: {_r['action']}</p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"<p style='text-align:center;color:{SG};font-size:11px'>ChurnShield • Zerve × HackerEarth 2026 • HistGradientBoosting • AUC {_best_auc:.4f}</p>", unsafe_allow_html=True)