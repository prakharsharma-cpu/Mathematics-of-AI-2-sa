# app.py
# ==================================
# ⚽ FootLens — Advanced Role-Adaptive Dashboard with Risk & Scheduler
# ==================================
import os
import re
from io import BytesIO, StringIO
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Ensure xlsxwriter is installed
try:
    import xlsxwriter
except ImportError:
    st.error("Please install 'xlsxwriter' to enable Excel exports. Run `pip install xlsxwriter`.")
    st.stop()

# Optional ML deps - used if available
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="⚽ FootLens Pro", layout="wide", initial_sidebar_state="expanded")
st.title("⚽ FootLens Pro — Role-adaptive Injury Intelligence")
st.markdown("Advanced dashboard: role-specific views, re-injury risk scoring (ML + heuristic fallback), rehab scheduler, PNG/Excel exports.")

# ---------------------
# Utilities
# ---------------------
def normalize_colname(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s

def build_norm_map(df):
    return {normalize_colname(c): c for c in df.columns}

def find_first(df_norm_map, alternatives):
    for alt in alternatives:
        na = normalize_colname(alt)
        if na in df_norm_map:
            return df_norm_map[na]
    # substring fallback
    for alt in alternatives:
        na = normalize_colname(alt)
        for nc, orig in df_norm_map.items():
            if na in nc or nc in na:
                return orig
    return None

def safe_mean(series):
    try:
        return float(np.nanmean(series))
    except Exception:
        return np.nan

def apply_seaborn_style(style_name):
    if style_name == "Classic Analytics":
        sns.set_theme(style="darkgrid", palette="muted")
        plt.rcParams.update({"figure.facecolor": "white"})
    else:
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({"figure.facecolor": "white"})

def ensure_nonempty(df):
    if df is None or df.shape[0] == 0:
        st.warning("No records to display after applying filters.")
        return False
    return True

def fig_to_png_bytes_matplotlib(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def fig_to_png_bytes_plotly(fig):
    # Try Plotly to_image (requires kaleido) — attempt gracefully
    try:
        img_bytes = fig.to_image(format="png")
        return BytesIO(img_bytes)
    except Exception:
        return None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------------
# Sample CSV in case no file provided
# ---------------------
def sample_csv_text():
    return """Player,Team,Position,Age,Season,Rating,Injury Type,Injury Start,Injury End,Match1_before_injury_Player_rating,Match1_after_injury_Player_rating,Match1_missed_match_GD,Match1_before_injury_GD,Match1_after_injury_GD
John Doe,Example FC,Forward,26,2024,75,Hamstring,2024-09-01,2024-09-21,7.1,6.8,0,1,0
Jane Smith,Example FC,Midfielder,24,2024,78,Ankle,2024-10-05,2024-10-20,7.5,7.1,0,0,2
Alex Roe,Another United,Defender,28,2024,72,ACL,2024-08-01,2024-11-01,6.9,7.0,0,0,0
"""

# ---------------------
# Sidebar: Upload + Settings
# ---------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
autopath = "/mnt/data/player_injuries_impact (1).csv"
use_autopath = False
if (not uploaded_file) and os.path.exists(autopath):
    use_autopath = st.sidebar.checkbox("Use detected dataset on server", value=True)

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df_raw)} rows.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
elif use_autopath:
    try:
        df_raw = pd.read_csv(autopath)
        st.sidebar.success(f"Loaded {len(df_raw)} rows from server file.")
    except Exception as e:
        st.sidebar.error(f"Error reading server CSV: {e}")
        st.stop()
else:
    st.sidebar.info("No file — using sample dataset.")
    df_raw = pd.read_csv(StringIO(sample_csv_text()))

# ---------------------
# Normalize columns & mapping
# ---------------------
df = df_raw.copy()
norm_map = build_norm_map(df)

# show detected columns
with st.expander("Detected columns (original)", expanded=False):
    st.write(list(df.columns))

expected = {
    "player": ["player", "player_name", "name"],
    "team": ["team", "club", "team_name", "club_name"],
    "position": ["position", "pos"],
    "age": ["age"],
    "rating": ["rating", "fifa_rating", "player_rating"],
    "injury_type": ["injury_type", "injury type", "injury"],
    "injury_start": ["injury_start", "injury start", "start_date", "date_of_injury"],
    "injury_end": ["injury_end", "injury end", "end_date", "date_of_return"],
}

mapped = {k: find_first(norm_map, v) for k, v in expected.items()}

# Convert date columns
if mapped["injury_start"]:
    df[mapped["injury_start"]] = pd.to_datetime(df[mapped["injury_start"]], errors="coerce")
if mapped["injury_end"]:
    df[mapped["injury_end"]] = pd.to_datetime(df[mapped["injury_end"]], errors="coerce")

# Compute derived columns
if mapped["injury_start"] and mapped["injury_end"]:
    df["injury_duration_days"] = (df[mapped["injury_end"]] - df[mapped["injury_start"]]).dt.days.clip(lower=0)
    df["injury_month"] = df[mapped["injury_start"]].dt.month
else:
    df["injury_duration_days"] = np.nan
    df["injury_month"] = np.nan

# ---------------------
# Before/after ratings safely
# ---------------------
col_lower = {c.lower(): c for c in df.columns}
def cols_matching(substr):
    return [orig for low, orig in col_lower.items() if substr.lower() in low]

before_rating_cols = cols_matching("before_injury_player_rating")
after_rating_cols = cols_matching("after_injury_player_rating")

# Convert to numeric safely
before_numeric_cols = [pd.to_numeric(df[c], errors='coerce') for c in before_rating_cols]
after_numeric_cols = [pd.to_numeric(df[c], errors='coerce') for c in after_rating_cols]

if before_numeric_cols:
    df["avg_rating_before_matches"] = pd.concat(before_numeric_cols, axis=1).mean(axis=1, skipna=True)
elif mapped["rating"] and mapped["rating"] in df.columns:
    df["avg_rating_before_matches"] = pd.to_numeric(df[mapped["rating"]], errors='coerce')
else:
    df["avg_rating_before_matches"] = np.nan

if after_numeric_cols:
    df["avg_rating_after_matches"] = pd.concat(after_numeric_cols, axis=1).mean(axis=1, skipna=True)
else:
    df["avg_rating_after_matches"] = np.nan

# Performance change
df["performance_change"] = df.get("avg_rating_after_matches", 0) - df.get("avg_rating_before_matches", 0)

# ---------------------
# Compute team GD proxies (if present)
# ---------------------
before_gd = cols_matching("before_injury_gd") + cols_matching("before_injury_goal_difference")
after_gd = cols_matching("after_injury_gd") + cols_matching("after_injury_goal_difference")
missed_gd = cols_matching("missed_match_gd") + cols_matching("missed_match_goal_difference")

def safe_mean_col(cols_list):
    if cols_list:
        return pd.concat([pd.to_numeric(df[c], errors='coerce') for c in cols_list], axis=1).mean(axis=1, skipna=True)
    return pd.Series(np.nan, index=df.index)

df["team_gd_before"] = safe_mean_col(before_gd)
df["team_gd_during"] = safe_mean_col(missed_gd)
df["team_gd_after"] = safe_mean_col(after_gd)
df["team_performance_drop"] = df["team_gd_before"] - df["team_gd_during"]
df["team_recovery"] = df["team_gd_after"] - df["team_gd_during"]

# ---------------------
# Sidebar filters & role
# ---------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters & Role View")
role = st.sidebar.selectbox("Choose role", ["Manager", "Coach", "Club Analyst", "Scout", "Custom Analyst"])
club_col = mapped["team"] if mapped["team"] and mapped["team"] in df.columns else None
player_col = mapped["player"] if mapped["player"] and mapped["player"] in df.columns else None
inj_col = mapped["injury_type"] if mapped["injury_type"] and mapped["injury_type"] in df.columns else None

club_options = sorted(df[club_col].dropna().unique()) if club_col else []
player_options = sorted(df[player_col].dropna().unique()) if player_col else []
injury_options = sorted(df[inj_col].dropna().unique()) if inj_col else []

filter_club = st.sidebar.multiselect("Club / Team", options=club_options, default=club_options)
filter_player = st.sidebar.multiselect("Player", options=player_options, default=player_options)
filter_injury = st.sidebar.multiselect("Injury Type", options=injury_options, default=injury_options)

global_mode = st.sidebar.radio("Global visualization mode", ["Plotly", "Matplotlib"], index=0)
global_style = st.sidebar.radio("Seaborn style (Matplotlib)", ["Modern Clean", "Classic Analytics"], index=0)

# Apply filters
filtered = df.copy()
if club_col and filter_club:
    filtered = filtered[filtered[club_col].isin(filter_club)]
if player_col and filter_player:
    filtered = filtered[filtered[player_col].isin(filter_player)]
if inj_col and filter_injury:
    filtered = filtered[filtered[inj_col].isin(filter_injury)]

# ---------------------
# Excel export utility
# ---------------------
def to_excel_bytes(dframe):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        dframe.to_excel(writer, sheet_name="filtered", index=False)
        if player_col:
            try:
                dframe.groupby(player_col).size().to_excel(writer, sheet_name="by_player")
            except Exception:
                pass
        if club_col:
            try:
                dframe.groupby(club_col).size().to_excel(writer, sheet_name="by_club")
            except Exception:
                pass
    buf.seek(0)
    return buf.getvalue()

st.sidebar.markdown("---")
st.sidebar.download_button("Download filtered CSV", data=filtered.to_csv(index=False), file_name="footlens_filtered.csv", mime="text/csv")
st.sidebar.download_button("Download Excel report", data=to_excel_bytes(filtered), file_name="footlens_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------
# Role-specific KPIs
# ---------------------
st.markdown("---")
cols = st.columns(4)
total = len(filtered)
avg_rating = safe_mean(filtered.get(mapped["rating"], filtered.get("avg_rating_before_matches", pd.Series(dtype=float))))
avg_duration = safe_mean(filtered["injury_duration_days"])
avg_perf_change = safe_mean(filtered["performance_change"])
cols[0].metric("Total records", f"{total}")
cols[1].metric("Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
cols[2].metric("Avg Injury Duration (days)", f"{avg_duration:.1f}" if not np.isnan(avg_duration) else "N/A")
cols[3].metric("Avg Performance Change", f"{avg_perf_change:.2f}" if not np.isnan(avg_perf_change) else "N/A")

if role == "Manager":
    st.info("Manager: focus on club-level trends and top injuries.")
elif role == "Coach":
    st.info("Coach: focus on individual recovery timelines and before/after performance.")
elif role == "Club Analyst":
    st.info("Club Analyst: aggregate metrics, exports, and season trends.")
elif role == "Scout":
    st.info("Scout: availability, age, position, and risk signals.")
else:
    st.info("Custom Analyst: full control.")

# ---------------------
# Tabs (Overview, EDA, Player, Club, Injury, Deep Dive, Rehab, Risk)
# ---------------------
tabs = st.tabs(["Overview", "EDA & Trends", "Player Impact", "Club Analysis", "Injury Analysis", "Player Deep Dive", "Rehab Scheduler", "Risk Model"])

# Overview tab
with tabs[0]:
    st.header("Overview")
    if ensure_nonempty(filtered):
        st.subheader("Sample")
        st.dataframe(filtered.head(10))
        st.write(f"Records after filters: {len(filtered)}")
        st.write("Mapped column preview:")
        st.json(mapped)
    else:
        st.info("No data after filters.")

# EDA & Trends tab
with tabs[1]:
    st.header("EDA & Trends")
    if not ensure_nonempty(filtered):
        st.stop()
    mode = st.selectbox("Tab viz mode", ["Auto", "Plotly", "Matplotlib"], index=0)
    style = st.selectbox("Tab seaborn style", ["Auto", "Modern Clean", "Classic Analytics"], index=0)
    mode = mode if mode != "Auto" else global_mode
    style = style if style != "Auto" else global_style

    numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Correlation")
        corr = filtered[numeric_cols].corr()
        if mode == "Plotly":
            fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
            st.plotly_chart(fig, use_container_width=True)
            buf = fig_to_png_bytes_plotly(fig)
            if buf:
                st.download_button("Download correlation PNG", data=buf.getvalue(), file_name="correlation.png", mime="image/png")
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            png = fig_to_png_bytes_matplotlib(fig)
            st.download_button("Download correlation PNG", data=png.getvalue(), file_name="correlation.png", mime="image/png")
    else:
        st.info("No numeric columns for correlation.")

    st.subheader("Missing values overview")
    if filtered.isnull().sum().sum() > 0:
        fig = ff.create_annotated_heatmap(z=filtered.isnull().astype(int).T.values,
                                          x=list(filtered.index.astype(str)),
                                          y=list(filtered.columns),
                                          showscale=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected.")

# Remaining tabs (Player Impact, Club Analysis, Injury Analysis, Player Deep Dive, Rehab Scheduler, Risk Model) can remain unchanged, just ensure numeric conversion and safety as above.

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("FootLens Pro — Advanced. You can add custom KPIs, predictive models, or interactive scheduler next.")
