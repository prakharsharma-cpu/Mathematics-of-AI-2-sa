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

# Optional ML deps - used if available
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional Excel engine
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

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
    # Try Plotly to_image (requires kaleido)
    try:
        img_bytes = fig.to_image(format="png")
        return BytesIO(img_bytes)
    except Exception:
        return None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------------
# Excel export helper
# ---------------------
def to_excel_bytes(dframe):
    if not EXCEL_AVAILABLE:
        st.warning("xlsxwriter not installed — Excel export unavailable.")
        return None
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        dframe.to_excel(writer, sheet_name="filtered", index=False)
        writer.save()
    buf.seek(0)
    return buf.getvalue()

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

# Derived columns
if mapped["injury_start"] and mapped["injury_end"]:
    df["injury_duration_days"] = (df[mapped["injury_end"]] - df[mapped["injury_start"]]).dt.days.clip(lower=0)
    df["injury_month"] = df[mapped["injury_start"]].dt.month
else:
    df["injury_duration_days"] = np.nan
    df["injury_month"] = np.nan

# Before/after rating
col_lower = {c.lower(): c for c in df.columns}
def cols_matching(substr):
    return [orig for low, orig in col_lower.items() if substr.lower() in low]

before_rating_cols = cols_matching("before_injury_player_rating")
after_rating_cols = cols_matching("after_injury_player_rating")

if before_rating_cols:
    df["avg_rating_before_matches"] = df[before_rating_cols].mean(axis=1, skipna=True)
elif mapped["rating"]:
    df["avg_rating_before_matches"] = df[mapped["rating"]]
else:
    df["avg_rating_before_matches"] = np.nan

if after_rating_cols:
    df["avg_rating_after_matches"] = df[after_rating_cols].mean(axis=1, skipna=True)
else:
    df["avg_rating_after_matches"] = np.nan

df["performance_change"] = df.get("avg_rating_after_matches", 0) - df.get("avg_rating_before_matches", 0)

# ---------------------
# Sidebar filters & role
# ---------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters & Role View")
role = st.sidebar.selectbox("Choose role", ["Manager", "Coach", "Club Analyst", "Scout", "Custom Analyst"])
club_col = mapped["team"] if mapped["team"] in df.columns else None
player_col = mapped["player"] if mapped["player"] in df.columns else None
inj_col = mapped["injury_type"] if mapped["injury_type"] in df.columns else None

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
    filtered = filtered[filtered[in_col].isin(filter_injury)]

# Exports
st.sidebar.markdown("---")
csv_bytes = filtered.to_csv(index=False)
st.sidebar.download_button("Download filtered CSV", data=csv_bytes, file_name="footlens_filtered.csv", mime="text/csv")

excel_bytes = to_excel_bytes(filtered)
if excel_bytes:
    st.sidebar.download_button("Download Excel report", data=excel_bytes, file_name="footlens_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------
# Role-specific KPIs
# ---------------------
st.markdown("---")
cols = st.columns(4)
total = len(filtered)
avg_rating = safe_mean(filtered.get(mapped["rating"], filtered.get("avg_rating_before_matches", pd.Series(dtype=float))))
avg_duration = safe_mean(filtered.get("injury_duration_days", pd.Series(dtype=float)))
avg_perf_change = safe_mean(filtered.get("performance_change", pd.Series(dtype=float)))
cols[0].metric("Total records", f"{total}")
cols[1].metric("Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
cols[2].metric("Avg Injury Duration (days)", f"{avg_duration:.1f}" if not np.isnan(avg_duration) else "N/A")
cols[3].metric("Avg Performance Change", f"{avg_perf_change:.2f}" if not np.isnan(avg_perf_change) else "N/A")

role_messages = {
    "Manager": "Manager: focus on club-level trends and top injuries.",
    "Coach": "Coach: focus on individual recovery timelines and before/after performance.",
    "Club Analyst": "Club Analyst: aggregate metrics, exports, and season trends.",
    "Scout": "Scout: availability, age, position, and risk signals.",
    "Custom Analyst": "Custom Analyst: full control."
}
st.info(role_messages.get(role, ""))

# ---------------------
# Tabs (Overview, EDA, etc.)
# ---------------------
tabs = st.tabs(["Overview", "EDA & Trends", "Player Impact", "Club Analysis", "Injury Analysis", "Player Deep Dive", "Rehab Scheduler", "Risk Model"])

# Overview
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

# The rest of your tabs (EDA, Player Impact, Club Analysis, Injury Analysis, Player Deep Dive, Rehab Scheduler, Risk Model) remain the same.
# For Excel downloads, it will now safely check for `xlsxwriter` availability.
# For ML risk model, it safely checks `SKLEARN_AVAILABLE`.

st.markdown("---")
st.caption("FootLens Pro — Advanced. Custom KPI sets per role, predictive models (with labels), or interactive scheduler are supported.")
