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

# Attempt to compute before/after ratings by pattern
col_lower = {c.lower(): c for c in df.columns}
def cols_matching(substr):
    return [orig for low, orig in col_lower.items() if substr.lower() in low]

before_rating_cols = cols_matching("before_injury_player_rating") + cols_matching("before_injury_player_rating".replace("_",""))
after_rating_cols = cols_matching("after_injury_player_rating") + cols_matching("after_injury_player_rating".replace("_",""))

if before_rating_cols:
    df["avg_rating_before_matches"] = df[before_rating_cols].mean(axis=1, skipna=True)
else:
    # fallback to mapped rating if only one rating present
    if mapped["rating"]:
        df["avg_rating_before_matches"] = df[mapped["rating"]]
    else:
        df["avg_rating_before_matches"] = np.nan

if after_rating_cols:
    df["avg_rating_after_matches"] = df[after_rating_cols].mean(axis=1, skipna=True)
else:
    df["avg_rating_after_matches"] = np.nan

if "avg_rating_before_matches" in df.columns and "avg_rating_after_matches" in df.columns:
    df["performance_change"] = df["avg_rating_after_matches"] - df["avg_rating_before_matches"]
else:
    df["performance_change"] = np.nan

# Compute team GD proxies (if present)
before_gd = cols_matching("before_injury_gd") + cols_matching("before_injury_goal_difference")
after_gd = cols_matching("after_injury_gd") + cols_matching("after_injury_goal_difference")
missed_gd = cols_matching("missed_match_gd") + cols_matching("missed_match_goal_difference")

if before_gd:
    df["team_gd_before"] = df[before_gd].mean(axis=1, skipna=True)
else:
    df["team_gd_before"] = np.nan
if missed_gd:
    df["team_gd_during"] = df[missed_gd].mean(axis=1, skipna=True)
else:
    df["team_gd_during"] = np.nan
if after_gd:
    df["team_gd_after"] = df[after_gd].mean(axis=1, skipna=True)
else:
    df["team_gd_after"] = np.nan

if "team_gd_before" in df.columns and "team_gd_during" in df.columns:
    df["team_performance_drop"] = df["team_gd_before"] - df["team_gd_during"]
else:
    df["team_performance_drop"] = np.nan
if "team_gd_during" in df.columns and "team_gd_after" in df.columns:
    df["team_recovery"] = df["team_gd_after"] - df["team_gd_during"]
else:
    df["team_recovery"] = np.nan

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

# Exports in sidebar
st.sidebar.markdown("---")
st.sidebar.download_button("Download filtered CSV", data=filtered.to_csv(index=False), file_name="footlens_filtered.csv", mime="text/csv")
# Excel export with multiple sheets
def to_excel_bytes(dframe):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        dframe.to_excel(writer, sheet_name="filtered", index=False)
        # small extra sheets
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
        writer.save()
    buf.seek(0)
    return buf.getvalue()

st.sidebar.download_button("Download Excel report", data=to_excel_bytes(filtered), file_name="footlens_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------
# Role-specific KPIs
# ---------------------
st.markdown("---")
cols = st.columns(4)
total = len(filtered)
avg_rating = safe_mean(filtered[mapped["rating"]] if mapped["rating"] and mapped["rating"] in filtered.columns else filtered.get("avg_rating_before_matches", pd.Series(dtype=float)))
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
# Tabs
# ---------------------
tabs = st.tabs(["Overview", "EDA & Trends", "Player Impact", "Club Analysis", "Injury Analysis", "Player Deep Dive", "Rehab Scheduler", "Risk Model"])

# ---------------
# Overview
# ---------------
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

# ---------------
# EDA & Trends
# ---------------
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
            # allow PNG download of this plotly figure
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

# ---------------------
# Player Impact
# ---------------------
with tabs[2]:
    st.header("Player Impact & Rankings")
    if not ensure_nonempty(filtered):
        st.stop()
    if player_col:
        player_summary = filtered.groupby(player_col).agg(
            injuries=(player_col, "count"),
            avg_injury_duration=("injury_duration_days", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            avg_perf_change=("performance_change", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            avg_rating=("avg_rating_before_matches", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan)
        ).reset_index().sort_values("injuries", ascending=False)
        st.dataframe(player_summary.head(200), use_container_width=True)
        # plot top impacted players
        if "avg_perf_change" in player_summary.columns and not player_summary["avg_perf_change"].isnull().all():
            tmp = player_summary.sort_values("avg_perf_change").head(20)
            fig = px.bar(tmp.rename(columns={player_col: "player"}), x="avg_perf_change", y="player", orientation="h", title="Players with largest negative perf change")
            st.plotly_chart(fig, use_container_width=True)
            buf = fig_to_png_bytes_plotly(fig)
            if buf:
                st.download_button("Download players impact PNG", data=buf.getvalue(), file_name="players_impact.png", mime="image/png")
    else:
        st.info("No player column available.")

# ---------------------
# Club Analysis
# ---------------------
with tabs[3]:
    st.header("Club Analysis")
    if not ensure_nonempty(filtered):
        st.stop()
    if club_col:
        club_summary = filtered.groupby(club_col).agg(
            injuries=(club_col, "count"),
            avg_injury_duration=("injury_duration_days", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            avg_team_drop=("team_performance_drop", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan)
        ).reset_index().sort_values("injuries", ascending=False)
        st.dataframe(club_summary, use_container_width=True)
        fig = px.bar(club_summary.head(20), x="injuries", y=club_col, orientation="h", title="Top clubs by injuries")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No club/team column available.")

# ---------------------
# Injury Analysis
# ---------------------
with tabs[4]:
    st.header("Injury Analysis & Trends")
    if not ensure_nonempty(filtered):
        st.stop()
    if inj_col:
        inj_counts = filtered[inj_col].value_counts().reset_index()
        inj_counts.columns = ["injury_type", "count"]
        st.dataframe(inj_counts.head(50))
        fig = px.bar(inj_counts.head(30), x="count", y="injury_type", orientation="h", title="Injury counts")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No injury column available.")

    if "injury_duration_days" in filtered.columns and not filtered["injury_duration_days"].isnull().all():
        fig = px.histogram(filtered, x="injury_duration_days", nbins=30, title="Injury duration (days)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No injury duration data present.")

    if "injury_month" in filtered.columns and not filtered["injury_month"].isnull().all():
        monthly = filtered.groupby("injury_month").size().reset_index(name="count").sort_values("injury_month")
        fig = px.line(monthly, x="injury_month", y="count", markers=True, title="Injuries by month")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------
# Player Deep Dive
# ---------------------
with tabs[5]:
    st.header("Player Deep Dive")
    if not ensure_nonempty(filtered):
        st.stop()
    if player_col:
        players = sorted(filtered[player_col].dropna().unique())
        selected = st.selectbox("Select player", players)
        p_df = filtered[filtered[player_col] == selected]
        st.dataframe(p_df, use_container_width=True)
        # timeline if dates
        if mapped["injury_start"] and mapped["injury_start"] in p_df.columns and p_df[mapped["injury_start"]].notna().any():
            timeline = p_df.copy()
            timeline["start"] = timeline[mapped["injury_start"]]
            timeline["end"] = timeline[mapped["injury_end"]] if mapped["injury_end"] and mapped["injury_end"] in timeline.columns else timeline["start"]
            timeline["label"] = timeline.get(inj_col, "injury").astype(str) + " (" + timeline["injury_duration_days"].astype(str) + "d)"
            fig = px.timeline(timeline, x_start="start", x_end="end", y="label", title=f"Injury timeline - {selected}")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No injury start/end dates for this player.")

        if "avg_rating_before_matches" in p_df.columns and "avg_rating_after_matches" in p_df.columns:
            if not p_df[["avg_rating_before_matches", "avg_rating_after_matches"]].dropna(how="all").empty:
                fig = px.scatter(p_df, x="avg_rating_before_matches", y="avg_rating_after_matches", hover_data=[inj_col, "injury_duration_days"], title="Before vs After ratings")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating before/after data for this player.")
    else:
        st.info("No player column available.")

# ---------------------
# Rehab Scheduler
# ---------------------
with tabs[6]:
    st.header("Rehab Scheduler")
    st.markdown("Generate estimated return dates and a simple rehab timeline per player. Uses injury duration when available; otherwise uses median duration from dataset.")

    if not ensure_nonempty(filtered):
        st.stop()

    # estimate median duration for fallback
    median_duration = np.nanmedian(filtered["injury_duration_days"]) if "injury_duration_days" in filtered.columns else np.nan
    st.write(f"Median injury duration (dataset): {median_duration if not np.isnan(median_duration) else 'N/A'} days")

    # choose players to schedule
    if player_col:
        selected_players = st.multiselect("Choose players to create schedule for", options=player_options, default=player_options[:5] if len(player_options)>0 else [])
        schedule_rows = []
        for p in selected_players:
            recs = filtered[filtered[player_col] == p]
            if recs.empty:
                continue
            # use latest injury record if multiple
            rec = recs.sort_values(mapped["injury_start"] if mapped["injury_start"] in recs.columns else recs.columns[0], ascending=False).iloc[0]
            start = rec[mapped["injury_start"]] if (mapped["injury_start"] and mapped["injury_start"] in recs.columns) else pd.NaT
            duration = rec.get("injury_duration_days", np.nan)
            if pd.isna(duration):
                duration = median_duration
            est_return = start + timedelta(days=int(duration)) if pd.notna(start) and not pd.isna(duration) else pd.NaT
            schedule_rows.append({
                "player": p,
                "injury_type": rec.get(inj_col, "N/A"),
                "injury_start": start,
                "estimated_return": est_return,
                "estimated_duration_days": duration
            })
        schedule_df = pd.DataFrame(schedule_rows)
        st.dataframe(schedule_df)
        if not schedule_df.empty:
            st.download_button("Download schedule CSV", data=schedule_df.to_csv(index=False), file_name="rehab_schedule.csv", mime="text/csv")
            st.download_button("Download schedule Excel", data=to_excel_bytes(schedule_df), file_name="rehab_schedule.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("No player column to create schedules for.")

# ---------------------
# Risk Model (ML + fallback heuristic)
# ---------------------
with tabs[7]:
    st.header("Re-injury Risk Model")
    st.markdown("Model: tries logistic regression if sklearn is available and dataset has a binary 'reinjury' column or enough features; otherwise computes heuristic risk score.")

    if not ensure_nonempty(filtered):
        st.stop()

    # if dataset has a 'reinjury' or 're-injury' or 'reinjured' column treat as label
    label_candidates = [c for c in filtered.columns if "reinj" in c.lower() or "re-inj" in c.lower()]
    label_col = label_candidates[0] if label_candidates else None

    if SKLEARN_AVAILABLE and label_col and filtered[label_col].dropna().nunique() > 1:
        st.success("Training logistic regression using detected label column: " + label_col)
        # choose features
        features = []
        for f in ["age", "injury_duration_days", "avg_rating_before_matches", "avg_rating_after_matches", "performance_change"]:
            if f in filtered.columns:
                features.append(f)
        if not features:
            st.warning("No numeric features available to train model. Falling back to heuristic.")
        else:
            X = filtered[features].fillna(0)
            y = filtered[label_col].astype(int).fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
            score = model.score(X_test_s, y_test)
            st.write(f"Model test accuracy: {score:.2f}")
            # predict risk for filtered rows (probability)
            proba = model.predict_proba(scaler.transform(X.fillna(0)))[:, 1]
            filtered["reinjury_risk_ml"] = proba
            st.dataframe(filtered[[player_col] if player_col else [] + ["reinjury_risk_ml"]].sort_values("reinjury_risk_ml", ascending=False).head(50))
            st.success("ML risk score added to filtered data (reinjury_risk_ml).")
    else:
        # heuristic fallback
        st.info("Using heuristic risk scoring (no binary label / sklearn unavailable). Heuristic combines age, injury duration, and performance drop.")
        # build z-scores where possible
        heur = pd.Series(0, index=filtered.index, dtype=float)
        weight_sum = 0.0
        # age: older -> higher risk
        if mapped["age"] and mapped["age"] in filtered.columns:
            age_z = (filtered[mapped["age"]] - filtered[mapped["age"]].mean()) / (filtered[mapped["age"]].std() + 1e-9)
            heur += 0.4 * age_z.fillna(0)
            weight_sum += 0.4
        # duration: longer -> higher risk
        if "injury_duration_days" in filtered.columns:
            dur_z = (filtered["injury_duration_days"] - filtered["injury_duration_days"].mean()) / (filtered["injury_duration_days"].std() + 1e-9)
            heur += 0.4 * dur_z.fillna(0)
            weight_sum += 0.4
        # perf change: large negative -> higher risk
        if "performance_change" in filtered.columns:
            perf_z = -1 * (filtered["performance_change"] - filtered["performance_change"].mean()) / (filtered["performance_change"].std() + 1e-9)
            heur += 0.2 * perf_z.fillna(0)
            weight_sum += 0.2
        # normalize
        if weight_sum == 0:
            st.warning("No features available for heuristic risk; defaulting to 0.1 risk.")
            filtered["reinjury_risk_heuristic"] = 0.1
        else:
            heur_scaled = heur / (weight_sum + 1e-9)
            # map to [0,1] via sigmoid
            risk_score = sigmoid(heur_scaled)
            filtered["reinjury_risk_heuristic"] = risk_score
            st.dataframe(filtered[[player_col] if player_col else [] + ["reinjury_risk_heuristic"]].sort_values("reinjury_risk_heuristic", ascending=False).head(50))
            st.success("Heuristic risk score added to filtered data (reinjury_risk_heuristic).")

    # allow download of risk-augmented dataset
    st.download_button("Download risk-augmented CSV", data=filtered.to_csv(index=False), file_name="footlens_with_risk.csv", mime="text/csv")
    st.download_button("Download risk-augmented Excel", data=to_excel_bytes(filtered), file_name="footlens_with_risk.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("FootLens Pro — Advanced. If you want a custom KPI set per role, predictive models (with labels), or interactive scheduler with calendar integrations, tell me which features to add next.")
