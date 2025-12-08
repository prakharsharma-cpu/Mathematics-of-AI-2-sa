# FootLens Analytics - Streamlit Dashboard
# Filename: footlens_streamlit_dashboard.py
# Purpose: Interactive dashboard to explore relationship between player injuries and team performance
# Author: FootLens AI Research & Insights (Junior Sports Data Analyst template)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="FootLens — Injuries vs Performance", page_icon="⚽")

# --- Additional upload for injury-level data --------------------------------
# Optional second CSV: player injuries (columns suggested: player, team, match_date, injury_type, days_out)
# Merged later for deeper insights

injury_player_df = None
if 'uploaded_player' not in st.session_state:
    st.session_state.uploaded_player = None

with st.sidebar:
    st.header("Player Injury Data (Optional)")
    uploaded_player = st.file_uploader("Upload player-level injury CSV", type=["csv"], key="player_csv")
    if uploaded_player is not None:
        try:
            injury_player_df = pd.read_csv(uploaded_player)
        except Exception as e:
            st.error(f"Error loading player-level CSV: {e}")

# --- Helper functions -------------------------------------------------
@st.cache_data
def load_sample_data(n_matches=400):
    # Create synthetic dataset with realistic-ish structure
    rng = np.random.default_rng(42)
    seasons = ["2021/22", "2022/23", "2023/24", "2024/25"]
    teams = [f"Team {c}" for c in list("ABCDEFGHIJ")[:10]]
    positions = ["GK", "DEF", "MID", "FWD"]

    rows = []
    start_date = datetime(2021, 8, 1)
    for i in range(n_matches):
        date = start_date + pd.Timedelta(days=int(i * 2))
        season = seasons[(i // 100) % len(seasons)]
        home = rng.choice(teams)
        away = rng.choice([t for t in teams if t != home])
        home_goals = rng.poisson(1.4)
        away_goals = rng.poisson(1.1)
        if home_goals > away_goals:
            result = "Home Win"
            points_home, points_away = 3, 0
        elif home_goals < away_goals:
            result = "Away Win"
            points_home, points_away = 0, 3
        else:
            result = "Draw"
            points_home = points_away = 1

        # Injuries simulated
        injuries_home = rng.poisson(0.6)
        injuries_away = rng.poisson(0.5)
        minutes_lost_home = int(injuries_home * rng.integers(15, 90))
        minutes_lost_away = int(injuries_away * rng.integers(15, 90))

        rows.append({
            "match_id": i,
            "match_date": date,
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "home_points": points_home,
            "away_points": points_away,
            "home_injuries": injuries_home,
            "away_injuries": injuries_away,
            "home_minutes_lost": minutes_lost_home,
            "away_minutes_lost": minutes_lost_away
        })

    df = pd.DataFrame(rows)
    return df

@st.cache_data
def preprocess_matches(df_raw):
    df = df_raw.copy()
    # Ensure datetime
    if not np.issubdtype(df["match_date"].dtype, np.datetime64):
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Melt to team-level rows (one row per team per match)
    home = df.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_points": "points",
        "home_injuries": "injuries",
        "home_minutes_lost": "minutes_lost"
    })
    home["venue"] = "Home"

    away = df.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_points": "points",
        "away_injuries": "injuries",
        "away_minutes_lost": "minutes_lost"
    })
    away["venue"] = "Away"

    cols = [c for c in home.columns if c in home.columns]
    team_level = pd.concat([home, away], ignore_index=True, sort=False)

    # Basic derived metrics
    team_level["goal_diff"] = team_level["goals_for"] - team_level["goals_against"]
    # Rolling injury burden per team
    team_level = team_level.sort_values(["team", "match_date"]) 
    team_level["injuries_rolling3"] = team_level.groupby("team")["injuries"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    team_level["minutes_lost_rolling3"] = team_level.groupby("team")["minutes_lost"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

    return team_level

# --- UI layout --------------------------------------------------------
st.title("⚽ FootLens — Injuries vs Team Performance Dashboard")
st.markdown(
    "Use this interactive dashboard to explore how player injuries (counts and minutes lost) correlate with match outcomes and team standings.\n\n"
    "Upload your dataset (CSV) or use the synthetic sample dataset to experiment. Expected columns for best results: `match_date`, `season`, `home_team`, `away_team`, `home_goals`, `away_goals`, `home_points`, `away_points`, `home_injuries`, `away_injuries`, `home_minutes_lost`, `away_minutes_lost`."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV file (match-level) — leave empty to use sample", type=["csv"])    
    sample_button = st.button("Load sample dataset")
    st.markdown("---")
    st.header("Filters")
    season_filter = st.multiselect("Seasons", options=[], default=None)
    team_filter = st.multiselect("Teams", options=[], default=None)
    venue_filter = st.selectbox("Venue", options=["All", "Home", "Away"], index=0)
    date_range = st.date_input("Date range", [])
    st.markdown("---")
    st.header("Export")
    st.write("Download filtered CSV or summary after applying filters.")

# Load data
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
elif sample_button or uploaded is None:
    df_raw = load_sample_data(420)

team_level = preprocess_matches(df_raw)

# Update sidebar filter options now that data is loaded
with st.sidebar:
    seasons = sorted(team_level["season"].dropna().unique().tolist())
    teams = sorted(team_level["team"].dropna().unique().tolist())
    season_filter = st.multiselect("Seasons", options=seasons, default=seasons)
    team_filter = st.multiselect("Teams", options=teams, default=teams[:5])

# Apply filters
df = team_level.copy()
if season_filter:
    df = df[df["season"].isin(season_filter)]
if team_filter:
    df = df[df["team"].isin(team_filter)]
if venue_filter != "All":
    df = df[df["venue"] == venue_filter]
if len(date_range) == 2:
    start, end = date_range
    df = df[(df["match_date"] >= pd.to_datetime(start)) & (df["match_date"] <= pd.to_datetime(end))]

st.sidebar.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8"), file_name="footlens_filtered.csv")

# --- Top-level KPIs --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Matches (team-rows)", int(len(df)))
with col2:
    st.metric("Avg injuries per match", round(df["injuries"].mean(), 2))
with col3:
    st.metric("Avg minutes lost per match", int(df["minutes_lost"].mean()))
with col4:
    st.metric("Avg points per match", round(df["points"].mean(), 2))

st.markdown("---")

# --- Time series: injuries vs points --------------------------------
st.subheader("Time series: Injuries and Points — selected teams")
team_for_ts = st.multiselect("Select teams to compare (time series)", options=teams, default=team_filter[:3])

if team_for_ts:
    df_ts = df[df["team"].isin(team_for_ts)].sort_values(["team", "match_date"]) 
    fig = go.Figure()
    # injuries (rolling)
    for t in team_for_ts:
        d = df_ts[df_ts["team"] == t]
        fig.add_trace(go.Scatter(x=d["match_date"], y=d["injuries_rolling3"], name=f"{t} — injuries (roll3)", mode="lines+markers", yaxis="y1"))
        fig.add_trace(go.Scatter(x=d["match_date"], y=d["points"].rolling(3, min_periods=1).mean(), name=f"{t} — points (roll3)", mode="lines", yaxis="y2"))

    # layout with two y axes
    fig.update_layout(
        xaxis_title="Match date",
        yaxis=dict(title="Injuries (rolling 3)", side="left"),
        yaxis2=dict(title="Points (rolling 3)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one team for time series view.")

st.markdown("---")

# --- Scatter: injuries vs points ------------------------------------
st.subheader("Team-level relationship: Injuries vs Points")
agg = df.groupby(["team", "season"]).agg(
    matches=("match_id", "nunique"),
    avg_injuries=("injuries", "mean"),
    avg_minutes_lost=("minutes_lost", "mean"),
    avg_points=("points", "mean")
).reset_index()

fig2 = px.scatter(
    agg, x="avg_injuries", y="avg_points", size="matches", color="season",
    hover_data=["team", "avg_minutes_lost", "matches"],
    labels={"avg_injuries": "Average injuries per match", "avg_points": "Average points per match"},
    title="Average injuries per match vs average points — circle size = matches played"
)
st.plotly_chart(fig2, use_container_width=True)

# Compute correlation
corr_text = "Correlation (Pearson) between avg_injuries and avg_points:"
if len(agg) >= 2:
    corr = agg[["avg_injuries", "avg_points"]].corr().iloc[0,1]
    st.write(corr_text, round(corr, 3))
else:
    st.write(corr_text, "Not enough data")

st.markdown("---")

# --- Injury heatmap by venue / season / team -------------------------
st.subheader("Injury burden heatmap")
heat = df.groupby(["team", "season"]).agg(total_injuries=("injuries", "sum"), total_minutes=("minutes_lost", "sum"))
heat = heat.reset_index()

heat_pivot = heat.pivot(index="team", columns="season", values="total_injuries").fillna(0)

fig3 = px.imshow(heat_pivot, labels=dict(x="Season", y="Team", color="Total injuries"), aspect="auto", title="Total injuries by team and season")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# --- Standings simulator (points table) ------------------------------
st.subheader("Simple standings: Points aggregation")
standings = df.groupby(["team", "season"]).agg(matches=("match_id","nunique"), points=("points", "sum"), goals_for=("goals_for","sum"), goals_against=("goals_against","sum")).reset_index()
standings["goal_diff"] = standings["goals_for"] - standings["goals_against"]
standings_sorted = standings.sort_values(["season", "points", "goal_diff"], ascending=[True, False, False])

season_choice = st.selectbox("Choose season to view standings", options=seasons, index=0)
st.dataframe(standings_sorted[standings_sorted["season"] == season_choice].reset_index(drop=True))

st.markdown("---")

# --- Squad risk table (minutes lost -> risk band) --------------------
st.subheader("Squad-level risk assessment (team-season)")
risk = heat.copy()
# Define simple risk bands
risk["risk_band"] = pd.cut(risk["total_minutes"], bins=[-1, 200, 500, 1000, 100000], labels=["Low", "Moderate", "High", "Critical"]) 
st.dataframe(risk.sort_values(["season", "total_minutes"], ascending=[True, False]).reset_index(drop=True))

st.markdown("---")

# --- Simple actionable insights generator ---------------------------
st.subheader("Actionable insights (auto-generated suggestions)")
insights = []

# Insight rules (simple heuristics)
for _, row in risk.iterrows():
    if row["total_minutes"] > 1000:
        insights.append(f"{row['team']} ({row['season']}): Critical minutes lost — review medical protocols, reduce training intensity, and prioritize recovery for key players.")
    elif row["total_minutes"] > 500:
        insights.append(f"{row['team']} ({row['season']}): High injury burden — consider increasing squad rotation and targeted conditioning for positions with repeated injuries.")
    elif row["total_minutes"] > 200:
        insights.append(f"{row['team']} ({row['season']}): Moderate injury load — monitor training loads and use sport science to individualize recovery.")
    else:
        insights.append(f"{row['team']} ({row['season']}): Low injury minutes — current load management appears effective; maintain preventive measures.")

# Deduplicate and show top suggestions
insights = list(dict.fromkeys(insights))
for s in insights[:12]:
    st.write("• ", s)

st.markdown("---")

# --- Appendix: show raw team-level data --------------------------------
with st.expander("Show filtered team-level data"):
    st.dataframe(df.reset_index(drop=True))

st.caption("FootLens — Dashboard prototype. Adapt column names and logic to match your production dataset schema.")

# End of file
