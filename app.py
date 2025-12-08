# footlens_dashboard_upload.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("FootLens Analytics: Player Injury Impact Dashboard")
st.markdown("""
Upload your dataset or use a generated sample to explore player injuries and team performance.
""")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
use_sample = False

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No CSV uploaded. Generating sample dataset for demo purposes.")
    use_sample = True

# --- Step 2: Data Preprocessing ---
@st.cache_data
def preprocess_data(df, use_sample=False):
    if use_sample:
        # Generate sample data
        seasons = ["2022-23", "2023-24"]
        teams = ["Team A", "Team B", "Team C"]
        players = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]
        data = []
        for season in seasons:
            for team in teams:
                for player in players:
                    start_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="days")
                    end_date = start_date + pd.to_timedelta(np.random.randint(5, 30), unit="days")
                    data.append({
                        "season": season,
                        "team": team,
                        "player_name": player,
                        "age": np.random.randint(18, 35),
                        "matches_played": np.random.randint(5, 38),
                        "injuries": np.random.randint(0, 5),
                        "goals_scored": np.random.randint(0, 20),
                        "assists": np.random.randint(0, 10),
                        "pre_injury_rating": np.random.uniform(5.0, 8.0),
                        "post_injury_rating": np.random.uniform(5.0, 8.0),
                        "injury_start_date": start_date,
                        "injury_end_date": end_date
                    })
        df = pd.DataFrame(data)

    # Handle missing values
    for col in ["goals_scored", "assists", "pre_injury_rating", "post_injury_rating"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Convert dates to datetime
    if "injury_start_date" in df.columns:
        df["injury_start_date"] = pd.to_datetime(df["injury_start_date"])
    if "injury_end_date" in df.columns:
        df["injury_end_date"] = pd.to_datetime(df["injury_end_date"])

    # Create new metrics
    if "pre_injury_rating" in df.columns and "post_injury_rating" in df.columns:
        df["rating_drop"] = df["pre_injury_rating"] - df["post_injury_rating"]
        df["rating_improvement"] = df["post_injury_rating"] - df["pre_injury_rating"]
    if "injury_start_date" in df.columns and "injury_end_date" in df.columns:
        df["injury_duration"] = (df["injury_end_date"] - df["injury_start_date"]).dt.days

    # Team performance drop index
    if "matches_played" in df.columns and "rating_drop" in df.columns:
        df["team_perf_drop_index"] = df["rating_drop"] * df["matches_played"] / df["matches_played"].max()

    # Injury month for heatmap
    if "injury_start_date" in df.columns:
        df["injury_month"] = df["injury_start_date"].dt.month

    return df

df = preprocess_data(df, use_sample)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
if "season" in df.columns:
    season_filter = st.sidebar.multiselect("Season", options=df["season"].unique(), default=df["season"].unique())
else:
    season_filter = df["season"].unique() if "season" in df.columns else None

if "team" in df.columns:
    team_filter = st.sidebar.multiselect("Team", options=df["team"].unique(), default=df["team"].unique())
else:
    team_filter = df["team"].unique() if "team" in df.columns else None

df_filtered = df.copy()
if season_filter is not None:
    df_filtered = df_filtered[df_filtered["season"].isin(season_filter)]
if team_filter is not None:
    df_filtered = df_filtered[df_filtered["team"].isin(team_filter)]

# --- Visualizations ---
st.subheader("Top Players by Injury Count")
if "player_name" in df_filtered.columns and "injuries" in df_filtered.columns:
    top_injured = df_filtered.groupby("player_name")["injuries"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_injured)

st.subheader("Team Performance Drop by Injury")
if "team" in df_filtered.columns and "team_perf_drop_index" in df_filtered.columns:
    team_drop = df_filtered.groupby("team")["team_perf_drop_index"].sum().sort_values(ascending=False)
    fig_team_drop = px.bar(team_drop, x=team_drop.index, y=team_drop.values, title="Team Performance Drop Index")
    st.plotly_chart(fig_team_drop)

st.subheader("Player Performance Timeline")
if "player_name" in df_filtered.columns:
    sample_player = st.selectbox("Select Player", df_filtered["player_name"].unique())
    player_df = df_filtered[df_filtered["player_name"] == sample_player].sort_values("injury_start_date")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=player_df["injury_start_date"],
        y=player_df["pre_injury_rating"],
        mode="lines+markers",
        name="Pre Injury Rating"
    ))
    fig_line.add_trace(go.Scatter(
        x=player_df["injury_start_date"],
        y=player_df["post_injury_rating"],
        mode="lines+markers",
        name="Post Injury Rating"
    ))
    fig_line.update_layout(title=f"Performance Timeline for {sample_player}", xaxis_title="Date", yaxis_title="Rating")
    st.plotly_chart(fig_line)

st.subheader("Injury Frequency Heatmap")
if "injury_month" in df_filtered.columns and "team" in df_filtered.columns:
    heatmap_data = df_filtered.pivot_table(index="team", columns="injury_month", values="injuries", aggfunc="sum", fill_value=0)
    fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Month", y="Team", color="Injuries"), title="Injury Frequency Heatmap")
    st.plotly_chart(fig_heatmap)

st.subheader("Player Age vs Performance Drop")
if "age" in df_filtered.columns and "rating_drop" in df_filtered.columns:
    fig_scatter = px.scatter(df_filtered, x="age", y="rating_drop", color="team", hover_data=["player_name"], title="Player Age vs Performance Drop")
    st.plotly_chart(fig_scatter)

st.subheader("Leaderboard: Biggest Comeback Players")
if "player_name" in df_filtered.columns and "rating_improvement" in df_filtered.columns:
    leaderboard = df_filtered.groupby("player_name")["rating_improvement"].mean().sort_values(ascending=False).head(10)
    st.table(leaderboard.reset_index().rename(columns={"rating_improvement": "Avg Rating Improvement"}))

# --- Step 5: What-If Simulation ---
st.subheader("What-If Simulation: Estimate Team Points Drop if Player is Injured")
if "team" in df_filtered.columns and "rating_drop" in df_filtered.columns:
    sim_player = st.selectbox("Select Player for Simulation", df_filtered["player_name"].unique(), key="sim_player")
    sim_team = df_filtered[df_filtered["player_name"] == sim_player]["team"].values[0]
    sim_rating_drop = df_filtered[df_filtered["player_name"] == sim_player]["rating_drop"].mean()
    sim_matches = df_filtered[df_filtered["player_name"] == sim_player]["matches_played"].mean()
    # Example simple simulation formula
    estimated_points_loss = sim_rating_drop * sim_matches / 2
    st.metric(label=f"Estimated Points Loss for {sim_team} if {sim_player} is injured", value=f"{estimated_points_loss:.1f}")
