# footlens_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("FootLens Analytics: Player Injury Impact Dashboard")
st.markdown("""
This dashboard analyzes how player injuries impact individual performance and team outcomes.
""")

# --- Step 2: Data Preprocessing and Cleaning ---
@st.cache_data
def load_and_clean_data(file_path="football_injuries.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("Dataset not found, generating sample data...")
        # Sample dataset generation
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
    df.fillna({
        "goals_scored": 0,
        "assists": 0,
        "pre_injury_rating": df["pre_injury_rating"].mean(),
        "post_injury_rating": df["post_injury_rating"].mean()
    }, inplace=True)

    # Convert dates to datetime
    df["injury_start_date"] = pd.to_datetime(df["injury_start_date"])
    df["injury_end_date"] = pd.to_datetime(df["injury_end_date"])

    # New metrics
    df["rating_drop"] = df["pre_injury_rating"] - df["post_injury_rating"]
    df["injury_duration"] = (df["injury_end_date"] - df["injury_start_date"]).dt.days
    df["team_perf_drop_index"] = df["rating_drop"] * df["matches_played"] / df["matches_played"].max()

    return df

df = load_and_clean_data()

# --- Step 3: Exploratory Data Analysis ---
st.sidebar.header("Filters")
season_filter = st.sidebar.multiselect("Season", options=df["season"].unique(), default=df["season"].unique())
team_filter = st.sidebar.multiselect("Team", options=df["team"].unique(), default=df["team"].unique())
df_filtered = df[(df["season"].isin(season_filter)) & (df["team"].isin(team_filter))]

# --- Insights ---
st.subheader("Top Players by Injury Count")
top_injured = df_filtered.groupby("player_name")["injuries"].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_injured)

st.subheader("Team Performance Drop by Injury")
team_drop = df_filtered.groupby("team")["team_perf_drop_index"].sum().sort_values(ascending=False)
fig_team_drop = px.bar(team_drop, x=team_drop.index, y=team_drop.values, title="Team Performance Drop Index")
st.plotly_chart(fig_team_drop)

st.subheader("Player Performance Timeline (Pre vs Post Injury)")
# Line chart example
sample_player = df_filtered["player_name"].iloc[0]
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
# Generate heatmap: months vs team
df_filtered["injury_month"] = df_filtered["injury_start_date"].dt.month
heatmap_data = df_filtered.pivot_table(index="team", columns="injury_month", values="injuries", aggfunc="sum", fill_value=0)
fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Month", y="Team", color="Injuries"), title="Injury Frequency Heatmap")
st.plotly_chart(fig_heatmap)

st.subheader("Player Age vs Performance Drop")
fig_scatter = px.scatter(df_filtered, x="age", y="rating_drop", color="team", hover_data=["player_name"], title="Player Age vs Performance Drop")
st.plotly_chart(fig_scatter)

st.subheader("Leaderboard: Players with Biggest Comeback")
df_filtered["rating_improvement"] = df_filtered["post_injury_rating"] - df_filtered["pre_injury_rating"]
leaderboard = df_filtered.groupby("player_name")["rating_improvement"].mean().sort_values(ascending=False).head(10)
st.table(leaderboard.reset_index().rename(columns={"rating_improvement": "Avg Rating Improvement"}))
