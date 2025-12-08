# footlens_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.title("FootLens Analytics: Player Injuries & Team Performance Dashboard")
st.markdown("""
This interactive dashboard helps visualize the impact of player injuries on team performance and match outcomes.
""")

# --- Load Dataset with Error Handling ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found: {file_path}")
        st.stop()  # Stop execution if file not found
    df = pd.read_csv(file_path)
    return df

# Path to your CSV dataset
dataset_path = "football_injuries.csv"  # Make sure the file is in the same folder

df = load_data(dataset_path)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
season_filter = st.sidebar.multiselect(
    "Select Season(s)",
    options=df["season"].unique(),
    default=df["season"].unique()
)

team_filter = st.sidebar.multiselect(
    "Select Team(s)",
    options=df["team"].unique(),
    default=df["team"].unique()
)

df_filtered = df[(df["season"].isin(season_filter)) & (df["team"].isin(team_filter))]

# --- Overview Metrics ---
st.subheader("Team Performance Overview")
total_matches = df_filtered["matches_played"].sum()
total_wins = df_filtered["wins"].sum()
total_draws = df_filtered["draws"].sum()
total_losses = df_filtered["losses"].sum()
total_injuries = df_filtered["injuries"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Matches Played", total_matches)
col2.metric("Wins", total_wins)
col3.metric("Draws", total_draws)
col4.metric("Losses", total_losses)
col5.metric("Total Injuries", total_injuries)

# --- Injuries vs Performance ---
st.subheader("Impact of Injuries on Team Points")
injury_points = df_filtered.groupby("team")[["injuries", "points"]].sum().reset_index()
fig1 = px.scatter(
    injury_points,
    x="injuries",
    y="points",
    size="points",
    color="team",
    hover_data=["team"],
    title="Total Injuries vs Team Points"
)
st.plotly_chart(fig1)

# --- Match Outcome Distribution ---
st.subheader("Match Outcome Distribution")
outcome_df = df_filtered.groupby("team")[["wins", "draws", "losses"]].sum().reset_index()
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=outcome_df["team"], y=outcome_df["wins"], name="Wins", marker_color="green"))
fig2.add_trace(go.Bar(x=outcome_df["team"], y=outcome_df["draws"], name="Draws", marker_color="gray"))
fig2.add_trace(go.Bar(x=outcome_df["team"], y=outcome_df["losses"], name="Losses", marker_color="red"))
fig2.update_layout(barmode='stack', title="Team Match Outcomes")
st.plotly_chart(fig2)

# --- Injuries Heatmap / Treemap ---
st.subheader("Injury Occurrences by Player")
player_injuries = df_filtered.groupby(["team", "player_name"])["injuries"].sum().reset_index()
fig3 = px.treemap(
    player_injuries,
    path=["team", "player_name"],
    values="injuries",
    color="injuries",
    color_continuous_scale="Reds",
    title="Injuries per Player (Treemap)"
)
st.plotly_chart(fig3)

# --- Insights / Recommendations ---
st.subheader("AI Insights & Recommendations")
st.markdown("""
- Teams with higher injury counts tend to accumulate fewer points; rotation and injury prevention are critical.
- Monitor players with frequent injuries and adjust training schedules.
- Consider squad depth when planning match line-ups to minimize performance drops.
""")
