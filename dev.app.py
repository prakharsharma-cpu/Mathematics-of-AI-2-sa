# ========================================================
# ⚽ Player Injury Impact Dashboard (Interactive Streamlit)
# ========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="⚽ Player Injury Impact Dashboard", layout="wide")
st.title("⚽ Player Injury Impact Dashboard")
st.markdown("Analyze how injuries affect player and team performance interactively!")

np.random.seed(42)
players = [f"Player_{i}" for i in range(1, 21)]
clubs = [f"Club_{i}" for i in range(1, 6)]
dates = pd.date_range("2020-01-01", "2022-12-31", freq="15D")

data = {
    "Player": np.random.choice(players, 200),
    "Club": np.random.choice(clubs, 200),
    "Rating": np.random.uniform(5, 9, 200),
    "Goals": np.random.randint(0, 5, 200),
    "Team_Goals_Before": np.random.randint(10, 30, 200),
    "Team_Goals_During": np.random.randint(5, 25, 200),
    "Age": np.random.randint(18, 35, 200),
    "Injury_Start": np.random.choice(dates, 200),
    "Injury_End": np.random.choice(dates, 200),
    "Status": np.random.choice(["Before", "During", "After"], 200)
}

df = pd.DataFrame(data)

df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Goals'] = df['Goals'].fillna(0)
df['Injury_Start'] = pd.to_datetime(df['Injury_Start'], errors='coerce')
df['Injury_End'] = pd.to_datetime(df['Injury_End'], errors='coerce')

# Add derived columns
df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
df['Month'] = df['Injury_Start'].dt.month

st.sidebar.header("🔍 Filters")
filter_club = st.sidebar.multiselect("Club", options=df['Club'].unique(), default=df['Club'].unique())
filter_player = st.sidebar.multiselect("Player", options=df['Player'].unique(), default=df['Player'].unique())
filter_status = st.sidebar.multiselect("Status", options=df['Status'].unique(), default=df['Status'].unique())

filtered_df = df[
    (df['Club'].isin(filter_club)) &
    (df['Player'].isin(filter_player)) &
    (df['Status'].isin(filter_status))
]

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
kpi2.metric("💥 Avg Performance Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
kpi3.metric("🩹 Total Injuries", f"{len(filtered_df)}")

tabs = st.tabs(["📊 Trends", "📈 Player Impact", "🔥 Club Analysis"])

# -------- 📊 Trends --------
with tabs[0]:
    st.subheader("Top Players with Highest Team Performance Drop")
    impact = (
        filtered_df.groupby("Player")['Team_Performance_Drop']
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    fig1 = px.bar(impact, x="Team_Performance_Drop", y="Player", orientation="h", color="Team_Performance_Drop",
                  color_continuous_scale="Reds")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Performance Timeline of Sample Players")
    sample_players = filtered_df['Player'].unique()[:5]
    fig2 = px.line(filtered_df[filtered_df['Player'].isin(sample_players)],
                   x="Injury_Start", y="Rating", color="Player", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# -------- 📈 Player Impact --------
with tabs[1]:
    st.subheader("Comeback Players Leaderboard (Rating Change)")
    leaderboard = (
        filtered_df.groupby('Player')['Performance_Change']
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    st.dataframe(leaderboard, use_container_width=True)

    st.subheader("Player Age vs Performance Drop")
    fig3 = px.scatter(filtered_df, x="Age", y="Team_Performance_Drop", color="Club", hover_data=["Player"])
    st.plotly_chart(fig3, use_container_width=True)

# -------- 🔥 Club Analysis --------
with tabs[2]:
    st.subheader("Injury Frequency by Month and Club")
    heatmap_data = filtered_df.groupby(['Club','Month']).size().reset_index(name="Count")
    fig4 = px.density_heatmap(heatmap_data, x="Month", y="Club", z="Count",
                              color_continuous_scale="Blues")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Club-wise Injury Counts")
    club_injuries = filtered_df.groupby("Club")['Injury_Start'].count().reset_index().rename(columns={"Injury_Start":"Injury_Count"})
    fig5 = px.bar(club_injuries, x="Club", y="Injury_Count", color="Injury_Count", color_continuous_scale="Viridis")
    st.plotly_chart(fig5, use_container_width=True)
    
st.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="injury_impact_filtered.csv",
    mime="text/csv"
)
