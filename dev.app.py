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

# --- Data Simulation (Improved for realistic injury dates) ---
np.random.seed(42)
players = [f"Player_{i}" for i in range(1, 21)]
clubs = [f"Club_{i}" for i in range(1, 6)]
dates = pd.date_range("2020-01-01", "2022-12-31", freq="15D")

injury_starts = np.random.choice(dates, 200)
injury_durations_days = np.random.randint(7, 90, 200) # Injuries last from 1 week to ~3 months

data = {
    "Player": np.random.choice(players, 200),
    "Club": np.random.choice(clubs, 200),
    "Rating": np.random.uniform(5, 9, 200),
    "Goals": np.random.randint(0, 5, 200),
    "Team_Goals_Before": np.random.randint(10, 30, 200),
    "Team_Goals_During": np.random.randint(5, 25, 200),
    "Age": np.random.randint(18, 35, 200),
    "Injury_Start": pd.to_datetime(injury_starts),
    "Injury_End": [start + pd.Timedelta(days=duration) for start, duration in zip(injury_starts, injury_durations_days)],
    "Status": np.random.choice(["Before", "During", "After"], 200)
}

df = pd.DataFrame(data)

df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Goals'] = df['Goals'].fillna(0)

# --- NEW: Calculate Injury Duration ---
df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days
# Ensure duration is non-negative
df['Injury_Duration'] = df['Injury_Duration'].apply(lambda x: x if x > 0 else 0)


# Add other derived columns
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
kpi2.metric("💥 Avg Team Performance Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
kpi3.metric("🩹 Total Injuries Recorded", f"{len(filtered_df)}")

# --- TABS: Added "Player Deep Dive" ---
tabs = st.tabs(["📊 Trends", "📈 Player Impact", "🔥 Club Analysis", "🔎 Player Deep Dive"])

# -------- 📊 Trends --------
with tabs[0]:
    st.subheader("Top Players with Highest Team Performance Drop")
    impact = (
        filtered_df.groupby("Player")['Team_Performance_Drop']
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    fig1 = px.bar(impact, x="Team_Performance_Drop", y="Player", orientation="h", color="Team_Performance_Drop",
                  color_continuous_scale="Reds", title="Impact of Player Absence on Team Goals")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Performance Timeline of Sample Players")
    sample_players = filtered_df['Player'].unique()[:5]
    fig2 = px.line(filtered_df[filtered_df['Player'].isin(sample_players)],
                   x="Injury_Start", y="Rating", color="Player", markers=True, title="Rating Fluctuation Around Injuries")
    st.plotly_chart(fig2, use_container_width=True)

# -------- 📈 Player Impact --------
with tabs[1]:
    st.subheader("Comeback Players Leaderboard (Rating Change Post-Injury)")
    leaderboard = (
        filtered_df.groupby('Player')['Performance_Change']
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    st.dataframe(leaderboard, use_container_width=True)

    st.subheader("Player Age vs Team Performance Drop")
    fig3 = px.scatter(filtered_df, x="Age", y="Team_Performance_Drop", color="Club", hover_data=["Player"],
                      title="Correlation between Age and Impact of Absence")
    st.plotly_chart(fig3, use_container_width=True)

# -------- 🔥 Club Analysis --------
with tabs[2]:
    st.subheader("Injury Frequency by Month and Club")
    heatmap_data = filtered_df.groupby(['Club','Month']).size().reset_index(name="Count")
    fig4 = px.density_heatmap(heatmap_data, x="Month", y="Club", z="Count",
                              color_continuous_scale="Blues", title="When Do Injuries Occur During the Season?")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Club-wise Injury Counts")
    club_injuries = filtered_df.groupby("Club")['Injury_Start'].count().reset_index().rename(columns={"Injury_Start":"Injury_Count"})
    fig5 = px.bar(club_injuries, x="Club", y="Injury_Count", color="Injury_Count", color_continuous_scale="Viridis",
                  title="Total Recorded Injuries per Club")
    st.plotly_chart(fig5, use_container_width=True)

    # --- NEW FEATURE ADDED HERE ---
    st.subheader("Which Clubs Suffer the Most from Player Injuries?")
    club_suffering = (
        filtered_df.groupby("Club")['Team_Performance_Drop']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_suffering = px.bar(
        club_suffering,
        x="Club",
        y="Team_Performance_Drop",
        title="Average Team Goal Drop When Players are Injured",
        color="Team_Performance_Drop",
        color_continuous_scale="OrRd",
        labels={"Team_Performance_Drop": "Avg. Goal Drop per Injury"}
    )
    st.plotly_chart(fig_suffering, use_container_width=True)


# -------- NEW FEATURE: 🔎 Player Deep Dive --------
with tabs[3]:
    st.subheader("🔎 Single Player Deep Dive")
    # Use the original df for the selectbox to ensure all players are available for selection
    player_to_analyze = st.selectbox("Select a Player to Analyze", options=sorted(df['Player'].unique()))

    if player_to_analyze:
        # Use the globally filtered_df to respect the sidebar filters
        player_df = filtered_df[filtered_df['Player'] == player_to_analyze].copy()
        st.markdown(f"### Analytics for: **{player_to_analyze}**")

        if not player_df.empty:
            # Player-specific KPIs
            kpi4, kpi5, kpi6 = st.columns(3)
            kpi4.metric("⚽ Average Rating", f"{player_df['Rating'].mean():.2f}")
            kpi5.metric("🩹 Total Injuries", f"{len(player_df)}")
            kpi6.metric("⏳ Avg. Injury Duration (Days)", f"{player_df['Injury_Duration'].mean():.1f}")

            # Player Injury History Table
            st.subheader("Injury History")
            display_cols = {
                'Injury_Start': 'From',
                'Injury_End': 'To',
                'Injury_Duration': 'Duration (Days)',
                'Team_Performance_Drop': 'Team Goal Drop'
            }
            st.dataframe(
                player_df[display_cols.keys()].rename(columns=display_cols).sort_values(by='From', ascending=False),
                use_container_width=True
            )

            # Player Performance Chart
            st.subheader("Performance Timeline")
            fig_player = px.line(player_df.sort_values(by='Injury_Start'),
                                 x="Injury_Start", y="Rating",
                                 title=f"Rating Over Time for {player_to_analyze}",
                                 markers=True, text="Rating")
            fig_player.update_traces(texttemplate='%{text:.2f}', textposition='top center')
            st.plotly_chart(fig_player, use_container_width=True)
        else:
            st.warning(f"No data available for **{player_to_analyze}** with the current sidebar filters applied.")

# --- Download Button for Filtered Data ---
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_injury_impact_data.csv",
    mime="text/csv"
)
