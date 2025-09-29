# streamlit_injury_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Player Injury Analysis Dashboard", layout="wide")
sns.set(style="whitegrid")

st.title("⚽ Player Injury Impact Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("Using sample dummy data.")
    np.random.seed(42)
    players = [f"Player_{i}" for i in range(1, 21)]
    clubs = [f"Club_{i}" for i in range(1, 6)]
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="15D")

    data = {
        "Player_Name": np.random.choice(players, 200),
        "Club_Name": np.random.choice(clubs, 200),
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

try:
    df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
    df['Goals'] = df['Goals'].fillna(0)
    df['Injury_Start'] = pd.to_datetime(df['Injury_Start'], errors='coerce')
    df['Injury_End'] = pd.to_datetime(df['Injury_End'], errors='coerce')

    df['Avg_Rating_Before'] = df.groupby('Player_Name')['Rating'].shift(1)
    df['Avg_Rating_After'] = df.groupby('Player_Name')['Rating'].shift(-1)
    df['Avg_Rating_Before'].fillna(df['Rating'], inplace=True)
    df['Avg_Rating_After'].fillna(df['Rating'], inplace=True)

    df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
    df.rename(columns={'Player_Name': 'Player', 'Club_Name': 'Club'}, inplace=True)
    df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

st.sidebar.header("Filters")
players_filter = st.sidebar.multiselect("Select Player(s)", options=df['Player'].unique(), default=df['Player'].unique()[:5])
clubs_filter = st.sidebar.multiselect("Select Club(s)", options=df['Club'].unique(), default=df['Club'].unique())
status_filter = st.sidebar.multiselect("Select Status", options=df['Status'].unique(), default=df['Status'].unique())
date_range = st.sidebar.date_input("Injury Start Date Range", [df['Injury_Start'].min(), df['Injury_Start'].max()])

filtered_df = df[
    (df['Player'].isin(players_filter)) &
    (df['Club'].isin(clubs_filter)) &
    (df['Status'].isin(status_filter)) &
    (df['Injury_Start'].dt.date >= date_range[0]) &
    (df['Injury_Start'].dt.date <= date_range[1])
]

st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(10))

# 4.1 Top Injuries with Highest Team Performance Drop
st.subheader("Top Injuries by Team Performance Drop")
impact = filtered_df[['Player', 'Team_Performance_Drop']].sort_values("Team_Performance_Drop", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Team_Performance_Drop', y='Player', data=impact, palette="Reds_r", ax=ax)
ax.set_title("Top Injuries with Highest Team Performance Drop")
st.pyplot(fig)

# 4.2 Player Performance Timeline
st.subheader("Player Performance Timeline")
fig, ax = plt.subplots(figsize=(12,6))
for player in filtered_df['Player'].unique():
    player_data = filtered_df[filtered_df['Player'] == player].sort_values('Injury_Start')
    ax.plot(player_data['Injury_Start'], player_data['Rating'], marker='o', label=player)
ax.set_ylabel("Rating")
ax.set_xlabel("Injury Start")
ax.legend()
st.pyplot(fig)

# 4.3 Injury Frequency Heatmap
st.subheader("Injury Frequency by Month and Club")
filtered_df['Month'] = filtered_df['Injury_Start'].dt.month
heatmap_data = filtered_df.groupby(['Club','Month']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt="d", linewidths=0.5, linecolor='gray', ax=ax)
ax.set_ylabel("Club")
ax.set_xlabel("Month")
st.pyplot(fig)

# 4.4 Scatter Plot: Age vs Performance Drop
st.subheader("Player Age vs Performance Drop")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=filtered_df['Age'], y=filtered_df['Team_Performance_Drop'], hue=filtered_df['Club'], ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Team Performance Drop")
st.pyplot(fig)

# 4.5 Leaderboard: Top Comeback Players
st.subheader("🏆 Comeback Players Leaderboard")
leaderboard = filtered_df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10).reset_index()
st.table(leaderboard)
