import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Display settings
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# Replace with your actual file path
# Example if in Google Drive: "/content/player_injuries_impact.csv"
# For now, let's create dummy data so you can run and test without errors
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

# Handle missing values
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Goals'] = df['Goals'].fillna(0)
df['Injury_Start'] = pd.to_datetime(df['Injury_Start'], errors='coerce')
df['Injury_End'] = pd.to_datetime(df['Injury_End'], errors='coerce')

# Create new columns
df['Avg_Rating_Before'] = df.groupby('Player_Name')['Rating'].shift(1)
df['Avg_Rating_After'] = df.groupby('Player_Name')['Rating'].shift(-1)

df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']

# Rename columns
df.rename(columns={
    'Player_Name': 'Player',
    'Club_Name': 'Club'
}, inplace=True)

# Summary stats
summary = df.groupby('Player').agg({
    'Rating': 'mean',
    'Goals': 'sum',
    'Team_Performance_Drop': 'mean',
    'Injury_Start': 'count'
}).rename(columns={'Injury_Start': 'Injury_Count'}).reset_index()

top_injured = summary.sort_values("Injury_Count", ascending=False).head(10)
club_injuries = df.groupby("Club")['Injury_Start'].count().reset_index().sort_values("Injury_Start", ascending=False)

df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
improvement = df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10)
decline = df.groupby('Player')['Performance_Change'].mean().sort_values().head(10)

impact = df[['Player', 'Team_Performance_Drop']].sort_values("Team_Performance_Drop", ascending=False).head(5)

pivot_perf = pd.pivot_table(
    df,
    values='Rating',
    index='Player',
    columns='Status',
    aggfunc='mean'
).fillna(0)

# 1. Bar Chart
plt.figure(figsize=(10,6))
sns.barplot(x='Team_Performance_Drop', y='Player', data=impact, palette="Reds_r")
plt.title("Top Injuries with Highest Team Performance Drop")
plt.show()

# 2. Line Chart
plt.figure(figsize=(12,6))
for player in df['Player'].unique()[:5]:
    player_data = df[df['Player'] == player].sort_values('Injury_Start')
    plt.plot(player_data['Injury_Start'], player_data['Rating'], label=player)
plt.legend()
plt.title("Player Performance Timeline (Before & After Injury)")
plt.show()

# 3. Heatmap
df['Month'] = df['Injury_Start'].dt.month
heatmap_data = df.groupby(['Club','Month']).size().unstack(fill_value=0)
plt.figure(figsize=(12,8))
sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt="d")
plt.title("Injury Frequency by Month and Club")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Age'], y=df['Team_Performance_Drop'], hue=df['Club'])
plt.title("Player Age vs. Performance Drop Index")
plt.show()

# 5. Leaderboard Table
leaderboard = df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10).reset_index()
print("🏆 Comeback Players Leaderboard")
print(leaderboard)
