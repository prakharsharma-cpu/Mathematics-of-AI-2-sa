# footlens_full_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import requests

st.set_page_config(page_title="FootLens AI Manager Dashboard", layout="wide")
st.title("FootLens AI Full Manager Dashboard")
st.markdown("""
Comprehensive dashboard for injury analytics, fatigue tracking, training load, lineup optimization, and real-time alerts.
""")

# --- Sidebar: Upload / Live Data ---
st.sidebar.header("Upload / Live Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (internal dataset)", type=["csv"])
api_key = st.sidebar.text_input("Football API Key (optional for live data)", type="password")
competition_code = st.sidebar.text_input("Competition Code (optional, e.g., PL, CL)")

# --- Load internal dataset ---
if uploaded_file:
    df_internal = pd.read_csv(uploaded_file)
    df_internal.fillna(0, inplace=True)
else:
    st.warning("Please upload an internal dataset CSV to proceed")
    st.stop()

# --- Filters ---
st.sidebar.header("Filters")
seasons = st.sidebar.multiselect("Select Season(s)", options=df_internal['Season'].unique(), default=df_internal['Season'].unique())
teams = st.sidebar.multiselect("Select Team(s)", options=df_internal['Team'].unique(), default=df_internal['Team'].unique())
df_filtered = df_internal[(df_internal['Season'].isin(seasons)) & (df_internal['Team'].isin(teams))]

st.subheader("Dataset Overview")
st.dataframe(df_filtered.head())

# --- Key Metrics ---
st.subheader("Team Metrics")
total_injuries = df_filtered['Injury_Count'].sum()
avg_points = df_filtered['Points'].mean()
avg_injuries = df_filtered.groupby('Team')['Injury_Count'].mean().mean()
col1, col2, col3 = st.columns(3)
col1.metric("Total Injuries", total_injuries)
col2.metric("Average Team Points", round(avg_points,2))
col3.metric("Average Injuries per Team", round(avg_injuries,2))

# --- Injury Distribution ---
st.subheader("Injury Distribution by Team")
injury_team = df_filtered.groupby('Team')['Injury_Count'].sum().reset_index()
fig1 = px.bar(injury_team, x='Team', y='Injury_Count', color='Injury_Count', color_continuous_scale='Reds')
st.plotly_chart(fig1, use_container_width=True)

# --- Player-Level Trends ---
st.subheader("Player-Level Injury Trends")
player = st.selectbox("Select Player", options=df_filtered['Player'].unique())
player_df = df_filtered[df_filtered['Player']==player]
fig2 = px.line(player_df, x='Season', y='Injury_Count', markers=True, title=f"Injury Trend for {player}")
st.plotly_chart(fig2, use_container_width=True)

# --- Injury Prediction ---
st.subheader("Injury Risk Prediction")
features = ['Minutes_Played','Goals_Scored','Assists','Injury_Count_Last_Season','Matches_Played']
if all(f in df_filtered.columns for f in features):
    X = df_filtered[features]
    y = df_filtered['Injury_Count']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df_filtered['Predicted_Injuries'] = model.predict(X)
    st.write(df_filtered[['Player','Team','Predicted_Injuries']].sort_values(by='Predicted_Injuries', ascending=False).head(10))
else:
    st.warning("Missing required columns for injury prediction.")

# --- Fatigue & Training Load ---
st.subheader("Fatigue & Training Load")
df_filtered['Fatigue'] = df_filtered['Minutes_Played']/90
df_filtered['Recommended_Training_Load'] = np.where(df_filtered['Predicted_Injuries']>1.5, 0.7, 1.0)
st.dataframe(df_filtered[['Player','Team','Predicted_Injuries','Fatigue','Recommended_Training_Load']])

# --- Dynamic Lineup Optimizer ---
st.subheader("Optimal Lineup")
selected_team = st.selectbox("Select Team to Optimize", options=teams)
team_df = df_filtered[df_filtered['Team']==selected_team].sort_values(by=['Predicted_Injuries','Fatigue'])
lineup = team_df.head(11)[['Player','Predicted_Injuries','Fatigue','Minutes_Played']]
st.dataframe(lineup)

# --- Scenario Simulation ---
st.subheader("Scenario Simulation")
rest_players = st.multiselect("Select players to rest", options=team_df['Player'])
if st.button("Simulate Team Performance with Rested Players"):
    temp_df = team_df.copy()
    temp_df['Simulated_Points'] = temp_df['Points']
    temp_df.loc[temp_df['Player'].isin(rest_players), 'Simulated_Points'] *= 0.9
    st.bar_chart(temp_df.groupby('Player')['Simulated_Points'].sum())

# --- Live Data Integration (optional) ---
if api_key and competition_code:
    st.subheader("Live Data Integration (Optional)")
    headers = {"X-Auth-Token": api_key}
    try:
        url = f"https://api.football-data.org/v4/matches?competitions={competition_code}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        live_matches = response.json().get("matches", [])
        st.write(f"Live matches fetched: {len(live_matches)}")
        st.json(live_matches[:5])  # show top 5 for preview
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")

# --- Alerts ---
st.subheader("High-Risk Player Alerts")
alerts = df_filtered[(df_filtered['Predicted_Injuries']>1.5) | (df_filtered['Fatigue']>0.9)]
st.dataframe(alerts[['Player','Team','Predicted_Injuries','Fatigue']])

# --- Visualizations ---
st.subheader("Fatigue vs Predicted Injury Risk")
fig3 = px.scatter(df_filtered, x='Fatigue', y='Predicted_Injuries', color='Team', hover_data=['Player'])
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Correlation Matrix")
if all(c in df_filtered.columns for c in ['Injury_Count','Points','Goals_Scored','Goals_Conceded']):
    corr = df_filtered[['Injury_Count','Points','Goals_Scored','Goals_Conceded']].corr()
    fig4 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
    st.plotly_chart(fig4, use_container_width=True)
