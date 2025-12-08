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

# --- Helper functions -------------------------------------------------
@st.cache_data
def load_sample_data(n_matches=400):
    rng = np.random.default_rng(42)
    seasons = ["2021/22", "2022/23", "2023/24", "2024/25"]
    teams = [f"Team {c}" for c in list("ABCDEFGHIJ")[:10]]

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
            points_home, points_away = 3, 0
        elif home_goals < away_goals:
            points_home, points_away = 0, 3
        else:
            points_home = points_away = 1

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

    # Ensure all required columns exist
    required_cols = ["match_id","match_date","season","home_team","away_team",
                     "home_goals","away_goals","home_points","away_points",
                     "home_injuries","away_injuries","home_minutes_lost","away_minutes_lost"]
    for col in required_cols:
        if col not in df.columns:
            # Fill missing with safe defaults
            if 'goals' in col or 'points' in col or 'injuries' in col or 'minutes' in col:
                df[col] = 0
            elif 'team' in col:
                df[col] = 'Team A'
            elif col == 'match_date':
                df[col] = pd.date_range(start='2021-08-01', periods=len(df), freq='2D')
            elif col == 'season':
                df[col] = '2021/22'

    # Convert match_date to datetime
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

    # Melt to team-level rows
    home = df.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_points": "points",
        "home_injuries": "injuries",
        "home_minutes_lost": "minutes_lost"
    })
    home['venue'] = 'Home'

    away = df.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_points": "points",
        "away_injuries": "injuries",
        "away_minutes_lost": "minutes_lost"
    })
    away['venue'] = 'Away'

    team_level = pd.concat([home, away], ignore_index=True, sort=False)
    team_level['goal_diff'] = team_level['goals_for'] - team_level['goals_against']

    # Rolling injury metrics
    team_level = team_level.sort_values(['team', 'match_date'])
    team_level['injuries_rolling3'] = team_level.groupby('team')['injuries'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    team_level['minutes_lost_rolling3'] = team_level.groupby('team')['minutes_lost'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

    return team_level

# --- Load sample data only -------------------------------------------
st.title("⚽ FootLens — Injuries vs Team Performance Dashboard")
df_raw = load_sample_data(400)
team_level = preprocess_matches(df_raw)
