# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# App Title and Description
# ------------------------------
st.set_page_config(page_title="FootLens Player Injury Dashboard", layout="wide")
st.title("FootLens Analytics: Player Injury & Performance Dashboard")
st.markdown("""
**Purpose:** Visualize the impact of injuries on team and player performance.  
**Audience:** Technical directors, coaches, and sports managers.  
""")

# ------------------------------
# Load Dataset DIRECTLY (no upload needed)
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("player_injuries_fixed.csv")

    # Parse match_date
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce', dayfirst=True)
    else:
        df['match_date'] = pd.NaT

    # Ensure numeric columns
    numeric_cols = ['performance_before', 'performance_after', 'age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Compute performance metrics
    df['performance_drop'] = df.get('performance_before', 0) - df.get('performance_after', 0)
    df['rating_improvement'] = df.get('performance_after', 0) - df.get('performance_before', 0)

    # Extract month
    if 'match_date' in df.columns:
        df['month'] = df['match_date'].dt.month
    else:
        df['month'] = np.nan

    return df

df = load_data()

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")

clubs = df['club'].dropna().unique().tolist() if 'club' in df.columns else []
players = df['player_name'].dropna().unique().tolist() if 'player_name' in df.columns else []
seasons = df['season'].dropna().unique().tolist() if 'season' in df.columns else []

selected_club = st.sidebar.multiselect("Select Club(s)", options=clubs, default=clubs)
selected_player = st.sidebar.multiselect("Select Player(s)", options=players, default=players)
selected_season = st.sidebar.multiselect("Select Season(s)", options=seasons, default=seasons)

df_filtered = df.copy()

# Apply filters
if 'club' in df_filtered.columns and selected_club:
    df_filtered = df_filtered[df_filtered['club'].isin(selected_club)]

if 'player_name' in df_filtered.columns and selected_player:
    df_filtered = df_filtered[df_filtered['player_name'].isin(selected_player)]

if 'season' in df_filtered.columns and selected_season:
    df_filtered = df_filtered[df_filtered['season'].isin(selected_season)]

if df_filtered.empty:
    st.info("No data available for the selected filters.")
    st.stop()

# ------------------------------
# Visualization 1: Top 10 Injuries by Performance Drop
# ------------------------------
st.subheader("Top 10 Injuries Causing Highest Team Performance Drop")
if 'injury_type' in df_filtered.columns and 'performance_drop' in df_filtered.columns:
    df_bar = df_filtered.dropna(subset=['injury_type', 'performance_drop'])
    if not df_bar.empty:
        injury_impact = df_bar.groupby('injury_type')['performance_drop'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_bar = px.bar(
            injury_impact,
            x='performance_drop',
            y='injury_type',
            orientation='h',
            color='performance_drop',
            color_continuous_scale='Reds',
            title="Top 10 Injuries Impacting Team Performance"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No valid data for Top Injuries chart.")
else:
    st.info("Columns 'injury_type' or 'performance_drop' missing.")

# ------------------------------
# Visualization 2: Player Performance Timeline
# ------------------------------
st.subheader("Player Performance Timeline (Before & After Injury)")
if 'player_name' in df_filtered.columns:
    player_name = st.selectbox("Select Player to View Timeline", df_filtered['player_name'].unique())
    player_df = df_filtered[df_filtered['player_name'] == player_name].sort_values('match_date')

    fig_line = go.Figure()
    if 'performance_before' in player_df.columns:
        fig_line.add_trace(go.Scatter(
            x=player_df['match_date'], y=player_df['performance_before'],
            mode='lines+markers', name='Performance Before Injury'
        ))
    if 'performance_after' in player_df.columns:
        fig_line.add_trace(go.Scatter(
            x=player_df['match_date'], y=player_df['performance_after'],
            mode='lines+markers', name='Performance After Injury'
        ))

    if fig_line.data:
        fig_line.update_layout(
            title=f"{player_name} Performance Timeline",
            xaxis_title="Match Date",
            yaxis_title="Performance Rating"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No performance data available for this player.")
else:
    st.info("Insufficient data to plot Player Performance Timeline.")

# ------------------------------
# Visualization 3: Injury Frequency Heatmap
# ------------------------------
st.subheader("Injury Frequency Across Months & Clubs")
if 'club' in df_filtered.columns and 'month' in df_filtered.columns:
    df_heat = df_filtered.dropna(subset=['club', 'month'])
    if not df_heat.empty:
        injury_freq = df_heat.groupby(['club', 'month']).size().reset_index(name='injury_count')
        fig_heatmap = px.density_heatmap(
            injury_freq,
            x='month',
            y='club',
            z='injury_count',
            color_continuous_scale='Viridis',
            title="Monthly Injury Frequency by Club"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No valid data for Injury Frequency Heatmap.")
else:
    st.info("Columns 'club' or 'month' missing.")

# ------------------------------
# Visualization 4: Player Age vs Performance Drop
# ------------------------------
st.subheader("Player Age vs Performance Drop")
if 'age' in df_filtered.columns and 'performance_drop' in df_filtered.columns:
    df_scatter = df_filtered.dropna(subset=['age', 'performance_drop'])
    if not df_scatter.empty:
        hover_cols = [col for col in ['player_name', 'injury_type', 'season'] if col in df_scatter.columns]
        fig_scatter = px.scatter(
            df_scatter,
            x='age',
            y='performance_drop',
            color='club' if 'club' in df_scatter.columns else None,
            size='performance_drop',
            hover_data=hover_cols,
            title="Player Age vs Performance Drop"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No valid data to plot Age vs Performance Drop.")
else:
    st.info("Columns 'age' or 'performance_drop' missing. Cannot display scatter plot.")

# ------------------------------
# Visualization 5: Leaderboard of Comeback Players
# ------------------------------
st.subheader("Leaderboard: Comeback Players")
if 'player_name' in df_filtered.columns and 'rating_improvement' in df_filtered.columns:
    df_leaderboard = df_filtered.dropna(subset=['player_name', 'rating_improvement'])
    if not df_leaderboard.empty:
        comeback_leaderboard = df_leaderboard.groupby('player_name')['rating_improvement'].mean().sort_values(ascending=False).head(10).reset_index()
        st.dataframe(comeback_leaderboard.style.background_gradient(subset=['rating_improvement'], cmap='Greens'))
    else:
        st.info("No valid data for Comeback Leaderboard.")
else:
    st.info("Columns 'player_name' or 'rating_improvement' missing.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("💡 **Insight:** Use this dashboard to track injury impacts, guide rotation strategies, and monitor player recovery trends.")
