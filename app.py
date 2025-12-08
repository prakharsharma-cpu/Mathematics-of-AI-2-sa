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
# Upload CSV Dataset
# ------------------------------
st.sidebar.header("Upload CSV (if required)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# ------------------------------
# Load & Preprocess Data (SAFE FIX)
# ------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    # Try local file first
    try:
        df = pd.read_csv("player_injuries_fixed.csv")
        st.success("Loaded local dataset: player_injuries_fixed.csv")
    except FileNotFoundError:
        # Fall back to uploaded file
        if uploaded_file is None:
            st.error("""
            ⚠️ No dataset found.

            The file **player_injuries_fixed.csv** is missing on the server.
            Please upload your CSV file using the left sidebar.
            """)
            st.stop()
        else:
            df = pd.read_csv(uploaded_file)
            st.success("Loaded uploaded dataset.")

    # Parse date column
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce', dayfirst=True)
    else:
        df['match_date'] = pd.NaT

    # Numeric columns
    for col in ['performance_before', 'performance_after', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Calculated metrics
    df['performance_drop'] = df.get('performance_before', 0) - df.get('performance_after', 0)
    df['rating_improvement'] = df.get('performance_after', 0) - df.get('performance_before', 0)

    # Month for heatmap
    if 'match_date' in df.columns:
        df['month'] = df['match_date'].dt.month
    else:
        df['month'] = np.nan

    return df

df = load_data(uploaded_file)

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")

clubs = df['club'].dropna().unique().tolist() if 'club' in df.columns else []
players = df['player_name'].dropna().unique().tolist() if 'player_name' in df.columns else []
seasons = df['season'].dropna().unique().tolist() if 'season' in df.columns else []

selected_club = st.sidebar.multiselect("Select Club(s)", clubs, default=clubs)
selected_player = st.sidebar.multiselect("Select Player(s)", players, default=players)
selected_season = st.sidebar.multiselect("Select Season(s)", seasons, default=seasons)

df_filtered = df.copy()

if 'club' in df.columns and selected_club:
    df_filtered = df_filtered[df_filtered['club'].isin(selected_club)]
if 'player_name' in df.columns and selected_player:
    df_filtered = df_filtered[df_filtered['player_name'].isin(selected_player)]
if 'season' in df.columns and selected_season:
    df_filtered = df_filtered[df_filtered['season'].isin(selected_season)]

if df_filtered.empty:
    st.info("No data available for the selected filters.")
    st.stop()

# ------------------------------
# Visualization 1: Injury Performance Drop
# ------------------------------
st.subheader("Top 10 Injuries Causing Highest Team Performance Drop")

if 'injury_type' in df.columns and 'performance_drop' in df.columns:
    df_bar = df_filtered.dropna(subset=['injury_type', 'performance_drop'])
    if not df_bar.empty:
        injury_impact = df_bar.groupby('injury_type')['performance_drop'] \
                              .mean().sort_values(ascending=False).head(10).reset_index()

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
        st.info("No valid data to display this chart.")

# ------------------------------
# Visualization 2: Player Performance Timeline
# ------------------------------
st.subheader("Player Performance Timeline (Before & After Injury)")

if 'player_name' in df.columns:
    player_name = st.selectbox("Select Player", df_filtered['player_name'].unique())
    player_df = df_filtered[df_filtered['player_name'] == player_name].sort_values('match_date')

    fig_line = go.Figure()
    if 'performance_before' in df.columns:
        fig_line.add_trace(go.Scatter(
            x=player_df['match_date'],
            y=player_df['performance_before'],
            mode='lines+markers',
            name="Performance Before Injury"
        ))

    if 'performance_after' in df.columns:
        fig_line.add_trace(go.Scatter(
            x=player_df['match_date'],
            y=player_df['performance_after'],
            mode='lines+markers',
            name="Performance After Injury"
        ))

    st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------
# Visualization 3: Injury Frequency Heatmap
# ------------------------------
st.subheader("Injury Frequency Across Months & Clubs")

if 'club' in df.columns and 'month' in df.columns:
    df_heat = df_filtered.dropna(subset=['club', 'month'])
    if not df_heat.empty:
        heat = df_heat.groupby(['club', 'month']).size().reset_index(name='injury_count')

        fig_heat = px.density_heatmap(
            heat,
            x='month',
            y='club',
            z='injury_count',
            color_continuous_scale='Viridis',
            title="Monthly Injury Frequency by Club"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------
# Visualization 4: Age vs Performance Drop
# ------------------------------
st.subheader("Player Age vs Performance Drop")

if 'age' in df.columns:
    df_scatter = df_filtered.dropna(subset=['age', 'performance_drop'])
    if not df_scatter.empty:
        hover_cols = [c for c in ['player_name','injury_type','season'] if c in df_scatter.columns]

        fig_scatter = px.scatter(
            df_scatter,
            x="age",
            y="performance_drop",
            color="club" if 'club' in df_scatter.columns else None,
            size="performance_drop",
            hover_data=hover_cols,
            title="Player Age vs Performance Drop"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------
# Visualization 5: Comeback Leaderboard
# ------------------------------
st.subheader("Leaderboard: Comeback Players")

if 'rating_improvement' in df.columns:
    come = df_filtered.groupby('player_name')['rating_improvement'] \
                      .mean().sort_values(ascending=False).head(10).reset_index()

    st.dataframe(
        come.style.background_gradient(subset=['rating_improvement'], cmap='Greens')
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("💡 **Insight:** Use this dashboard to track injury impacts, guide rotation strategies, and monitor player recovery trends.")
