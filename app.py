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
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    st.warning("Please upload a CSV file to use the dashboard.")
    st.stop()

# ------------------------------
# Load & Preprocess Data
# ------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Parse match_date if exists
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce', dayfirst=True)
    else:
        st.warning("Warning: 'match_date' column not found in CSV. Some visualizations may not work.")
        df['match_date'] = pd.NaT
    
    # Ensure numeric columns exist
    numeric_cols = ['performance_before', 'performance_after', 'age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    
    # Compute performance drop and rating improvement
    df['performance_drop'] = df.get('performance_before', 0) - df.get('performance_after', 0)
    df['rating_improvement'] = df.get('performance_after', 0) - df.get('performance_before', 0)
    
    # Add month column for heatmap
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
clubs = df['club'].dropna().unique() if 'club' in df.columns else []
players = df['player_name'].dropna().unique() if 'player_name' in df.columns else []
seasons = df['season'].dropna().unique() if 'season' in df.columns else []

selected_club = st.sidebar.multiselect("Select Club(s)", options=clubs, default=clubs)
selected_player = st.sidebar.multiselect("Select Player(s)", options=players, default=players)
selected_season = st.sidebar.multiselect("Select Season(s)", options=seasons, default=seasons)

df_filtered = df.copy()
if clubs: df_filtered = df_filtered[df_filtered['club'].isin(selected_club)]
if players: df_filtered = df_filtered[df_filtered['player_name'].isin(selected_player)]
if seasons: df_filtered = df_filtered[df_filtered['season'].isin(selected_season)]

if df_filtered.empty:
    st.info("No data available for the selected filters.")
    st.stop()

# ------------------------------
# Visualization 1: Top 10 Injuries by Performance Drop
# ------------------------------
st.subheader("Top 10 Injuries Causing Highest Team Performance Drop")
if 'injury_type' in df_filtered.columns:
    injury_impact = df_filtered.groupby('injury_type')['performance_drop'].mean().sort_values(ascending=False).head(10).reset_index()
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
    st.info("Column 'injury_type' missing. Cannot display Top Injuries chart.")

# ------------------------------
# Visualization 2: Player Performance Timeline
# ------------------------------
st.subheader("Player Performance Timeline (Before & After Injury)")
if 'player_name' in df_filtered.columns and ('performance_before' in df_filtered.columns or 'performance_after' in df_filtered.columns):
    player_name = st.selectbox("Select Player to View Timeline", df_filtered['player_name'].unique())
    player_df = df_filtered[df_filtered['player_name'] == player_name].sort_values('match_date')
    
    fig_line = go.Figure()
    if 'performance_before' in player_df.columns:
        fig_line.add_trace(go.Scatter(x=player_df['match_date'], y=player_df['performance_before'],
                                      mode='lines+markers', name='Performance Before Injury'))
    if 'performance_after' in player_df.columns:
        fig_line.add_trace(go.Scatter(x=player_df['match_date'], y=player_df['performance_after'],
                                      mode='lines+markers', name='Performance After Injury'))
    fig_line.update_layout(title=f"{player_name} Performance Timeline",
                           xaxis_title="Match Date", yaxis_title="Performance Rating")
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Insufficient data to plot Player Performance Timeline.")

# ------------------------------
# Visualization 3: Injury Frequency Heatmap
# ------------------------------
st.subheader("Injury Frequency Across Months & Clubs")
if 'club' in df_filtered.columns and 'month' in df_filtered.columns:
    injury_freq = df_filtered.groupby(['club', 'month']).size().reset_index(name='injury_count')
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
    st.info("Insufficient data to display Injury Frequency Heatmap.")

# ------------------------------
# Visualization 4: Player Age vs Performance Drop
# ------------------------------
st.subheader("Player Age vs Performance Drop")
if 'age' in df_filtered.columns and 'performance_drop' in df_filtered.columns:
    fig_scatter = px.scatter(
        df_filtered,
        x='age',
        y='performance_drop',
        color='club' if 'club' in df_filtered.columns else None,
        size='performance_drop',
        hover_data=['player_name', 'injury_type', 'season'],
        title="Player Age vs Performance Drop"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Insufficient data to display Age vs Performance Drop.")

# ------------------------------
# Visualization 5: Leaderboard of Comeback Players
# ------------------------------
st.subheader("Leaderboard: Comeback Players")
if 'player_name' in df_filtered.columns and 'rating_improvement' in df_filtered.columns:
    comeback_leaderboard = df_filtered.groupby('player_name')['rating_improvement'].mean().sort_values(ascending=False).head(10).reset_index()
    st.dataframe(comeback_leaderboard.style.background_gradient(subset=['rating_improvement'], cmap='Greens'))
else:
    st.info("Insufficient data to display Comeback Leaderboard.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("💡 **Insight:** Use this dashboard to track injury impacts, guide rotation strategies, and monitor player recovery trends.")
