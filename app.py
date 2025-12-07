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

📌 **Note:** Upload a CSV dataset containing player injury and performance data to unlock the full capabilities of this dashboard.
""")

# ------------------------------
# CSV Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['match_date'], dayfirst=True)
    
    # ------------------------------
    # Data Preprocessing
    # ------------------------------
    required_columns = ['player_name', 'club', 'season', 'match_date', 'injury_type', 'performance_before', 'performance_after', 'age']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Uploaded CSV is missing these required columns: {missing_cols}")
    else:
        st.success("Dataset loaded successfully! Preprocessing data...")

        # Ensure numeric columns
        df['performance_before'] = pd.to_numeric(df['performance_before'], errors='coerce')
        df['performance_after'] = pd.to_numeric(df['performance_after'], errors='coerce')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

        # Remove rows with missing critical data
        df.dropna(subset=['performance_before', 'performance_after', 'age', 'match_date'], inplace=True)

        # Compute performance drop and rating improvement
        df['performance_drop'] = df['performance_before'] - df['performance_after']
        df['rating_improvement'] = df['performance_after'] - df['performance_before']
        df['month'] = df['match_date'].dt.month

        # ------------------------------
        # Sidebar Filters
        # ------------------------------
        st.sidebar.header("Filters")
        clubs = df['club'].unique()
        players = df['player_name'].unique()
        seasons = df['season'].unique()

        selected_club = st.sidebar.multiselect("Select Club(s)", options=clubs, default=clubs)
        selected_player = st.sidebar.multiselect("Select Player(s)", options=players, default=players)
        selected_season = st.sidebar.multiselect("Select Season(s)", options=seasons, default=seasons)

        df_filtered = df[
            (df['club'].isin(selected_club)) &
            (df['player_name'].isin(selected_player)) &
            (df['season'].isin(selected_season))
        ]

        if df_filtered.empty:
            st.warning("No data available for selected filters.")
        else:
            # ------------------------------
            # Visualization 1: Top 10 Injuries by Team Performance Drop
            # ------------------------------
            st.subheader("Top 10 Injuries Causing Highest Team Performance Drop")
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

            # ------------------------------
            # Visualization 2: Player Performance Timeline
            # ------------------------------
            st.subheader("Player Performance Timeline (Before & After Injury)")
            player_name = st.selectbox("Select Player to View Timeline", df_filtered['player_name'].unique())
            player_df = df_filtered[df_filtered['player_name'] == player_name].sort_values('match_date')

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=player_df['match_date'], y=player_df['performance_before'],
                                          mode='lines+markers', name='Performance Before Injury'))
            fig_line.add_trace(go.Scatter(x=player_df['match_date'], y=player_df['performance_after'],
                                          mode='lines+markers', name='Performance After Injury'))
            fig_line.update_layout(title=f"{player_name} Performance Timeline", xaxis_title="Match Date", yaxis_title="Performance Rating")
            st.plotly_chart(fig_line, use_container_width=True)

            # ------------------------------
            # Visualization 3: Injury Frequency Heatmap
            # ------------------------------
            st.subheader("Injury Frequency Across Months & Clubs")
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

            # ------------------------------
            # Visualization 4: Player Age vs Performance Drop
            # ------------------------------
            st.subheader("Player Age vs Performance Drop")
            fig_scatter = px.scatter(
                df_filtered,
                x='age',
                y='performance_drop',
                color='club',
                size='performance_drop',
                hover_data=['player_name', 'injury_type', 'season'],
                title="Player Age vs Performance Drop"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # ------------------------------
            # Visualization 5: Leaderboard of Comeback Players
            # ------------------------------
            st.subheader("Leaderboard: Comeback Players")
            comeback_leaderboard = df_filtered.groupby('player_name')['rating_improvement'].mean().sort_values(ascending=False).head(10).reset_index()
            st.dataframe(comeback_leaderboard.style.background_gradient(subset=['rating_improvement'], cmap='Greens'))

            # ------------------------------
            # Footer
            # ------------------------------
            st.markdown("---")
            st.markdown("💡 **Insight:** Upload your dataset to explore injury impacts, guide rotation strategies, and monitor player recovery trends in real time.")

else:
    st.info("Please upload a CSV file to unlock the dashboard.")
