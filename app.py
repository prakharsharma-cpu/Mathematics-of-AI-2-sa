# footlens_dashboard_debugged.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="FootLens Analytics Dashboard", layout="wide")

st.title("⚽ FootLens Analytics Dashboard")
st.markdown("""
Interactive dashboard combining Plotly, Matplotlib, and Seaborn to visualize the impact of player injuries 
on team performance. Filters allow season, team, and injury type selection for detailed insights.
""")

# ------------------------------
# Upload Dataset
# ------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
    
    # ------------------------------
    # Basic Checks & Preprocessing
    # ------------------------------
    expected_columns = ['player_name','team_name','match_date','player_rating','goals',
                        'injury_type','injury_start','injury_end','team_points_before','team_points_during','player_age']
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"The dataset is missing some expected columns: {missing_cols}. Default values will be used where possible.")
    
    # Rename columns safely
    df.rename(columns={
        'player_name': 'Player',
        'team_name': 'Team',
        'match_date': 'Date'
    }, inplace=True)
    
    # Convert dates safely
    for date_col in ['Date','injury_start','injury_end']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            df[date_col] = pd.NaT
    
    # Fill missing numeric columns
    for col in ['player_rating','goals','team_points_before','team_points_during','player_age']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Fill missing string columns
    for col in ['Player','Team','injury_type']:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown')
    
    # Derived columns
    df['avg_rating_before_injury'] = df.groupby('Player')['player_rating'].transform(lambda x: x.shift(1))
    df['avg_rating_after_injury'] = df.groupby('Player')['player_rating'].transform(lambda x: x.shift(-1))
    df['performance_drop_index'] = df['team_points_before'] - df['team_points_during']
    df['rating_improvement'] = df['avg_rating_after_injury'] - df['avg_rating_before_injury']
    
    # ------------------------------
    # Sidebar Filters
    # ------------------------------
    st.sidebar.header("Filters")
    
    # Season filter
    if 'season' in df.columns:
        seasons = df['season'].dropna().unique()
        if len(seasons) > 0:
            selected_season = st.sidebar.selectbox("Select Season", seasons, index=0)
            df = df[df['season'] == selected_season]
    
    # Team filter
    teams = df['Team'].unique()
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=list(teams))
    df = df[df['Team'].isin(selected_teams)]
    
    # Injury type filter
    injuries = df['injury_type'].unique()
    selected_injuries = st.sidebar.multiselect("Select Injury Types", injuries, default=list(injuries))
    df = df[df['injury_type'].isin(selected_injuries)]
    
    # ------------------------------
    # Dashboard Overview
    # ------------------------------
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.markdown(f"**Number of Matches:** {len(df)}")
    st.markdown(f"**Number of Players:** {df['Player'].nunique()}")
    st.markdown(f"**Number of Teams:** {df['Team'].nunique()}")
    
    # ------------------------------
    # Interactive Visualizations (Plotly + Seaborn)
    # ------------------------------
    
    # Top Injured Players
    st.subheader("Top Injured Players")
    top_injured = df.groupby('Player')['injury_type'].count().sort_values(ascending=False).head(10)
    if not top_injured.empty:
        fig1 = px.bar(
            x=top_injured.values,
            y=top_injured.index,
            orientation='h',
            labels={'x':'Number of Injuries','y':'Player'},
            color=top_injured.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No injury data available for selected filters.")
    
    # Player Performance Timeline
    st.subheader("Player Performance Timeline")
    if 'Player' in df.columns and not df['Player'].empty:
        player_selected = st.selectbox("Select Player", df['Player'].unique())
        player_data = df[df['Player'] == player_selected]
        if not player_data.empty:
            fig2 = px.line(
                player_data, x='Date', y='player_rating', markers=True,
                title=f"{player_selected}'s Rating Timeline"
            )
            if not player_data['injury_start'].isna().all():
                fig2.add_vline(x=player_data['injury_start'].min(), line_dash="dash", line_color="red", annotation_text="Injury Start")
            if not player_data['injury_end'].isna().all():
                fig2.add_vline(x=player_data['injury_end'].max(), line_dash="dash", line_color="green", annotation_text="Injury End")
            st.plotly_chart(fig2, use_container_width=True)
    
    # Injury Heatmap
    st.subheader("Injury Frequency Heatmap")
    if 'Date' in df.columns:
        df['month'] = df['Date'].dt.month.fillna(0).astype(int)
        heatmap_data = df.pivot_table(index='Team', columns='month', values='injury_type', aggfunc='count', fill_value=0)
        if not heatmap_data.empty:
            plt.figure(figsize=(12,6))
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
            plt.xlabel("Month")
            plt.ylabel("Team")
            st.pyplot(plt)
    
    # Age vs Performance Drop Scatter
    st.subheader("Player Age vs Performance Drop")
    if not df.empty:
        fig3 = px.scatter(
            df,
            x='player_age', y='performance_drop_index', color='Team',
            size='performance_drop_index', hover_data=['Player','injury_type'],
            title="Player Age vs Performance Drop Index"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Top Comeback Players Leaderboard
    st.subheader("Top Comeback Players by Rating Improvement")
    comeback_table = df.groupby('Player')['rating_improvement'].mean().sort_values(ascending=False).head(10)
    st.table(comeback_table)
    
    # Top Injuries Impacting Team Performance
    st.subheader("Top Injuries Impacting Team Performance")
    top_injuries = df.groupby('injury_type')['performance_drop_index'].mean().sort_values(ascending=False).head(10)
    if not top_injuries.empty:
        fig4 = px.bar(
            x=top_injuries.values, y=top_injuries.index, orientation='h',
            labels={'x':'Performance Drop Index','y':'Injury Type'},
            color=top_injuries.values, color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
else:
    st.info("Please upload a CSV file to get started.")
