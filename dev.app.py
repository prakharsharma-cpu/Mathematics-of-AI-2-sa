# ==================================
# ⚽ Player Injury Impact Dashboard 
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="⚽ Player Injury Impact Dashboard", layout="wide")
st.title("⚽ Player Injury Impact Dashboard")
st.markdown("Analyze how injuries affect player and team performance interactively!")

# --- Data Simulation (Improved with Injury Type) ---
np.random.seed(42)
players = [f"Player_{i}" for i in range(1, 21)]
clubs = [f"Club_{i}" for i in range(1, 6)]
dates = pd.date_range("2020-01-01", "2022-12-31", freq="15D")
injury_types = ["Hamstring", "Groin", "ACL", "Ankle", "Calf", "Back"]

injury_starts = np.random.choice(dates, 200)
injury_durations_days = np.random.randint(7, 90, 200)  # 1 week to ~3 months

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
    "Status": np.random.choice(["Before", "During", "After"], 200),
    "Injury_Type": np.random.choice(injury_types, 200)
}

df = pd.DataFrame(data)

# --- Data Cleaning and Derived Metrics ---
df.drop_duplicates(inplace=True)
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Goals'] = df['Goals'].fillna(0)
df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days
df['Injury_Duration'] = df['Injury_Duration'].apply(lambda x: x if x > 0 else 0)

df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
df['Month'] = df['Injury_Start'].dt.month
df['Impact_Index'] = df['Team_Performance_Drop'] / df['Injury_Duration'].replace(0, np.nan)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
filter_club = st.sidebar.multiselect("Club", options=df['Club'].unique(), default=df['Club'].unique())
filter_player = st.sidebar.multiselect("Player", options=df['Player'].unique(), default=df['Player'].unique())
filter_injury = st.sidebar.multiselect("Injury Type", options=df['Injury_Type'].unique(), default=df['Injury_Type'].unique())

filtered_df = df[
    (df['Club'].isin(filter_club)) &
    (df['Player'].isin(filter_player)) &
    (df['Injury_Type'].isin(filter_injury))
]

# --- KPIs ---
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
kpi2.metric("💥 Avg Team Performance Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
kpi3.metric("🩹 Total Injuries Recorded", f"{len(filtered_df)}")

# --- Tabs ---
tabs = st.tabs([
    "🧾 Dataset Overview", 
    "📊 Trends", 
    "📈 Player Impact", 
    "🔥 Club Analysis", 
    "🔬 Injury Analysis", 
    "🔎 Player Deep Dive"
])

# -------- 🧾 Dataset Overview --------
with tabs[0]:
    st.subheader("📄 Dataset Preview and Summary")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"**Total Records:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    with st.expander("📈 Statistical Summary and Correlation"):
        st.dataframe(df[['Age', 'Rating', 'Team_Performance_Drop', 'Injury_Duration']].describe())
        st.write("Correlation Heatmap:")
        corr = df[['Age', 'Rating', 'Team_Performance_Drop', 'Injury_Duration']].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix',
            labels=dict(x="Metric", y="Metric", color="Correlation")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# -------- 📊 Trends --------
with tabs[1]:
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
                   x="Injury_Start", y="Rating", color="Player", markers=True, 
                   title="Rating Fluctuation Around Injuries")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📆 Monthly Injury Trend")
    monthly_trend = df.groupby('Month').size().reset_index(name='Injury_Count')
    fig_trend = px.line(monthly_trend, x='Month', y='Injury_Count', markers=True,
                        title='Injury Frequency Over the Season')
    st.plotly_chart(fig_trend, use_container_width=True)

# -------- 📈 Player Impact --------
with tabs[2]:
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

    st.subheader("📈 Average Rating Before, During, and After Injury")
    status_avg = filtered_df.groupby('Status')['Rating'].mean().reset_index()
    fig_status = px.bar(status_avg, x='Status', y='Rating',
                        color='Status', title='Average Rating by Injury Phase',
                        color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_status, use_container_width=True)

# -------- 🔥 Club Analysis --------
with tabs[3]:
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

    st.subheader("⚽ Team Goals Before vs During Injury Period")
    team_goals = df.groupby('Club')[['Team_Goals_Before', 'Team_Goals_During']].mean().reset_index()
    fig_goals = px.bar(team_goals, x='Club', y=['Team_Goals_Before', 'Team_Goals_During'],
                       barmode='group', title='Average Team Goals Before and During Injury Period')
    st.plotly_chart(fig_goals, use_container_width=True)

    st.subheader("🏆 Top 5 Clubs by Average Performance Drop")
    top_clubs = (
        filtered_df.groupby('Club')['Team_Performance_Drop']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    st.dataframe(top_clubs, use_container_width=True)

# -------- 🔬 Injury Analysis --------
with tabs[4]:
    st.subheader("Analysis by Injury Type")
    injury_counts = filtered_df['Injury_Type'].value_counts().reset_index()
    injury_counts.columns = ['Injury_Type', 'Count']
    fig_injury_counts = px.bar(
        injury_counts.sort_values('Count', ascending=False),
        x='Injury_Type', y='Count', color='Count',
        color_continuous_scale='Plasma', title='Most Common Injury Types'
    )
    st.plotly_chart(fig_injury_counts, use_container_width=True)
    
    fig_injury_impact = px.box(
        filtered_df,
        x='Injury_Type', y='Team_Performance_Drop', color='Injury_Type',
        title='Team Performance Drop by Injury Type',
        labels={'Team_Performance_Drop': 'Team Goal Drop'}
    )
    st.plotly_chart(fig_injury_impact, use_container_width=True)

    st.subheader("📊 Injury Impact Index by Injury Type")
    impact_df = df.groupby('Injury_Type')['Impact_Index'].mean().reset_index()
    fig_impact = px.bar(impact_df, x='Injury_Type', y='Impact_Index',
                        color='Impact_Index', color_continuous_scale='Inferno',
                        title='Average Impact Index (Performance Drop ÷ Injury Duration)')
    st.plotly_chart(fig_impact, use_container_width=True)

# -------- 🔎 Player Deep Dive --------
with tabs[5]:
    st.subheader("🔎 Single Player Deep Dive")
    player_to_analyze = st.selectbox("Select a Player to Analyze", options=sorted(df['Player'].unique()))

    if player_to_analyze:
        player_df = filtered_df[filtered_df['Player'] == player_to_analyze].copy()
        st.markdown(f"### Analytics for: **{player_to_analyze}**")

        if not player_df.empty:
            kpi4, kpi5, kpi6 = st.columns(3)
            kpi4.metric("⚽ Average Rating", f"{player_df['Rating'].mean():.2f}")
            kpi5.metric("🩹 Total Injuries", f"{len(player_df)}")
            kpi6.metric("⏳ Avg. Injury Duration (Days)", f"{player_df['Injury_Duration'].mean():.1f}")

            st.subheader("Injury History")
            display_cols = {
                'Injury_Start': 'From',
                'Injury_End': 'To',
                'Injury_Type': 'Injury',
                'Injury_Duration': 'Duration (Days)',
                'Team_Performance_Drop': 'Team Goal Drop'
            }
            st.dataframe(
                player_df[display_cols.keys()].rename(columns=display_cols).sort_values(by='From', ascending=False),
                use_container_width=True
            )

            st.subheader("Performance Timeline")
            fig_player = px.line(player_df.sort_values(by='Injury_Start'),
                                 x="Injury_Start", y="Rating",
                                 title=f"Rating Over Time for {player_to_analyze}",
                                 markers=True, text="Rating")
            fig_player.update_traces(texttemplate='%{text:.2f}', textposition='top center')
            st.plotly_chart(fig_player, use_container_width=True)
        else:
            st.warning(f"No data available for **{player_to_analyze}** with the current sidebar filters applied.")

# --- About Section ---
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    This dashboard helps sports analysts and club managers visualize 
    how player injuries affect team goals, player performance, and overall club statistics.
    Explore various tabs for trends, comparisons, and detailed analysis by player or injury type.
    Use sidebar filters to narrow down results and download the dataset.
    """)

# --- Download Button ---
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_injury_impact_data.csv",
    mime="text/csv"
)

st.markdown("<hr><center>© 2025 FootLens Analytics | Developed by Parkar</center>", unsafe_allow_html=True)
