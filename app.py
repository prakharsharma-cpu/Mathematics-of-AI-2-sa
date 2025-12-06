# ==================================
# ⚽ Player Injury Impact Dashboard (CSV Powered)
# Dual-mode: Plotly + Matplotlib/Seaborn with style selection
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="⚽ Player Injury Impact Dashboard", layout="wide")
st.title("⚽ Player Injury Impact Dashboard")
st.markdown("Interactive dashboard with Plotly + Matplotlib/Seaborn. **Upload your CSV** or explore the demo data.")

# ---------------------
# Helper utilities
# ---------------------
def get_mode(per_tab_choice, global_mode):
    return per_tab_choice if per_tab_choice in ["Plotly", "Matplotlib"] else global_mode

def get_style(per_tab_style, global_style):
    """Return effective seaborn style name."""
    return per_tab_style if per_tab_style in ["Modern Clean", "Classic Analytics"] else global_style

def apply_seaborn_style(style_name):
    """Apply seaborn theme based on friendly name."""
    if style_name == "Classic Analytics":
        sns.set_theme(style="darkgrid", palette="muted")
        plt.rcParams.update({"figure.facecolor":"white"})
    else:  # Modern Clean
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({"figure.facecolor":"white"})

def render_plotly(fig):
    st.plotly_chart(fig, use_container_width=True)

def render_matplotlib(fig):
    st.pyplot(fig)

def generate_sample_data():
    """Generates the original random simulation data if no CSV is uploaded."""
    np.random.seed(42)
    players = [f"Player_{i}" for i in range(1, 21)]
    clubs = [f"Club_{i}" for i in range(1, 6)]
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="15D")
    injury_types = ["Hamstring", "Groin", "ACL", "Ankle", "Calf", "Back"]

    injury_starts = np.random.choice(dates, 200)
    injury_durations_days = np.random.randint(7, 90, 200)

    data = {
        "Player": np.random.choice(players, 200),
        "Club": np.random.choice(clubs, 200),
        "Rating": np.random.uniform(5, 9, 200),
        "Goals": np.random.randint(0, 5, 200),
        "Team_Goals_Before": np.random.randint(10, 30, 200),
        "Team_Goals_During": np.random.randint(5, 25, 200),
        "Team_Goals_After": np.random.randint(10, 30, 200), # Included in base for consistency
        "Age": np.random.randint(18, 35, 200),
        "Injury_Start": pd.to_datetime(injury_starts),
        "Injury_End": [start + pd.Timedelta(days=duration) for start, duration in zip(injury_starts, injury_durations_days)],
        "Status": np.random.choice(["Before", "During", "After"], 200),
        "Injury_Type": np.random.choice(injury_types, 200)
    }
    return pd.DataFrame(data)

# ---------------------
# Data Loading & Sidebar
# ---------------------
st.sidebar.header("📂 Data Source")

# CSV Template creation for user convenience
sample_csv = generate_sample_data().head(5).to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="⬇️ Download CSV Template",
    data=sample_csv,
    file_name="injury_data_template.csv",
    mime="text/csv",
    help="Use this structure to prepare your own data."
)

uploaded_file = st.sidebar.file_uploader("Upload your Injury CSV", type=["csv"])

st.sidebar.header("🔍 Visualization Settings")
filter_club = []
filter_player = []
filter_injury = []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Ensure dates are datetime objects
        df['Injury_Start'] = pd.to_datetime(df['Injury_Start'], errors='coerce')
        df['Injury_End'] = pd.to_datetime(df['Injury_End'], errors='coerce')
        
        # Ensure basic numeric columns exist (fill with 0 if missing to prevent crash)
        required_numerics = ['Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After', 'Rating', 'Goals']
        for col in required_numerics:
            if col not in df.columns:
                df[col] = 0  # Fill missing calc columns with 0
        
        st.success("✅ Custom CSV Loaded Successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = generate_sample_data()
else:
    st.info("ℹ️ Showing Demo Data. Upload a CSV to see your analytics.")
    df = generate_sample_data()

# ---------------------
# Derived metrics and cleaning
# ---------------------
# These transformations run on BOTH the uploaded CSV and the Sample Data
df.dropna(subset=['Injury_Start'], inplace=True) # Essential for time plotting
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Goals'] = df['Goals'].fillna(0)

# Calculate duration if not provided or just consistency
if 'Injury_Duration' not in df.columns:
    df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days

df['Injury_Duration'] = df['Injury_Duration'].apply(lambda x: x if x > 0 else 0)

# Calculate Shifts and Diffs
# Note: In a raw CSV, user needs to ensure multiple rows per player for shifts to work efficiently
df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
df['Month'] = df['Injury_Start'].dt.month
df['Impact_Index'] = df['Team_Performance_Drop'] / df['Injury_Duration'].replace(0, np.nan)
df['Team_Recovery'] = df['Team_Goals_After'] - df['Team_Goals_During']

# ---------------------
# Filters Implementation
# ---------------------
all_clubs = sorted(df['Club'].astype(str).unique())
all_players = sorted(df['Player'].astype(str).unique())
all_injuries = sorted(df['Injury_Type'].astype(str).unique())

filter_club = st.sidebar.multiselect("Club", options=all_clubs, default=all_clubs)
filter_player = st.sidebar.multiselect("Player", options=all_players, default=all_players)
filter_injury = st.sidebar.multiselect("Injury Type", options=all_injuries, default=all_injuries)

global_mode = st.sidebar.radio("🧭 Global Visualization Mode", options=["Plotly", "Matplotlib"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Seaborn Theme (Matplotlib mode)**")
global_style = st.sidebar.radio("Global Seaborn Style", options=["Modern Clean", "Classic Analytics"], index=0)
st.sidebar.markdown("Tip: select a per-tab override at the top of any tab.")

# Apply filters
filtered_df = df[
    (df['Club'].isin(filter_club)) &
    (df['Player'].isin(filter_player)) &
    (df['Injury_Type'].isin(filter_injury))
].copy()

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your sidebar selections.")
    st.stop()

# ---------------------
# KPIs
# ---------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
kpi2.metric("💥 Avg Team Performance Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
kpi3.metric("🩹 Total Injuries Recorded", f"{len(filtered_df)}")

# ---------------------
# Tabs (global + per-tab override)
# ---------------------
tabs = st.tabs([
    "🧾 Dataset Overview",
    "📊 Trends",
    "📈 Player Impact",
    "🔥 Club Analysis",
    "🔬 Injury Analysis",
    "🔎 Player Deep Dive"
])

# ---------- Tab 0: Dataset Overview ----------
with tabs[0]:
    st.subheader("📄 Dataset Preview and Summary")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0)
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0)
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    st.dataframe(filtered_df.head(), use_container_width=True)
    st.write(f"**Total Records:** {filtered_df.shape[0]} | **Columns:** {filtered_df.shape[1]}")

    with st.expander("📈 Statistical Summary and Correlation"):
        # Select numeric columns only for correlation
        numeric_df = filtered_df[['Age', 'Rating', 'Team_Performance_Drop', 'Injury_Duration']]
        st.dataframe(numeric_df.describe())
        corr = numeric_df.corr()

        if mode == "Plotly":
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                 title='Correlation Matrix (Plotly)', labels=dict(x="Metric", y="Metric", color="Correlation"))
            render_plotly(fig_corr)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("Correlation Matrix (Matplotlib / Seaborn)")
            plt.tight_layout()
            render_matplotlib(fig)

# ---------- Tab 1: Trends ----------
with tabs[1]:
    st.subheader("📊 Trends")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0, key="tab_trends_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0, key="tab_trends_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    # Top players by avg Team_Performance_Drop
    impact = filtered_df.groupby("Player")['Team_Performance_Drop'].mean().sort_values(ascending=False).head(10).reset_index()

    if mode == "Plotly":
        fig1 = px.bar(impact, x="Team_Performance_Drop", y="Player", orientation="h",
                      color="Team_Performance_Drop", color_continuous_scale="Reds",
                      title="Impact of Player Absence on Team Goals (Top 10)")
        render_plotly(fig1)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=impact, y="Player", x="Team_Performance_Drop", palette="Reds_r", ax=ax)
        ax.set_title("Impact of Player Absence on Team Goals (Top 10) - Matplotlib")
        ax.set_xlabel("Avg Team Performance Drop")
        plt.tight_layout()
        render_matplotlib(fig)

    st.markdown("---")
    # Performance timeline for sample players
    sample_players = filtered_df['Player'].unique()[:5]
    timeline_df = filtered_df[filtered_df['Player'].isin(sample_players)].sort_values('Injury_Start')
    
    if not timeline_df.empty:
        if mode == "Plotly":
            fig2 = px.line(timeline_df, x="Injury_Start", y="Rating", color="Player", markers=True,
                           title="Rating Fluctuation Around Injuries (Top 5 Selected Players)")
            render_plotly(fig2)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(10, 5))
            for player in sample_players:
                p_df = timeline_df[timeline_df['Player'] == player]
                ax.plot(p_df['Injury_Start'], p_df['Rating'], marker='o', label=player)
            ax.set_title("Rating Fluctuation Around Injuries - Matplotlib")
            ax.set_xlabel("Injury Start")
            ax.set_ylabel("Rating")
            ax.legend()
            plt.tight_layout()
            render_matplotlib(fig)

    st.markdown("---")
    monthly_trend = filtered_df.groupby('Month').size().reset_index(name='Injury_Count').sort_values('Month')
    if mode == "Plotly":
        fig_trend = px.line(monthly_trend, x='Month', y='Injury_Count', markers=True, title='Injury Frequency Over the Season')
        render_plotly(fig_trend)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(monthly_trend['Month'], monthly_trend['Injury_Count'], marker='o')
        ax.set_title("Monthly Injury Trend - Matplotlib")
        ax.set_xlabel("Month")
        ax.set_ylabel("Injury Count")
        plt.tight_layout()
        render_matplotlib(fig)

# ---------- Tab 2: Player Impact ----------
with tabs[2]:
    st.subheader("📈 Player Impact")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0, key="tab_player_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0, key="tab_player_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    leaderboard = filtered_df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10).reset_index()
    st.markdown("**Comeback Players Leaderboard (Rating Change Post-Injury)**")
    st.dataframe(leaderboard, use_container_width=True)
    st.markdown("---")

    if mode == "Plotly":
        fig3 = px.scatter(filtered_df, x="Age", y="Team_Performance_Drop", color="Club", hover_data=["Player"],
                          title="Age vs Team Performance Drop (Plotly)")
        render_plotly(fig3)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=filtered_df, x="Age", y="Team_Performance_Drop", hue="Club", ax=ax)
        ax.set_title("Age vs Team Performance Drop (Matplotlib)")
        plt.tight_layout()
        render_matplotlib(fig)

    st.markdown("---")
    status_avg = filtered_df.groupby('Status')['Rating'].mean().reset_index()
    if mode == "Plotly":
        fig_status = px.bar(status_avg, x='Status', y='Rating', color='Status', title='Average Rating by Injury Phase (Plotly)')
        render_plotly(fig_status)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=status_avg, x='Status', y='Rating', palette="pastel", ax=ax)
        ax.set_title("Average Rating by Injury Phase (Matplotlib)")
        plt.tight_layout()
        render_matplotlib(fig)

    st.markdown("---")
    player_perf = filtered_df.groupby('Player')[['Avg_Rating_Before', 'Avg_Rating_After']].mean().dropna().reset_index().head(20) # Limit for chart clarity
    if not player_perf.empty:
        if mode == "Plotly":
            fig_player_perf = px.bar(player_perf, x='Player', y=['Avg_Rating_Before', 'Avg_Rating_After'], barmode='group',
                                     title='Average Player Rating: Before vs After Injury (First 20 Players)')
            render_plotly(fig_player_perf)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(10, 5))
            player_perf_melt = player_perf.melt(id_vars='Player', value_vars=['Avg_Rating_Before', 'Avg_Rating_After'], var_name='Phase', value_name='Rating')
            sns.barplot(data=player_perf_melt, x='Player', y='Rating', hue='Phase', ax=ax)
            ax.set_title("Average Player Rating: Before vs After Injury (Matplotlib)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            render_matplotlib(fig)
    else:
        st.info("No before/after rating data available for selected filters.")

# ---------- Tab 3: Club Analysis ----------
with tabs[3]:
    st.subheader("🔥 Club Analysis")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0, key="tab_club_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0, key="tab_club_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    heatmap_data = filtered_df.groupby(['Club', 'Month']).size().reset_index(name="Count")
    if heatmap_data.empty:
        st.info("No data for the selected filters.")
    else:
        if mode == "Plotly":
            fig4 = px.density_heatmap(heatmap_data, x="Month", y="Club", z="Count", color_continuous_scale="Blues",
                                      title="When Do Injuries Occur During the Season? (Plotly)")
            render_plotly(fig4)
        else:
            apply_seaborn_style(style)
            pivot = heatmap_data.pivot(index='Club', columns='Month', values='Count').fillna(0)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap='Blues', ax=ax)
            ax.set_title("When Do Injuries Occur During the Season? (Matplotlib)")
            plt.tight_layout()
            render_matplotlib(fig)

    st.markdown("---")
    club_injuries = filtered_df.groupby("Club")['Injury_Start'].count().reset_index().rename(columns={"Injury_Start":"Injury_Count"})
    if mode == "Plotly":
        fig5 = px.bar(club_injuries, x="Club", y="Injury_Count", color="Injury_Count", color_continuous_scale="Viridis",
                      title="Total Recorded Injuries per Club (Plotly)")
        render_plotly(fig5)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=club_injuries, x='Club', y='Injury_Count', palette="viridis", ax=ax)
        ax.set_title("Total Recorded Injuries per Club (Matplotlib)")
        plt.tight_layout()
        render_matplotlib(fig)

    st.markdown("---")
    team_goals = filtered_df.groupby('Club')[['Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After']].mean().reset_index()
    if team_goals.empty:
        st.info("No team goals data for selected filters.")
    else:
        if mode == "Plotly":
            fig_goals = px.bar(team_goals, x='Club', y=['Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After'],
                               barmode='group', title='Average Team Goals Before, During and After Injury (Plotly)')
            render_plotly(fig_goals)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(10, 5))
            tg_melt = team_goals.melt(id_vars='Club', value_vars=['Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After'], var_name='Phase', value_name='Goals')
            sns.barplot(data=tg_melt, x='Club', y='Goals', hue='Phase', ax=ax)
            ax.set_title("Average Team Goals Before, During and After Injury (Matplotlib)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            render_matplotlib(fig)

    st.markdown("---")
    top_clubs = filtered_df.groupby('Club')['Team_Performance_Drop'].mean().sort_values(ascending=False).head(5).reset_index()
    st.markdown("### Highest Avg Performance Drop by Club")
    st.dataframe(top_clubs, use_container_width=True)

# ---------- Tab 4: Injury Analysis ----------
with tabs[4]:
    st.subheader("🔬 Injury Analysis")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0, key="tab_injury_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0, key="tab_injury_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    if mode == "Matplotlib":
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=filtered_df, x='Injury_Type', y='Team_Performance_Drop', ax=ax, palette="Set2")
        ax.set_title("Team Performance Drop by Injury Type (Matplotlib)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        render_matplotlib(fig)
    else:
        fig_injury_impact = px.box(filtered_df, x='Injury_Type', y='Team_Performance_Drop', color='Injury_Type',
                                   title='Team Performance Drop by Injury Type (Plotly)')
        render_plotly(fig_injury_impact)

    st.markdown("---")
    injury_counts = filtered_df['Injury_Type'].value_counts().reset_index()
    injury_counts.columns = ['Injury_Type', 'Count']
    if mode == "Plotly":
        fig_injury_counts = px.bar(injury_counts.sort_values('Count', ascending=False),
                                   x='Injury_Type', y='Count', color='Count',
                                   color_continuous_scale='Plasma', title='Most Common Injury Types (Plotly)')
        render_plotly(fig_injury_counts)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=injury_counts, x='Injury_Type', y='Count', palette="plasma", ax=ax)
        ax.set_title("Most Common Injury Types (Matplotlib)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        render_matplotlib(fig)

    st.markdown("---")
    impact_df = filtered_df.groupby('Injury_Type')['Impact_Index'].mean().reset_index()
    if mode == "Plotly":
        fig_impact = px.bar(impact_df, x='Injury_Type', y='Impact_Index', color='Impact_Index',
                            color_continuous_scale='Inferno', title='Average Impact Index (Plotly)')
        render_plotly(fig_impact)
    else:
        apply_seaborn_style(style)
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(data=impact_df, x='Injury_Type', y='Impact_Index', palette="inferno", ax=ax)
        ax.set_title("Average Impact Index (Matplotlib)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        render_matplotlib(fig)

# ---------- Tab 5: Player Deep Dive ----------
with tabs[5]:
    st.subheader("🔎 Player Deep Dive")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0, key="tab_deep_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0, key="tab_deep_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    player_to_analyze = st.selectbox("Select a Player to Analyze", options=sorted(df['Player'].unique()))
    if player_to_analyze:
        player_df = filtered_df[filtered_df['Player'] == player_to_analyze].copy()
        st.markdown(f"### Analytics for: **{player_to_analyze}**")

        if player_df.empty:
            st.warning("No data for this player with the current filters.")
        else:
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
                'Team_Performance_Drop': 'Team Goal Drop',
                'Team_Recovery': 'Team Goal Recovery After'
            }
            # Only show columns that exist
            actual_cols = [c for c in display_cols.keys() if c in player_df.columns]
            st.dataframe(player_df[actual_cols].rename(columns=display_cols).sort_values(by='From', ascending=False), use_container_width=True)

            st.subheader("Performance Timeline")
            if mode == "Plotly":
                fig_player = px.line(player_df.sort_values(by='Injury_Start'), x="Injury_Start", y="Rating", markers=True,
                                     title=f"Rating Over Time for {player_to_analyze}")
                fig_player.update_traces(text=player_df['Rating'], textposition='top center', hovertemplate='Date: %{x}<br>Rating: %{y:.2f}')
                render_plotly(fig_player)
            else:
                apply_seaborn_style(style)
                fig, ax = plt.subplots(figsize=(8, 4))
                sorted_df = player_df.sort_values('Injury_Start')
                ax.plot(sorted_df['Injury_Start'], sorted_df['Rating'], marker='o')
                ax.set_title(f"Rating Over Time for {player_to_analyze} (Matplotlib)")
                ax.set_xlabel("Injury Start")
                ax.set_ylabel("Rating")
                plt.tight_layout()
                render_matplotlib(fig)

            st.markdown("---")
            st.subheader("📈 Player Rating vs Team Goals: Before / During / After")
            plot_df = player_df.sort_values('Injury_Start').reset_index(drop=True)

            if mode == "Plotly":
                try:
                    # Melt safely assuming columns exist
                    cols_to_melt = ['Avg_Rating_Before', 'Rating', 'Avg_Rating_After', 'Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After']
                    # Verify cols exist before melting
                    available_cols = [c for c in cols_to_melt if c in plot_df.columns]
                    
                    plot_df_melt = plot_df.melt(id_vars=['Injury_Start'], value_vars=available_cols,
                                                var_name='Metric', value_name='Value')
                    
                    fig_dynamic = px.line(plot_df_melt, x='Injury_Start', y='Value', color='Metric', markers=True,
                                          title=f"{player_to_analyze}: Rating vs Team Goals (Plotly)")
                    fig_dynamic.update_traces(mode='lines+markers')
                    render_plotly(fig_dynamic)
                except Exception as e:
                    st.error(f"Could not visualize team goal data: {e}")
            else:
                apply_seaborn_style(style)
                fig, ax = plt.subplots(figsize=(9, 4))
                sorted_df = plot_df
                ax.plot(sorted_df['Injury_Start'], sorted_df['Rating'], marker='o', label='Player Rating (During)', linewidth=2)
                
                if 'Avg_Rating_Before' in sorted_df:
                    ax.plot(sorted_df['Injury_Start'], sorted_df['Avg_Rating_Before'], marker='o', linestyle='--', label='Player Rating - Before')
                if 'Avg_Rating_After' in sorted_df:
                    ax.plot(sorted_df['Injury_Start'], sorted_df['Avg_Rating_After'], marker='o', linestyle=':', label='Player Rating - After')
                ax.set_ylabel('Player Rating')
                ax.set_xlabel('Injury Start')

                # Twin axis for goals if they exist
                if all(x in sorted_df.columns for x in ['Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After']):
                    ax2 = ax.twinx()
                    ax2.plot(sorted_df['Injury_Start'], sorted_df['Team_Goals_Before'], marker='s', linestyle='--', label='Team Goals - Before', color='tab:green')
                    ax2.plot(sorted_df['Injury_Start'], sorted_df['Team_Goals_During'], marker='s', linestyle='-', label='Team Goals - During', color='tab:orange')
                    ax2.plot(sorted_df['Injury_Start'], sorted_df['Team_Goals_After'], marker='s', linestyle=':', label='Team Goals - After', color='tab:purple')
                    ax2.set_ylabel('Team Goals')
                    lines2, labels2 = ax2.get_legend_handles_labels()
                else:
                    lines2, labels2 = [], []

                lines, labels = ax.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2, fontsize='small')
                ax.set_title(f"{player_to_analyze}: Rating vs Team Goals (Matplotlib)")
                plt.tight_layout()
                render_matplotlib(fig)

# ---------------------
# About + Download
# ---------------------
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    This dashboard supports both Plotly (interactive) and Matplotlib/Seaborn (static) visuals.
    **Data Source:** Use the file uploader in the sidebar to supply your own CSV. If no file is provided,
    demo data is shown.
    """)

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_injury_impact_data.csv",
    mime="text/csv"
)

st.markdown("<hr><center>© 2025 FootLens Analytics | Developed by Parkar</center>", unsafe_allow_html=True)
