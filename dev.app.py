# ==================================
# ⚽ FootLens — Ultra Elite Player Injury Impact & Prediction Dashboard (Final Unified Version)
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import zipfile

# ---------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="⚽ FootLens — Ultra Elite Dashboard", layout="wide")
st.title("⚽ FootLens — Ultra Elite Player Injury Impact & Prediction Dashboard")
st.markdown("Analyze, Predict, and Compare Player Injuries, Performance, and Recovery with Professional Insights!")

# ---------------------------------------------
# Data Simulation & Cleaning
# ---------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    players = [f"Player_{i}" for i in range(1, 21)]
    clubs = [f"Club_{i}" for i in range(1, 6)]
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="15D")
    injury_types = ["Hamstring", "Groin", "ACL", "Ankle", "Calf", "Back"]

    injury_starts = np.random.choice(dates, 300)
    injury_durations_days = np.random.randint(7, 120, 300)

    data = {
        "Player": np.random.choice(players, 300),
        "Club": np.random.choice(clubs, 300),
        "Rating": np.random.uniform(5, 9, 300),
        "Goals": np.random.randint(0, 6, 300),
        "Team_Goals_Before": np.random.randint(10, 30, 300),
        "Team_Goals_During": np.random.randint(5, 25, 300),
        "Age": np.random.randint(18, 35, 300),
        "Injury_Start": pd.to_datetime(injury_starts),
        "Injury_End": [start + pd.Timedelta(days=dur) for start, dur in zip(injury_starts, injury_durations_days)],
        "Status": np.random.choice(["Before", "During", "After"], 300),
        "Injury_Type": np.random.choice(injury_types, 300)
    }

    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days.clip(lower=1)
    df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
    df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
    df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
    df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
    df['Impact_Index'] = df['Team_Performance_Drop'] / df['Injury_Duration']
    df['Severity_Score'] = df['Injury_Duration'] * (10 - df['Rating']) / 10
    df['Month'] = df['Injury_Start'].dt.month
    return df

df = generate_data()

# ---------------------------------------------
# Sidebar Filters
# ---------------------------------------------
st.sidebar.header("🔍 Filters")
mode = st.sidebar.radio("Dashboard Mode", ["Executive", "Analyst"], index=0)
filter_club = st.sidebar.multiselect("Club", df['Club'].unique(), default=df['Club'].unique())
filter_player = st.sidebar.multiselect("Player", df['Player'].unique(), default=df['Player'].unique())
filter_injury = st.sidebar.multiselect("Injury Type", df['Injury_Type'].unique(), default=df['Injury_Type'].unique())

filtered_df = df[(df['Club'].isin(filter_club)) & (df['Player'].isin(filter_player)) & (df['Injury_Type'].isin(filter_injury))]

# ---------------------------------------------
# KPIs
# ---------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
kpi2.metric("💥 Avg Team Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
kpi3.metric("🩹 Total Injuries", f"{len(filtered_df)}")
kpi4.metric("🔥 Avg Severity", f"{filtered_df['Severity_Score'].mean():.2f}")

# ---------------------------------------------
# Tabs
# ---------------------------------------------
tabs = st.tabs([
    "🧾 Dataset Overview", 
    "📊 Trends & Club Impact", 
    "📈 Player Performance & Comeback", 
    "🧠 Prediction & Similarity", 
    "🔎 Player Deep Dive"
])

# -------- 🧾 Dataset Overview --------
with tabs[0]:
    st.subheader("📄 Dataset Overview")
    st.dataframe(filtered_df.head(), use_container_width=True)

    with st.expander("📊 Summary Stats & Correlation"):
        st.dataframe(df[['Age', 'Rating', 'Team_Performance_Drop', 'Injury_Duration']].describe())
        corr = df[['Age', 'Rating', 'Team_Performance_Drop', 'Injury_Duration', 'Severity_Score']].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# -------- 📊 Trends & Club Impact --------
with tabs[1]:
    st.subheader("Top 10 Players with Highest Team Impact")
    impact = filtered_df.groupby('Player')['Team_Performance_Drop'].mean().sort_values(ascending=False).head(10).reset_index()
    fig1 = px.bar(impact, x='Team_Performance_Drop', y='Player', orientation='h', color='Team_Performance_Drop', color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📆 Monthly Injury Trend")
    monthly_trend = df.groupby('Month').size().reset_index(name='Injury_Count')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_trend['Month'], monthly_trend['Injury_Count'], marker='o', color='blue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Injuries')
    ax.set_title('Monthly Injury Trend')
    st.pyplot(fig)

    st.subheader("Club Injury Heatmap")
    heatmap_data = filtered_df.groupby(['Club', 'Month']).size().reset_index(name='Count')
    fig4 = px.density_heatmap(heatmap_data, x='Month', y='Club', z='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig4, use_container_width=True)

# -------- 📈 Player Performance & Comeback --------
with tabs[2]:
    st.subheader("Comeback Players Leaderboard (Rating Change Post-Injury)")
    leaderboard = filtered_df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10).reset_index()
    st.dataframe(leaderboard, use_container_width=True)

    st.subheader("Player Age vs Team Performance Drop")
    fig3 = px.scatter(filtered_df, x='Age', y='Team_Performance_Drop', color='Club', hover_data=['Player'])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Average Rating Before, During, After Injury")
    status_avg = filtered_df.groupby('Status')['Rating'].mean().reset_index()
    fig_status = px.bar(status_avg, x='Status', y='Rating', color='Status', color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_status, use_container_width=True)

# -------- 🧠 Prediction & Similarity --------
with tabs[3]:
    st.subheader("Injury Recurrence Prediction (Demo)")
    model_data = filtered_df[['Age', 'Rating', 'Injury_Duration', 'Severity_Score']].dropna()
    y = np.random.choice([0, 1], len(model_data))
    X = StandardScaler().fit_transform(model_data)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    preds = clf.predict_proba(X)[:, 1]
    filtered_df.loc[model_data.index, 'Recurrence_Prob'] = preds
    fig_pred = px.scatter(filtered_df, x='Severity_Score', y='Recurrence_Prob', color='Club', hover_data=['Player'])
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("Player Similarity Engine")
    features = ['Age', 'Rating', 'Severity_Score', 'Injury_Duration']
    sim_data = filtered_df[features].dropna()
    sim_matrix = cosine_similarity(StandardScaler().fit_transform(sim_data))
    player_select = st.selectbox('Select a Player for Similarity', options=filtered_df['Player'].unique())
    if player_select in filtered_df['Player'].values:
        idx = filtered_df[filtered_df['Player'] == player_select].index[0]
        sims = sim_matrix[idx]
        sim_df = pd.DataFrame({'Player': filtered_df.iloc[:len(sims)]['Player'], 'Similarity': sims})
        st.dataframe(sim_df.sort_values('Similarity', ascending=False).head(5))

# -------- 🔎 Player Deep Dive --------
with tabs[4]:
    st.subheader("Player Deep Dive — Full Recovery Analysis")
    player_analyze = st.selectbox('Select Player to Analyze', options=sorted(df['Player'].unique()))
    player_df = filtered_df[filtered_df['Player'] == player_analyze]
    if not player_df.empty:
        kpiA, kpiB, kpiC = st.columns(3)
        kpiA.metric('⚽ Average Rating', f"{player_df['Rating'].mean():.2f}")
        kpiB.metric('🩹 Injury Count', f"{len(player_df)}")
        kpiC.metric('⏳ Avg Duration (Days)', f"{player_df['Injury_Duration'].mean():.1f}")
        fig5 = px.line(player_df, x='Injury_Start', y='Rating', title='Performance Over Time', markers=True)
        st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------
# Export Functionality
# ---------------------------------------------
with st.expander("📦 Export Filtered Data"):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        zf.writestr('filtered_data.csv', filtered_df.to_csv(index=False))
    st.download_button(label='📥 Download Filtered Dataset (ZIP)', data=buffer.getvalue(), file_name='footlens_export.zip', mime='application/zip')

st.markdown("<hr><center>© 2025 FootLens Analytics | Unified Hybrid Elite Version</center>", unsafe_allow_html=True)
