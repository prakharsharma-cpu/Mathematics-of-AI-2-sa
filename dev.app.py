# ==================================
# ⚽ FootLens — Ultimate Elite Player Injury Dashboard with Full Features
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
# Streamlit Page Config
# ---------------------------------------------
st.set_page_config(page_title="⚽ FootLens — Ultimate Elite Dashboard", layout="wide")
st.title("⚽ FootLens — Ultimate Elite Player Injury Dashboard")
st.markdown("Analyze, Predict, Compare, and Simulate Player Injuries and Performance with Pro Insights!")

# ---------------------------------------------
# Data Upload / Simulation
# ---------------------------------------------
st.sidebar.header("🔍 Data Input & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV for EDA & Analysis", type=['csv'])

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Injury_Start','Injury_End'], dayfirst=True, infer_datetime_format=True)
    else:
        np.random.seed(42)
        players = [f"Player_{i}" for i in range(1,21)]
        clubs = [f"Club_{i}" for i in range(1,6)]
        dates = pd.date_range("2020-01-01","2023-12-31", freq="15D")
        injury_types = ["Hamstring","Groin","ACL","Ankle","Calf","Back"]
        injury_starts = np.random.choice(dates, 300)
        injury_durations = np.random.randint(7,120,300)
        data = {
            "Player": np.random.choice(players,300),
            "Club": np.random.choice(clubs,300),
            "Rating": np.random.uniform(5,9,300),
            "Goals": np.random.randint(0,6,300),
            "Team_Goals_Before": np.random.randint(10,30,300),
            "Team_Goals_During": np.random.randint(5,25,300),
            "Age": np.random.randint(18,35,300),
            "Injury_Start": pd.to_datetime(injury_starts),
            "Injury_End": [s+pd.Timedelta(days=d) for s,d in zip(injury_starts,injury_durations)],
            "Status": np.random.choice(["Before","During","After"],300),
            "Injury_Type": np.random.choice(injury_types,300)
        }
        df = pd.DataFrame(data)

    df.drop_duplicates(inplace=True)
    df['Injury_Duration'] = (df['Injury_End']-df['Injury_Start']).dt.days.clip(lower=1)
    df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
    df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
    df['Performance_Change'] = df['Avg_Rating_After']-df['Avg_Rating_Before']
    df['Team_Performance_Drop'] = df['Team_Goals_Before']-df['Team_Goals_During']
    df['Impact_Index'] = df['Team_Performance_Drop']/df['Injury_Duration']
    df['Severity_Score'] = df['Injury_Duration']*(10-df['Rating'])/10
    df['Month'] = df['Injury_Start'].dt.month

    # --- 20+ Advanced Features ---
    df['ERI'] = df['Severity_Score']/df['Injury_Duration']  # Expected Recovery Index
    df['Recurrence_180d'] = np.random.randint(0,2,len(df))
    df['Club_Resilience'] = df.groupby('Club')['Impact_Index'].transform('mean')
    df['Goals_Lost'] = df['Team_Goals_Before']-df['Goals']
    df['Rating_Drop'] = df['Avg_Rating_Before']-df['Rating']
    df['Age_Category'] = pd.cut(df['Age'], bins=[17,20,25,30,35], labels=['Young','Early','Prime','Late'])
    df['High_Impact'] = df['Impact_Index']>df['Impact_Index'].quantile(0.75)
    df['Recovery_Trend'] = df['Avg_Rating_After'] - df['Rating']
    df['Injury_Frequency'] = df.groupby('Player')['Player'].transform('count')
    df['Performance_Consistency'] = df.groupby('Player')['Rating'].transform(lambda x: x.std())
    df['Team_Performance_Percent_Drop'] = df['Team_Performance_Drop']/df['Team_Goals_Before']*100
    df['Severity_Category'] = pd.qcut(df['Severity_Score'],4,labels=['Low','Moderate','High','Critical'])
    df['Days_Since_Last_Injury'] = (df['Injury_Start']-df.groupby('Player')['Injury_End'].shift(1)).dt.days.fillna(0)
    df['Cumulative_Injury_Duration'] = df.groupby('Player')['Injury_Duration'].cumsum()
    df['Recent_Injury'] = (df['Injury_Start'] > pd.Timestamp('2022-01-01'))
    df['Impact_Per_Goal'] = df['Impact_Index']/df['Goals'].replace(0,1)
    df['Normalized_Impact'] = (df['Impact_Index']-df['Impact_Index'].min())/(df['Impact_Index'].max()-df['Impact_Index'].min())
    df['Rating_Per_Day'] = df['Rating']/df['Injury_Duration']
    df['Goal_Contribution'] = df['Goals']/df['Team_Goals_Before']
    df['Weighted_Impact'] = df['Impact_Index']*df['Severity_Score']

    return df

# Load Data
filtered_df = load_data(uploaded_file)

# Sidebar Filters
mode = st.sidebar.radio("Dashboard Mode", ["Executive","Analyst"], index=0)
filter_club = st.sidebar.multiselect("Club", filtered_df['Club'].unique(), default=filtered_df['Club'].unique())
filter_player = st.sidebar.multiselect("Player", filtered_df['Player'].unique(), default=filtered_df['Player'].unique())
filter_injury = st.sidebar.multiselect("Injury Type", filtered_df['Injury_Type'].unique(), default=filtered_df['Injury_Type'].unique())
filtered_df = filtered_df[(filtered_df['Club'].isin(filter_club)) & (filtered_df['Player'].isin(filter_player)) & (filtered_df['Injury_Type'].isin(filter_injury))]

# ---------------------------------------------
# KPIs
# ---------------------------------------------
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
k2.metric("💥 Avg Team Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
k3.metric("🩹 Total Injuries", f"{len(filtered_df)}")
k4.metric("🔥 Avg Severity", f"{filtered_df['Severity_Score'].mean():.2f}")
k5.metric("📊 Avg Impact Index", f"{filtered_df['Impact_Index'].mean():.2f}")
k6.metric("📈 Avg Weighted Impact", f"{filtered_df['Weighted_Impact'].mean():.2f}")

# ---------------------------------------------
# Tabs
# ---------------------------------------------
tabs = st.tabs(["🧾 Dataset Overview","📊 Trends & Club Impact","📈 Player Performance","🧠 Prediction & Similarity","🔎 Deep Dive & Simulation"])

# Dataset Overview
with tabs[0]:
    st.subheader("Dataset Preview & Stats")
    st.dataframe(filtered_df.head(), use_container_width=True)
    with st.expander("Statistical Summary & Correlation"):
        st.dataframe(filtered_df.describe())
        corr = filtered_df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

# Trends & Club Impact
with tabs[1]:
    st.subheader("Top Players by Impact")
    top_players = filtered_df.groupby('Player')['Team_Performance_Drop'].mean().sort_values(ascending=False).head(10).reset_index()
    fig1 = px.bar(top_players, x='Team_Performance_Drop', y='Player', orientation='h', color='Team_Performance_Drop', color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Club Monthly Heatmap")
    heatmap_data = filtered_df.groupby(['Club','Month']).size().reset_index(name='Count')
    fig2 = px.density_heatmap(heatmap_data, x='Month', y='Club', z='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig2, use_container_width=True)

# Player Performance
with tabs[2]:
    st.subheader("Comeback Leaderboard")
    leaderboard = filtered_df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10).reset_index()
    st.dataframe(leaderboard, use_container_width=True)

    st.subheader("Avg Rating by Injury Phase")
    status_avg = filtered_df.groupby('Status')['Rating'].mean().reset_index()
    fig3 = px.bar(status_avg, x='Status', y='Rating', color='Status', color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig3, use_container_width=True)

# Prediction & Similarity
with tabs[3]:
    st.subheader("Recurrence Risk Prediction (Demo)")
    model_data = filtered_df[['Age','Rating','Injury_Duration','Severity_Score']].dropna()
    y = np.random.choice([0,1], len(model_data))
    X = StandardScaler().fit_transform(model_data)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X,y)
    filtered_df.loc[model_data.index,'Recurrence_Prob'] = clf.predict_proba(X)[:,1]
    fig4 = px.scatter(filtered_df, x='Severity_Score', y='Recurrence_Prob', color='Club', hover_data=['Player'])
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Player Similarity Engine")
    features = ['Age','Rating','Severity_Score','Injury_Duration']
    sim_matrix = cosine_similarity(StandardScaler().fit_transform(filtered_df[features]))
    player_choice = st.selectbox("Select Player for Similarity", options=filtered_df['Player'].unique())
    idx = filtered_df[filtered_df['Player']==player_choice].index[0]
    sim_scores = sim_matrix[idx]
    sim_df = pd.DataFrame({'Player': filtered_df['Player'],'Similarity': sim_scores}).sort_values('Similarity', ascending=False).head(6)
    st.dataframe
