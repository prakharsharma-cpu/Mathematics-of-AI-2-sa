# ==================================
# ⚽ Ultimate Player Injury Impact Dashboard
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="⚽ Ultimate Player Injury Impact Dashboard", layout="wide")
st.title("⚽ Ultimate Player Injury Impact Dashboard")
st.markdown("Upload your CSV and explore injury impact analytics with full automated EDA and interactive visualizations.")

# ---------------------
# Helper functions
# ---------------------
def get_mode(per_tab_choice, global_mode):
    return per_tab_choice if per_tab_choice in ["Plotly","Matplotlib"] else global_mode

def get_style(per_tab_style, global_style):
    return per_tab_style if per_tab_style in ["Modern Clean","Classic Analytics"] else global_style

def apply_seaborn_style(style_name):
    if style_name=="Classic Analytics":
        sns.set_theme(style="darkgrid", palette="muted")
        plt.rcParams.update({"figure.facecolor":"white"})
    else:
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({"figure.facecolor":"white"})

def render_plotly(fig):
    st.plotly_chart(fig, use_container_width=True)

def render_matplotlib(fig):
    st.pyplot(fig)

# ---------------------
# CSV Upload
# ---------------------
uploaded_file = st.file_uploader("Upload CSV file with injury data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} records.")
else:
    st.warning("Upload CSV to continue.")
    st.stop()

# ---------------------
# Data preprocessing
# ---------------------
df.drop_duplicates(inplace=True)

# Convert date columns
for date_col in ["Injury_Start","Injury_End"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Fill numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Derived metrics
if all(c in df.columns for c in ["Injury_Start","Injury_End"]):
    df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days.clip(lower=0)
    df['Month'] = df['Injury_Start'].dt.month
else:
    df['Injury_Duration'] = np.nan
    df['Month'] = np.nan

if all(c in df.columns for c in ["Team_Goals_Before","Team_Goals_During"]):
    df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
else:
    df['Team_Performance_Drop'] = np.nan

if all(c in df.columns for c in ["Team_Goals_During","Team_Goals_After"]):
    df['Team_Recovery'] = df['Team_Goals_After'] - df['Team_Goals_During']
else:
    df['Team_Recovery'] = np.nan

if all(c in df.columns for c in ["Avg_Rating_Before","Avg_Rating_After"]):
    df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
else:
    df['Performance_Change'] = np.nan

# ---------------------
# Sidebar Filters + Mode/Style
# ---------------------
st.sidebar.header("🔍 Filters & Visualization Settings")
filter_club = st.sidebar.multiselect("Club", options=df['Club'].unique() if 'Club' in df.columns else [], default=df['Club'].unique() if 'Club' in df.columns else [])
filter_player = st.sidebar.multiselect("Player", options=df['Player'].unique() if 'Player' in df.columns else [], default=df['Player'].unique() if 'Player' in df.columns else [])
filter_injury = st.sidebar.multiselect("Injury Type", options=df['Injury_Type'].unique() if 'Injury_Type' in df.columns else [], default=df['Injury_Type'].unique() if 'Injury_Type' in df.columns else [])

global_mode = st.sidebar.radio("🧭 Global Visualization Mode", ["Plotly","Matplotlib"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Seaborn Theme (Matplotlib)")
global_style = st.sidebar.radio("Global Seaborn Style", ["Modern Clean","Classic Analytics"], index=0)

# Apply filters
filtered_df = df.copy()
if 'Club' in df.columns:
    filtered_df = filtered_df[filtered_df['Club'].isin(filter_club)]
if 'Player' in df.columns:
    filtered_df = filtered_df[filtered_df['Player'].isin(filter_player)]
if 'Injury_Type' in df.columns:
    filtered_df = filtered_df[filtered_df['Injury_Type'].isin(filter_injury)]

# ---------------------
# KPIs
# ---------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}" if 'Rating' in filtered_df.columns else "N/A")
kpi2.metric("💥 Avg Team Performance Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}" if 'Team_Performance_Drop' in filtered_df.columns else "N/A")
kpi3.metric("🩹 Total Injuries", f"{len(filtered_df)}")

# ---------------------
# Tabs
# ---------------------
tabs = st.tabs([
    "🧾 Dataset Overview",
    "📊 Full EDA & Trends",
    "📈 Player Impact",
    "🔥 Club Analysis",
    "🔬 Injury Analysis",
    "🔎 Player Deep Dive"
])

# =====================================
# Tab 0: Dataset Overview
# =====================================
with tabs[0]:
    st.subheader("📄 Dataset Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)
    st.write(f"**Total Records:** {filtered_df.shape[0]} | **Columns:** {filtered_df.shape[1]}")
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    st.dataframe(filtered_df.describe(include='all').transpose())

# =====================================
# Tab 1: Full EDA & Trends
# =====================================
with tabs[1]:
    st.subheader("📊 Automated EDA & Core Trends")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", ["Auto","Plotly","Matplotlib"], index=0, key="eda_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", ["Auto","Modern Clean","Classic Analytics"], index=0, key="eda_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(exclude=np.number).columns.tolist()

    # Correlation Heatmap
    if numeric_cols:
        st.markdown("### 🔗 Correlation Heatmap")
        corr = filtered_df[numeric_cols].corr()
        if mode=="Plotly":
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            plt.tight_layout(); render_matplotlib(fig)

    # Missing Values Heatmap
    st.markdown("### ❗ Missing Values Heatmap")
    if filtered_df.isnull().sum().sum() > 0:
        if mode=="Plotly":
            fig = ff.create_annotated_heatmap(z=filtered_df.isnull().astype(int).T.values, x=list(filtered_df.index), y=list(filtered_df.columns))
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(filtered_df.isnull(), cbar=False, cmap='viridis', ax=ax)
            plt.tight_layout(); render_matplotlib(fig)
    else:
        st.info("No missing values detected.")

    # Histograms
    if numeric_cols:
        st.markdown("### 📊 Numeric Feature Distributions")
        for col in numeric_cols:
            if mode=="Plotly":
                fig = px.histogram(filtered_df, x=col, nbins=20, title=f"Distribution of {col}")
                render_plotly(fig)
            else:
                apply_seaborn_style(style)
                fig, ax = plt.subplots(figsize=(6,3))
                sns.histplot(filtered_df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}"); plt.tight_layout(); render_matplotlib(fig)

    # Boxplots
    if numeric_cols and categorical_cols:
        st.markdown("### 📦 Boxplots by Categorical Features")
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if mode=="Plotly":
                    fig = px.box(filtered_df, x=cat_col, y=num_col, color=cat_col, title=f"{num_col} by {cat_col}")
                    render_plotly(fig)
                else:
                    apply_seaborn_style(style)
                    fig, ax = plt.subplots(figsize=(8,4))
                    sns.boxplot(data=filtered_df, x=cat_col, y=num_col, palette="Set3", ax=ax)
                    ax.set_title(f"{num_col} by {cat_col}"); plt.xticks(rotation=45)
                    plt.tight_layout(); render_matplotlib(fig)

    # Top 10 injuries by performance drop
    if 'Injury_Type' in filtered_df.columns and 'Team_Performance_Drop' in filtered_df.columns:
        top_injuries = filtered_df.groupby('Injury_Type')['Team_Performance_Drop'].mean().sort_values(ascending=False).head(10).reset_index()
        st.markdown("### 🩹 Top 10 Injuries by Avg Team Performance Drop")
        if mode=="Plotly":
            fig = px.bar(top_injuries, x='Team_Performance_Drop', y='Injury_Type', orientation='h', color='Team_Performance_Drop', color_continuous_scale='Reds')
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=top_injuries, y='Injury_Type', x='Team_Performance_Drop', palette='Reds_r', ax=ax)
            plt.tight_layout(); render_matplotlib(fig)
