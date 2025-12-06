# ==================================
# ⚽ Player Injury Impact Dashboard (CSV-ready)
# Dual-mode: Plotly + Matplotlib/Seaborn with style selection
# ==================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="⚽ Player Injury Impact Dashboard", layout="wide")
st.title("⚽ Player Injury Impact Dashboard")
st.markdown("Interactive dashboard with Plotly + Matplotlib/Seaborn. Upload your CSV file and explore the data.")

# ---------------------
# Helper utilities
# ---------------------
def get_mode(per_tab_choice, global_mode):
    return per_tab_choice if per_tab_choice in ["Plotly", "Matplotlib"] else global_mode

def get_style(per_tab_style, global_style):
    return per_tab_style if per_tab_style in ["Modern Clean", "Classic Analytics"] else global_style

def apply_seaborn_style(style_name):
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

# ---------------------
# CSV Upload
# ---------------------
uploaded_file = st.file_uploader("Upload CSV file with injury data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Successfully loaded {len(df)} records.")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ---------------------
# Data Cleaning & Derived Metrics
# ---------------------
df.drop_duplicates(inplace=True)

# Ensure required columns exist
required_cols = ["Player","Club","Rating","Goals","Team_Goals_Before","Team_Goals_During",
                 "Age","Injury_Start","Injury_End","Status","Injury_Type"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# Fill missing numeric data
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['Goals'] = df['Goals'].fillna(0)

# Convert date columns
df['Injury_Start'] = pd.to_datetime(df['Injury_Start'])
df['Injury_End'] = pd.to_datetime(df['Injury_End'])

# Derived metrics
df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days.clip(lower=0)
df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
df['Month'] = df['Injury_Start'].dt.month
df['Impact_Index'] = df['Team_Performance_Drop'] / df['Injury_Duration'].replace(0, np.nan)
df['Team_Recovery'] = df['Team_Goals_After'] - df['Team_Goals_During']

# ---------------------
# Sidebar: filters + global mode + style
# ---------------------
st.sidebar.header("🔍 Filters & Visualization Settings")

filter_club = st.sidebar.multiselect("Club", options=sorted(df['Club'].unique()), default=sorted(df['Club'].unique()))
filter_player = st.sidebar.multiselect("Player", options=sorted(df['Player'].unique()), default=sorted(df['Player'].unique()))
filter_injury = st.sidebar.multiselect("Injury Type", options=sorted(df['Injury_Type'].unique()), default=sorted(df['Injury_Type'].unique()))

global_mode = st.sidebar.radio("🧭 Global Visualization Mode", options=["Plotly", "Matplotlib"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Seaborn Theme (Matplotlib mode)")
global_style = st.sidebar.radio("Global Seaborn Style", options=["Modern Clean", "Classic Analytics"], index=0)

# Apply filters
filtered_df = df[
    (df['Club'].isin(filter_club)) &
    (df['Player'].isin(filter_player)) &
    (df['Injury_Type'].isin(filter_injury))
].copy()

# ---------------------
# KPIs
# ---------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
kpi2.metric("💥 Avg Team Performance Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}")
kpi3.metric("🩹 Total Injuries Recorded", f"{len(filtered_df)}")

# ---------------------
# Tabs
# ---------------------
tabs = st.tabs([
    "🧾 Dataset Overview",
    "📊 Trends",
    "📈 Player Impact",
    "🔥 Club Analysis",
    "🔬 Injury Analysis",
    "🔎 Player Deep Dive"
])

# ---------------------------------
# Tab 0: Dataset Overview
# ---------------------------------
with tabs[0]:
    st.subheader("📄 Dataset Preview")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", options=["Auto", "Plotly", "Matplotlib"], index=0)
    tab_style = st.selectbox("Seaborn style for this tab (override)", options=["Auto", "Modern Clean", "Classic Analytics"], index=0)
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    st.dataframe(filtered_df.head(), use_container_width=True)
    st.write(f"**Total Records:** {filtered_df.shape[0]} | **Columns:** {filtered_df.shape[1]}")

    with st.expander("📈 Statistical Summary and Correlation"):
        numeric_cols = ['Age','Rating','Team_Performance_Drop','Injury_Duration']
        st.dataframe(filtered_df[numeric_cols].describe())
        corr = filtered_df[numeric_cols].corr()

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

# ---------------------
# Remaining tabs...
# ---------------------
# The structure for tabs 1–5 will follow exactly as in your code
# just replace any `df` with `filtered_df` and ensure numeric conversions where needed
# Also make sure to handle empty dataframes with `if filtered_df.empty: st.info("No data")`

# ---------------------
# About + Download
# ---------------------
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    This dashboard supports both Plotly (interactive) and Matplotlib/Seaborn (static) visuals.
    Use the global toggles in the sidebar or override per-tab using the dropdowns at the top of each tab.
    """)

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_injury_impact_data.csv",
    mime="text/csv"
)

st.markdown("<hr><center>© 2025 FootLens Analytics | Developed by Parkar</center>", unsafe_allow_html=True)
