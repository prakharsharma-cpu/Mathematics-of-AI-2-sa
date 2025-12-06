import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# 1. Page & Styles
# ---------------------
st.set_page_config(page_title="⚽ Player Injury Dashboard", layout="wide")
st.title("⚽ Player Injury Impact Dashboard")
st.markdown("Data-driven injury analysis. **Upload CSV** or use **Demo Data**. Pre-processing auto-calculates impact metrics.")

# ---------------------
# 2. Logic: Data & Preprocessing
# ---------------------
def generate_demo_data():
    """Generates synthetic data if no CSV is uploaded."""
    np.random.seed(42)
    rows = 300
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="7D")
    data = {
        "Player": np.random.choice([f"Player_{i}" for i in range(1, 26)], rows),
        "Club": np.random.choice([f"FC Club_{i}" for i in range(1, 6)], rows),
        "Injury_Start": np.random.choice(dates, rows),
        "Injury_Type": np.random.choice(["Hamstring", "Knee", "Ankle", "Muscle", "Back"], rows),
        "Rating": np.random.uniform(5.5, 9.0, rows),
        "Team_Goals_Before": np.random.randint(15, 30, rows),
        "Team_Goals_During": np.random.randint(5, 20, rows),
        "Age": np.random.randint(19, 36, rows)
    }
    df = pd.DataFrame(data)
    # Add random duration to create End Date
    df["Injury_End"] = df["Injury_Start"] + pd.to_timedelta(np.random.randint(5, 90, rows), unit='D')
    # Randomly calculate Goals After
    df["Team_Goals_After"] = df["Team_Goals_During"] + np.random.randint(-2, 5, rows)
    df["Status"] = "Recovered"
    return df

def preprocess_csv(df):
    """
    Core Preprocessing Logic: 
    Accepts raw CSV data and calculates the missing 'Impact' metrics required by the dashboard.
    """
    df = df.copy()
    
    # A. Date Handling
    cols_to_date = [c for c in df.columns if 'date' in c.lower() or 'start' in c.lower() or 'end' in c.lower()]
    for c in cols_to_date:
        try: df[c] = pd.to_datetime(df[c])
        except: pass

    # Sort for shift calculations
    if 'Injury_Start' in df.columns:
        df = df.sort_values(['Player', 'Injury_Start'])
        df['Month'] = df['Injury_Start'].dt.month_name()
    
    # B. Missing Stats Calculation
    # 1. Injury Duration
    if 'Injury_Duration' not in df.columns and 'Injury_End' in df.columns:
        df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days.clip(lower=1)
    
    # 2. Rating Shifting (Before/After)
    if 'Rating' in df.columns:
        df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1).fillna(df['Rating'])
        df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1).fillna(df['Rating'])
        df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
    
    # 3. Team Impact (Drops and Recoveries)
    if {'Team_Goals_Before', 'Team_Goals_During'}.issubset(df.columns):
        df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
        
        # Avoid division by zero
        if 'Injury_Duration' in df.columns:
            df['Impact_Index'] = df['Team_Performance_Drop'] / df['Injury_Duration']
    else:
        # Fallbacks if columns are totally missing to prevent crashes
        df['Team_Performance_Drop'] = 0
        df['Impact_Index'] = 0

    if 'Team_Goals_After' not in df.columns:
        df['Team_Goals_After'] = df.get('Team_Goals_During', 0) # Fallback
    
    # Fill remaining NaNs for safety
    return df.fillna(0)

# ---------------------
# 3. Sidebar: Upload & Settings
# ---------------------
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df = preprocess_csv(raw_df)
    st.sidebar.success(f"CSV Loaded: {len(df)} rows")
else:
    df = preprocess_csv(generate_demo_data())
    st.sidebar.info("Using **Demo Data**")

st.sidebar.divider()
st.sidebar.header("🎨 Visuals")
# Global Filters
clubs = st.sidebar.multiselect("Club", options=df['Club'].unique(), default=df['Club'].unique())
injuries = st.sidebar.multiselect("Injury Type", options=df['Injury_Type'].unique(), default=df['Injury_Type'].unique())
# Mode Selection
global_mode = st.sidebar.radio("Global Mode", ["Plotly", "Matplotlib"], horizontal=True)
global_style = st.sidebar.selectbox("Seaborn Style", ["whitegrid", "darkgrid", "ticks"])

# Filter Data
dff = df[df['Club'].isin(clubs) & df['Injury_Type'].isin(injuries)]

# ---------------------
# 4. Helpers for Visuals
# ---------------------
def tab_config(key_prefix):
    """Mini helper to handle per-tab override UI"""
    c1, c2 = st.columns([1, 4])
    mode = c1.selectbox("Mode", ["Global", "Plotly", "Matplotlib"], key=f"{key_prefix}_m")
    active_mode = global_mode if mode == "Global" else mode
    
    # Set Seaborn style if in Matplotlib mode
    if active_mode == "Matplotlib":
        sns.set_theme(style=global_style)
        plt.rcParams['figure.figsize'] = (10, 5)
    return active_mode

def render(fig, mode):
    if mode == "Plotly": st.plotly_chart(fig, use_container_width=True)
    else: st.pyplot(fig)

# ---------------------
# 5. Main Dashboard Tabs
# ---------------------
# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Injuries", len(dff))
k2.metric("Avg Rating", f"{dff['Rating'].mean():.2f}")
k3.metric("Avg Team Goal Drop", f"{dff['Team_Performance_Drop'].mean():.2f}")
k4.metric("Avg Injury Duration", f"{dff.get('Injury_Duration', pd.Series([0])).mean():.1f} Days")

tabs = st.tabs(["📊 Trends", "📉 Impact", "🏟 Club Deep Dive", "🔬 Raw Data"])

# --- Tab 1: Trends ---
with tabs[0]:
    mode = tab_config("tab1")
    st.subheader("Monthly Injury Frequency")
    
    monthly = dff.groupby('Month', as_index=False)['Player'].count()
    if mode == "Plotly":
        fig = px.bar(monthly, x='Month', y='Player', title="Injuries by Month", color='Player')
        render(fig, mode)
    else:
        fig, ax = plt.subplots()
        sns.barplot(data=monthly, x='Month', y='Player', ax=ax, palette="viridis")
        ax.set_title("Injuries by Month")
        render(fig, mode)

# --- Tab 2: Player Impact ---
with tabs[1]:
    mode = tab_config("tab2")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("#### Team Performance Drop vs. Injury Type")
        if mode == "Plotly":
            fig = px.box(dff, x="Injury_Type", y="Team_Performance_Drop", color="Injury_Type")
            render(fig, mode)
        else:
            fig, ax = plt.subplots()
            sns.boxplot(data=dff, x="Injury_Type", y="Team_Performance_Drop", ax=ax)
            plt.xticks(rotation=45)
            render(fig, mode)
            
    with c2:
        st.write("#### Player Rating Change (Post Injury)")
        agg = dff.groupby("Player")['Performance_Change'].mean().reset_index().sort_values('Performance_Change').head(10)
        if mode == "Plotly":
            fig = px.bar(agg, y="Player", x="Performance_Change", orientation='h', color="Performance_Change", color_continuous_scale="RdBu")
            render(fig, mode)
        else:
            fig, ax = plt.subplots()
            sns.barplot(data=agg, y="Player", x="Performance_Change", ax=ax, palette="vlag")
            render(fig, mode)

# --- Tab 3: Club Deep Dive ---
with tabs[2]:
    mode = tab_config("tab3")
    
    st.write("#### Average Team Goals: Before vs During vs After")
    club_stats = dff.groupby("Club")[['Team_Goals_Before', 'Team_Goals_During', 'Team_Goals_After']].mean().reset_index()
    melted = club_stats.melt(id_vars="Club", var_name="Period", value_name="Avg Goals")
    
    if mode == "Plotly":
        fig = px.bar(melted, x="Club", y="Avg Goals", color="Period", barmode="group", title="Goal Analysis")
        render(fig, mode)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=melted, x="Club", y="Avg Goals", hue="Period", ax=ax)
        plt.xticks(rotation=45)
        render(fig, mode)

# --- Tab 4: Raw Data & Download ---
with tabs[3]:
    st.dataframe(dff.sort_values("Injury_Start", ascending=False), use_container_width=True)
    
    # Template Generator for Users
    example_cols = ["Player", "Club", "Injury_Start", "Injury_End", "Rating", 
                   "Team_Goals_Before", "Team_Goals_During", "Injury_Type"]
    template_csv = pd.DataFrame(columns=example_cols).to_csv(index=False)
    
    st.download_button("📥 Download CSV Template", template_csv, "template.csv", "text/csv")
