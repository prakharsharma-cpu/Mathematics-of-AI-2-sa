# ==================================
# ⚽ Player Injury Impact Dashboard (Date-Optional)
# Fixed: 'Injury_Start' is now OPTIONAL. App works without it.
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
st.markdown("Interactive dashboard. **Upload your CSV**. (Date columns are optional!)")

# ---------------------
# Helper Utilities
# ---------------------
def get_mode(per_tab_choice, global_mode):
    return per_tab_choice if per_tab_choice in ["Plotly", "Matplotlib"] else global_mode

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

def generate_sample_data():
    """Generates robust random simulation data."""
    np.random.seed(42)
    num_rows = 150
    players = [f"Player_{i}" for i in range(1, 15)]
    clubs = [f"Club_{i}" for i in range(1, 5)]
    dates = pd.date_range("2021-01-01", "2023-01-01", freq="W")
    
    data = {
        "Player": np.random.choice(players, num_rows),
        "Club": np.random.choice(clubs, num_rows),
        # Sample data HAS date, but user upload doesn't need to.
        "Injury_Start": np.random.choice(dates, num_rows),
        "Injury_Type": np.random.choice(["Hamstring", "Knee", "Ankle", "Calf", "Back"], num_rows),
        "Rating": np.random.uniform(5, 9, num_rows).round(1),
        "Team_Goals_Before": np.random.randint(10, 30, num_rows),
        "Team_Goals_During": np.random.randint(5, 20, num_rows)
    }
    
    # Explicitly calculate Duration
    duration = np.random.randint(5, 60, num_rows)
    df = pd.DataFrame(data)
    df["Injury_Duration"] = duration
    return df

# ---------------------
# SMART DATA LOADING
# ---------------------
def load_and_standardize_csv(uploaded_file):
    try:
        # Flexible Reader
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip() # Remove hidden spaces

        # Rename map for variations, BUT NOT strictly required
        col_map = {
            'player': 'Player', 'name': 'Player',
            'club': 'Club', 'team': 'Club',
            'rating': 'Rating',
            'injury_type': 'Injury_Type',
            'duration': 'Injury_Duration', 'days': 'Injury_Duration',
            'injury_start': 'Injury_Start', 'date': 'Injury_Start' # Still map if it exists
        }

        # Case-insensitive Rename
        actual_rename = {}
        for col in df.columns:
            clean_col = col.lower().replace('.', '_').replace(' ', '_')
            if clean_col in col_map:
                actual_rename[col] = col_map[clean_col]
        
        df = df.rename(columns=actual_rename)
        return df
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None

# ---------------------
# Sidebar & Load Logic
# ---------------------
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_and_standardize_csv(uploaded_file)
else:
    df = generate_sample_data()

# ---------------------
# CRITICAL FIX: Safe Processing
# ---------------------
if df is not None:
    # 1. Player column is the ONLY strict requirement
    if 'Player' not in df.columns:
        # If Player missing, create dummy players
        df['Player'] = [f"Record_{i}" for i in range(len(df))]
        st.sidebar.warning("⚠️ Column 'Player' not found. Using Row IDs.")

    # 2. Handle MISSING 'Injury_Start'
    has_date = False
    if 'Injury_Start' not in df.columns:
        # User uploaded data WITHOUT date. 
        # Create a Dummy Date (Index based) to prevent crashing charts
        df['Injury_Start'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        df['Is_Fake_Date'] = True
    else:
        # Attempt conversion
        df['Injury_Start'] = pd.to_datetime(df['Injury_Start'], dayfirst=True, errors='coerce')
        df['Injury_Start'].fillna(pd.Timestamp("2023-01-01"), inplace=True)
        has_date = True
        df['Is_Fake_Date'] = False

    # 3. Handle numeric columns safely
    numeric_defaults = {
        'Rating': 7.0,
        'Team_Goals_Before': 0,
        'Team_Goals_During': 0,
        'Goals': 0,
        'Age': 25,
        'Injury_Duration': 7 # Default to 1 week if duration missing
    }
    
    for col, val in numeric_defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    # 4. Final Feature Engineering
    df['Month'] = df['Injury_Start'].dt.month_name()
    
    # Only calculate meaningful metrics if possible
    df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']

    if 'Club' not in df.columns: df['Club'] = 'All'
    if 'Injury_Type' not in df.columns: df['Injury_Type'] = 'General'

    # ---------------------
    # Filters
    # ---------------------
    st.sidebar.divider()
    
    clubs = sorted(df['Club'].astype(str).unique())
    sel_clubs = st.sidebar.multiselect("Club", clubs, default=clubs)
    
    inj_types = sorted(df['Injury_Type'].astype(str).unique())
    sel_injuries = st.sidebar.multiselect("Injury Type", inj_types, default=inj_types)
    
    # Filter DataFrame
    filtered_df = df[
        (df['Club'].astype(str).isin(sel_clubs)) &
        (df['Injury_Type'].astype(str).isin(sel_injuries))
    ]

    # Global Style
    g_mode = st.sidebar.radio("Display Mode", ["Plotly", "Matplotlib"], horizontal=True)
    g_style = "Modern Clean" 
    if g_mode == "Matplotlib":
        g_style = st.sidebar.selectbox("Theme", ["Modern Clean", "Classic Analytics"])

    # ---------------------
    # Dashboard Tabs
    # ---------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Injuries Recorded", len(filtered_df))
    k2.metric("Avg Duration", f"{filtered_df['Injury_Duration'].mean():.0f} Days")
    k3.metric("Avg Perf. Drop", f"{filtered_df['Team_Performance_Drop'].mean():.1f}")
    
    if not has_date and uploaded_file:
         st.warning("⚠️ NOTE: Your CSV did not contain an 'Injury_Start' date column. Timeline charts below use artificial dates.")

    tabs = st.tabs(["📊 Trends", "🏥 Injury Types", "📉 Player Impact", "📋 Data"])

    # --- TAB 1: TRENDS (Handles missing date gracefully) ---
    with tabs[0]:
        st.subheader("Seasonal & Monthly Trends")
        
        if df['Is_Fake_Date'].all() and uploaded_file:
            st.info("ℹ️ Trends chart skipped (No date data in CSV).")
        else:
            # Determine order for months
            months = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
            m_data = filtered_df['Month'].value_counts().reindex(months).reset_index()
            m_data.columns = ['Month', 'Count']
            
            if g_mode == "Plotly":
                fig = px.bar(m_data, x="Month", y="Count", title="Injury Frequency by Month")
                render_plotly(fig)
            else:
                apply_seaborn_style(g_style)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=m_data, x="Month", y="Count", color="steelblue", ax=ax)
                plt.xticks(rotation=45)
                render_matplotlib(fig)

    # --- TAB 2: INJURY TYPES ---
    with tabs[1]:
        st.subheader("Distribution of Injury Types")
        counts = filtered_df['Injury_Type'].value_counts().reset_index()
        counts.columns = ['Injury_Type', 'Count']

        col1, col2 = st.columns(2)
        with col1:
             if g_mode == "Plotly":
                fig = px.pie(counts, values='Count', names='Injury_Type', hole=0.4)
                render_plotly(fig)
             else:
                apply_seaborn_style(g_style)
                fig, ax = plt.subplots()
                ax.pie(counts['Count'], labels=counts['Injury_Type'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
                render_matplotlib(fig)
        with col2:
             # Impact by Type
             impact = filtered_df.groupby('Injury_Type')['Team_Performance_Drop'].mean().reset_index()
             if g_mode == "Plotly":
                fig2 = px.bar(impact, x="Team_Performance_Drop", y="Injury_Type", orientation='h', title="Goal Drop by Injury")
                render_plotly(fig2)
             else:
                fig2, ax = plt.subplots()
                sns.barplot(data=impact, x="Team_Performance_Drop", y="Injury_Type", ax=ax, palette="Reds")
                render_matplotlib(fig2)

    # --- TAB 3: PLAYER IMPACT ---
    with tabs[2]:
        st.subheader("Player Analytics")
        # Ensure we have Player data to group
        p_stats = filtered_df.groupby('Player').agg({
            'Injury_Duration': 'sum', 
            'Rating': 'mean',
            'Team_Performance_Drop': 'mean'
        }).reset_index().sort_values('Injury_Duration', ascending=False).head(15)

        if g_mode == "Plotly":
             fig3 = px.scatter(p_stats, x="Team_Performance_Drop", y="Rating", size="Injury_Duration", hover_name="Player",
                               color="Player", title="Player Impact Bubble Chart")
             render_plotly(fig3)
        else:
             apply_seaborn_style(g_style)
             fig3, ax = plt.subplots(figsize=(9,5))
             sns.scatterplot(data=p_stats, x="Team_Performance_Drop", y="Rating", size="Injury_Duration", hue="Player", ax=ax, sizes=(20, 200))
             ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
             render_matplotlib(fig3)

    # --- TAB 4: DATA ---
    with tabs[3]:
        st.write("Current Filtered Data")
        st.dataframe(filtered_df)
