# ==================================
# ⚽ Player Injury Impact Dashboard (Smart Column Fix)
# Auto-detects column names to prevent KeyErrors
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
st.markdown("Interactive dashboard with Plotly + Matplotlib. **Upload a CSV** or use the demo data.")

# ---------------------
# Helper Utilities
# ---------------------
def get_mode(per_tab_choice, global_mode):
    return per_tab_choice if per_tab_choice in ["Plotly", "Matplotlib"] else global_mode

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
    """Generates robust random simulation data."""
    np.random.seed(42)
    num_rows = 200
    players = [f"Player_{i}" for i in range(1, 15)]
    clubs = [f"Club_{i}" for i in range(1, 5)]
    dates = pd.date_range("2021-01-01", "2023-01-01", freq="W")
    
    data = {
        "Player": np.random.choice(players, num_rows),
        "Club": np.random.choice(clubs, num_rows),
        "Injury_Start": np.random.choice(dates, num_rows),
        "Injury_Type": np.random.choice(["Hamstring", "Knee", "Ankle", "Calf", "Back"], num_rows),
        "Rating": np.random.uniform(5, 9, num_rows).round(1),
        "Team_Goals_Before": np.random.randint(10, 30, num_rows),
        "Team_Goals_During": np.random.randint(5, 20, num_rows),
        "Team_Goals_After": np.random.randint(10, 30, num_rows),
        "Age": np.random.randint(19, 34, num_rows)
    }
    
    # Generate explicit end date for demo
    duration = np.random.randint(5, 60, num_rows)
    data["Injury_End"] = [start + pd.Timedelta(days=int(d)) for start, d in zip(data["Injury_Start"], duration)]
    
    return pd.DataFrame(data).sort_values("Injury_Start")

# ---------------------
# SMART DATA LOADING FUNCTION
# ---------------------
def load_and_standardize_csv(uploaded_file):
    try:
        # 1. Flexible Reader (Auto-detects ; vs , separator)
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # 2. Strip whitespace from header
        df.columns = df.columns.str.strip()

        # 3. Flexible Column Mapping (Case-insensitive matching)
        # We look for columns that 'look like' the target and rename them to standard key
        col_map = {
            'injury_start': 'Injury_Start', 
            'date': 'Injury_Start', 
            'start_date': 'Injury_Start',
            'injury start': 'Injury_Start',
            
            'injury_end': 'Injury_End', 
            'end_date': 'Injury_End',
            'injury end': 'Injury_End',
            
            'player': 'Player',
            'name': 'Player',
            'player_name': 'Player',
            
            'club': 'Club',
            'team': 'Club',
            
            'rating': 'Rating',
            'avg_rating': 'Rating'
        }

        # Create rename dictionary by checking current columns lower-cased
        actual_rename = {}
        for col in df.columns:
            col_lower = col.lower().replace('.', '_') # standardized key
            if col_lower in col_map:
                actual_rename[col] = col_map[col_lower]
        
        df = df.rename(columns=actual_rename)

        return df
        
    except Exception as e:
        st.error(f"❌ Failed to parse CSV: {e}")
        return None

# ---------------------
# Sidebar
# ---------------------
st.sidebar.header("📂 Data Configuration")

sample_df = generate_sample_data()
csv_template = sample_df.head(5).to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    "⬇️ Download CSV Template", 
    data=csv_template, 
    file_name="template.csv",
    help="Use this headers: Player, Club, Injury_Start, Injury_End, Rating, Team_Goals_Before, Team_Goals_During"
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
data_source = "demo"
df = None

# ---------------------
# Load Data
# ---------------------
if uploaded_file:
    df = load_and_standardize_csv(uploaded_file)
    if df is not None:
        data_source = "upload"
else:
    df = sample_df.copy()

# ---------------------
# Data Processing (Robust)
# ---------------------
if df is not None:
    # 1. Validate Crucial Column existence
    required_cols = ['Injury_Start', 'Player']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"⚠️ Critical Error: The following required columns are missing (or misspelled): **{missing_cols}**")
        st.write("Found columns:", list(df.columns))
        st.stop()
    
    # 2. Ensure DateTime (Safely)
    try:
        df['Injury_Start'] = pd.to_datetime(df['Injury_Start'], dayfirst=True, errors='coerce')
        # Drop rows where Date is invalid (NaT)
        invalid_dates = df['Injury_Start'].isna().sum()
        if invalid_dates > 0:
            st.sidebar.warning(f"⚠️ Found {invalid_dates} rows with invalid dates. They have been excluded.")
        df = df.dropna(subset=['Injury_Start'])
        
        if 'Injury_End' in df.columns:
            df['Injury_End'] = pd.to_datetime(df['Injury_End'], dayfirst=True, errors='coerce')
        else:
            # Create dummy end date if missing
            df['Injury_End'] = df['Injury_Start'] + pd.Timedelta(days=14)

    except Exception as e:
        st.error(f"Date conversion error: {e}")
        st.stop()

    # 3. Create Missing Numeric Columns with Defaults
    expected_numerics = {
        'Rating': 7.0, 
        'Team_Goals_Before': 10, 
        'Team_Goals_During': 10, 
        'Team_Goals_After': 10,
        'Goals': 0,
        'Age': 25
    }
    
    for col, default_val in expected_numerics.items():
        if col not in df.columns:
            df[col] = default_val # fill constant
        else:
            # Convert to number, coerce invalid strings to default
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)

    # 4. Feature Engineering
    # Calculate duration
    if 'Injury_Duration' not in df.columns:
        df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days
    
    # Fix negative durations
    df['Injury_Duration'] = df['Injury_Duration'].clip(lower=1)
    df['Month'] = df['Injury_Start'].dt.month_name()
    
    # Drop calculation
    df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
    
    # Ratings Calc (Sorting Essential)
    df = df.sort_values(by=['Player', 'Injury_Start'])
    df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
    df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
    
    if 'Status' not in df.columns:
        df['Status'] = 'Recovered'
    if 'Injury_Type' not in df.columns:
        df['Injury_Type'] = 'Unspecified'

    # ---------------------
    # Filters
    # ---------------------
    st.sidebar.markdown("---")
    
    # safe list comprehension in case of mixed types
    all_clubs = sorted([str(x) for x in df['Club'].unique()]) if 'Club' in df.columns else ['Unknown']
    all_players = sorted([str(x) for x in df['Player'].unique()])
    all_types = sorted([str(x) for x in df['Injury_Type'].unique()])

    sel_club = st.sidebar.multiselect("Club", all_clubs, default=all_clubs)
    sel_player = st.sidebar.multiselect("Player", all_players, default=all_players)
    sel_type = st.sidebar.multiselect("Injury Type", all_types, default=all_types)

    # Filtering Logic
    filtered_df = df[
        (df['Club'].astype(str).isin(sel_club)) &
        (df['Player'].astype(str).isin(sel_player)) &
        (df['Injury_Type'].astype(str).isin(sel_type))
    ]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters.")
        st.stop()
        
    # Global Settings
    st.sidebar.markdown("---")
    g_mode = st.sidebar.radio("Display Mode", ["Plotly", "Matplotlib"])
    g_style = "Modern Clean"
    if g_mode == "Matplotlib":
        g_style = st.sidebar.selectbox("Matplotlib Theme", ["Modern Clean", "Classic Analytics"])

    # ---------------------
    # VISUALIZATION LOGIC
    # ---------------------
    
    # --- Top KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Injuries", len(filtered_df))
    k2.metric("Avg Duration", f"{filtered_df['Injury_Duration'].mean():.0f} days")
    k3.metric("Avg Perf Drop", f"{filtered_df['Team_Performance_Drop'].mean():.1f}")
    k4.metric("Active Players", filtered_df['Player'].nunique())

    # --- TABS ---
    t_trend, t_player, t_stats, t_data = st.tabs(["📊 Trends", "🏃 Player Analysis", "🔬 Deep Stats", "🧾 Raw Data"])

    # === Tab 1: Trends ===
    with t_trend:
        # Determine month order correctly
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        trends = filtered_df['Month'].value_counts().reindex(month_order).reset_index()
        trends.columns = ['Month', 'Count']

        st.subheader("Seasonal Injury Trends")
        if g_mode == "Plotly":
            fig = px.bar(trends, x='Month', y='Count', title="Injuries by Month", template="plotly_white")
            render_plotly(fig)
        else:
            apply_seaborn_style(g_style)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=trends, x='Month', y='Count', color="steelblue", ax=ax)
            plt.xticks(rotation=45)
            render_matplotlib(fig)

    # === Tab 2: Player Analysis ===
    with t_player:
        st.subheader("Individual Player Impact")
        
        # Safe aggregation
        impact = filtered_df.groupby("Player").agg({
            'Team_Performance_Drop': 'mean',
            'Rating': 'mean',
            'Injury_Duration': 'count'
        }).rename(columns={'Injury_Duration':'Count'}).reset_index().sort_values("Team_Performance_Drop", ascending=False).head(10)

        if g_mode == "Plotly":
            fig = px.scatter(impact, x="Team_Performance_Drop", y="Rating", size="Count", color="Player",
                             title="Avg Goal Drop vs Player Rating (Size = Num Injuries)")
            render_plotly(fig)
        else:
            apply_seaborn_style(g_style)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=impact, x="Team_Performance_Drop", y="Rating", size="Count", hue="Player", ax=ax)
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            render_matplotlib(fig)
            
        st.caption("Higher 'Team Performance Drop' indicates the team scores significantly less when this player is injured.")

    # === Tab 3: Deep Stats ===
    with t_stats:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Impact by Injury Type**")
            inj_impact = filtered_df.groupby('Injury_Type')['Team_Performance_Drop'].mean().sort_values()
            st.bar_chart(inj_impact)
            
        with c2:
            st.write("**Recovery Time Distribution**")
            if g_mode == "Plotly":
                fig = px.histogram(filtered_df, x="Injury_Duration", nbins=15, title="Days Out Distribution")
                render_plotly(fig)
            else:
                fig, ax = plt.subplots()
                sns.histplot(filtered_df['Injury_Duration'], bins=15, kde=True, ax=ax)
                render_matplotlib(fig)

    # === Tab 4: Raw Data ===
    with t_data:
        st.dataframe(filtered_df)
        csv_dl = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data", data=csv_dl, file_name="analysis.csv", mime="text/csv")
