# ============
# ⚽ FootLens 
# ============
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="⚽ FootLens", layout="wide", initial_sidebar_state="expanded")
st.title("⚽ FootLens")
st.markdown("Upload your CSV and explore injury impact analytics with automated EDA and interactive visualizations. "
            "If you don't have a CSV, use the sample CSV download link in the sidebar.")

# ---------------------
# Helpers & Utilities
# ---------------------
@st.cache_data
def load_csv_bytes(uploaded_file):
    return pd.read_csv(uploaded_file)

def safe_mean(series):
    try:
        return float(np.nanmean(series))
    except Exception:
        return np.nan

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
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plotly render error: {e}")

def render_matplotlib(fig):
    try:
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Matplotlib render error: {e}")

def ensure_nonempty(df):
    if df is None or df.shape[0] == 0:
        st.warning("No records to display after applying filters.")
        return False
    return True

def sample_csv_text():
    csv = """Player,Club,Injury_Type,Injury_Start,Injury_End,Team_Goals_Before,Team_Goals_During,Team_Goals_After,Avg_Rating_Before,Avg_Rating_After,Rating
John Doe,Example FC,Hamstring,2024-09-01,2024-09-21,1.2,0.6,1.0,7.1,6.9,6.9
Jane Smith,Example FC,Ankle,2024-10-05,2024-10-20,1.5,1.0,1.4,7.5,7.3,7.4
Alex Roe,Another United,ACL,2024-08-01,2024-11-01,1.0,0.2,0.8,6.9,7.0,6.95
"""
    return csv

# ---------------------
# Sidebar: Upload + Settings
# ---------------------
st.sidebar.header("🔍 Upload & Settings")
st.sidebar.markdown("Upload a CSV with injury & performance data. Recommended columns: "
                    "`Player, Club, Injury_Type, Injury_Start, Injury_End, Team_Goals_Before, Team_Goals_During, Team_Goals_After, Avg_Rating_Before, Avg_Rating_After, Rating`.")

sample_csv = sample_csv_text()
st.sidebar.download_button("⬇️ Download sample CSV", data=sample_csv, file_name="footlens_sample.csv", mime="text/csv")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ---------------------
# Load data
# ---------------------
if uploaded_file:
    try:
        df = load_csv_bytes(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} records.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.sidebar.info("No file uploaded — using sample dataset for demonstration.")
    df = pd.read_csv(StringIO(sample_csv))

# ---------------------
# Basic preprocessing
# ---------------------
# defensive copy
df = df.copy()

# drop exact duplicates
df.drop_duplicates(inplace=True)

# standardize column names (strip)
df.columns = [c.strip() for c in df.columns]

# convert date columns if present
for date_col in ["Injury_Start","Injury_End"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# fill numeric NaNs with column mean (but keep NaN for derived metrics if needed)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().any():
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
# Sidebar: Filters + Mode/Style
# ---------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters & Visualization Settings")

filter_club = st.sidebar.multiselect("Club", options=sorted(df['Club'].dropna().unique()) if 'Club' in df.columns else [], default=sorted(df['Club'].dropna().unique()) if 'Club' in df.columns else [])
filter_player = st.sidebar.multiselect("Player", options=sorted(df['Player'].dropna().unique()) if 'Player' in df.columns else [], default=sorted(df['Player'].dropna().unique()) if 'Player' in df.columns else [])
filter_injury = st.sidebar.multiselect("Injury Type", options=sorted(df['Injury_Type'].dropna().unique()) if 'Injury_Type' in df.columns else [], default=sorted(df['Injury_Type'].dropna().unique()) if 'Injury_Type' in df.columns else [])

global_mode = st.sidebar.radio("🧭 Global Visualization Mode", ["Plotly","Matplotlib"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Seaborn Theme (Matplotlib)")
global_style = st.sidebar.radio("Global Seaborn Style", ["Modern Clean","Classic Analytics"], index=0)

# Apply filters (defensive)
filtered_df = df.copy()
if 'Club' in df.columns and filter_club:
    filtered_df = filtered_df[filtered_df['Club'].isin(filter_club)]
if 'Player' in df.columns and filter_player:
    filtered_df = filtered_df[filtered_df['Player'].isin(filter_player)]
if 'Injury_Type' in df.columns and filter_injury:
    filtered_df = filtered_df[filtered_df['Injury_Type'].isin(filter_injury)]

# ---------------------
# KPIs
# ---------------------
kpi1, kpi2, kpi3 = st.columns(3)
avg_rating = safe_mean(filtered_df['Rating']) if 'Rating' in filtered_df.columns else np.nan
avg_drop = safe_mean(filtered_df['Team_Performance_Drop']) if 'Team_Performance_Drop' in filtered_df.columns else np.nan
total_injuries = len(filtered_df)
kpi1.metric("⚽ Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
kpi2.metric("💥 Avg Team Performance Drop", f"{avg_drop:.2f}" if not np.isnan(avg_drop) else "N/A")
kpi3.metric("🩹 Total Injuries", f"{total_injuries}")

st.markdown("---")

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
    st.write("First 10 rows (after filters):")
    if ensure_nonempty(filtered_df):
        st.dataframe(filtered_df.head(10), use_container_width=True)
        st.write(f"**Total Records:** {filtered_df.shape[0]} | **Columns:** {filtered_df.shape[1]}")
        st.markdown("---")
        st.subheader("📈 Statistical Summary")
        try:
            st.dataframe(filtered_df.describe(include='all').transpose())
        except Exception as e:
            st.warning(f"Could not compute full describe: {e}")
    else:
        st.info("Upload or filter to see dataset preview.")

# =====================================
# Tab 1: Full EDA & Trends
# =====================================
with tabs[1]:
    st.subheader("📊 Automated EDA & Core Trends")
    tab_mode = st.selectbox("Visualization mode for this tab (override)", ["Auto","Plotly","Matplotlib"], index=0, key="eda_mode")
    tab_style = st.selectbox("Seaborn style for this tab (override)", ["Auto","Modern Clean","Classic Analytics"], index=0, key="eda_style")
    mode = get_mode(tab_mode if tab_mode != "Auto" else None, global_mode)
    style = get_style(tab_style if tab_style != "Auto" else None, global_style)

    if not ensure_nonempty(filtered_df):
        st.stop()

    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(exclude=np.number).columns.tolist()

    # Correlation Heatmap
    if numeric_cols:
        st.markdown("### 🔗 Correlation Heatmap")
        corr = filtered_df[numeric_cols].corr()
        if mode=="Plotly":
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation matrix")
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title("Correlation matrix")
            plt.tight_layout(); render_matplotlib(fig)
    else:
        st.info("No numeric columns available for correlation heatmap.")

    # Missing Values Heatmap
    st.markdown("### ❗ Missing Values Heatmap")
    if filtered_df.isnull().sum().sum() > 0:
        if mode=="Plotly":
            z = filtered_df.isnull().astype(int).T.values
            x = list(filtered_df.index.astype(str))
            y = list(filtered_df.columns)
            fig = ff.create_annotated_heatmap(z=z, x=x, y=y, annotation_text=None, showscale=True)
            fig.update_layout(title="Missing values (1 = missing)", xaxis_title="Row index", yaxis_title="Columns")
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(filtered_df.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title("Missing values heatmap")
            plt.tight_layout(); render_matplotlib(fig)
    else:
        st.info("No missing values detected.")

    # Numeric distributions
    if numeric_cols:
        st.markdown("### 📊 Numeric Feature Distributions")
        for col in numeric_cols:
            with st.expander(f"Distribution: {col}", expanded=False):
                if mode=="Plotly":
                    fig = px.histogram(filtered_df, x=col, nbins=30, title=f"Distribution of {col}")
                    render_plotly(fig)
                else:
                    apply_seaborn_style(style)
                    fig, ax = plt.subplots(figsize=(6,3))
                    sns.histplot(filtered_df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    plt.tight_layout(); render_matplotlib(fig)
    else:
        st.info("No numeric columns to show distributions for.")

    # Boxplots by categorical features
    if numeric_cols and categorical_cols:
        st.markdown("### 📦 Boxplots by Categorical Features")
        # to avoid explosion, limit combinations
        max_combinations = 12
        combos_shown = 0
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if combos_shown >= max_combinations:
                    break
                with st.expander(f"{num_col} by {cat_col}", expanded=False):
                    if mode=="Plotly":
                        try:
                            fig = px.box(filtered_df, x=cat_col, y=num_col, color=cat_col, title=f"{num_col} by {cat_col}")
                            render_plotly(fig)
                        except Exception as e:
                            st.error(f"Plotly box error: {e}")
                    else:
                        apply_seaborn_style(style)
                        fig, ax = plt.subplots(figsize=(8,4))
                        try:
                            sns.boxplot(data=filtered_df, x=cat_col, y=num_col, ax=ax)
                            ax.set_title(f"{num_col} by {cat_col}")
                            plt.xticks(rotation=45); plt.tight_layout()
                            render_matplotlib(fig)
                        except Exception as e:
                            st.error(f"Matplotlib box error: {e}")
                combos_shown += 1
            if combos_shown >= max_combinations:
                st.info("Limited boxplot combinations to keep UI responsive.")
                break
    else:
        st.info("Not enough numeric/categorical columns to plot boxplots.")

    # Top injury types by team performance drop
    if 'Injury_Type' in filtered_df.columns and 'Team_Performance_Drop' in filtered_df.columns:
        st.markdown("### 🩹 Top 10 Injuries by Avg Team Performance Drop")
        top_injuries = filtered_df.groupby('Injury_Type')['Team_Performance_Drop'].mean().sort_values(ascending=False).head(10).reset_index()
        if mode=="Plotly":
            fig = px.bar(top_injuries, x='Team_Performance_Drop', y='Injury_Type', orientation='h', color='Team_Performance_Drop', title="Top injuries by team performance drop", labels={'Team_Performance_Drop':'Avg Drop','Injury_Type':'Injury'})
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=top_injuries, y='Injury_Type', x='Team_Performance_Drop', ax=ax)
            ax.set_title("Top injuries by team performance drop")
            plt.tight_layout(); render_matplotlib(fig)
    else:
        st.info("Columns 'Injury_Type' or 'Team_Performance_Drop' missing; cannot compute top injuries by drop.")

# =====================================
# Tab 2: Player Impact
# =====================================
with tabs[2]:
    st.subheader("📈 Player Impact Overview")
    if not ensure_nonempty(filtered_df):
        st.stop()

    # summarize by player
    if 'Player' in filtered_df.columns:
        player_summary = filtered_df.groupby('Player').agg(
            Injuries=('Player','count'),
            Avg_Injury_Duration=('Injury_Duration', lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            Avg_Perf_Change=('Performance_Change', lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            Avg_Rating=('Rating', lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan)
        ).reset_index().sort_values('Injuries', ascending=False)
        st.dataframe(player_summary.head(50), use_container_width=True)

        # Top impacted players (by avg drop or perf change)
        st.markdown("### Players with largest average performance change (drop)")
        if 'Performance_Change' in filtered_df.columns:
            tmp = player_summary.sort_values('Avg_Perf_Change').head(10).reset_index(drop=True)
            fig = px.bar(tmp, x='Avg_Perf_Change', y='Player', orientation='h', title="Players with largest negative performance change")
            render_plotly(fig)
        else:
            st.info("No 'Performance_Change' column available.")
    else:
        st.info("No 'Player' column in dataset to compute player-level impact.")

# =====================================
# Tab 3: Club Analysis
# =====================================
with tabs[3]:
    st.subheader("🔥 Club Analysis")
    if not ensure_nonempty(filtered_df):
        st.stop()

    if 'Club' in filtered_df.columns:
        club_summary = filtered_df.groupby('Club').agg(
            Injuries=('Club','count'),
            Avg_Injury_Duration=('Injury_Duration', lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            Avg_Team_Drop=('Team_Performance_Drop', lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan)
        ).reset_index().sort_values('Injuries', ascending=False)
        st.dataframe(club_summary, use_container_width=True)

        # top clubs by injuries
        fig = px.bar(club_summary.sort_values('Injuries',ascending=False).head(15), x='Injuries', y='Club', orientation='h', title="Top clubs by injury count")
        render_plotly(fig)
    else:
        st.info("No 'Club' column available to show club-level analysis.")

# =====================================
# Tab 4: Injury Analysis
# =====================================
with tabs[4]:
    st.subheader("🔬 Injury Analysis & Time Trends")
    if not ensure_nonempty(filtered_df):
        st.stop()

    # Injury counts by type
    if 'Injury_Type' in filtered_df.columns:
        inj_counts = filtered_df['Injury_Type'].value_counts().reset_index().rename(columns={'index':'Injury_Type','Injury_Type':'Count'})
        st.markdown("### Injury counts by type")
        fig = px.bar(inj_counts.head(30), x='Count', y='Injury_Type', orientation='h', title="Injury counts")
        render_plotly(fig)
    else:
        st.info("No 'Injury_Type' column to show injury distribution.")

    # Injury duration distribution
    if 'Injury_Duration' in filtered_df.columns and not filtered_df['Injury_Duration'].isnull().all():
        st.markdown("### Injury duration distribution")
        fig = px.histogram(filtered_df, x='Injury_Duration', nbins=30, title="Injury duration (days)")
        render_plotly(fig)
    else:
        st.info("No 'Injury_Duration' values to show distribution.")

    # Monthly trend of injuries (if date present)
    if 'Month' in filtered_df.columns and not filtered_df['Month'].isnull().all():
        monthly = filtered_df.groupby('Month').size().reset_index(name='Count').sort_values('Month')
        st.markdown("### Monthly injury counts")
        fig = px.line(monthly, x='Month', y='Count', markers=True, title="Injuries by month")
        render_plotly(fig)
    else:
        st.info("No date information to compute monthly trends (Injury_Start missing).")

# =====================================
# Tab 5: Player Deep Dive
# =====================================
with tabs[5]:
    st.subheader("🔎 Player Deep Dive")
    if not ensure_nonempty(filtered_df):
        st.stop()

    if 'Player' in filtered_df.columns:
        player_options = sorted(filtered_df['Player'].dropna().unique())
        selected_player = st.selectbox("Select player", options=player_options)
        player_df = filtered_df[filtered_df['Player']==selected_player]
        st.write(f"Showing {len(player_df)} record(s) for **{selected_player}**")

        with st.expander("Player raw records", expanded=False):
            st.dataframe(player_df, use_container_width=True)

        # Player timeline: injury durations on a Gantt-like bar (Plotly)
        if 'Injury_Start' in player_df.columns and player_df['Injury_Start'].notna().any():
            timeline_df = player_df.copy()
            timeline_df['start'] = timeline_df['Injury_Start']
            timeline_df['end'] = timeline_df['Injury_End'].fillna(timeline_df['Injury_Start'])
            timeline_df['Injury_Label'] = timeline_df.apply(lambda r: f"{r.get('Injury_Type','Unknown')} ({r.get('Injury_Duration', '')}d)", axis=1)
            try:
                fig = px.timeline(timeline_df, x_start="start", x_end="end", y="Injury_Label", title=f"Injury timeline for {selected_player}")
                fig.update_yaxes(autorange="reversed")
                render_plotly(fig)
            except Exception as e:
                st.error(f"Could not render timeline: {e}")
        else:
            st.info("No Injury_Start dates for this player to create a timeline.")

        # Performance before/after scatter
        if all(c in player_df.columns for c in ['Avg_Rating_Before','Avg_Rating_After']):
            fig = px.scatter(player_df, x='Avg_Rating_Before', y='Avg_Rating_After', hover_data=['Injury_Type','Injury_Duration'], title=f"Before vs After ratings for {selected_player}")
            render_plotly(fig)
        else:
            st.info("Avg_Rating_Before/Avg_Rating_After columns missing for before/after analysis.")

    else:
        st.info("No 'Player' column in dataset to do player deep dive.")

# ---------------------
# Footer / Tips
# ---------------------
st.markdown("---")
st.caption("Tip: If you see 'N/A' or missing charts, check your CSV column names and date formats. "
           "Recommended date format: YYYY-MM-DD. For large datasets, apply filters to speed up plotting.")
