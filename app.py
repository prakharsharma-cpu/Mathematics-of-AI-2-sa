# ==================================
# ⚽ FootLens - Robust Streamlit App (auto column-mapping)
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="⚽ FootLens", layout="wide", initial_sidebar_state="expanded")
st.title("⚽ FootLens")
st.markdown("Upload your CSV and explore injury impact analytics. This app automatically standardizes and maps columns so it works with many CSV header variants.")

# ---------------------
# Utilities: normalization & fuzzy mapping
# ---------------------
def normalize_colname(name: str) -> str:
    """Normalize a column name to a canonical form: lower, underscores, alnum only."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)          # remove punctuation
    s = re.sub(r"\s+", "_", s)             # spaces -> underscore
    return s

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {c: normalize_colname(c) for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    return df

def find_column(df_cols, alternatives):
    """Return the first matching column in df_cols found in alternatives (both normalized)."""
    norm_cols = {normalize_colname(c): c for c in df_cols}
    for alt in alternatives:
        nalt = normalize_colname(alt)
        if nalt in norm_cols:
            return norm_cols[nalt]
    # fallback: try partial contains matching
    for alt in alternatives:
        nalt = normalize_colname(alt)
        for nc, orig in norm_cols.items():
            if nalt in nc or nc in nalt:
                return orig
    return None

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
    csv = """Player,Club,Injury Type,Injury Start,Injury End,Team Goals Before,Team Goals During,Team Goals After,Avg Rating Before,Avg Rating After,Rating
John Doe,Example FC,Hamstring,2024-09-01,2024-09-21,1.2,0.6,1.0,7.1,6.9,6.9
Jane Smith,Example FC,Ankle,2024-10-05,2024-10-20,1.5,1.0,1.4,7.5,7.3,7.4
Alex Roe,Another United,ACL,2024-08-01,2024-11-01,1.0,0.2,0.8,6.9,7.0,6.95
"""
    return csv

# ---------------------
# Sidebar: Upload + Settings
# ---------------------
st.sidebar.header("🔍 Upload & Settings")
st.sidebar.markdown("Upload a CSV. This app will attempt to map common column name variants (e.g., 'Injury Type' or 'injury_type').")

sample_csv = sample_csv_text()
st.sidebar.download_button("⬇️ Download sample CSV", data=sample_csv, file_name="footlens_sample.csv", mime="text/csv")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ---------------------
# Load data
# ---------------------
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df_raw)} records from file.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.sidebar.info("No file uploaded — using sample dataset for demonstration.")
    df_raw = pd.read_csv(StringIO(sample_csv))

# ---------------------
# Normalize columns and keep mapping
# ---------------------
df = df_raw.copy()
original_columns = list(df.columns)
df = normalize_df_columns(df)  # columns become normalized keys, but values remain same; store originals mapping
# Build reverse mapping: normalized -> normalized (since df columns now normalized)
df_cols = list(df.columns)

# For user clarity, show detected columns (normalized)
with st.expander("Detected columns (normalized)", expanded=False):
    st.write(df_cols)

# ---------------------
# Attempt fuzzy mapping to expected fields
# ---------------------
expected_alternatives = {
    "player": ["player", "player_name", "name"],
    "club": ["club", "team", "club_name", "team_name"],
    "injury_type": ["injury_type", "injury type", "injury", "injurytype", "type_of_injury"],
    "injury_start": ["injury_start", "injury start", "start_date", "start", "injury_start_date", "startdate"],
    "injury_end": ["injury_end", "injury end", "end_date", "end", "injury_end_date", "enddate"],
    "team_goals_before": ["team_goals_before", "team goals before", "goals_before", "goals before", "goals_before_match"],
    "team_goals_during": ["team_goals_during", "team goals during", "goals_during", "goals during"],
    "team_goals_after": ["team_goals_after", "team goals after", "goals_after", "goals after"],
    "avg_rating_before": ["avg_rating_before", "avg rating before", "avg_rating_before_match", "average_rating_before"],
    "avg_rating_after": ["avg_rating_after", "avg rating after", "average_rating_after"],
    "rating": ["rating", "player_rating", "rating_overall", "avg_rating"]
}

mapped = {}
for key, alts in expected_alternatives.items():
    found = find_column(df.columns, alts)
    mapped[key] = found  # may be None

# ---------------------
# Preprocessing & derived metrics (use mapped names)
# ---------------------
# Convert date-like columns
if mapped["injury_start"]:
    df[mapped["injury_start"]] = pd.to_datetime(df[mapped["injury_start"]], errors="coerce")
if mapped["injury_end"]:
    df[mapped["injury_end"]] = pd.to_datetime(df[mapped["injury_end"]], errors="coerce")

# fill numeric NaNs in numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mean())

# Derived metrics
if mapped["injury_start"] and mapped["injury_end"]:
    df["injury_duration_days"] = (df[mapped["injury_end"]] - df[mapped["injury_start"]]).dt.days.clip(lower=0)
    df["injury_month"] = df[mapped["injury_start"]].dt.month
else:
    df["injury_duration_days"] = np.nan
    df["injury_month"] = np.nan

if mapped["team_goals_before"] and mapped["team_goals_during"]:
    df["team_performance_drop"] = df[mapped["team_goals_before"]] - df[mapped["team_goals_during"]]
else:
    df["team_performance_drop"] = np.nan

if mapped["team_goals_during"] and mapped["team_goals_after"]:
    df["team_recovery"] = df[mapped["team_goals_after"]] - df[mapped["team_goals_during"]]
else:
    df["team_recovery"] = np.nan

if mapped["avg_rating_before"] and mapped["avg_rating_after"]:
    df["performance_change"] = df[mapped["avg_rating_after"]] - df[mapped["avg_rating_before"]]
else:
    df["performance_change"] = np.nan

# ---------------------
# Sidebar: Filters + Mode/Style
# ---------------------
st.sidebar.markdown("---")
st.sidebar.header("Filters & Visualization Settings")

club_options = sorted(df[mapped["club"]].dropna().unique()) if mapped["club"] else []
player_options = sorted(df[mapped["player"]].dropna().unique()) if mapped["player"] else []
injury_options = sorted(df[mapped["injury_type"]].dropna().unique()) if mapped["injury_type"] else []

filter_club = st.sidebar.multiselect("Club", options=club_options, default=club_options)
filter_player = st.sidebar.multiselect("Player", options=player_options, default=player_options)
filter_injury = st.sidebar.multiselect("Injury Type", options=injury_options, default=injury_options)

global_mode = st.sidebar.radio("🧭 Global Visualization Mode", ["Plotly","Matplotlib"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Seaborn Theme (Matplotlib)")
global_style = st.sidebar.radio("Global Seaborn Style", ["Modern Clean","Classic Analytics"], index=0)

# Apply filters defensively
filtered_df = df.copy()
if mapped["club"] and filter_club:
    filtered_df = filtered_df[filtered_df[mapped["club"]].isin(filter_club)]
if mapped["player"] and filter_player:
    filtered_df = filtered_df[filtered_df[mapped["player"]].isin(filter_player)]
if mapped["injury_type"] and filter_injury:
    filtered_df = filtered_df[filtered_df[mapped["injury_type"]].isin(filter_injury)]

# ---------------------
# KPIs
# ---------------------
kpi1, kpi2, kpi3 = st.columns(3)
avg_rating = safe_mean(filtered_df[mapped["rating"]]) if mapped["rating"] else np.nan
avg_drop = safe_mean(filtered_df["team_performance_drop"]) if "team_performance_drop" in filtered_df.columns else np.nan
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
    st.write("First 10 rows (after normalization & filters):")
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
    categorical_cols = [c for c in filtered_df.select_dtypes(exclude=np.number).columns.tolist() if c not in [mapped.get(k) for k in mapped.keys()] or c in [mapped.get("player"), mapped.get("club"), mapped.get("injury_type")] ]

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

    # Boxplots by categorical features (limited combinations)
    if numeric_cols and categorical_cols:
        st.markdown("### 📦 Boxplots by Categorical Features")
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

    # Top injury types by team performance drop (safe check)
    inj_col = mapped["injury_type"]
    if inj_col and "team_performance_drop" in filtered_df.columns:
        st.markdown("### 🩹 Top 10 Injuries by Avg Team Performance Drop")
        top_injuries = filtered_df.groupby(inj_col)["team_performance_drop"].mean().sort_values(ascending=False).head(10).reset_index()
        # Ensure column names in plot match dataframe
        top_injuries = top_injuries.rename(columns={inj_col: "injury_type", "team_performance_drop": "avg_drop"})
        if mode=="Plotly":
            fig = px.bar(top_injuries, x='avg_drop', y='injury_type', orientation='h', title="Top injuries by team performance drop", labels={'avg_drop':'Avg Drop','injury_type':'Injury'})
            render_plotly(fig)
        else:
            apply_seaborn_style(style)
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=top_injuries, y='injury_type', x='avg_drop', ax=ax)
            ax.set_title("Top injuries by team performance drop")
            plt.tight_layout(); render_matplotlib(fig)
    else:
        st.info("Columns for injury impact by drop missing (injury type or team performance metrics).")

# =====================================
# Tab 2: Player Impact
# =====================================
with tabs[2]:
    st.subheader("📈 Player Impact Overview")
    if not ensure_nonempty(filtered_df):
        st.stop()

    if mapped["player"]:
        player = mapped["player"]
        player_summary = filtered_df.groupby(player).agg(
            Injuries=(player,'count'),
            Avg_Injury_Duration=("injury_duration_days", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            Avg_Perf_Change=("performance_change", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            Avg_Rating=(mapped["rating"] if mapped["rating"] else player, lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan)  # fallback
        ).reset_index().sort_values('Injuries', ascending=False)
        st.dataframe(player_summary.head(50), use_container_width=True)

        st.markdown("### Players with largest average performance change (most negative)")
        if "performance_change" in filtered_df.columns and not filtered_df["performance_change"].isnull().all():
            tmp = player_summary.sort_values('Avg_Perf_Change').head(10).reset_index(drop=True)
            fig = px.bar(tmp, x='Avg_Perf_Change', y=player, orientation='h', title="Players with largest negative performance change")
            render_plotly(fig)
        else:
            st.info("No 'performance_change' values available.")
    else:
        st.info("No player column detected to compute player-level impact.")

# =====================================
# Tab 3: Club Analysis
# =====================================
with tabs[3]:
    st.subheader("🔥 Club Analysis")
    if not ensure_nonempty(filtered_df):
        st.stop()

    if mapped["club"]:
        club = mapped["club"]
        club_summary = filtered_df.groupby(club).agg(
            Injuries=(club,'count'),
            Avg_Injury_Duration=("injury_duration_days", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan),
            Avg_Team_Drop=("team_performance_drop", lambda s: np.nanmean(s) if len(s.dropna())>0 else np.nan)
        ).reset_index().sort_values('Injuries', ascending=False)
        st.dataframe(club_summary, use_container_width=True)

        fig = px.bar(club_summary.sort_values('Injuries',ascending=False).head(15), x='Injuries', y=club, orientation='h', title="Top clubs by injury count")
        render_plotly(fig)
    else:
        st.info("No club column detected for club-level analysis.")

# =====================================
# Tab 4: Injury Analysis
# =====================================
with tabs[4]:
    st.subheader("🔬 Injury Analysis & Time Trends")
    if not ensure_nonempty(filtered_df):
        st.stop()

    inj_col = mapped["injury_type"]
    # Injury counts by type (safe)
    if inj_col and inj_col in filtered_df.columns:
        inj_counts = filtered_df[inj_col].value_counts().reset_index()
        inj_counts.columns = ["injury_type", "count"]
        st.markdown("### Injury counts by type")
        # Plotly expects the DataFrame to have 'count' and 'injury_type'
        try:
            fig = px.bar(inj_counts.head(30), x='count', y='injury_type', orientation='h', title="Injury counts")
            render_plotly(fig)
        except Exception as e:
            st.error(f"Could not create injury counts bar: {e}")
    else:
        st.info("No injury type column available to show injury distribution.")

    # Injury duration distribution
    if "injury_duration_days" in filtered_df.columns and not filtered_df["injury_duration_days"].isnull().all():
        st.markdown("### Injury duration distribution (days)")
        fig = px.histogram(filtered_df, x='injury_duration_days', nbins=30, title="Injury duration (days)")
        render_plotly(fig)
    else:
        st.info("No injury duration information available (needs start & end dates).")

    # Monthly trend of injuries (if present)
    if "injury_month" in filtered_df.columns and not filtered_df["injury_month"].isnull().all():
        monthly = filtered_df.groupby("injury_month").size().reset_index(name='count').sort_values('injury_month')
        st.markdown("### Monthly injury counts")
        fig = px.line(monthly, x='injury_month', y='count', markers=True, title="Injuries by month")
        render_plotly(fig)
    else:
        st.info("No date information to compute monthly trends (Injury start missing).")

# =====================================
# Tab 5: Player Deep Dive
# =====================================
with tabs[5]:
    st.subheader("🔎 Player Deep Dive")
    if not ensure_nonempty(filtered_df):
        st.stop()

    if mapped["player"]:
        player_col = mapped["player"]
        options = sorted(filtered_df[player_col].dropna().unique())
        if not options:
            st.info("No players found after filtering.")
        else:
            selected_player = st.selectbox("Select player", options=options)
            player_df = filtered_df[filtered_df[player_col]==selected_player]
            st.write(f"Showing {len(player_df)} record(s) for **{selected_player}**")

            with st.expander("Player raw records", expanded=False):
                st.dataframe(player_df, use_container_width=True)

            # Timeline (if dates)
            if mapped["injury_start"] and player_df[mapped["injury_start"]].notna().any():
                timeline_df = player_df.copy()
                timeline_df["start"] = timeline_df[mapped["injury_start"]]
                timeline_df["end"] = timeline_df[mapped["injury_end"]].fillna(timeline_df[mapped["injury_start"]]) if mapped["injury_end"] else timeline_df[mapped["injury_start"]]
                timeline_df["label"] = timeline_df.get(mapped["injury_type"], "injury").astype(str) + " (" + timeline_df["injury_duration_days"].astype(str) + "d)"
                try:
                    fig = px.timeline(timeline_df, x_start="start", x_end="end", y="label", title=f"Injury timeline for {selected_player}")
                    fig.update_yaxes(autorange="reversed")
                    render_plotly(fig)
                except Exception as e:
                    st.error(f"Could not render timeline: {e}")
            else:
                st.info("No injury start dates for this player to create a timeline.")

            # Before vs After scatter
            if mapped["avg_rating_before"] and mapped["avg_rating_after"]:
                if mapped["avg_rating_before"] in player_df.columns and mapped["avg_rating_after"] in player_df.columns:
                    fig = px.scatter(player_df, x=mapped["avg_rating_before"], y=mapped["avg_rating_after"], hover_data=[mapped.get("injury_type"), "injury_duration_days"], title=f"Before vs After ratings for {selected_player}")
                    render_plotly(fig)
                else:
                    st.info("Before/after rating columns not present for this player.")
            else:
                st.info("Avg rating before/after columns not available for before/after analysis.")
    else:
        st.info("No 'player' column detected to do player deep dive.")

# ---------------------
# Footer / Tips
# ---------------------
st.markdown("---")
st.caption("Tip: If charts say 'No column available', check your CSV headers. This app normalizes many common variants but if your dataset uses very different names, rename them or share the header list.")
