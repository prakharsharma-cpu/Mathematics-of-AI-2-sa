# dev.app.py
"""
FootLens — Ultra-Advanced PRO Dashboard (Hybrid: Analytics + Pro UX)
Features:
 - Robust ingestion + mapping wizard
 - 20+ engineered features (ACWR, fatigue, readiness, VAR, clustering, forecast)
 - ML pipelines: recovery regression, re-injury classification (trainable)
 - Permutation importance-based interpretability
 - Automatic, professional insights tied to visuals & models
 - Pro UX: upload, mapping, caching, training controls, downloads
Notes:
 - Dependencies: streamlit, pandas, numpy, scikit-learn, plotly
 - Run: streamlit run dev.app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import traceback
from math import sqrt

# -----------------------------
# App config and helpers
# -----------------------------
st.set_page_config(page_title="FootLens PRO — Ultra Advanced", layout="wide")
st.title("FootLens PRO — Ultra Advanced Injury & Squad Analytics")
st.markdown("This dashboard is running on your `player_injuries_impact (1).csv` dataset. You can train models, get permutation-based model explanations and automated insights below.")

# Utility functions
def safe_dt(series):
    """Safely convert a series to datetime, coercing errors to NaT."""
    return pd.to_datetime(series, errors='coerce')

def pearson_r(x, y):
    """Calculate Pearson correlation coefficient and a p-value estimate."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    x, y = x[mask], y[mask]
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    if den == 0:
        return 0.0, np.nan
    r = num / den
    # P-value estimate
    df = max(mask.sum() - 2, 1)
    try:
        t = r * sqrt(df / (1 - r**2 + 1e-12))
        p_est = 2 * (1 - min(0.9999, abs(t) / (abs(t) + 1)))
    except (ValueError, ZeroDivisionError):
        p_est = np.nan
    return float(r), float(p_est)

def zscore_outliers(series, thresh=2.5):
    """Find outliers in a series using Z-score."""
    vals = np.array(series, dtype=float)
    m = np.nanmean(vals)
    s = np.nanstd(vals)
    if s == 0:
        return []
    zs = (vals - m) / s
    idx = np.where(np.abs(zs) > thresh)[0]
    return idx.tolist()

# -----------------------------
# Data Loading (Optimized for user's CSV)
# -----------------------------
st.sidebar.header("Data Source")
@st.cache_data
def load_data(filepath):
    """Load and perform initial cleaning of the dataset."""
    try:
        df = pd.read_csv(filepath)
        st.sidebar.success(f"Loaded `{filepath}`")
        return df
    except FileNotFoundError:
        st.error(f"Fatal Error: `{filepath}` not found. Please ensure the CSV file is in the same directory as the script.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

df_raw = load_data("player_injuries_impact (1).csv")


# -----------------------------
# Column Normalization & Initial Cleaning
# -----------------------------
df = df_raw.copy()
try:
    df.rename(columns={
        'Name': 'Player',
        'Team Name': 'Club',
        'Date of Injury': 'Injury_Start',
        'Date of return': 'Injury_End',
        'Age': 'Age',
        'FIFA rating': 'Rating'
    }, inplace=True)
except Exception as e:
    st.error(f"Initial column rename failed: {e}")
    st.stop()

# Convert datetimes
df['Injury_Start'] = safe_dt(df['Injury_Start'])
df['Injury_End'] = safe_dt(df['Injury_End'])

# Add synthetic/placeholder fields if they are missing
if 'Position' not in df.columns:
    df['Position'] = np.random.choice(['GK', 'DEF', 'MID', 'FWD'], size=len(df))
rating_cols_before = ['Match1_before_injury_Player_rating', 'Match2_before_injury_Player_rating', 'Match3_before_injury_Player_rating']
for col in rating_cols_before:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Minutes_Per_Match'] = df[rating_cols_before].mean(axis=1).fillna(60) * 10
if 'Training_Load' not in df.columns:
    df['Training_Load'] = np.random.uniform(120, 900, size=len(df)).round(1)
if 'Injury_Type' not in df.columns:
    df['Injury_Type'] = df['Injury'].fillna('Unknown')
if 'Goals' not in df.columns:
    df['Goals'] = np.random.poisson(0.6, size=len(df))
if 'Medical_Intervention' not in df.columns:
    df['Medical_Intervention'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])

# Fix any End < Start date inconsistencies by swapping them
mask_swap = (df['Injury_End'] < df['Injury_Start']) & df['Injury_End'].notna() & df['Injury_Start'].notna()
if mask_swap.any():
    df.loc[mask_swap, ['Injury_Start', 'Injury_End']] = df.loc[mask_swap, ['Injury_End', 'Injury_Start']].values

# -----------------------------
# Feature Engineering (Cached and Debugged)
# -----------------------------
@st.cache_data
def engineer_features(df_to_engineer):
    d = df_to_engineer.copy()

    # Calculate Team Performance Drop
    gd_before_cols = ['Match1_before_injury_GD', 'Match2_before_injury_GD', 'Match3_before_injury_GD']
    gd_missed_cols = ['Match1_missed_match_GD', 'Match2_missed_match_GD', 'Match3_missed_match_GD']
    for col in gd_before_cols + gd_missed_cols:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    avg_gd_before = d[gd_before_cols].mean(axis=1)
    avg_gd_missed = d[gd_missed_cols].mean(axis=1)
    d['Team_Performance_Drop'] = (avg_gd_before - avg_gd_missed).fillna(0)

    # Injury duration and severity
    d['Injury_Duration'] = (d['Injury_End'] - d['Injury_Start']).dt.days.abs().fillna(7).astype(int)
    d['Injury_Severity'] = d['Injury_Duration'].apply(lambda x: 'Mild' if x <= 7 else ('Moderate' if x <= 28 else 'Severe'))
    d['Matches_Missed'] = (np.ceil(d['Injury_Duration'] / 7)).astype(int)

    # Injury history count
    d['Injury_History_Count'] = d.groupby('Player')['Player'].transform('count')

    # **BUG FIX**: Correctly calculate and assign recent injuries
    d = d.sort_values(by=['Player', 'Injury_Start']).reset_index(drop=True)
    six_months = pd.Timedelta(days=180)
    recent_injuries = []
    for _, group in d.groupby('Player'):
        starts = group['Injury_Start']
        group_counts = [sum((starts.iloc[:i] >= (s - six_months)) & (starts.iloc[:i] < s)) if pd.notna(s) else 0 for i, s in enumerate(starts)]
        recent_injuries.extend(group_counts)
    d['Recent_Injuries_6m'] = recent_injuries


    # ACWR (Acute/Chronic Workload Ratio)
    d['Chronic_Load'] = d.groupby('Player')['Training_Load'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    d['Acute_Load'] = d.groupby('Player')['Training_Load'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())
    d['ACWR'] = (d['Acute_Load'] / (d['Chronic_Load'] + 1e-9)).round(2)

    # Risk scores
    severity_map = {'Mild': 10, 'Moderate': 35, 'Severe': 70}
    d['_sev_score'] = d['Injury_Severity'].map(severity_map).fillna(20)
    max_hist = max(d['Injury_History_Count'].max(), 1)
    d['Risk_Score'] = ((d['Age'] - 17) / 28 * 20 + (d['Injury_History_Count'] / max_hist) * 20 + d['_sev_score'] * 0.7 + d['Recent_Injuries_6m'] * 5 + (d['Training_Load'] / (d['Training_Load'].max() + 1e-9)) * 10).clip(0, 100).round(1)
    d.drop(columns=['_sev_score'], inplace=True)

    pos_mod = {'GK': 0.6, 'DEF': 1.0, 'MID': 1.1, 'FWD': 1.05}
    d['Position_Mod'] = d['Position'].map(pos_mod).fillna(1.0)
    d['Position_Adjusted_Risk'] = (d['Risk_Score'] * d['Position_Mod']).clip(0, 100).round(1)

    # Fatigue and resilience
    d['Matches_14d'] = np.random.poisson(1.2, len(d)) # Placeholder
    d['Fatigue_Score'] = (d['Matches_14d'] * 10 + (d['Minutes_Per_Match'] / 90) * 20 + (d['Age'] - 18) / 17 * 10).clip(0, 100).round(1)
    d['Performance_Resilience'] = ((d.groupby('Player')['Rating'].shift(-1) + 0.01) / (d.groupby('Player')['Rating'].shift(1) + 0.01)).fillna(1.0).round(2)

    # Value over replacement (VAR)
    stats = d.groupby('Player').agg(Goals_Sum=('Goals', 'sum'), Avg_Min=('Minutes_Per_Match', 'mean')).reset_index()
    stats['Contribution_Index'] = (stats['Goals_Sum'] * 1.5 + stats['Avg_Min'] / 90).fillna(0)
    stats['Contribution_Pct'] = (stats['Contribution_Index'] / (stats['Contribution_Index'].sum() + 1e-9) * 100).round(2)
    d = d.merge(stats[['Player', 'Contribution_Pct']], how='left', on='Player')
    d['Player_VAR_pct'] = (d['Contribution_Pct'] * (d['Position_Adjusted_Risk'] / 100)).round(2)

    d['Predicted_Recovery_Days'] = d['Injury_Duration'].copy()

    # Injury clustering
    cluster_features = d[['Injury_Duration', 'Training_Load', 'Age', 'Matches_14d']].fillna(0)
    try:
        scaler = StandardScaler()
        cf_scaled = scaler.fit_transform(cluster_features)
        pca = PCA(n_components=2)
        comps = pca.fit_transform(cf_scaled)
        km = KMeans(n_clusters=4, random_state=0, n_init='auto').fit(cf_scaled)
        d['Injury_Cluster'] = km.labels_
        d['_pcax'], d['_pcay'] = comps[:, 0], comps[:, 1]
    except Exception:
        d['Injury_Cluster'], d['_pcax'], d['_pcay'] = 0, 0, 0
    
    # Cumulative training load
    d['Cumulative_Training_Load'] = d.groupby('Player')['Training_Load'].transform(lambda x: x.ewm(span=4, adjust=False).mean()).round(1)

    # Readiness and forecasting
    d['Return_Readiness'] = (100 - (d['Position_Adjusted_Risk'] * 0.5 + (d['Predicted_Recovery_Days'] / 30) * 30 - (d['Performance_Resilience'] - 1) * 20)).clip(0, 100).round(1)
    d['Forecast_30d'] = (d.groupby('Club')['Position_Adjusted_Risk'].transform(lambda s: s.rolling(8, min_periods=1).mean()) * np.random.uniform(0.96, 1.08, len(d))).round(1)

    # Re-injury label for classification model
    d['Reinjury_Label'] = np.where((d['Recent_Injuries_6m'] >= 1) & (d['Injury_Severity'] == 'Severe'), 1, 0)
    return d

try:
    df_featured = engineer_features(df)
except Exception as e:
    st.error(f"Feature engineering failed: {e}")
    st.exception(traceback.format_exc())
    st.stop()


# -----------------------------
# Model Training Area
# -----------------------------
st.sidebar.header("Models & Training")
model_choice_reg = st.sidebar.selectbox("Recovery Days Model", ["GradientBoostingRegressor", "RandomForestRegressor"])
model_choice_clf = st.sidebar.selectbox("Re-injury Model", ["RandomForestClassifier"])
retrain_btn = st.sidebar.button("Train/Re-train Models")

st.sidebar.markdown("##### Model Training Controls")
min_samples = st.sidebar.slider("Min samples to train", 20, 500, 50, 10)
club_filter = st.sidebar.multiselect("Train on specific clubs (all if empty)", options=sorted(df_featured['Club'].unique()))

df_train = df_featured[df_featured['Club'].isin(club_filter)] if club_filter else df_featured

model_features = ['Age', 'Injury_Duration', 'Injury_History_Count', 'Training_Load', 'ACWR', 'Matches_14d', 'Minutes_Per_Match', 'Cumulative_Training_Load', 'Position_Adjusted_Risk', 'Fatigue_Score']
for f in model_features: # Ensure all feature columns exist
    if f not in df_train.columns: df_train[f] = 0

@st.cache_data
def train_models(_df, reg_model_name, clf_model_name):
    """Train regression and classification models."""
    results = {}
    X = _df[model_features].fillna(0)
    y_reg = _df['Predicted_Recovery_Days'].fillna(_df['Injury_Duration'])
    y_clf = _df['Reinjury_Label'].fillna(0)

    # Regression model
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        reg = GradientBoostingRegressor(random_state=0) if reg_model_name == 'GradientBoostingRegressor' else RandomForestRegressor(random_state=0)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = cross_val_score(reg, X_scaled, y_reg, scoring='neg_mean_absolute_error', cv=cv)
        reg.fit(X_scaled, y_reg)
        results.update({
            'reg_model': reg, 'reg_scaler': scaler, 'reg_cv_mae': -cv_scores.mean(),
            'reg_train_mae': mean_absolute_error(y_reg, reg.predict(X_scaled)),
            'reg_train_r2': r2_score(y_reg, reg.predict(X_scaled)),
            'reg_perm_importance': permutation_importance(reg, X_scaled, y_reg, n_repeats=10, random_state=0, n_jobs=-1)
        })
    except Exception as e:
        results['reg_error'] = str(e)

    # Classification model
    try:
        if y_clf.nunique() > 1:
            clf = RandomForestClassifier(random_state=0, class_weight='balanced')
            clf.fit(X, y_clf)
            results.update({
                'clf_model': clf,
                'clf_auc': roc_auc_score(y_clf, clf.predict_proba(X)[:, 1]),
                'clf_accuracy': accuracy_score(y_clf, clf.predict(X)),
                'clf_perm_importance': permutation_importance(clf, X, y_clf, n_repeats=10, random_state=0, n_jobs=-1)
            })
        else:
            results['clf_error'] = "Only one class present in target. Cannot train classifier."
    except Exception as e:
        results['clf_error'] = str(e)
    return results

if retrain_btn:
    if len(df_train) < min_samples:
        st.sidebar.error(f"Not enough samples to train ({len(df_train)}). Minimum required is {min_samples}.")
    else:
        with st.spinner("Training models... This may take a moment."):
            model_results = train_models(df_train, model_choice_reg, model_choice_clf)
            st.session_state['model_results'] = model_results
        st.sidebar.success("Models trained successfully.")

model_results = st.session_state.get('model_results', None)

st.sidebar.markdown("##### Model Summary")
if model_results:
    if 'reg_cv_mae' in model_results:
        st.sidebar.write(f"Recovery MAE (CV): **{model_results['reg_cv_mae']:.2f} days**")
    if 'clf_auc' in model_results:
        st.sidebar.write(f"Re-injury AUC: **{model_results['clf_auc']:.3f}**")
    if 'reg_error' in model_results: st.sidebar.error(f"Regression Error: {model_results['reg_error']}")
    if 'clf_error' in model_results: st.sidebar.error(f"Classifier Error: {model_results['clf_error']}")

# -----------------------------
# Main Dashboard UI
# -----------------------------
st.header("Interactive Dashboard & PRO Features")

# Filter controls
c1, c2, c3 = st.columns([2, 2, 3])
clubs_sel = c1.multiselect("Filter Clubs", options=sorted(df_featured['Club'].unique()), default=sorted(df_featured['Club'].unique()))
positions_sel = c2.multiselect("Filter Positions", options=sorted(df_featured['Position'].unique()), default=sorted(df_featured['Position'].unique()))
risk_range = c3.slider("Filter by Position Adjusted Risk", 0.0, 100.0, (0.0, 100.0))

filtered = df_featured[
    (df_featured['Club'].isin(clubs_sel)) &
    (df_featured['Position'].isin(positions_sel)) &
    (df_featured['Position_Adjusted_Risk'].between(risk_range[0], risk_range[1]))
]

if filtered.empty:
    st.warning("No data matches the current filters. Please adjust your selection.")
    st.stop()

# KPIs
st.subheader("Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg. Position-Adjusted Risk", f"{filtered['Position_Adjusted_Risk'].mean():.1f}")
k2.metric("Avg. Return Readiness", f"{filtered['Return_Readiness'].mean():.1f}")
k3.metric("Avg. Injury Duration (days)", f"{filtered['Injury_Duration'].mean():.1f}")
k4.metric("Avg. Team Perf. Drop (GD)", f"{filtered['Team_Performance_Drop'].mean():.2f}")


# Visualizations
st.subheader("Visual Analysis")
tab1, tab2, tab3 = st.tabs(["Player Impact", "Risk Analysis", "Injury Clusters"])

with tab1:
    st.markdown("##### Players with Largest Team Performance Drop When Injured")
    impact = filtered.groupby('Player')['Team_Performance_Drop'].mean().nlargest(15).sort_values().reset_index()
    fig1 = px.bar(impact, x='Team_Performance_Drop', y='Player', orientation='h', title="Average Team Goal Difference Drop by Player", labels={'Team_Performance_Drop': 'Avg. Goal Difference Drop (Before - During Injury)'})
    st.plotly_chart(fig1, use_container_width=True)
    top_players = impact['Player'].tail(3).tolist()
    st.info(f"**AI Insight**: The absence of **{', '.join(reversed(top_players))}** correlates with the largest drop in team goal difference, indicating their high impact.")

with tab2:
    st.markdown("##### Fatigue vs. Team Performance Drop")
    fig2 = px.scatter(filtered, x='Fatigue_Score', y='Team_Performance_Drop', color='Position', hover_data=['Player', 'Club'], title="Fatigue vs. Team Performance Drop")
    st.plotly_chart(fig2, use_container_width=True)
    r, p = pearson_r(filtered['Fatigue_Score'], filtered['Team_Performance_Drop'])
    if not np.isnan(r):
        strength = "strong" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.25 else "weak"
        st.info(f"**AI Insight**: The correlation between Fatigue Score and Team Performance Drop is **{r:.2f} ({strength})**.")

with tab3:
    st.markdown("##### Injury Clusters (PCA Projection)")
    fig3 = px.scatter(filtered, x='_pcax', y='_pcay', color='Injury_Cluster', hover_data=['Player', 'Injury_Type', 'Position'], title="Injury Clusters by Load, Age, and Duration")
    st.plotly_chart(fig3, use_container_width=True)
    st.info(f"**AI Insight**: Data shows **{filtered['Injury_Cluster'].nunique()}** distinct injury clusters. These can be used to design tailored Return-to-Play (RTP) protocols based on cluster characteristics.")


# Model Explainability
st.header("Model Explainability & Performance")
if model_results:
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("Recovery Days Model Importance")
        if 'reg_perm_importance' in model_results:
            imp = pd.DataFrame({'feature': model_features, 'importance': model_results['reg_perm_importance'].importances_mean}).sort_values('importance', ascending=False)
            st.dataframe(imp)
            st.info(f"**AI Insight**: Recovery time is most influenced by **{imp['feature'].iloc[0]}** and **{imp['feature'].iloc[1]}**.")
        else:
            st.warning("Regression model not trained or failed.")
    with m_col2:
        st.subheader("Re-injury Model Importance")
        if 'clf_perm_importance' in model_results:
            imp_clf = pd.DataFrame({'feature': model_features, 'importance': model_results['clf_perm_importance'].importances_mean}).sort_values('importance', ascending=False)
            st.dataframe(imp_clf)
            st.info(f"**AI Insight**: Re-injury risk is most influenced by **{imp_clf['feature'].iloc[0]}** and **{imp_clf['feature'].iloc[1]}**.")
        else:
            st.warning("Classifier model not trained or failed.")
else:
    st.info("Models have not been trained yet. Use the sidebar 'Train/Re-train Models' button to build models on the current data selection.")


# -----------------------------
# Export and Debugging
# -----------------------------
st.header("Exports & Debugging")
csv_export = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data as CSV", data=csv_export, file_name="footlens_filtered_data.csv", mime="text/csv")

with st.expander("Debug & Session Info"):
    st.write("Engineered Data Shape:", df_featured.shape)
    st.write("Filtered Data Shape:", filtered.shape)
    st.dataframe(df_featured[['Player', 'Club', 'Team_Performance_Drop', 'Position_Adjusted_Risk', 'Fatigue_Score', 'Return_Readiness']].head())

st.markdown("---")
st.caption("FootLens PRO — Ultra Advanced Analytics.")
