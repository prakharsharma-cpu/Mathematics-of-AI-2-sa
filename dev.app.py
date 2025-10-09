pip install streamlit pandas numpy scikit-learn plotly

streamlit run dev.app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io
import base64
import traceback
from math import sqrt

st.set_page_config(page_title="FootLens PRO — Ultra Advanced", layout="wide")
st.title("FootLens PRO — Ultra Advanced Injury & Squad Analytics")
st.markdown("Ultra-tuned analytics + UX. This dashboard is running on your `player_injuries_impact (1).csv` dataset. You can train models, get permutation-based model explanations and automated insights below.")

# Utility functions
def safe_dt(series):
    try:
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return pd.Series([pd.NaT]*len(series))

def pearson_r(x, y):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    x = x[mask]; y = y[mask]
    xm = x.mean(); ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    if den == 0:
        return 0.0, np.nan
    r = num / den
    # crude p-like estimate
    df = max(mask.sum() - 2, 1)
    try:
        t = r * sqrt(df / (1 - r**2 + 1e-12))
        p_est = 2 * (1 - min(0.9999, abs(t) / (abs(t) + 1)))
    except Exception:
        p_est = np.nan
    return float(r), float(p_est)

def zscore_outliers(series, thresh=2.5):
    vals = np.array(series).astype(float)
    m = np.nanmean(vals); s = np.nanstd(vals)
    if s == 0:
        return []
    zs = (vals - m) / s
    idx = np.where(np.abs(zs) > thresh)[0]
    return idx.tolist()

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generate a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        csv = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    else:
        b64 = base64.b64encode(object_to_download).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

st.sidebar.header("Data")
try:
    # Directly load the user's provided CSV file
    df_raw = pd.read_csv("player_injuries_impact (1).csv")
    st.sidebar.success("Loaded `player_injuries_impact (1).csv`")
except FileNotFoundError:
    st.error("Fatal Error: `player_injuries_impact (1).csv` not found. Please ensure the CSV file is in the same directory as the script.")
    st.stop()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Copy working DF and normalize column names
df = df_raw.copy()

# Standardize column names based on the user's CSV structure
# This replaces the interactive mapping wizard
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
    st.sidebar.error(f"Initial column rename failed: {e}")
    st.stop()


# Ensure required columns exist or create placeholders
required_cols = ['Player', 'Club', 'Injury_Start', 'Injury_End', 'Age', 'Rating']
for rc in required_cols:
    if rc not in df.columns:
        st.error(f"A required column '{rc}' was not found in the CSV after renaming. Please check the file.")
        st.stop()


# Convert datetimes
df['Injury_Start'] = safe_dt(df['Injury_Start'])
df['Injury_End'] = safe_dt(df['Injury_End'])

# Add synthetic fields if they are missing from the input CSV
if 'Position' not in df.columns:
    df['Position'] = np.random.choice(['GK','DEF','MID','FWD'], size=len(df))
if 'Minutes_Per_Match' not in df.columns:
    # Use player ratings before injury as a proxy for minutes
    rating_cols_before = ['Match1_before_injury_Player_rating', 'Match2_before_injury_Player_rating', 'Match3_before_injury_Player_rating']
    # Convert to numeric, coercing errors
    for col in rating_cols_before:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Minutes_Per_Match'] = df[rating_cols_before].mean(axis=1).fillna(60) * 10  # Simple heuristic
if 'Training_Load' not in df.columns:
    df['Training_Load'] = np.random.uniform(120,900,size=len(df)).round(1)
if 'Injury_Type' not in df.columns:
    df['Injury_Type'] = df['Injury'].fillna('Unknown') # Use injury column
if 'Goals' not in df.columns:
    df['Goals'] = np.random.poisson(0.6, size=len(df))
if 'Medical_Intervention' not in df.columns:
    df['Medical_Intervention'] = np.random.choice([0,1], size=len(df), p=[0.85,0.15])


# Fix End < Start by swapping
mask_swap = (df['Injury_End'] < df['Injury_Start']) & df['Injury_End'].notna() & df['Injury_Start'].notna()
if mask_swap.any():
    tmp_start = df.loc[mask_swap,'Injury_Start'].copy()
    df.loc[mask_swap,'Injury_Start'] = df.loc[mask_swap,'Injury_End']
    df.loc[mask_swap,'Injury_End'] = tmp_start

@st.cache_data
def engineer_features(df):
    d = df.copy()

    # **BUG FIX & NEW FEATURE**: Calculate Team Performance Drop
    gd_before_cols = ['Match1_before_injury_GD', 'Match2_before_injury_GD', 'Match3_before_injury_GD']
    gd_missed_cols = ['Match1_missed_match_GD', 'Match2_missed_match_GD', 'Match3_missed_match_GD']

    for col in gd_before_cols + gd_missed_cols:
        d[col] = pd.to_numeric(d[col], errors='coerce')

    avg_gd_before = d[gd_before_cols].mean(axis=1)
    avg_gd_missed = d[gd_missed_cols].mean(axis=1)

    # The drop is the performance before minus the performance during the player's absence
    d['Team_Performance_Drop'] = (avg_gd_before - avg_gd_missed).fillna(0)


    # Injury duration
    d['Injury_Duration'] = (d['Injury_End'] - d['Injury_Start']).dt.days.abs()
    d['Injury_Duration'] = d['Injury_Duration'].fillna(7).astype(int)
    # severity
    def sev(x):
        if x <= 7: return 'Mild'
        if x <=28: return 'Moderate'
        return 'Severe'
    d['Injury_Severity'] = d['Injury_Duration'].apply(sev)
    # matches missed
    d['Matches_Missed'] = (np.ceil(d['Injury_Duration'] / 7)).astype(int)
    # injury history count
    inj_hist = d.groupby('Player').size().rename('Injury_History_Count')
    d = d.merge(inj_hist.reset_index(), how='left', on='Player')
    # recent injuries 6m
    six_months = pd.Timedelta(days=180)
    d['Recent_Injuries_6m'] = 0
    d = d.sort_values('Injury_Start')
    for p in d['Player'].unique():
        idx = d['Player']==p
        rows = d.loc[idx]
        starts = rows['Injury_Start'].tolist()
        counts=[]
        for i,s in enumerate(starts):
            if pd.isna(s):
                counts.append(0); continue
            c = sum((pd.Series(starts[:i]) >= (s - six_months)) & (pd.Series(starts[:i]) < s)) if i>0 else 0
            counts.append(int(c))
        d.loc[rows.index,'Recent_Injuries_6m'] = counts
    # ACWR (Acute/Chronic) using rolling approximations grouped by player
    d = d.sort_values(['Player','Injury_Start'])
    d['Chronic_Load'] = d.groupby('Player')['Training_Load'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    d['Acute_Load'] = d.groupby('Player')['Training_Load'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())
    d['ACWR'] = (d['Acute_Load'] / (d['Chronic_Load'] + 1e-9)).round(2)
    # base risk
    severity_score = {'Mild':10,'Moderate':35,'Severe':70}
    d['_sev_score'] = d['Injury_Severity'].map(severity_score).fillna(20)
    max_hist = max(d['Injury_History_Count'].max(),1)
    d['Risk_Score'] = (
        (d['Age'] - 17) / (45 - 17) * 20
        + (d['Injury_History_Count'] / max_hist) * 20
        + d['_sev_score'] * 0.7
        + d['Recent_Injuries_6m'] * 5
        + (d['Training_Load'] / (d['Training_Load'].max() + 1e-9)) * 10
    ).clip(0,100).round(1)
    d.drop(columns=['_sev_score'], inplace=True)
    d['Load_Adjusted_Risk'] = (d['Risk_Score'] * (1 + (d['ACWR'] - 1) * 0.6)).clip(0,100).round(1)
    # position modifier
    pos_mod = {'GK':0.6,'DEF':1.0,'MID':1.1,'FWD':1.05}
    d['Position_Mod'] = d['Position'].map(pos_mod).fillna(1.0)
    d['Position_Adjusted_Risk'] = (d['Load_Adjusted_Risk'] * d['Position_Mod']).clip(0,100).round(1)
    # fatigue score
    d['Matches_14d'] = np.random.poisson(1.2, len(d)) # placeholder; real fixture data recommended
    d['Fatigue_Score'] = (d['Matches_14d'] * 10 + (d['Minutes_Per_Match']/90)*20 + (d['Age']-18)/17*10).clip(0,100).round(1)
    # performance resilience
    d['Avg_Rating_Before'] = d.groupby('Player')['Rating'].shift(1)
    d['Avg_Rating_After'] = d.groupby('Player')['Rating'].shift(-1)
    d['Performance_Resilience'] = ((d['Avg_Rating_After'] + 0.01) / (d['Avg_Rating_Before'] + 0.01)).fillna(1.0).round(2)
    # contribution / VAR
    stats = d.groupby('Player').agg({'Goals':'sum','Minutes_Per_Match':'mean'}).rename(columns={'Goals':'Goals_Sum','Minutes_Per_Match':'Avg_Min'}).reset_index()
    stats['Contribution_Index'] = (stats['Goals_Sum'] * 1.5 + stats['Avg_Min']/90).fillna(0)
    total_contrib = max(stats['Contribution_Index'].sum(),1)
    stats['Contribution_Pct'] = (stats['Contribution_Index'] / total_contrib * 100).round(2)
    d = d.merge(stats[['Player','Contribution_Pct']], how='left', on='Player')
    d['Player_VAR_pct'] = (d['Contribution_Pct'] * (d['Position_Adjusted_Risk']/100)).round(2)
    # recovery days placeholder if available as a column use it otherwise set to Injury_Duration
    d['Predicted_Recovery_Days'] = d['Injury_Duration'].copy()
    # cluster
    cf = d[['Injury_Duration','Training_Load','Age','Matches_14d']].fillna(0)
    try:
        scaler = StandardScaler()
        cf_scaled = scaler.fit_transform(cf)
        pca = PCA(n_components=2)
        comps = pca.fit_transform(cf_scaled)
        km = KMeans(n_clusters=4, random_state=0, n_init=10).fit(cf_scaled)
        d['Injury_Cluster'] = km.labels_
        d['_pcax'] = comps[:,0]; d['_pcay'] = comps[:,1]
    except Exception as e:
        print(f"Clustering failed: {e}")
        d['Injury_Cluster'] = 0; d['_pcax'] = 0; d['_pcay'] = 0
    # cumulative load
    d['Cumulative_Training_Load'] = d.groupby('Player')['Training_Load'].transform(lambda x: x.ewm(span=4, adjust=False).mean()).round(1)
    # injury type risk
    inj_type_risk = d.groupby('Injury_Type')['Risk_Score'].mean().rename('InjuryType_Risk')
    d = d.merge(inj_type_risk.reset_index(), how='left', on='Injury_Type')
    # medical interventions aggregated
    medp = d.groupby('Player')['Medical_Intervention'].sum().rename('Med_Interventions_Player')
    medc = d.groupby('Club')['Medical_Intervention'].sum().rename('Med_Interventions_Club')
    d = d.merge(medp.reset_index(), how='left', on='Player')
    d = d.merge(medc.reset_index(), how='left', on='Club')
    # readiness composite
    d['Return_Readiness'] = (100 - (d['Position_Adjusted_Risk']*0.5 + (d['Predicted_Recovery_Days']/30)*30 - (d['Performance_Resilience']-1)*20)).clip(0,100).round(1)
    # club aggregates
    d['Club_Avg_Readiness'] = d.groupby('Club')['Return_Readiness'].transform('mean').round(1)
    # forecasting placeholder - rolling mean of Load_Adjusted_Risk
    d['RecentMeanRisk'] = d.groupby('Club')['Load_Adjusted_Risk'].transform(lambda s: s.rolling(8, min_periods=1).mean())
    d['Forecast_30d'] = (d['RecentMeanRisk'] * np.random.uniform(0.96,1.08,len(d))).round(1)
    # reinjury label heuristic for training classifier (if you have true labels replace this)
    d['Reinjury_Label'] = np.where((d['Recent_Injuries_6m'] >= 1) & (d['Injury_Severity']=='Severe'), 1, 0)
    return d

# Apply engineering
try:
    df_featured = engineer_features(df)
except Exception as e:
    st.error(f"Feature engineering failed: {e}")
    st.exception(traceback.format_exc())
    st.stop()

st.sidebar.header("Models & Training")
model_choice_reg = st.sidebar.selectbox("Recovery days model", ["GradientBoostingRegressor","RandomForestRegressor"])
model_choice_clf = st.sidebar.selectbox("Re-injury model", ["RandomForestClassifier"])
retrain_btn = st.sidebar.button("Train/Re-train models (use current filtered data)")

# Training dataset selection controls
st.sidebar.markdown("Model training controls")
min_samples = st.sidebar.slider("Min samples required to train", min_value=20, max_value=500, value=50, step=10)

# Filter dataset controls for training
club_filter = st.sidebar.multiselect("Train on clubs (all if empty)", options=sorted(df_featured['Club'].unique()))
if len(club_filter) > 0:
    df_train = df_featured[df_featured['Club'].isin(club_filter)].copy()
else:
    df_train = df_featured.copy()

# Prepare features
model_features = ['Age','Injury_Duration','Injury_History_Count','Training_Load','ACWR','Matches_14d','Minutes_Per_Match','Cumulative_Training_Load','Position_Adjusted_Risk','Fatigue_Score']
for f in model_features:
    if f not in df_train.columns:
        df_train[f] = 0

# Regression: Predicted_Recovery_Days target
reg_target = 'Predicted_Recovery_Days'
# Classification: Reinjury_Label target
clf_target = 'Reinjury_Label'

# Model training function with caching
@st.cache_data
def train_models(X_reg, y_reg, X_clf, y_clf, reg_model_name, clf_model_name):
    results = {}
    # scale features for regressor
    scaler = StandardScaler()
    X_reg_scaled = scaler.fit_transform(X_reg)
    # regression model choice
    if reg_model_name == 'GradientBoostingRegressor':
        reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=0)
    else:
        reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0)
    # train / cv
    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = cross_val_score(reg, X_reg_scaled, y_reg, scoring='neg_mean_absolute_error', cv=cv)
        reg.fit(X_reg_scaled, y_reg)
        y_pred = reg.predict(X_reg_scaled)
        results['reg_model'] = reg
        results['reg_scaler'] = scaler
        results['reg_cv_mae'] = -cv_scores.mean()
        results['reg_train_mae'] = mean_absolute_error(y_reg, y_pred)
        results['reg_train_r2'] = r2_score(y_reg, y_pred)
        # permutation importance
        pi = permutation_importance(reg, X_reg_scaled, y_reg, n_repeats=10, random_state=0, n_jobs=-1)
        results['reg_perm_importance'] = pi
    except Exception as e:
        results['reg_error'] = str(e)
    # classifier
    try:
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, class_weight='balanced')
        if len(np.unique(y_clf))>1:
            clf.fit(X_clf, y_clf)
            y_pred_clf = clf.predict_proba(X_clf)[:,1]
            results['clf_model'] = clf
            # metrics
            try:
                results['clf_auc'] = roc_auc_score(y_clf, y_pred_clf)
            except:
                results['clf_auc'] = None
            results['clf_accuracy'] = accuracy_score(y_clf, clf.predict(X_clf))
            pi2 = permutation_importance(clf, X_clf, y_clf, n_repeats=10, random_state=0, n_jobs=-1)
            results['clf_perm_importance'] = pi2
        else:
            results['clf_error'] = "Insufficient label variation for training classifier"
    except Exception as e:
        results['clf_error'] = str(e)
    return results

# Only train if retrain clicked and enough samples
models_container = st.sidebar.empty()
model_results = st.session_state.get('model_results', None)

if retrain_btn:
    if len(df_train) < min_samples:
        st.sidebar.error(f"Not enough samples to train: have {len(df_train)}, need {min_samples}.")
    else:
        # training features and targets
        X_reg = df_train[model_features].fillna(0)
        y_reg = df_train[reg_target].fillna(df_train['Injury_Duration']).astype(float) # fallback target
        X_clf = df_train[model_features].fillna(0)
        y_clf = df_train[clf_target].fillna(0).astype(int)
        try:
            with st.spinner("Training models..."):
                model_results = train_models(X_reg, y_reg, X_clf, y_clf, model_choice_reg, model_choice_clf)
            st.session_state['model_results'] = model_results
            st.sidebar.success("Models trained and cached.")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")
            st.session_state['model_results'] = None
else:
    model_results = st.session_state.get('model_results', None)

# Model summary panel
st.sidebar.markdown("### Model summary")
if model_results:
    if 'reg_cv_mae' in model_results:
        st.sidebar.write(f"Recovery model CV MAE: {model_results['reg_cv_mae']:.2f}, Train MAE: {model_results['reg_train_mae']:.2f}, R²: {model_results['reg_train_r2']:.2f}")
    if 'clf_auc' in model_results and model_results.get('clf_auc') is not None:
        st.sidebar.write(f"Re-injury model AUC: {model_results['clf_auc']:.3f}, Acc: {model_results['clf_accuracy']:.3f}")
    if 'reg_error' in model_results:
        st.sidebar.error("Regression error: " + model_results['reg_error'])
    if 'clf_error' in model_results:
        st.sidebar.error("Classifier error: " + model_results['clf_error'])

st.header("Interactive Dashboard & PRO Features")

# Filter controls
col1, col2, col3 = st.columns([2,1,1])
with col1:
    clubs_sel = st.multiselect("Clubs", options=sorted(df_featured['Club'].unique()), default=sorted(df_featured['Club'].unique()))
with col2:
    positions_sel = st.multiselect("Positions", options=sorted(df_featured['Position'].unique()), default=sorted(df_featured['Position'].unique()))
with col3:
    risk_range = st.slider("Position Adjusted Risk Range", 0.0, 100.0, (0.0,100.0))

filtered = df_featured[
    (df_featured['Club'].isin(clubs_sel)) &
    (df_featured['Position'].isin(positions_sel)) &
    (df_featured['Position_Adjusted_Risk'] >= risk_range[0]) &
    (df_featured['Position_Adjusted_Risk'] <= risk_range[1])
].copy()

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()


st.subheader("Top KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Rating", f"{filtered['Rating'].mean():.2f}")
k2.metric("Avg Position-Adjusted Risk", f"{filtered['Position_Adjusted_Risk'].mean():.1f}")
k3.metric("Avg Return Readiness", f"{filtered['Return_Readiness'].mean():.1f}")
k4.metric("Avg Injury Duration (days)", f"{filtered['Injury_Duration'].mean():.1f}")
k5.metric("Avg Re-injury Prob (%)", f"{filtered['Reinjury_Label'].mean()*100:.1f}")

# KPI auto insight
kpi_insight = []
kpi_insight.append(f"Rating mean={filtered['Rating'].mean():.2f}, std={filtered['Rating'].std():.2f}")
kpi_insight.append(f"Position_Adjusted_Risk mean={filtered['Position_Adjusted_Risk'].mean():.1f}")
kpi_insight.append(f"Return_Readiness mean={filtered['Return_Readiness'].mean():.1f}")
st.info(" | ".join(kpi_insight))

# Trends chart + auto analysis
st.subheader("Players with largest average Team Performance Drop")
impact = filtered.groupby('Player')['Team_Performance_Drop'].mean().sort_values(ascending=True).tail(15).reset_index()
fig1 = px.bar(impact, x='Team_Performance_Drop', y='Player', orientation='h', color='Team_Performance_Drop', title="Average Team Performance Drop by Player", labels={'Team_Performance_Drop': 'Avg Goal Difference Drop (Before - During Injury)', 'Player': 'Player Name'})
fig1.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig1, use_container_width=True)

if len(impact)>0:
    top_players = impact.sort_values('Team_Performance_Drop', ascending=False)['Player'].head(3).tolist()
    st.info(f"AI Insight: Highest average team performance drops associated with: {', '.join(top_players)}. These players' absences correlate with the biggest drop in team goal difference.")

# Fatigue vs performance drop
st.subheader("Fatigue vs Team Performance Drop")
fig2 = px.scatter(filtered, x='Fatigue_Score', y='Team_Performance_Drop', color='Position', hover_data=['Player','Club'], title="Fatigue vs Team Performance Drop")
st.plotly_chart(fig2, use_container_width=True)
r,p = pearson_r(filtered['Fatigue_Score'], filtered['Team_Performance_Drop'])
if not np.isnan(r):
    strength = "strong" if abs(r)>=0.5 else "moderate" if abs(r)>=0.25 else "weak"
    st.info(f"AI Insight: Correlation between Fatigue Score and Team Performance Drop is r={r:.2f} ({strength}).")

# Clustering visualization
st.subheader("Injury Clusters (PCA scatter)")
fig3 = px.scatter(filtered, x='_pcax', y='_pcay', color='Injury_Cluster', hover_data=['Player','Injury_Type','Position'], title="Injury clusters by load/age/duration")
st.plotly_chart(fig3, use_container_width=True)
# cluster insight
clusters = filtered['Injury_Cluster'].unique()
st.info(f"AI Insight: Found {len(clusters)} cluster(s) in current selection. Use cluster labels to design tailored RTP (return-to-play) protocols.")

# Model explainability panels (if models trained)
st.header("Model Explainability & Performance")
if model_results:
    if 'reg_model' in model_results:
        st.subheader("Recovery Days Model (Permutation Importance)")
        reg_pi = model_results.get('reg_perm_importance', None)
        if reg_pi is not None:
            # map back to features
            importances = pd.DataFrame({
                'feature': model_features,
                'importance_mean': reg_pi.importances_mean,
                'importance_std': reg_pi.importances_std
            }).sort_values('importance_mean', ascending=True)
            fig_imp = px.bar(importances.tail(10), x='importance_mean', y='feature', orientation='h', title="Top 10 Features for Recovery Days Model")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info(f"AI Insight: Top regressor features: {importances['feature'].iloc[-1]}, {importances['feature'].iloc[-2]}")
        else:
            st.write("No permutation importance available for regression.")
    if 'clf_model' in model_results:
        st.subheader("Re-injury Classifier (Permutation Importance)")
        clf_pi = model_results.get('clf_perm_importance', None)
        if clf_pi is not None:
            importances_clf = pd.DataFrame({
                'feature': model_features,
                'importance_mean': clf_pi.importances_mean,
                'importance_std': clf_pi.importances_std
            }).sort_values('importance_mean', ascending=True)
            fig_imp_clf = px.bar(importances_clf.tail(10), x='importance_mean', y='feature', orientation='h', title="Top 10 Features for Re-Injury Classifier")
            st.plotly_chart(fig_imp_clf, use_container_width=True)
            st.info(f"AI Insight: Top classifier features: {importances_clf['feature'].iloc[-1]}, {importances_clf['feature'].iloc[-2]}")
        else:
            st.write("No permutation importance available for classifier.")

else:
    st.info("Models not trained yet. Use the sidebar 'Train/Re-train models' button to build models on the current selection.")


# Automated natural-language insights that combine stats + model explanations
st.header("Automated Analyst — Insights & Actions")
insights = []

# Example insight: Age vs Risk
r_age_risk, p_age_risk = pearson_r(filtered['Age'], filtered['Position_Adjusted_Risk'])
if not np.isnan(r_age_risk):
    desc = "positive" if r_age_risk>0 else "negative"
    insights.append(f"Players' age has a {desc} correlation with Position_Adjusted_Risk (r={r_age_risk:.2f}). Action: review veteran load management.")

# Fatigue and drop
r_fat_drop, _ = pearson_r(filtered['Fatigue_Score'], filtered['Team_Performance_Drop'])
if not np.isnan(r_fat_drop) and abs(r_fat_drop) >= 0.25:
    insights.append(f"Fatigue shows a meaningful association with team performance drop (r={r_fat_drop:.2f}). Action: prioritize rotations during congested periods.")

# Top clubs by forecasted risk
club_forecast = filtered.groupby('Club')['Forecast_30d'].mean().sort_values(ascending=False)
if len(club_forecast)>0:
    top_club = club_forecast.index[0]
    insights.append(f"{top_club} has the highest 30-day forecasted risk ({club_forecast.iloc[0]:.1f}). Action: perform targeted medical reviews and depth planning.")

# Outliers detection in fatigue
outliers_idx = zscore_outliers(filtered['Fatigue_Score'])
if outliers_idx:
    sample_out = filtered.iloc[outliers_idx][['Player','Club','Fatigue_Score']].head(5)
    insights.append(f"Found high-fatigue outliers (examples: {', '.join(sample_out['Player'].tolist())}). Action: immediate load reduction and monitoring recommended.")

# Model-driven insight
if model_results and 'reg_perm_importance' in model_results and model_results.get('reg_perm_importance') is not None:
    pi_df = pd.DataFrame({
        'feature': model_features,
        'importance': model_results['reg_perm_importance'].importances_mean
    }).sort_values('importance', ascending=False)
    top_feats = pi_df['feature'].head(3).tolist()
    insights.append(f"Model insight: Recovery days are strongly influenced by {', '.join(top_feats)}. Consider tailoring rehab to these drivers.")

# Show insights (concise)
if len(insights) == 0:
    st.info("No strong automated insights found for the current filters.")
else:
    for ins in insights:
        st.success("💡 AI Insight: " + ins)

st.header("Exports")
st.markdown("Download the fully-engineered, filtered dataset as a CSV file.")

# CSV download
csv = filtered.to_csv(index=False)
st.download_button("Download filtered CSV", data=csv, file_name="footlens_engineered_filtered.csv", mime="text/csv")

with st.expander("Debug & Notes (click to expand)"):
    st.write("Data shape (after feature engineering):", df_featured.shape)
    st.write("Sample of engineered features:")
    st.dataframe(df_featured[['Player', 'Club', 'Team_Performance_Drop', 'Position_Adjusted_Risk', 'Fatigue_Score', 'Return_Readiness']].head(10))
    st.write("Current filter length:", len(filtered))
    if model_results:
        st.write("Models trained and cached in session_state.")
    else:
        st.write("Models not trained in this session. Press 'Train/Re-train models' in the sidebar to build them.")

st.markdown("---")
st.caption("FootLens PRO — Ultra Advanced. For production: replace synthetic placeholders (fixture logs, GPS load, true recovery labels) with real inputs for improved accuracy.")
