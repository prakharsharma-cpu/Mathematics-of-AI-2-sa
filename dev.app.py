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
st.set_page_config(page_title="⚽ FootLens — Ultimate Elite Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("⚽ FootLens — Ultimate Elite Player Injury Dashboard")
st.markdown("Analyze, Predict, Compare, and Simulate Player Injuries and Performance with Pro Insights!")

# ---------------------------------------------
# Data Upload / Simulation
# ---------------------------------------------

# Using st.cache_data to ensure data loading and processing happens only once.
@st.cache_data
def load_data(uploaded_file=None):
    """
    Loads data from an uploaded CSV or generates a simulated dataset.
    It then performs extensive feature engineering to create advanced metrics for analysis.
    """
    if uploaded_file is not None:
        # Read uploaded data, ensuring date columns are parsed correctly.
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['Injury_Start', 'Injury_End'], dayfirst=True)
        except Exception as e:
            st.error(f"Error parsing the uploaded file: {e}")
            return pd.DataFrame()
    else:
        # Generate a realistic-looking simulated dataset if no file is uploaded.
        np.random.seed(42)
        players = [f"Player_{i}" for i in range(1, 21)]
        clubs = [f"Club_{i}" for i in range(1, 6)]
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="15D")
        injury_types = ["Hamstring", "Groin", "ACL", "Ankle", "Calf", "Back"]
        injury_starts = np.random.choice(dates, 300)
        injury_durations = np.random.randint(7, 120, 300)
        data = {
            "Player": np.random.choice(players, 300),
            "Club": np.random.choice(clubs, 300),
            "Rating": np.random.uniform(5, 9, 300),
            "Goals": np.random.randint(0, 6, 300),
            "Team_Goals_Before": np.random.randint(10, 30, 300),
            "Team_Goals_During": np.random.randint(5, 25, 300),
            "Age": np.random.randint(18, 35, 300),
            "Injury_Start": pd.to_datetime(injury_starts),
            "Injury_End": [s + pd.Timedelta(days=d) for s, d in zip(injury_starts, injury_durations)],
            "Status": np.random.choice(["Before", "During", "After"], 300),
            "Injury_Type": np.random.choice(injury_types, 300)
        }
        df = pd.DataFrame(data)

    # --- Data Cleaning and Core Feature Engineering ---
    df.drop_duplicates(inplace=True)
    
    # CRITICAL FIX: Sort data by Player and Injury_Start date to ensure calculations like shift() are correct.
    df.sort_values(['Player', 'Injury_Start'], inplace=True)
    
    df['Injury_Duration'] = (df['Injury_End'] - df['Injury_Start']).dt.days.clip(lower=1)
    
    # Correctly calculate metrics based on the previous and next injury for each player.
    df['Avg_Rating_Before'] = df.groupby('Player')['Rating'].shift(1)
    df['Avg_Rating_After'] = df.groupby('Player')['Rating'].shift(-1)
    df['Performance_Change'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
    df['Days_Since_Last_Injury'] = (df['Injury_Start'] - df.groupby('Player')['Injury_End'].shift(1)).dt.days.fillna(0)
    
    # Continue with other feature engineering
    df['Team_Performance_Drop'] = df['Team_Goals_Before'] - df['Team_Goals_During']
    df['Impact_Index'] = (df['Team_Performance_Drop'] / df['Injury_Duration']).replace([np.inf, -np.inf], 0).fillna(0)
    df['Severity_Score'] = df['Injury_Duration'] * (10 - df['Rating']) / 10
    df['Month'] = df['Injury_Start'].dt.month

    # --- Advanced Feature Engineering (20+ Features) ---
    df['ERI'] = df['Severity_Score'] / df['Injury_Duration'].replace(0, 1)  # Expected Recovery Index
    df['Recurrence_180d'] = np.random.randint(0, 2, len(df)) # For demo purposes
    df['Club_Resilience'] = df.groupby('Club')['Impact_Index'].transform('mean')
    df['Goals_Lost'] = df['Team_Goals_Before'] - df['Goals']
    df['Rating_Drop'] = df['Avg_Rating_Before'] - df['Rating']
    df['Age_Category'] = pd.cut(df['Age'], bins=[17, 20, 25, 30, 35], labels=['Young', 'Early', 'Prime', 'Late'])
    df['High_Impact'] = (df['Impact_Index'] > df['Impact_Index'].quantile(0.75)).astype(int)
    df['Recovery_Trend'] = df['Avg_Rating_After'] - df['Rating']
    df['Injury_Frequency'] = df.groupby('Player')['Player'].transform('count')
    df['Performance_Consistency'] = df.groupby('Player')['Rating'].transform(lambda x: x.std())
    df['Team_Performance_Percent_Drop'] = (df['Team_Performance_Drop'] / df['Team_Goals_Before'].replace(0, 1)) * 100
    df['Severity_Category'] = pd.qcut(df['Severity_Score'], 4, labels=['Low', 'Moderate', 'High', 'Critical'], duplicates='drop')
    df['Cumulative_Injury_Duration'] = df.groupby('Player')['Injury_Duration'].cumsum()
    df['Recent_Injury'] = (df['Injury_Start'] > pd.Timestamp('2022-01-01'))
    df['Impact_Per_Goal'] = df['Impact_Index'] / df['Goals'].replace(0, 1)
    df['Normalized_Impact'] = (df['Impact_Index'] - df['Impact_Index'].min()) / (df['Impact_Index'].max() - df['Impact_Index'].min())
    df['Rating_Per_Day'] = df['Rating'] / df['Injury_Duration'].replace(0, 1)
    df['Goal_Contribution'] = df['Goals'] / df['Team_Goals_Before'].replace(0, 1)
    df['Weighted_Impact'] = df['Impact_Index'] * df['Severity_Score']

    return df.reset_index(drop=True)

# ---------------------------------------------
# Sidebar & Filtering
# ---------------------------------------------
st.sidebar.header("🔍 Data Input & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV for EDA & Analysis", type=['csv'])

# Load data
df = load_data(uploaded_file)

# NEW: Implement Dashboard Mode functionality
mode = st.sidebar.radio("Dashboard Mode", ["Executive", "Analyst"], index=1, help="Executive mode shows high-level summaries. Analyst mode provides deep-dive tools.")

st.sidebar.header("📊 Filter Your Data")
# Apply filters based on the loaded data
if not df.empty:
    filter_club = st.sidebar.multiselect("Club", df['Club'].unique(), default=df['Club'].unique())
    filter_player = st.sidebar.multiselect("Player", df['Player'].unique(), default=df['Player'].unique())
    filter_injury = st.sidebar.multiselect("Injury Type", df['Injury_Type'].unique(), default=df['Injury_Type'].unique())
    
    # Filter the dataframe based on selections
    filtered_df = df[(df['Club'].isin(filter_club)) & (df['Player'].isin(filter_player)) & (df['Injury_Type'].isin(filter_injury))]
else:
    filtered_df = pd.DataFrame()

# ---------------------------------------------
# Main Panel
# ---------------------------------------------

# NEW: Add a check to handle empty dataframe after filtering
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your selections in the sidebar.")
else:
    # ---------------------------------------------
    # KPIs Section
    # ---------------------------------------------
    st.markdown("### 🚀 Key Performance Indicators")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("⚽ Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
    k2.metric("💥 Avg Team Drop", f"{filtered_df['Team_Performance_Drop'].mean():.2f}", help="Average drop in team goals when a player is injured.")
    k3.metric("🩹 Total Injuries", f"{len(filtered_df)}")
    k4.metric("🔥 Avg Severity", f"{filtered_df['Severity_Score'].mean():.2f}", help="A measure combining injury duration and player rating.")
    k5.metric("📊 Avg Impact Index", f"{filtered_df['Impact_Index'].mean():.2f}", help="Team performance drop per day of injury.")
    k6.metric("📈 Avg Weighted Impact", f"{filtered_df['Weighted_Impact'].mean():.2f}", help="Impact Index weighted by the Severity Score.")
    st.markdown("---")

    # ---------------------------------------------
    # Dashboard Mode Implementation
    # ---------------------------------------------
    
    # --- EXECUTIVE MODE ---
    if mode == 'Executive':
        st.header("Executive Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 10 Most Impactful Players")
            top_players = filtered_df.groupby('Player')['Impact_Index'].mean().sort_values(ascending=False).head(10).reset_index()
            fig1 = px.bar(top_players, x='Impact_Index', y='Player', orientation='h', 
                          color='Impact_Index', color_continuous_scale='Reds',
                          labels={'Player': 'Player Name', 'Impact_Index': 'Average Impact Index'})
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.subheader("Injury Counts by Club")
            club_counts = filtered_df['Club'].value_counts().reset_index()
            club_counts.columns = ['Club', 'Count']
            fig2 = px.pie(club_counts, names='Club', values='Count', hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig2, use_container_width=True)

    # --- ANALYST MODE ---
    else:
        # ---------------------------------------------
        # Tabs for detailed analysis
        # ---------------------------------------------
        tabs = st.tabs(["🧾 Dataset Overview", "📊 Trends & Club Impact", "📈 Player Performance", "🧠 Prediction & Similarity", "🔎 Deep Dive & Simulation"])

        # Tab 1: Dataset Overview
        with tabs[0]:
            st.subheader("Dataset Preview & Stats")
            st.dataframe(filtered_df.head(), use_container_width=True)
            with st.expander("Show Statistical Summary & Correlation Matrix"):
                st.write("Descriptive Statistics")
                st.dataframe(filtered_df.describe())
                st.write("Correlation Heatmap")
                corr = filtered_df.select_dtypes(include=np.number).corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)

        # Tab 2: Trends & Club Impact
        with tabs[1]:
            st.subheader("Top Players by Team Performance Drop")
            top_players = filtered_df.groupby('Player')['Team_Performance_Drop'].mean().sort_values(ascending=False).head(10).reset_index()
            fig1 = px.bar(top_players, x='Team_Performance_Drop', y='Player', orientation='h', color='Team_Performance_Drop', color_continuous_scale='Reds')
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Club Monthly Injury Heatmap")
            heatmap_data = filtered_df.groupby(['Club', 'Month']).size().reset_index(name='Count')
            fig2 = px.density_heatmap(heatmap_data, x='Month', y='Club', z='Count', color_continuous_scale='Blues',
                                      labels={'Month': 'Month of Year', 'Club': 'Club', 'Count': 'Number of Injuries'})
            st.plotly_chart(fig2, use_container_width=True)

        # Tab 3: Player Performance
        with tabs[2]:
            st.subheader("Comeback Kings: Top 10 Players by Performance Change Post-Injury")
            leaderboard = filtered_df.groupby('Player')['Performance_Change'].mean().sort_values(ascending=False).head(10).reset_index()
            st.dataframe(leaderboard, use_container_width=True)

            st.subheader("Average Rating by Injury Phase")
            status_avg = filtered_df.groupby('Status')['Rating'].mean().reset_index()
            fig3 = px.bar(status_avg, x='Status', y='Rating', color='Status', color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig3, use_container_width=True)

        # Tab 4: Prediction & Similarity
        with tabs[4]:
            st.header("🔎 Player Deep Dive")
            player_dive = st.selectbox("Select a Player for Deep Dive Analysis", filtered_df['Player'].unique())
            player_data = filtered_df[filtered_df['Player'] == player_dive]
            
            st.subheader(f"Injury Timeline for {player_dive}")
            fig_timeline = px.timeline(player_data, x_start="Injury_Start", x_end="Injury_End", y="Injury_Type", color="Severity_Category",
                                       hover_data=['Injury_Duration', 'Severity_Score'])
            fig_timeline.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_timeline, use_container_width=True)

            st.subheader("⚙️ What-If Scenario Simulation")
            duration_increase = st.slider("Increase Avg Injury Duration by (%)", 0, 100, 10)
            
            original_avg_duration = player_data['Injury_Duration'].mean()
            simulated_avg_duration = original_avg_duration * (1 + duration_increase / 100)
            
            original_total_severity = player_data['Severity_Score'].sum()
            simulated_total_severity = (player_data['Severity_Score'] / player_data['Injury_Duration'] * simulated_avg_duration).sum()
            
            c1, c2 = st.columns(2)
            c1.metric("Original Avg Duration (Days)", f"{original_avg_duration:.1f}")
            c1.metric("Simulated Avg Duration (Days)", f"{simulated_avg_duration:.1f}")
            c2.metric("Original Total Severity Score", f"{original_total_severity:.1f}")
            c2.metric("Simulated Total Severity Score", f"{simulated_total_severity:.1f}", delta=f"{simulated_total_severity - original_total_severity:.1f}")
            
        # Tab 5: Deep Dive & Simulation (NEW)
        with tabs[3]:
            st.subheader("High-Impact Injury Prediction")
            
            # --- Machine Learning Model ---
            # CORRECTED LOGIC: Predict a meaningful binary target like 'High_Impact' instead of random data.
            features_for_model = ['Age', 'Rating', 'Injury_Duration', 'Severity_Score', 'Days_Since_Last_Injury']
            model_data = filtered_df[features_for_model].dropna()
            
            if not model_data.empty:
                X = StandardScaler().fit_transform(model_data)
                y = filtered_df.loc[model_data.index, 'High_Impact'] # Using a logical, derived target
                
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X, y)
                
                # Assign probabilities back to the main dataframe for visualization
                filtered_df.loc[model_data.index, 'High_Impact_Prob'] = clf.predict_proba(X)[:, 1]
                
                fig4 = px.scatter(filtered_df, x='Severity_Score', y='High_Impact_Prob', color='Club',
                                  hover_data=['Player'], title="Predicted Probability of a High-Impact Injury")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("Not enough data to train the prediction model.")

            st.markdown("---")
            st.subheader("Player Similarity Engine")
            
            # Use a smaller, unique set of players for the similarity matrix to avoid confusion
            sim_features_df = filtered_df.groupby('Player').agg({
                'Age': 'mean',
                'Rating': 'mean',
                'Severity_Score': 'mean',
                'Injury_Duration': 'mean'
            }).reset_index()

            if len(sim_features_df) > 1:
                features = ['Age', 'Rating', 'Severity_Score', 'Injury_Duration']
                # Scale features for accurate similarity calculation
                scaled_features = StandardScaler().fit_transform(sim_features_df[features])
                sim_matrix = cosine_similarity(scaled_features)

                player_choice = st.selectbox("Select Player for Similarity", options=sim_features_df['Player'].unique())
                
                # Find the index for the chosen player
                idx = sim_features_df[sim_features_df['Player'] == player_choice].index[0]
                
                sim_scores = list(enumerate(sim_matrix[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] # Get top 5 similar
                
                similar_players_indices = [i[0] for i in sim_scores]
                similar_players_scores = [i[1] for i in sim_scores]
                
                # Create a dataframe to display results
                result_df = sim_features_df.iloc[similar_players_indices].copy()
                result_df['Similarity_Score'] = similar_players_scores
                
                st.dataframe(result_df[['Player', 'Similarity_Score'] + features], use_container_width=True)
            else:
                st.warning("Not enough players to run the similarity engine. Please broaden your filters.")


    # ---------------------------------------------
    # NEW: Download Report Feature
    # ---------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export & Download")
    if st.sidebar.button("Generate & Download Report"):
        # Create a zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            # Save filtered data to CSV
            csv_buffer = BytesIO()
            filtered_df.to_csv(csv_buffer, index=False)
            zip_file.writestr("filtered_data.csv", csv_buffer.getvalue())

            # Save a key plot (e.g., top players chart)
            if 'fig1' in locals():
                fig_buffer = BytesIO()
                fig1.write_image(fig_buffer, format='png')
                zip_file.writestr("top_impact_players.png", fig_buffer.getvalue())

        st.sidebar.download_button(
            label="Download ZIP Report",
            data=zip_buffer.getvalue(),
            file_name="FootLens_Report.zip",
            mime="application/zip"
        )
