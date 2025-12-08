import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="FootLens Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Professional Look ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Step 1 & 2: Data Loading & Preprocessing ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None

    # 1. Handle Missing Values
    df.replace("N.A.", np.nan, inplace=True)
    
    # 2. Convert Dates
    df['Date of Injury'] = pd.to_datetime(df['Date of Injury'], format='%b %d, %Y', errors='coerce')
    df['Date of return'] = pd.to_datetime(df['Date of return'], format='%b %d, %Y', errors='coerce')
    
    # 3. Clean Numeric Columns (Function to handle '(S)' and strings)
    def clean_numeric(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            val = val.replace('(S)', '').strip()
            try:
                return float(val)
            except ValueError:
                return np.nan
        return float(val)

    # Identify numeric columns for cleaning
    rating_cols_before = [f'Match{i}_before_injury_Player_rating' for i in range(1, 4)]
    rating_cols_after = [f'Match{i}_after_injury_Player_rating' for i in range(1, 4)]
    gd_cols_before = [f'Match{i}_before_injury_GD' for i in range(1, 4)]
    gd_cols_missed = [f'Match{i}_missed_match_GD' for i in range(1, 4)]
    
    numeric_cols = rating_cols_before + rating_cols_after + gd_cols_before + gd_cols_missed
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)

    # 4. Feature Engineering
    # Calculate Averages
    df['Avg_Rating_Before'] = df[rating_cols_before].mean(axis=1)
    df['Avg_Rating_After'] = df[rating_cols_after].mean(axis=1)
    df['Avg_GD_Before'] = df[gd_cols_before].mean(axis=1)
    df['Avg_GD_Missed'] = df[gd_cols_missed].mean(axis=1)
    
    # Calculate KPIs
    # Performance Drop: If GD Before (e.g. +1) is higher than Missed (e.g. -1), Index is 2 (Big Drop)
    df['Performance_Drop_Index'] = df['Avg_GD_Before'] - df['Avg_GD_Missed']
    df['Rating_Improvement'] = df['Avg_Rating_After'] - df['Avg_Rating_Before']
    df['Recovery_Duration'] = (df['Date of return'] - df['Date of Injury']).dt.days
    
    # Month Name for Seasonality Analysis
    df['Injury_Month'] = df['Date of Injury'].dt.month_name()
    
    # Filter out invalid recovery durations (data errors)
    df = df[df['Recovery_Duration'] > 0]
    
    return df

# Load Data
DATA_FILE = 'player_injuries_impact (2).csv'
df = load_data(DATA_FILE)

# --- Sidebar Controls ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/53/53283.png", width=100)
st.sidebar.title("FootLens Analytics")
st.sidebar.markdown("Filter the dashboard views:")

if df is not None:
    # Filters
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=sorted(df['Team Name'].unique()),
        default=sorted(df['Team Name'].unique())[:5]
    )
    
    selected_positions = st.sidebar.multiselect(
        "Select Positions",
        options=sorted(df['Position'].unique()),
        default=sorted(df['Position'].unique())
    )
    
    # Apply Filters
    if not selected_teams:
        df_filtered = df.copy()
    else:
        df_filtered = df[df['Team Name'].isin(selected_teams)]
        
    if selected_positions:
        df_filtered = df_filtered[df_filtered['Position'].isin(selected_positions)]

    # --- Main Dashboard Layout ---
    st.title("⚽ Player Injury Impact & Performance Dashboard")
    st.markdown("### Interactive Insights for Squad Planning")

    # 1. Top Level Metrics (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Injuries Analysis", f"{len(df_filtered)}")
    with col2:
        st.metric("Avg Recovery Time", f"{df_filtered['Recovery_Duration'].mean():.0f} days")
    with col3:
        avg_drop = df_filtered['Performance_Drop_Index'].mean()
        st.metric("Avg Team Performance Drop", f"{avg_drop:.2f}", delta_color="inverse")
    with col4:
        avg_comeback = df_filtered['Rating_Improvement'].mean()
        st.metric("Avg Comeback Rating", f"{avg_comeback:.2f}", delta=f"{avg_comeback:.2f}")

    st.markdown("---")

    # 2. Visualizations Row 1
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("📉 Top 10 Injuries: Highest Team Performance Drop")
        st.caption("Which injuries caused the team's Goal Difference (GD) to suffer the most?")
        
        # Prepare Data
        top_drop = df_filtered.sort_values(by='Performance_Drop_Index', ascending=False).head(10)
        
        fig_bar = px.bar(
            top_drop,
            x='Performance_Drop_Index',
            y='Name',
            color='Team Name',
            orientation='h',
            hover_data=['Injury', 'Recovery_Duration'],
            labels={'Performance_Drop_Index': 'GD Drop Index (Higher = Worse Impact)'},
            text_auto='.1f'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("🗓️ Injury Seasonality")
        st.caption("Frequency of injuries by month and club.")
        
        # Order months correctly
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        
        # Aggregate data for heatmap
        heatmap_data = df_filtered.groupby(['Team Name', 'Injury_Month']).size().reset_index(name='Count')
        
        fig_heat = px.density_heatmap(
            heatmap_data,
            x='Injury_Month',
            y='Team Name',
            z='Count',
            category_orders={'Injury_Month': months_order},
            color_continuous_scale='Reds',
            labels={'Count': 'Injuries'}
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # 3. Visualizations Row 2
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("👴 Age vs. Performance Drop")
        st.caption("Do older players leave a bigger gap in the team when injured?")
        
        fig_scatter = px.scatter(
            df_filtered,
            x='Age',
            y='Performance_Drop_Index',
            color='Position',
            size='Recovery_Duration',
            hover_name='Name',
            hover_data=['Injury', 'Team Name'],
            title="Age Impact Analysis (Bubble Size = Recovery Days)"
        )
        # Add a reference line at 0
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right2:
        st.subheader("🚀 Comeback Leaderboard")
        st.caption("Players with biggest rating improvement after returning.")
        
        top_comebacks = df_filtered.sort_values(by='Rating_Improvement', ascending=False).head(10)
        
        # Create a nice table using Streamlit's dataframe configuration
        st.dataframe(
            top_comebacks[['Name', 'Team Name', 'Injury', 'Rating_Improvement', 'Recovery_Duration']],
            column_config={
                "Rating_Improvement": st.column_config.ProgressColumn(
                    "Rating Boost",
                    help="Improvement in FIFA rating post-injury",
                    format="%.2f",
                    min_value=-2,
                    max_value=4,
                ),
                "Recovery_Duration": st.column_config.NumberColumn(
                    "Days Out",
                    format="%d days"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    # 4. Deep Dive Section
    st.markdown("---")
    st.subheader("🔍 Player Timeline Deep Dive")
    
    player_list = sorted(df['Name'].unique())
    selected_player = st.selectbox("Select a Player to view their Injury Timeline", player_list)
    
    player_data = df[df['Name'] == selected_player].sort_values('Date of Injury')
    
    if not player_data.empty:
        # Create a dual-line plot for Before vs After ratings
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=player_data['Date of Injury'], 
            y=player_data['Avg_Rating_Before'],
            mode='lines+markers',
            name='Rating Before Injury',
            line=dict(color='#ff4b4b', dash='dash')
        ))
        
        fig_line.add_trace(go.Scatter(
            x=player_data['Date of return'], 
            y=player_data['Avg_Rating_After'],
            mode='lines+markers',
            name='Rating After Return',
            line=dict(color='#00c853')
        ))
        
        fig_line.update_layout(
            title=f"Performance Timeline: {selected_player}",
            xaxis_title="Timeline",
            yaxis_title="Average Match Rating",
            hovermode="x unified"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Show raw data for this player below the chart
        with st.expander(f"View Raw Data for {selected_player}"):
            st.dataframe(player_data[['Injury', 'Date of Injury', 'Date of return', 'Recovery_Duration', 'Performance_Drop_Index', 'Rating_Improvement']])
            
    else:
        st.warning("No data found for this player.")

else:
    st.error("Dataset not found. Please ensure 'player_injuries_impact (2).csv' is in the directory.")
