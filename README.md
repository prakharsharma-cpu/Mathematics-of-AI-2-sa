⚽ Player Injury and Team Performance Dashboard

This project is a Streamlit-based analytical dashboard designed to explore the impact of football player injuries on team performance, match outcomes, and player recovery/comeback ratings. This dashboard was developed as the Summative Assessment (SA) for the 'Mathematics for AI-II' course.

🔗 Links

Component
LinkLive Streamlit App
https://mathematics-of-ai-2-sa-3e9znjttnwalmh3jgankda.

streamlit.app 
GitHub Repository
https://github.com/prakharsharma-cpu/Mathematics-of-AI-2-sa 3

🎯 Objective

The main goal was to create an interactive dashboard that provides a full analytical view of player injuries across seasons.

Key questions addressed by the analysis:
Which injuries caused the biggest team performance drop (Goal Difference)?
Which clubs and months have the highest number of injuries (Seasonality)?
How does a player’s rating change after returning from an injury?
Does age affect performance drop or recovery duration?

✨ Features and Methodology

The project included a complete data science pipeline:

1. Data Processing & Feature Engineering 

Cleaning: Handled missing values (N.A./empty values converted to NaN), converted dates to datetime, and cleaned numeric columns.
Engineered Features: Created new columns essential for the analysis:
Performance_Drop_Index = GD before − GD missed 
Rating_Improvement = Rating after − Rating before 
Recovery_Duration in days 
Injury_Month (for seasonality) 

2. Dashboard Interface (UI/UX) 

Filters: Interactive sidebar filters for Team, Position, and Player selection.
KPI Cards: Quick summary statistics are displayed, including Total Injuries, Average Recovery Duration, Average Performance Drop, and Average Comeback Rating.

3. Visualisations (Plotly)

The dashboard features more than five interactive Plotly charts, directly addressing the business questions.

Top 10 Injuries by GD Drop (Bar Chart) 

Injury Seasonality by Month & Club (Heatmap)

Age vs Performance Drop (Scatter plot, bubble size = recovery duration) 

Player Rating Timeline (Line Chart showing 'Rating Before' vs. 'Rating After' injury)

🚀 Deployment

The final dashboard was deployed using GitHub and Streamlit Cloud.
Repository Contents: app.py (main code), requirements.txt (dependencies), and README.md.
Process: The GitHub repository was linked to Streamlit Cloud, and the app.py file was selected for deployment, generating the live application link.
