# Milestone 4 – Dashboard for Health Anomaly Insights

## Objective
The objective of this milestone is to develop an interactive dashboard that visualizes health data from fitness devices and highlights anomalous patterns related to heart rate, sleep, and physical activity. The dashboard enables dynamic analysis and report generation for health insights.

##**Dataset used :-** 
final_dataset(1)

## Dashboard Workflow
1. User uploads fitness data (CSV) through the Streamlit dashboard.
2. Uploaded data is sent to the backend for preprocessing.
3. Cleaned data is used for feature extraction and anomaly detection.
4. Users can interactively explore trends and anomalies using filters.
5. Anomaly summaries and reports can be downloaded as CSV files.(health_alerts.csv in data file)

## Tools & Libraries Used
- Streamlit (Dashboard UI)
- FastAPI (Backend API)
- Pandas, NumPy (Data processing)
- Plotly (Interactive visualizations)
- Prophet (Time-series modeling)
- Scikit-learn (Clustering & anomaly detection)
- Google Colaboratory
- ngrok (Public access)

## Key Insights from the Dashboard
- Heart rate trends help identify abnormal spikes and drops.
- Sleep duration analysis highlights insufficient or excessive sleep patterns.
- Step count behavior reveals irregular physical activity.
- Anomaly severity is categorized into Low, Medium, and High risk levels.

## Screenshot References
- Dashboard UI: `screenshots/dashboard_ui.png`
- Report Download: `screenshots/report_download.png`

