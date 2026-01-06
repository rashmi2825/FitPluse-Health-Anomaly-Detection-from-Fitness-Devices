
# Milestone 3: Anomaly Detection and Visualization

## Objective

The objective of Milestone 3 is to detect, label, and visualize anomalies in fitness time-series data generated from wearable devices. The focus is on identifying unusual patterns in heart rate, sleep duration, and physical activity to support early health risk detection and preventive healthcare insights. This milestone builds on the preprocessed dataset generated in Milestone 1.

## Dataset Used

The analysis is performed using the **final merged dataset created in Milestone 1**, which integrates multiple fitness metrics into a unified structure.

### Key features used:
- **Id** – Unique user identifier  
- **Date / Timestamp** – Temporal information  
- **HeartRate** – Minute-level heart rate readings  
- **TotalSteps** – Daily step count  
- **SleepFlag / TotalSleepMinutes** – Sleep activity information  

A subset of users was selected to ensure clarity, interpretability, and efficient analysis.

## Steps Followed

### 1. Residual Analysis (Heart Rate)

- Heart rate data was aggregated at the daily level.
- A time-series forecasting model was applied to estimate expected heart rate trends.
- **Residuals** were computed as the difference between observed and predicted values.
- Large residuals indicated potential heart rate anomalies.

### 2. Threshold-Based Anomaly Detection

- Statistical thresholds were applied to residual values.
- Heart rate deviations beyond acceptable limits were flagged as anomalies.
- Sleep duration anomalies were detected using domain rules:
  - Very low sleep duration
  - Excessively high sleep duration
 
### 3. Cluster-Based Detection (Behavioral Patterns)

- User behavior patterns were analyzed across multiple metrics.
- Clustering techniques were used to group similar activity profiles.
- Outlier clusters were interpreted as abnormal or risky behavioral patterns.

### 4. Visualization

Multiple visualizations were created to clearly communicate results:

- **Heart Rate Anomaly Plots** highlighting abnormal fluctuations
- **Sleep Pattern Analysis** showing irregular sleep durations
- **Step Count Visualizations** comparing daily activity against recommended thresholds
- **User-wise Health Summary Charts** aggregating normal and anomalous days

These visualizations help translate technical results into intuitive health insights.

## Tools and Technologies Used

- **Python**
- **Pandas & NumPy** – Data processing
- **Matplotlib** – Visualization
- **Time-series modeling techniques**
- **Clustering algorithms**
- **Google Colab** – Development environment

## Key Insights

- Several users exhibited irregular heart rate patterns that deviated significantly from expected trends.
- Sleep anomalies were clearly identifiable when daily aggregation was used instead of minute-level data.
- Users with consistently low activity levels were easily detectable through step-count analysis.
- Combining multiple metrics provided a more reliable indicator of overall health anomalies than analyzing a single parameter.

## Outcome

This milestone successfully demonstrates how AI-based anomaly detection can be applied to wearable fitness data to uncover hidden health risks. The approach supports proactive health monitoring and lays the foundation for personalized health alerts and real-world deployment scenarios.
