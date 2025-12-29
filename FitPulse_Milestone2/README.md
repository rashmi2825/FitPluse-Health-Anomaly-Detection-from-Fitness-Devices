**Milestone 2: Feature Extraction, Modeling, and Clustering**

**Objective of Milestone 2**

- The objective of Milestone 2 is to transform raw fitness device time-series data into meaningful representations, model temporal health patterns, and identify anomalous user behavior using unsupervised learning techniques.
- This milestone focuses on feature extraction, trend modeling, and behavioral clustering to support health anomaly detection.
  
**Dataset Used:**
[minute_level_data.csv](https://drive.google.com/file/d/1vZOK-StZ2se297M78bVjfkNpQ-SEI8H_/view?usp=drive_link)

**Steps Performed:**

**1. Feature Extraction**

Selected **14** unique users from the dataset.
For each user, the first **600 heart-rate records** were considered.
Applied **TSFresh** using** MinimalFCParameters** to extract statistical features.

**Extracted Features Include**

Mean
Median
Standard deviation
Variance
Root mean square (RMS)
Minimum and maximum heart-rate values
Sum of values
Length of time series

**2. Trend Modeling**

Applied Facebook Prophet on heart-rate time-series data for individual users.

Modeled:
Overall trend
Daily seasonality
Weekly seasonality
Visualized Prophet forecasts and trend components.
Analyzed deviations between actual and predicted values to understand unusual behavior patterns.

**3. Clustering and Anomaly Detection**
Standardized extracted TSFresh features using StandardScaler.
Applied DBSCAN clustering to group users based on behavioral similarity.

**DBSCAN labeled users as:**
Cluster 0 → Normal behavior
Cluster −1 → Anomalous behavior (outliers)
Reason for Choosing DBSCAN

**4. Visualization**

-Applied **Principal Component Analysis (PCA) **to reduce high-dimensional features to two dimensions.
-Visualized DBSCAN clustering results using a 2D scatter plot.
-Clearly distinguished normal users from anomalous users.

**Results and Observations**

-TSFresh successfully extracted meaningful statistical features from raw heart-rate data.
-Prophet effectively captured trends and seasonality in physiological behavior.
-DBSCAN identified a dense cluster representing normal behavior.
-Users with extreme heart-rate values, high variance, or irregular patterns were detected as anomalies.
-PCA visualization confirmed clear separation between normal and anomalous users.

**Tools and Libraries Used**

Python
Pandas
NumPy
TSFresh
Facebook Prophet
Scikit-learn
Matplotlib
Google Colab

