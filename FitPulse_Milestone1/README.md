# FitPluse-Health-Anomaly-Detection-from-Fitness-Devices
MILESTONE 1:

Objectives:-

1.Implement fitness data ingestion logic for CSV/JSON files (heart rate, steps, sleep logs).
2.Normalize timestamps to UTC and handle missing/null values.
3.Align data to a consistent 1-minute interval frequency.
4.Generate a clean, consolidated dataset ready for analysis.

Dataset Source:
The dataset was sourced from Kaggle and is titled “Fitbit Fitness Tracker Data.”(https://www.kaggle.com/datasets/arashnic/fitbit/data)
Datasets taken:
- dailyActivity_merged.csv(https://drive.google.com/file/d/1cd2tNwRDfkeZnHjO5g77uadw5-3VSBOv/view?usp=drive_link)
- heartrate_seconds_merged.csv(https://drive.google.com/file/d/105iw-Nu_eOxcWd3RSsnbxyTE0FDCgUKx/view?usp=drive_link)
- minuteSleep_merged.csv(https://drive.google.com/file/d/1Bl6iLPV73US5meKGlcW0m1O_0bwBUY0d/view?usp=drive_link)
- the final dataset(minute_wise_data) used was (https://drive.google.com/file/d/1vZOK-StZ2se297M78bVjfkNpQ-SEI8H_/view?usp=drive_link)
  
Steps Performed:
1.	 Uploaded the raw datasets from Google Drive into the Google Colab notebook.
2.	Selected the necessary columns from the datasets such as heart rate, step count, sleep data, and timestamp.
3.	Processed each dataset separately by converting the date and time columns into datetime format for proper processing.
4.	Checked and handled missing or duplicate values and sorted the data to ensure consistent time order.
5.	Aggregated heart rate data to per-minute level and sleep data to daily sleep duration.
6.	 Merged all three datasets into a single final preprocessed dataset named as final_merged_data(https://drive.google.com/file/d/1ABFjZfd1vRiYr-JSCNLtgXrx7KPR3yJu/view?usp=drive_link) and uploaded it back to Google Drive
   
Tools and Technologies Used:

•	Python
•	Pandas
•	Numpy
•	Matplotlib

Key Insights :

•	Daily step count varies significantly across different users and days, showing different activity levels.
•	Heart rate data at minute level helps in observing small changes in physical activity and health conditions.
•	Sleep duration is not consistent every day, which can influence overall daily performance and fitness.


