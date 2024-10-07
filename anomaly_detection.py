import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv('data/ML.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print("NaT values:", df['date'].isna().sum())

# Feature engineering
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Define features
features = df[['sourceid', 'destinationid', 'amountofmoney', 'hour', 'day_of_week', 'month']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = model.fit_predict(scaled_features)

# Check if 'anomaly' column is created
print("Columns in DataFrame:", df.columns)
print(df.head())  # Print the first few rows of the DataFrame

# Identify anomalies
if 'anomaly' in df.columns:
    anomalies = df[df['anomaly'] == -1]
    print("Detected Anomalies:")
    print(anomalies)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(df['date'], df['amountofmoney'], c=df['anomaly'], cmap='coolwarm', label='Data points')
    plt.scatter(anomalies['date'], anomalies['amountofmoney'], color='red', label='Anomalies', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Amount of Money')
    plt.title('Anomaly Detection in Transactions')
    plt.legend()
    plt.show()

    # Save the model
    joblib.dump(model, 'USL_ML.pkl')
    print("Model saved to USL_ML.pkl")
else:
    print("Anomaly column not found in DataFrame.")
