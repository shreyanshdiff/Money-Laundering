import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, precision_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Title of the App
st.title("Transaction Fraud Detection")

# Sidebar for selection
option = st.sidebar.selectbox("Choose Approach", ["Unsupervised Learning (Anomaly Detection)", "Supervised Learning (Fraud Classification)"])

# ---------- Unsupervised Approach (Anomaly Detection) ----------
if option == "Unsupervised Learning (Anomaly Detection)":
    st.subheader("Unsupervised Learning: Anomaly Detection")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        features = df[['sourceid', 'destinationid', 'amountofmoney', 'hour', 'day_of_week', 'month']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Train the Isolation Forest model
        model = IsolationForest(contamination=0.1, random_state=42)
        df['anomaly'] = model.fit_predict(scaled_features)

        # Display anomalies
        anomalies = df[df['anomaly'] == -1]
        st.write(f"Total Anomalies Detected: {anomalies.shape[0]}")
        st.write(anomalies)

        # Plotting the anomalies
        plt.figure(figsize=(10, 6))
        plt.scatter(df['date'], df['amountofmoney'], c=df['anomaly'], cmap='coolwarm', label='Data points')
        plt.scatter(anomalies['date'], anomalies['amountofmoney'], color='red', label='Anomalies', marker='x')
        plt.xlabel('Date')
        plt.ylabel('Amount of Money')
        plt.title('Anomaly Detection in Transactions')
        plt.legend()
        st.pyplot(plt)

        # Save unsupervised model
        joblib.dump(model, 'USL_ML.pkl')
        st.success("Unsupervised model saved to USL_ML.pkl")

# ---------- Supervised Approach (XGBoost Classifier) ----------
elif option == "Supervised Learning (Fraud Classification)":
    st.subheader("Supervised Learning: Fraud Classification")

    # User inputs for making predictions
    typeofaction = st.selectbox("Type of Action", [0, 1, 2, 3])  # Replace with appropriate action labels
    sourceid = st.number_input("Source ID", min_value=0)
    destinationid = st.number_input("Destination ID", min_value=0)
    amountofmoney = st.number_input("Amount of Money", min_value=0.0)
    hour = st.slider("Transaction Hour", min_value=0, max_value=23)
    day_of_week = st.slider("Day of the Week", min_value=0, max_value=6)
    month = st.slider("Month", min_value=1, max_value=12)

    # Load or train the XGBoost model
    if st.button("Predict Fraud"):
        try:
            # Load the saved model
            best_model = joblib.load('supervised_xgb.pkl')
        except:
            # If model not saved, train and save it
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['hour'] = df['date'].dt.hour
                df['day_of_week'] = df['date'].dt.dayofweek
                df['month'] = df['date'].dt.month

                # Label encoding for categorical features
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                df['typeofaction'] = encoder.fit_transform(df['typeofaction'])
                df['typeoffraud'] = encoder.fit_transform(df['typeoffraud'])

                # Define features and target variable
                X = df[['typeofaction', 'sourceid', 'destinationid', 'amountofmoney', 'hour', 'day_of_week', 'month']]
                y = df['isfraud']

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train XGBoost model
                best_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
                best_model.fit(X_train, y_train)

                # Save the model after training
                joblib.dump(best_model, 'supervised_xgb.pkl')
                st.success("Supervised model saved to supervised_xgb.pkl")

        # Make prediction
        input_data = np.array([[typeofaction, sourceid, destinationid, amountofmoney, hour, day_of_week, month]])
        prediction = best_model.predict(input_data)
        if prediction == 1:
            st.warning("The transaction is predicted to be fraudulent!")
        else:
            st.success("The transaction is predicted to be non-fraudulent.")
