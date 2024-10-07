

---

# 💸 Transaction Fraud Detection App

This Streamlit application allows users to detect fraudulent transactions using two different machine learning approaches: **Unsupervised Learning (Anomaly Detection)** and **Supervised Learning (Fraud Classification)**.

Try the app live here:  
🔗 [Transaction Fraud Detection App](https://money-laundering-vqsugk4dxypxeshe9gkzrx.streamlit.app/)

---

## 🧠 Approaches in the App

### 1. **Unsupervised Learning (Anomaly Detection)**  
This approach uses **Isolation Forest** to detect anomalous transactions that may indicate fraud.

- **How It Works**:
  - User uploads a CSV containing transaction details.
  - Features such as `sourceid`, `destinationid`, `amountofmoney`, `hour`, `day_of_week`, and `month` are extracted.
  - Anomalies (possible fraudulent transactions) are detected based on the input features.
  - Detected anomalies are visualized in a scatter plot and displayed in a table.

- **Model Used**:
  - **Isolation Forest** with a contamination rate of 0.1 (assuming 10% of the transactions are anomalies).

---

### 2. **Supervised Learning (Fraud Classification)**  
This approach uses **XGBoost Classifier** to classify transactions as fraudulent or non-fraudulent based on labeled data.

- **How It Works**:
  - Users can either load a pre-trained XGBoost model or train one using their own dataset.
  - If no pre-trained model is available, users can upload a CSV with labeled transaction data and the model will be trained on it.
  - The model takes user inputs such as transaction details and predicts whether the transaction is fraudulent or not.

- **Model Used**:
  - **XGBoost Classifier** with a multi-log loss evaluation metric.

---

## 📊 Features

- **Anomaly Detection** using Isolation Forest.
- **Supervised Classification** using XGBoost.
- **Interactive User Interface**:
  - File uploader for input CSVs.
  - Real-time anomaly detection visualization.
  - Predict transaction fraud using input fields.
- **Model Saving**: Both unsupervised and supervised models are saved after training for future use.

---

## 🚀 How to Run Locally

### Prerequisites

Make sure you have the following installed:

- Python 3.10+
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🔗 App Live Demo

Click the link to try the app directly in your browser:  
🔗 [Transaction Fraud Detection App](https://money-laundering-vqsugk4dxypxeshe9gkzrx.streamlit.app/)

---

## 📁 Example CSV Structure

For both approaches, the CSV file should include the following columns:

- **date**: Date of transaction (YYYY-MM-DD)
- **sourceid**: ID of the source account
- **destinationid**: ID of the destination account
- **amountofmoney**: Transaction amount
- **typeofaction**: (For supervised learning) Type of action taken in the transaction
- **isfraud**: (For supervised learning) Label indicating if the transaction is fraudulent (1) or not (0)

Example:

| date       | sourceid | destinationid | amountofmoney | typeofaction | isfraud |
|------------|----------|---------------|---------------|--------------|---------|
| 2023-10-01 | 12345    | 54321         | 1000          | 1            | 0       |
| 2023-10-02 | 67890    | 98765         | 2500          | 2            | 1       |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-learn, XGBoost
- **Model Storage**: Joblib
- **Visualization**: Matplotlib
- **Deployment**: Streamlit Cloud

---

## 📜 License

This project is licensed under the MIT License.

---

