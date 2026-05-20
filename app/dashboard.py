import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Financial Crime Monitoring System", layout="wide")

st.title("AI Financial Crime Monitoring Dashboard")

st.write(
"""
This dashboard monitors suspicious credit card transactions using
machine learning fraud detection and anomaly detection models.
"""
)

# -------------------------
# Load Data
# -------------------------

alerts = pd.read_csv("data/high_risk_alerts.csv")

fraud_model = joblib.load("models/fraud_detection_model.pkl")

# -------------------------
# System Metrics
# -------------------------

st.subheader("System Metrics")

total_transactions = 284807
total_alerts = len(alerts)
fraud_detected = alerts["Class"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total_transactions)
col2.metric("High Risk Alerts", total_alerts)
col3.metric("Fraud Detected", fraud_detected)

# -------------------------
# Risk Filter
# -------------------------

st.subheader("Filter Alerts by Risk Score")

min_risk = st.slider(
    "Minimum Risk Score",
    float(alerts["risk_score"].min()),
    float(alerts["risk_score"].max()),
    float(alerts["risk_score"].quantile(0.90))
)

filtered_alerts = alerts[alerts["risk_score"] >= min_risk]

st.write("Filtered Alerts:", len(filtered_alerts))

# -------------------------
# Top Suspicious Transactions
# -------------------------

st.subheader("Top Suspicious Transactions")

top_alerts = filtered_alerts.sort_values(
    "risk_score",
    ascending=False
).head(10)

st.table(
    top_alerts[
        ["Time", "Amount", "fraud_probability", "risk_score", "Class"]
    ]
)

# -------------------------
# Alert Table
# -------------------------

st.subheader("High Risk Transactions")

alerts_sorted = filtered_alerts.sort_values(
    "risk_score",
    ascending=False
)

st.dataframe(alerts_sorted.head(50))

# -------------------------
# Risk Score Distribution
# -------------------------

st.subheader("Risk Score Distribution")

st.bar_chart(alerts["risk_score"])

# -------------------------
# Fraud Probability Distribution
# -------------------------

st.subheader("Fraud Probability Distribution")

st.bar_chart(alerts["fraud_probability"])

# -------------------------
# Fraud vs Normal Alerts
# -------------------------

st.subheader("Fraud vs Normal Alerts")

st.bar_chart(alerts["Class"].value_counts())

# -------------------------
# Investigation Panel
# -------------------------

st.subheader("Investigate Transaction")

transaction_id = st.selectbox(
    "Select Alert Index",
    filtered_alerts.index
)

st.write(filtered_alerts.loc[transaction_id])
