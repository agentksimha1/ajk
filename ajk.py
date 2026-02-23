import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------------
# Load Models
# -------------------------------
gbn_model = joblib.load(r"C:\Users\Krishna\Downloads\gbn_model.pkl")
rf_model = joblib.load(r"C:\Users\Krishna\Downloads\rf_model.pkl")

# -------------------------------
# Reason Mapping
# -------------------------------
reason_map = {
    0: "No Reason",
    1: "Sore Throat",
    2: "In Jamshedpur",
    3: "Monday after fest",
    4: "Servicing",
    5: "Class preponed 1 hour",
    6: "Class timing 11:30",
    7: "Quiz Day"
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Professor Lateness Predictor")

# User selects a date
input_date = st.date_input("Select a Date")

if st.button("Predict"):

    base_date = datetime(2026, 1, 13)

    # Convert input date to datetime
    selected_date = datetime.combine(input_date, datetime.min.time())

    # Calculate days since Jan 13
    days_since_start = (selected_date - base_date).days

    # Reshape for model
    X = np.array([[days_since_start]])

    # Predict Reason
    predicted_reason_class = gbn_model.predict(X)[0]
    predicted_reason = reason_map.get(predicted_reason_class, "Unknown")

    # Predict Minutes Late
    predicted_minutes = rf_model.predict(X)[0]

    # Display Results
    st.subheader("Prediction Results")
    st.write(f"Days since Jan 13, 2026: {days_since_start}")
    st.write(f"Predicted Reason: {predicted_reason}")
    st.write(f"Predicted Minutes Late: {round(predicted_minutes, 2)} minutes")