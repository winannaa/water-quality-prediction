import streamlit as st
import numpy as np
import joblib

st.title("🔮 Water Quality Prediction")

# Load model terbaik (misal pilih XGBoost Panel A)
model = joblib.load("models/xgb_A.pkl")
scaler = joblib.load("models/scaler_A.pkl")
encoder = joblib.load("models/label_encoder_A.pkl")

flow1 = st.number_input("Flow1", min_value=0.0)
turbidity = st.number_input("Turbidity", min_value=0.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0)
tds = st.number_input("TDS", min_value=0.0)

if st.button("Predict"):
    X = np.array([[flow1, turbidity, ph, tds]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    label = encoder.inverse_transform(pred)[0]

    st.success(f"**Kualitas Air: {label}**")

