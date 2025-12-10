import streamlit as st
import numpy as np
import pickle

st.title("🔮 Prediksi Kualitas Air")

flow1 = st.number_input("Flow 1", 0.0)
turbidity = st.number_input("Turbidity", 0.0)
ph = st.number_input("pH", 0.0)
tds = st.number_input("TDS", 0.0)

model_type = st.selectbox("Model", ["XGBoost Panel A", "XGBoost Panel B"])

# Load model
model_path = "models/xgb_panelA.pkl" if "A" in model_type else "models/xgb_panelB.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

if st.button("Predict"):
    features = np.array([[flow1, turbidity, ph, tds]])
    pred = model.predict(features)[0]
    st.success(f"🏷 Kualitas Air: **{pred}**")
