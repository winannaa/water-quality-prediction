import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import os
import requests

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Kualitas Air IoT", layout="wide")
st.title("💧 Prediksi & Rekomendasi Kualitas Air Berbasis IoT")

# =========================================================
# GOOGLE DRIVE FILE IDs
# =========================================================
FILE_ID_PANEL_A = "1fZCYvQ8JDaJmuy2E1IcdU7ijfsGp4mbz"
FILE_ID_PANEL_B = "1CdtMOSwjHd8OLfrbJy1AxVRajB51OcvQ"

URL_A = f"https://drive.google.com/uc?id={FILE_ID_PANEL_A}"
URL_B = f"https://drive.google.com/uc?id={FILE_ID_PANEL_B}"

# =========================================================
# LOAD DATA FROM GOOGLE DRIVE
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists("tmp_data"):
        os.makedirs("tmp_data")

    path_A = "tmp_data/panelA_clean.csv"
    path_B = "tmp_data/panelB_clean.csv"

    if not os.path.exists(path_A):
        gdown.download(URL_A, path_A, quiet=False)
    if not os.path.exists(path_B):
        gdown.download(URL_B, path_B, quiet=False)

    panelA = pd.read_csv(path_A)
    panelB = pd.read_csv(path_B)
    return panelA, panelB


# =========================================================
# LOAD MODELS (MODEL, SCALER, LABEL ENCODER)
# =========================================================
@st.cache_resource
def load_models():
    modelA = pickle.load(open("models/modelA_best.pkl", "rb"))
    modelB = pickle.load(open("models/modelB_best.pkl", "rb"))

    scalerA = pickle.load(open("models/scalerA.pkl", "rb"))
    scalerB = pickle.load(open("models/scalerB.pkl", "rb"))

    leA = pickle.load(open("models/labelA.pkl", "rb"))
    leB = pickle.load(open("models/labelB.pkl", "rb"))

    return modelA, modelB, scalerA, scalerB, leA, leB


modelA, modelB, scalerA, scalerB, leA, leB = load_models()

panelA, panelB = load_data()

# =========================================================
# SIDEBAR MENU
# =========================================================
menu = st.sidebar.radio(
    "Navigasi", 
    ["Data & Info", "Model & Evaluasi", "Prediksi + Rekomendasi AI"]
)

# =========================================================
# 1. DATA PAGE
# =========================================================
if menu == "Data & Info":
    st.header("📊 Dataset Sensor IoT dari Google Drive")

    tab1, tab2 = st.tabs(["Panel A", "Panel B"])

    with tab1:
        st.subheader("Contoh Data Panel A")
        st.dataframe(panelA.head())

        st.subheader("Distribusi Label Panel A")
        fig, ax = plt.subplots()
        panelA["label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Contoh Data Panel B")
        st.dataframe(panelB.head())

        st.subheader("Distribusi Label Panel B")
        fig, ax = plt.subplots()
        panelB["label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# =========================================================
# 2. MODEL & EVALUATION PAGE
# =========================================================
elif menu == "Model & Evaluasi":
    st.header("🤖 Evaluasi Model Machine Learning")

    if os.path.exists("evaluation/cmA.csv"):
        cmA = pd.read_csv("evaluation/cmA.csv", index_col=0)
        st.subheader("Confusion Matrix Panel A")
        fig, ax = plt.subplots()
        sns.heatmap(cmA, annot=True, cmap="Blues", fmt="d")
        st.pyplot(fig)

    if os.path.exists("evaluation/cmB.csv"):
        cmB = pd.read_csv("evaluation/cmB.csv", index_col=0)
        st.subheader("Confusion Matrix Panel B")
        fig, ax = plt.subplots()
        sns.heatmap(cmB, annot=True, cmap="Blues", fmt="d")
        st.pyplot(fig)

# =========================================================
# 3. PREDIKSI + REKOMENDASI AI PAGE
# =========================================================
elif menu == "Prediksi + Rekomendasi AI":
    st.header("🧪 Prediksi Kualitas Air")

    st.markdown("### Masukkan nilai sensor (4 fitur)")

    col1, col2 = st.columns(2)

    with col1:
        flow1 = st.number_input("Flow 1 (L/min)", min_value=0.0, value=0.0)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=0.0)

    with col2:
        tds = st.number_input("TDS (ppm)", min_value=0.0, value=0.0)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)

    st.write("")
    panel_choice = st.selectbox("Pilih Panel Model", ["Panel A", "Panel B"])

    st.write("")
    predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True)

    if predict_btn:
        # Prepare input
        input_data = np.array([[flow1, turbidity, ph, tds]])

        # Predict
        if panel_choice == "Panel A":
            Xs = scalerA.transform(input_data)
            pred = modelA.predict(Xs)[0]
            label = leA.inverse_transform([pred])[0]
        else:
            Xs = scalerB.transform(input_data)
            pred = modelB.predict(Xs)[0]
            label = leB.inverse_transform([pred])[0]

        # OUTPUT PREDIKSI
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background:#eef6ff; border:1px solid #c8defc;">
                <h3 style="color:#0b63c7;">Hasil Prediksi</h3>
                <p style="font-size:22px;"><b>{label}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================================================
        # GEMINI PRO — FIXED AND SAFE
        # =========================================================
        API_KEY = st.secrets["GEMINI_KEY"]

        prompt = f"""
        Data sensor air:
        - Flow1: {flow1}
        - Turbidity: {turbidity}
        - TDS: {tds}
        - pH: {ph}

        Prediksi kualitas air: {label}

        Berikan:
        1. Analisis kualitas air
        2. Risiko jika tidak ditangani
        3. Rekomendasi perbaikan kualitas air
        """

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"

        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        response = requests.post(url, json=payload)
        result = response.json()

        # Debug (muncul di logs Cloud)
        print("DEBUG GEMINI:", result)

        if "candidates" in result:
            ai_text = result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            ai_text = f"""
            ❌ AI gagal memberikan rekomendasi.

            **Detail respons API:**
            ```
            {result}
            ```
            """

        # OUTPUT
        st.markdown(
            "<h3 style='margin-top:25px;'>🔍 Rekomendasi dari Gemini AI</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background:#f9f9f9; 
                padding:20px; 
                border-radius:10px; 
                border-left:5px solid #0b63c7;
                font-size:16px;
                line-height:1.6;
            ">
                {ai_text}
            </div>
            """,
            unsafe_allow_html=True
        )
