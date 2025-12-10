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
# GOOGLE DRIVE FILE IDs (sudah diisi dari link kamu)
# =========================================================
FILE_ID_PANEL_A = "1fZCYvQ8JDaJmuy2E1IcdU7ijfsGp4mbz"
FILE_ID_PANEL_B = "1CdtMOSwjHd8OLfrbJy1AxVRajB51OcvQ"

URL_A = f"https://drive.google.com/uc?id={FILE_ID_PANEL_A}"
URL_B = f"https://drive.google.com/uc?id={FILE_ID_PANEL_B}"

# =========================================================
# LOAD DATA FROM GOOGLE DRIVE (AUTO DOWNLOAD)
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists("tmp_data"):
        os.makedirs("tmp_data")

    path_A = "tmp_data/panelA_clean.csv"
    path_B = "tmp_data/panelB_clean.csv"

    # Download dataset jika belum ada
    if not os.path.exists(path_A):
        gdown.download(URL_A, path_A, quiet=False)

    if not os.path.exists(path_B):
        gdown.download(URL_B, path_B, quiet=False)

    panelA = pd.read_csv(path_A)
    panelB = pd.read_csv(path_B)
    return panelA, panelB

# =========================================================
# LOAD MODEL, SCALER, LABEL ENCODER
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

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Kualitas Air IoT", layout="wide")
st.title("💧 Prediksi & Rekomendasi Kualitas Air Berbasis IoT")

menu = st.sidebar.radio("Navigasi", ["Data & Info", "Model & Evaluasi", "Prediksi + Rekomendasi AI"])

# =========================================================
# LOAD DATA (once)
# =========================================================
panelA, panelB = load_data()

# =========================================================
# 1. HALAMAN DATA
# =========================================================
if menu == "Data & Info":
    st.header("📊 Dataset Sensor IoT dari Google Drive")

    tab1, tab2 = st.tabs(["Panel A", "Panel B"])

    # ==== PANEL A ====
    with tab1:
        st.subheader("Contoh Data Panel A")
        st.dataframe(panelA.head())

        st.subheader("Distribusi Label Panel A")
        fig, ax = plt.subplots()
        panelA["label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # ==== PANEL B ====
    with tab2:
        st.subheader("Distribusi Label Panel B")
        fig, ax = plt.subplots()
        panelB["label"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# =========================================================
# 2. HALAMAN MODEL & EVALUASI
# =========================================================
elif menu == "Model & Evaluasi":
    st.header("🤖 Evaluasi Model Machine Learning")

    st.info("""
    Akurasi dan confusion matrix berasal dari hasil training.
    File dataset besar tetap berada di Google Drive, tetapi model .pkl digunakan di Streamlit.
    """)

    # Confusion Matrix Panel A
    if os.path.exists("evaluation/cmA.csv"):
        cmA = pd.read_csv("evaluation/cmA.csv", index_col=0)
        st.subheader("Confusion Matrix Panel A")
        fig, ax = plt.subplots()
        sns.heatmap(cmA, annot=True, cmap="Blues", fmt="d")
        st.pyplot(fig)
    else:
        st.warning("Confusion matrix Panel A tidak tersedia.")

    # Confusion Matrix Panel B
    if os.path.exists("evaluation/cmB.csv"):
        cmB = pd.read_csv("evaluation/cmB.csv", index_col=0)
        st.subheader("Confusion Matrix Panel B")
        fig, ax = plt.subplots()
        sns.heatmap(cmB, annot=True, cmap="Blues", fmt="d")
        st.pyplot(fig)
    else:
        st.warning("Confusion matrix Panel B tidak tersedia.")

# =========================================================
# 3. HALAMAN PREDIKSI + REKOMENDASI AI
# =========================================================
elif menu == "Prediksi + Rekomendasi AI":
    st.header("🧪 Prediksi Kualitas Air")

    flow1 = st.number_input("Flow 1")
    turbidity = st.number_input("Turbidity")
    tds = st.number_input("TDS")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0)

    panel_choice = st.selectbox("Pilih Panel", ["Panel A", "Panel B"])

    if st.button("Predict"):
        input_data = np.array([[flow1, turbidity, ph, tds]])

        # PILIH MODEL
        if panel_choice == "Panel A":
            Xs = scalerA.transform(input_data)
            pred = modelA.predict(Xs)[0]
            label = leA.inverse_transform([pred])[0]
        else:
            Xs = scalerB.transform(input_data)
            pred = modelB.predict(Xs)[0]
            label = leB.inverse_transform([pred])[0]

        st.success(f"Prediksi Kualitas Air: **{label}**")

        # ============ AI RECOMMENDATION (Gemini) ============
        API_KEY = st.secrets["GEMINI_KEY"]

        prompt = f"""
        Nilai sensor:
        - Flow1: {flow1}
        - Turbidity: {turbidity}
        - TDS: {tds}
        - pH: {ph}

        Prediksi label: {label}

        Berikan:
        1. Analisis kondisi air berdasarkan label
        2. Rekomendasi treatment untuk meningkatkan kualitas air
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"
        
        data = { "contents": [{ "parts": [{ "text": prompt }] }] }
        
        response = requests.post(url, json=data)
        result = response.json()

        ai_text = result["candidates"][0]["content"]["parts"][0]["text"]

        st.subheader("🔍 Rekomendasi dari Gemini AI")
        st.write(ai_text)
