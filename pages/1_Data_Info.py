import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Data & Info")

# Load dataset
panelA = pd.read_csv("data/panelA_clean.csv")
panelB = pd.read_csv("data/panelB_clean.csv")

st.subheader("🔹 Panel A — Sample Data")
st.dataframe(panelA.head())

st.subheader("🔹 Panel B — Sample Data")
st.dataframe(panelB.head())

# Label distribution
st.subheader("📌 Distribusi Label (Panel A)")
fig1, ax1 = plt.subplots()
panelA['label'].value_counts().plot(kind='bar', ax=ax1)
st.pyplot(fig1)

st.subheader("📌 Distribusi Label (Panel B)")
fig2, ax2 = plt.subplots()
panelB['label'].value_counts().plot(kind='bar', ax=ax2)
st.pyplot(fig2)

