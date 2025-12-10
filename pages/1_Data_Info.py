import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_panelA, load_panelB

st.title("📊 Data & Info")

panelA = load_panelA()
panelB = load_panelB()

st.subheader("Dataset Panel A (Raw Cleaned)")
st.write(panelA.head())
st.write(panelA.describe())

st.subheader("Dataset Panel B (Raw Cleaned)")
st.write(panelB.head())
st.write(panelB.describe())

# Distribusi label
fig, ax = plt.subplots(1, 2, figsize=(14,5))
sns.countplot(x=panelA["label"], ax=ax[0])
sns.countplot(x=panelB["label"], ax=ax[1])
ax[0].set_title("Distribusi Label - Panel A")
ax[1].set_title("Distribusi Label - Panel B")
st.pyplot(fig)
