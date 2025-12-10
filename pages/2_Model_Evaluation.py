import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🤖 Model & Evaluation")

# Load model
xgb_A = joblib.load("models/xgb_A.pkl")
xgb_B = joblib.load("models/xgb_B.pkl")

logreg_A = joblib.load("models/logreg_A.pkl")
logreg_B = joblib.load("models/logreg_B.pkl")

st.write("Berikut adalah hasil evaluasi model Anda:")

def show_metrics(name, model, X, y):
    st.subheader(name)
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    st.write(f"**Accuracy: {acc:.4f}**")

    report = classification_report(y, pred, output_dict=True)
    st.dataframe(report)

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    st.pyplot(fig)

# Load data
panelA = pd.read_csv("data/panelA_clean.csv")
panelB = pd.read_csv("data/panelB_clean.csv")

X_A = panelA[['flow1','turbidity','ph','tds']]
y_A = panelA['label']

X_B = panelB[['flow1','turbidity','ph','tds']]
y_B = panelB['label']

# Show evaluation
show_metrics("XGBoost – Panel A", xgb_A, X_A, y_A)
show_metrics("Logistic Regression – Panel A", logreg_A, X_A, y_A)
show_metrics("XGBoost – Panel B", xgb_B, X_B, y_B)
show_metrics("Logistic Regression – Panel B", logreg_B, X_B, y_B)

