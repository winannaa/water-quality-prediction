import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_panelA, load_panelB

st.title("📈 Model & Evaluasi")

# Load models
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

models = {
    "XGBoost Panel A": "models/xgb_panelA.pkl",
    "LogReg Panel A": "models/logreg_panelA.pkl",
    "XGBoost Panel B": "models/xgb_panelB.pkl",
    "LogReg Panel B": "models/logreg_panelB.pkl"
}

selected = st.selectbox("Pilih Model:", list(models.keys()))
model = load_model(models[selected])

panel = load_panelA() if "Panel A" in selected else load_panelB()

X = panel[['flow1','turbidity','ph','tds']]
y = panel['label']

# Predict for evaluation
y_pred = model.predict(X)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y, y_pred)
st.write(f"### Accuracy: **{acc:.4f}**")

# Classification report
st.text(classification_report(y, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {selected}")
st.pyplot(fig)
