"""
app.py
------
Streamlit demo app for BITS Pilani M.Tech AIML - ML Assignment 2.
Dataset: Mobile Price Classification (4-class: price_range 0-3)

Features:
    a. CSV upload (test data only, per free-tier size constraints)
    b. Model selection dropdown
    c. Evaluation metrics display
    d. Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Mobile Price Classification", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "model"
TARGET = "price_range"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return scaler, feature_names, models


scaler, feature_names, models = load_artifacts()

st.title("📱 Mobile Price Classification — Model Comparison")
st.caption(
    "BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2. "
    "Dataset: Mobile Price Classification (Kaggle) — 20 features, 4 price-range classes."
)

# ---------------------------------------------------------------------
# a. Dataset upload
# ---------------------------------------------------------------------
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (features + optional 'price_range' true-label column)",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload the `test_data.csv` from the repo to see live predictions and metrics.")
    st.stop()

data = pd.read_csv(uploaded_file)
st.write(f"Loaded **{data.shape[0]}** rows, **{data.shape[1]}** columns.")
st.dataframe(data.head())

has_labels = TARGET in data.columns
missing_cols = [c for c in feature_names if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required feature columns: {missing_cols}")
    st.stop()

X = data[feature_names]
X_scaled = scaler.transform(X)

# ---------------------------------------------------------------------
# b. Model selection dropdown
# ---------------------------------------------------------------------
st.header("2. Select a Model")
model_choice = st.selectbox("Choose a classification model", list(models.keys()))
model = models[model_choice]

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

result_df = data.copy()
result_df["predicted_price_range"] = y_pred
st.subheader("Predictions")
st.dataframe(result_df.head(20))

# ---------------------------------------------------------------------
# c. Evaluation metrics (only possible if true labels are present)
# ---------------------------------------------------------------------
st.header("3. Evaluation Metrics")

if has_labels:
    y_true = data[TARGET]
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # ---------------------------------------------------------------
    # d. Confusion matrix + classification report
    # ---------------------------------------------------------------
    st.header("4. Confusion Matrix & Classification Report")
    col_a, col_b = st.columns(2)

    with col_a:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)

    with col_b:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3))
else:
    st.warning(
        "No 'price_range' column found in the uploaded file — showing predictions only. "
        "Upload a file with true labels to see accuracy/AUC/precision/recall/F1/MCC and the confusion matrix."
    )

# ---------------------------------------------------------------------
# Bonus: compare all 5 models side by side on the uploaded data
# ---------------------------------------------------------------------
if has_labels:
    st.header("5. All Models — Side-by-Side Comparison")
    rows = []
    for name, m in models.items():
        yp = m.predict(X_scaled)
        ypr = m.predict_proba(X_scaled)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, yp), 4),
            "AUC": round(roc_auc_score(y_true, ypr, multi_class="ovr", average="macro"), 4),
            "Precision": round(precision_score(y_true, yp, average="macro", zero_division=0), 4),
            "Recall": round(recall_score(y_true, yp, average="macro", zero_division=0), 4),
            "F1": round(f1_score(y_true, yp, average="macro", zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_true, yp), 4),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"))
