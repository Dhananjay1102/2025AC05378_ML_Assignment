"""
train_models.py
----------------
Trains 5 classification models on the Mobile Price Classification dataset
(BITS Pilani M.Tech AIML - ML Assignment 2).

Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Gaussian Naive Bayes
    5. Random Forest Classifier (Ensemble)

For each model, computes: Accuracy, AUC (macro, One-vs-Rest), Precision (macro),
Recall (macro), F1 (macro), Matthews Correlation Coefficient.

Outputs:
    - model/*.pkl              -> trained model objects
    - model/scaler.pkl         -> StandardScaler fitted on training data
    - model/feature_names.pkl  -> ordered list of feature columns
    - test_data.csv            -> held-out test split (features + true label),
                                   this is what gets uploaded to the Streamlit app
    - model/comparison_table.csv -> the metrics table used in README.md
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(BASE_DIR / "train_full.csv")

TARGET = "price_range"
X = df.drop(columns=[TARGET])
y = df[TARGET]

feature_names = list(X.columns)
print(f"Dataset shape: {df.shape}  |  Features: {len(feature_names)}  |  Classes: {sorted(y.unique())}")

# ---------------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only, applied to all models for consistency
#    in the deployed app; tree-based models are scale-invariant so this is safe)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / "scaler.pkl", compress=3)
joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

# ---------------------------------------------------------------------
# 4. Save the held-out test set as test_data.csv (features + true label)
#    -> this is the file you upload to the Streamlit app / submit as test data
# ---------------------------------------------------------------------
test_export = X_test.copy()
test_export[TARGET] = y_test.values
test_export.to_csv(BASE_DIR / "test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_export)} rows")

# ---------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
    "kNN": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=120, random_state=RANDOM_STATE, max_depth=10
    ),
}

results = []
n_classes = len(np.unique(y))

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    # save model (compressed)
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, MODEL_DIR / f"{fname}.pkl", compress=3)

    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f}  AUC={auc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
    print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(MODEL_DIR / "comparison_table.csv", index=False)
print("\n\n===== FINAL COMPARISON TABLE =====")
print(results_df.to_string(index=False))
