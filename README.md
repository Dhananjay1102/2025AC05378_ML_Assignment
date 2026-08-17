# Mobile Price Range Classification — ML Assignment 2

**BITS Pilani WILP — M.Tech (AIML) — Machine Learning**
Student ID: 2025AC05378

---

## a. Problem Statement

A mobile phone manufacturer wants to estimate the **price range** of a phone
(rather than its exact price) based on its hardware specifications — RAM,
battery capacity, camera resolution, processor cores, screen size, connectivity
features (3G/4G/Wi-Fi/Bluetooth), and so on. This is framed as a **multi-class
classification problem**: given a phone's specs, predict which of four price
tiers it falls into. Five classification models are trained on the same
dataset and compared using standard classification metrics to identify the
best-performing approach.

## b. Dataset Description

- **Source:** Mobile Price Classification dataset (Kaggle)
- **Instances:** 2,000
- **Features:** 20 numeric features (≥ the required minimum of 12), including
  `battery_power`, `ram`, `px_height`, `px_width`, `mobile_wt`, `n_cores`,
  `int_memory`, `talk_time`, `fc`/`pc` (camera MP), and binary flags for
  `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi`.
- **Target variable:** `price_range` — 4 classes:
  - `0` = Low cost
  - `1` = Medium cost
  - `2` = High cost
  - `3` = Very high cost
- **Missing values:** None
- **Train/test split:** 80% / 20%, stratified by class (1,600 train / 400 test)

## c. GitHub Repository Link

> **`https://github.com/Dhananjay1102/2025AC05378_ML_Assignment.git`**

## d. Streamlit App Link

> **`https://github.com/Dhananjay1102/2025AC05378_ML_Assignment.git`**

## e. Models Used

All 5 models were trained on an identical 80/20 stratified train/test split of
the same dataset, with features standardized (`StandardScaler`, fit on the
training set only). Metrics below are computed on the **held-out test set**
(400 samples, same set exported as `test_data.csv`).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8500 | 0.9125 | 0.8523 | 0.8500 | 0.8501 | 0.8007 |
| kNN | 0.5625 | 0.7893 | 0.5753 | 0.5625 | 0.5674 | 0.4174 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8975 | 0.9801 | 0.8975 | 0.8975 | 0.8973 | 0.8634 |

*(Precision/Recall/F1/AUC use macro-averaging since this is a 4-class
problem; AUC uses the One-vs-Rest scheme.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | By far the strongest model on this dataset. `ram` has an almost linear relationship with `price_range`, which plays directly to a linear model's strengths — confusion is limited to adjacent price tiers (e.g. class 1 vs 2), which is expected since those boundaries are the hardest to separate. |
| Decision Tree | Solid but visibly overfits relative to its cross-validated depth; a single tree splits on `ram` first and then fragments on weaker features, which loses some of the smooth linear signal Logistic Regression captures. |
| kNN | Weakest performer by a wide margin. With 20 features on comparable but not identical scales, distance metrics get diluted by low-signal binary flags (`blue`, `wifi`, `touch_screen`), so the curse of dimensionality hurts it more than the other models — this matches what's commonly reported on this dataset. |
| Naive Bayes | Reasonable AUC (0.9506) despite modest accuracy — its probability *ranking* is decent, but the Gaussian independence assumption is violated by correlated features like `px_height`/`px_width`, which hurts hard-label accuracy more than ranking quality. |
| Random Forest (Ensemble) | Comfortably the second-best model. Averaging across 120 trees smooths out the overfitting a single Decision Tree shows, and it comes closest to Logistic Regression's AUC — but doesn't quite match its accuracy/MCC, suggesting the underlying relationship here is close to linear enough that ensembling trees adds less than it typically would on noisier data. |
| **Overall Winner** | **Logistic Regression** — highest score on every single metric (Accuracy, AUC, Precision, Recall, F1, MCC). |

---

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split (features + true label)
│-- train_full.csv          # full source dataset (used only for training)
│-- model/
│   │-- train_models.py     # trains all 5 models, computes metrics, saves artifacts
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl
│   │-- feature_names.pkl
│   │-- comparison_table.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional: re-trains models and regenerates test_data.csv
streamlit run app.py
```

