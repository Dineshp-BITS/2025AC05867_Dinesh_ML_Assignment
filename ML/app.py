"""
==============================================================================
Streamlit App — ML Assignment 2 (Heart Disease Classification)
BITS M.Tech (AIML/DSE) — Machine Learning
==============================================================================
Required features (as per the assignment brief):
  (a) Dataset upload option (CSV)          — st.file_uploader
  (b) Model selection dropdown             — st.selectbox
  (c) Display of evaluation metrics        — accuracy, AUC, precision, recall,
                                              F1, MCC shown as metric cards
  (d) Confusion matrix & classification report — seaborn heatmap + report
==============================================================================
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
MODEL_DIR   = "ML\\model"
TEST_CSV    = "ML\\test_data.csv"   # bundled test data so the app works out-of-the-box

FEATURE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET_COL   = "num"

LABELS = ["No Disease", "Disease"]

MODEL_FILES = {
    "Logistic Regression": f"{MODEL_DIR}/logistic_regression.pkl",
    "Decision Tree":       f"{MODEL_DIR}/decision_tree.pkl",
    "kNN":                 f"{MODEL_DIR}/knn.pkl",
    "Naive Bayes":         f"{MODEL_DIR}/naive_bayes.pkl",
    "Random Forest":       f"{MODEL_DIR}/random_forest.pkl",
}

# ---------------------------------------------------------------------------
# Preprocessing helpers (same pipeline used during training)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, imputer=None, scaler=None, fitted=False):
    """Replicate the training-time preprocessing: mode imputation for
    categorical-coded columns, median for numeric columns, then
    StandardScaler."""
    df = df.copy()
    for col in FEATURE_COLS:
        col_data = df[col]
        numeric = pd.to_numeric(col_data, errors="coerce")
        if numeric.notna().sum() >= col_data.notna().sum() / 2:
            df[col] = numeric.fillna(numeric.median())
        else:
            # Categorical coded as string — encode to integers on the fly.
            counter = col_data.astype(str).dropna().value_counts()
            if not counter.empty:
                fill_val = counter.idxmax()
            else:
                fill_val = ""
            col_data = col_data.astype(str).fillna(fill_val)
            # Use a deterministic integer mapping per column
            le = LabelEncoder()
            df[col] = le.fit_transform(col_data)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encode any residual non-numeric values using most-frequent imputation
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not fitted:
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        X_imp   = pd.DataFrame(imputer.fit_transform(df[FEATURE_COLS]),
                               columns=FEATURE_COLS)
        X_s     = scaler.fit_transform(X_imp)
        return X_s, imputer, scaler
    else:
        X_imp = pd.DataFrame(imputer.transform(df[FEATURE_COLS]),
                             columns=FEATURE_COLS)
        X_s   = scaler.transform(X_imp)
        return X_s

# Fit preprocessing objects once on the bundled test data so that uploaded
# files are transformed with the same fitted pipeline.
_default_X_test = pd.read_csv(TEST_CSV)
X_test_prepared, IMPUTER, SCALER = preprocess(_default_X_test, fitted=False)
y_test_true = _default_X_test[TARGET_COL].values

# Load all saved models
@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}

MODELS = load_models()


# ---------------------------------------------------------------------------
# App UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

st.title("Heart Disease Classification — ML Assignment 2")
st.markdown(
    "BITS M.Tech (AIML/DSE) — Machine Learning  |  "
    "Dataset: **UCI Heart Disease** (920 instances, 14 attributes, binary target). "
    "Six evaluation metrics are computed for every selected model."
)

st.sidebar.header("Settings")

# ---- (a) Dataset upload option ------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Upload only test data (Streamlit free tier has limited capacity). "
         "The file must contain the 14 feature columns "
         f"{', '.join(FEATURE_COLS)} and the target column '{TARGET_COL}'. "
         "Leaving this empty uses the bundled test_data.csv (230 rows).")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = _default_X_test

# Validate uploaded / bundled dataframe
st.sidebar.subheader("Loaded data")
st.sidebar.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing feature columns: {missing_cols}. "
             "Please upload a file with the correct schema.")
    st.stop()

if TARGET_COL not in df.columns:
    st.warning(f"Target column '{TARGET_COL}' not found — metrics cannot be "
               "computed. Only the confusion matrix layout will be shown.")
    y_true = None
else:
    y_true = df[TARGET_COL].values

st.subheader("Preview of test data")
st.dataframe(df.head(10), use_container_width=True)

# ---- (b) Model selection dropdown ----------------------------------------
chosen = st.sidebar.selectbox("Select a model", list(MODELS.keys()))

# ---- Prediction -----------------------------------------------------------
X_s = preprocess(df, imputer=IMPUTER, scaler=SCALER, fitted=True)
model = MODELS[chosen]
y_pred = model.predict(X_s)
y_prob = model.predict_proba(X_s)[:, 1]

# ---- (c) Display of evaluation metrics -------------------------------------
st.subheader(f"Evaluation Metrics — {chosen}")

if y_true is not None:
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
else:
    acc = auc = prec = rec = f1 = mcc = np.nan

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}" if not np.isnan(acc) else "—")
col2.metric("AUC (ROC)", f"{auc:.4f}" if not np.isnan(auc) else "—")
col3.metric("Precision", f"{prec:.4f}" if not np.isnan(prec) else "—")
col4.metric("Recall", f"{rec:.4f}" if not np.isnan(rec) else "—")
col5.metric("F1 Score", f"{f1:.4f}" if not np.isnan(f1) else "—")
col6.metric("MCC", f"{mcc:.4f}" if not np.isnan(mcc) else "—")

# ---- (d) Confusion matrix & classification report --------------------------
st.subheader("Confusion Matrix & Classification Report")

cm_col, rep_col = st.columns(2)
with cm_col:
    cm = confusion_matrix(y_true, y_pred) if y_true is not None else np.array([[0, 0], [0, 0]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"{chosen}\n(confusion matrix)")
    st.pyplot(fig)

with rep_col:
    if y_true is not None:
        report = classification_report(y_true, y_pred, target_names=LABELS,
                                       output_dict=False)
        st.text(report)
    else:
        st.info("Provide the target column to see the full classification "
                "report.")

# ---- Compare all models ----------------------------------------------------
st.subheader("All Models — Side by Side (bundled test data)")
rows = []
for name, m in MODELS.items():
    p  = m.predict(X_test_prepared)
    po = m.predict_proba(X_test_prepared)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test_true, p), 4),
        "AUC": round(roc_auc_score(y_test_true, po), 4),
        "Precision": round(precision_score(y_test_true, p), 4),
        "Recall": round(recall_score(y_test_true, p), 4),
        "F1": round(f1_score(y_test_true, p), 4),
        "MCC": round(matthews_corrcoef(y_test_true, p), 4),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.caption(
    "Model implementation: model/model_training.py | Saved models: model/*.pkl | "
    "Source dataset: UCI Heart Disease (Kaggle) — "
    "https://archive.ics.uci.edu/dataset/45/heart+disease"
)
