"""
==============================================================================
ML Assignment 2 — Heart Disease Classification
BITS M.Tech (AIML/DSE) — Machine Learning
==============================================================================
Problem statement :
    Predict whether a patient has heart disease (binary classification) based
    on 14 clinical attributes. Dataset: UCI Heart Disease (Kaggle version),
    sourced from the Cleveland, Hungary, Switzerland and VA Long Beach
    databases.

Models implemented :
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors Classifier
    4. Gaussian Naive Bayes Classifier
    5. Random Forest Classifier  (Ensemble)

Metrics computed for every model :
    Accuracy | AUC (ROC) | Precision | Recall | F1-Score | MCC
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

import joblib

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "heart_disease_uci.csv")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
TEST_CSV   = os.path.join(OUT_DIR, "test_data.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.25

# ---------------------------------------------------------------------------
# 2. Load and prepare the dataset
# ---------------------------------------------------------------------------
print("[1/6] Loading dataset ...")
df = pd.read_csv(DATA_FILE)
print(f"      Original shape : {df.shape}")

# Keep only the clinically used subset (14 attribute columns + target)
feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
target_col   = "num"

data = df[feature_cols + [target_col]].copy()

# The original 'num' values range 0..4 (severity). Following the standard
# Cleveland protocol, convert to a binary target:
#   0 = no heart disease,  1..4 = heart disease present
data[target_col] = (data[target_col] > 0).astype(int)
print(f"      Binary target  : {data[target_col].value_counts().to_dict()}")

# Encode categorical string columns numerically
label_encoders = {}
for col in feature_cols:
    if data[col].dtype == object or pd.api.types.is_bool_dtype(data[col]):
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Fill any non-numeric junk with NaN first, then impute, then encode.
# Order matters: columns like 'sex'/'cp'/'thal' arrived as strings with a
# leading non-breaking character, so they became NaN via to_numeric. We
# impute the missing string-coded columns with their most-frequent value
# BEFORE numeric coercion so no feature is lost.
from collections import Counter

for col in feature_cols:
    col_data = data[col]
    numeric = pd.to_numeric(col_data, errors="coerce")
    if numeric.notna().sum() >= col_data.notna().sum() / 2:
        # Treat as numeric: impute missing with median, then coerce
        data[col] = numeric.fillna(numeric.median())
    else:
        # Treat as categorical: impute with mode, then label-encode
        counter = Counter(col_data.astype(str).dropna())
        fill_val = counter.most_common(1)[0][0] if counter else ""
        col_data = col_data.astype(str).fillna(fill_val)
        le = LabelEncoder()
        data[col] = le.fit_transform(col_data)
        label_encoders[col] = le
    # Final numeric-safe coercion (fallback for any residual NaN)
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

print("      Columns remaining non-numeric:",
      [c for c in feature_cols if data[c].dtype == object])

# ---------------------------------------------------------------------------
# 3. Split into training and test sets
# ---------------------------------------------------------------------------
print("[2/6] Splitting data (75% train / 25% test) ...")
X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Save the labelled test data (feature columns + target) as test_data.csv.
# Streamlit free tier has limited capacity, so only the test slice is used.
test_data = X_test.copy()
test_data[target_col] = y_test.values
test_data.to_csv(TEST_CSV, index=False)
print(f"      Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")
print(f"      Saved test data to : {TEST_CSV}")

# ---------------------------------------------------------------------------
# 4. Preprocessing pipeline
# ---------------------------------------------------------------------------
print("[3/6] Preprocessing (imputation + standardisation) ...")
imputer = StandardScaler  # placeholder, replaced below
imputer = SimpleImputer(strategy="median")

X_train_imp = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=X_train.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),
                           columns=X_test.columns)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_imp)
X_test_s  = scaler.transform(X_test_imp)

# ---------------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        C=0.1, max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------------------
# 6. Train, evaluate and save
# ---------------------------------------------------------------------------
print("[4/6] Training models and computing metrics ...")
metrics_rows = []
results = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred      = model.predict(X_test_s)
    y_prob      = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # 5-fold cross-validated accuracy for robustness
    cv_acc = cross_val_score(model, X_train_s, y_train,
                             cv=5, scoring="accuracy").mean()

    metrics_rows.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "CV Accuracy (5-fold)": round(cv_acc, 4),
    })

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred,
                                        target_names=["No Disease", "Disease"]),
    }

    # Save each trained model
    joblib.dump(model, os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}.pkl"))
    print(f"      Saved : {name}.pkl")

# ---------------------------------------------------------------------------
# 7. Print / save the comparison table
# ---------------------------------------------------------------------------
print("[5/6] Evaluation results")
metrics_df = pd.DataFrame(metrics_rows)
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
print(f"      Saved metric table to model_metrics.csv")

with open(os.path.join(OUT_DIR, "classification_reports.txt"), "w") as f:
    for name, res in results.items():
        f.write(f"===== {name} =====\n")
        f.write(res["report"])
        f.write("\n")

# ---------------------------------------------------------------------------
# 8. Visualisation — confusion matrices
# ---------------------------------------------------------------------------
print("[6/6] Generating visualisations ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Confusion Matrices — Heart Disease Classification (Test Set)",
             fontsize=16)
names = list(results.keys())
for ax, name in zip(axes.flat, names):
    cm = results[name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    acc = metrics_df.loc[metrics_df["ML Model Name"] == name, "Accuracy"].iloc[0]
    auc = metrics_df.loc[metrics_df["ML Model Name"] == name, "AUC"].iloc[0]
    ax.set_title(f"{name}\nAcc = {acc:.3f}, AUC = {auc:.3f}")
axes.flat[5].axis("off")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150)
print(f"      Saved : {cm_path}")

# Metric comparison bar chart
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
plot_df = metrics_df.set_index("ML Model Name")[["Accuracy", "AUC", "F1", "MCC"]]
plot_df.plot(kind="bar", ax=ax2, colormap="tab10")
ax2.set_title("Model Comparison — Test Set Metrics", fontsize=14)
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.0)
ax2.legend(loc="lower right")
ax2.tick_params(axis="x", rotation=0)
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "metric_comparison.png")
plt.savefig(bar_path, dpi=150)
print(f"      Saved : {bar_path}")

# ---------------------------------------------------------------------------
# 9. Winner summary
# ---------------------------------------------------------------------------
best_idx = metrics_df["AUC"].idxmax()
winner = metrics_df.loc[best_idx, "ML Model Name"]
print(f"\nOverall winner on this dataset (highest AUC): {winner}")
print("\nDone.")
