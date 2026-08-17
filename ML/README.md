# ML Assignment 2 — Heart Disease Classification

BITS M.Tech (AIML/DSE) — Machine Learning (Work Integrated Learning Programmes Division)

---

## a. Problem Statement

Cardiovascular disease remains one of the leading causes of mortality worldwide, and early
diagnosis significantly improves patient outcomes. In this project we frame heart disease
prediction as a **binary classification problem**: given a set of clinical measurements
recorded for a patient, predict whether the patient has heart disease (presence) or not
(absence).

Six well-known supervised classification algorithms — Logistic Regression, Decision Tree,
K-Nearest Neighbors, Gaussian Naive Bayes and Random Forest (ensemble) — are trained on the
same dataset and compared against six evaluation metrics: **Accuracy, AUC (ROC), Precision,
Recall, F1-Score and Matthews Correlation Coefficient (MCC)**. An interactive Streamlit
application is built to serve the trained models and visualise their results, and it is
deployed on Streamlit Community Cloud.

---

## b. Dataset Description

The dataset used is the **UCI Heart Disease dataset** (hosted on Kaggle as
`heart_disease_uci.csv`), which combines four medical databases collected in 1988:
Cleveland Clinic Foundation, Hungarian Institute of Cardiology, University Hospitals
(VA Long Beach, San Diego) and University Hospital (Zurich, Switzerland).

| Property | Value |
| --- | --- |
| Source | UCI Machine Learning Repository — [Heart Disease (Dataset 45)](https://archive.ics.uci.edu/dataset/45/heart+disease) / Kaggle [redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| Total instances | 920 (satisfies the >= 500 minimum) |
| Attributes | 14 clinical attributes (satisfies the >= 12 minimum) |
| Task | Binary classification |
| Target | `num`: original severity 0–4; binarised as **0 = No disease, 1 = Disease (1–4)** |
| Class distribution (full data) | No disease: 411, Disease: 509 |
| Train / Test split | 690 / 230 (75% / 25%, stratified, random state 42) |

**Attributes (14):** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood
pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar), `restecg` (resting
ECG results), `thalch` (maximum heart rate achieved), `exang` (exercise induced angina),
`oldpeak` (ST depression induced by exercise), `slope` (slope of peak exercise ST segment),
`ca` (number of major vessels coloured by fluoroscopy) and `thal` (thalassemia — a blood
disorder).

The dataset contains missing values (most notably in `ca` and `thal`); these were imputed
with the **median** for numeric columns and the **mode** for categorical-coded columns
before scaling with a `StandardScaler`.

---

## c. GitHub Repository Link

> https://github.com/Dineshp-BITS/2025AC05867_Dinesh_ML_Assignment

---

## d. Models Used — Comparison Table

All models were trained with `random_state = 42` for reproducibility. The metrics below
are computed on the held-out **test set (230 instances)** after median/mode imputation
and standardisation.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8261 | 0.9008 | 0.8271 | 0.8661 | 0.8462 | 0.6473 |
| Decision Tree | 0.8261 | 0.8665 | 0.8271 | 0.8661 | 0.8462 | 0.6473 |
| kNN | 0.8304 | 0.8866 | 0.8492 | 0.8425 | 0.8458 | 0.6575 |
| Naive Bayes | 0.8174 | 0.8885 | 0.8244 | 0.8504 | 0.8372 | 0.6298 |
| Random Forest (Ensemble) | 0.8304 | 0.9232 | 0.8188 | 0.8898 | 0.8528 | 0.6568 |

Additional 5-fold cross-validated accuracy on the training set: Logistic Regression
0.8101, Decision Tree 0.7638, kNN 0.7971, Naive Bayes 0.8029, Random Forest 0.8246.

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Solid linear baseline. Highest precision (0.8271) and a strong AUC of 0.9008, showing that a simple linear decision boundary already separates the classes well on the scaled features. Recall is slightly better than precision, indicating a mild tendency to predict disease. |
| Decision Tree | Matches Logistic Regression on accuracy (0.8261) and recall, but its AUC drops to 0.8665, the lowest of the five. The tree captures non-linear interactions but overfits slightly on this noisy clinical data; its 5-fold CV accuracy (0.7638) is also the lowest, confirming weaker generalisation. |
| kNN | Best accuracy (tied, 0.8304) and best MCC (0.6575), meaning the strongest overall agreement between predictions and true labels. It benefits from the StandardScaler preprocessing. However, its AUC (0.8866) is lower than Logistic Regression, suggesting less confident probability estimates. |
| Naive Bayes | Slightly lower accuracy (0.8174) and the lowest MCC (0.6298). The Gaussian assumption on this small, partially non-normal feature set is not perfectly valid (e.g., `ca` and `thal`), but it still produces a respectable AUC (0.8885) and is by far the fastest to train. |
| Random Forest (Ensemble) | Best AUC (0.9232), best recall (0.8898) and best F1 (0.8528), while tying for best accuracy (0.8304) and best CV accuracy (0.8246). The ensemble approach most effectively reduces overfitting and yields the most reliable probability estimates, at the cost of slightly lower precision. |
| **Overall Winner for this dataset** | **Random Forest** — it leads in AUC, recall, F1 and cross-validated accuracy, giving the most balanced and trustworthy performance. kNN is a close runner-up when MCC is the priority metric. |

**Takeaway:** on this modest-sized, mixed-type clinical dataset with missing values, the
ensemble model (Random Forest) consistently outperforms single learners. Logistic
Regression remains a strong, interpretable baseline, while the Naive Bayes independence
assumption and the single tree's overfitting cost them the top spot.

---

## e. Project Structure

```text
project-folder/
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── test_data.csv              # Held-out test data (230 rows, used for demo)
├── data/
│   └── heart_disease_uci.csv  # Full original dataset (920 rows)
└── model/
    ├── model_training.py      # Training script for all 5 models
    ├── model_metrics.csv      # Metric comparison table
    ├── classification_reports.txt
    ├── confusion_matrices.png
    ├── metric_comparison.png
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## f. How to Run Locally

```bash
pip install -r requirements.txt
python model/model_training.py      # retrains models and regenerates metrics/plots
streamlit run app.py                # launches the web app on http://localhost:8501
```

## g. Streamlit App Features

1. **CSV upload option** — upload only the test data (Streamlit free tier has limited
   capacity).
2. **Model selection dropdown** — choose among all 5 trained models.
3. **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC shown as
   metric cards.
4. **Confusion matrix + classification report** — rendered with seaborn and scikit-learn.
5. Side-by-side comparison table of all models on the bundled test data.

## h. Deployment

Deployed on **Streamlit Community Cloud**:

> https://dinesh-panda-2025ac05867.streamlit.app/

---

*Prepared as part of the BITS M.Tech (AIML/DSE) Machine Learning Assignment 2.
Deadline: 18 August 2026.*
