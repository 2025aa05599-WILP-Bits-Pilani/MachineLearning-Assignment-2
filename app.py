import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Bank Marketing Classification App")
st.write("Compare Multiple ML Models")

# ----------------------------
# Load Models
# ----------------------------

log_model = joblib.load("models/logistic_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
nb_model = joblib.load("models/naive_bayes_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ----------------------------
# Model Selection Dropdown
# ----------------------------

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# ----------------------------
# File Upload
# ----------------------------

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file, sep=';')

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Encode target
    data['y'] = data['y'].map({'no': 0, 'yes': 1})

    # Encode categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop('y', axis=1)
    y = data['y']

    X = scaler.transform(X)

    # Model selection logic
    if model_choice == "Logistic Regression":
        model = log_model
    elif model_choice == "Decision Tree":
        model = dt_model
    elif model_choice == "KNN":
        model = knn_model
    elif model_choice == "Naive Bayes":
        model = nb_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col1.metric("AUC", round(auc, 4))

    col2.metric("Precision", round(precision, 4))
    col2.metric("Recall", round(recall, 4))

    col3.metric("F1 Score", round(f1, 4))
    col3.metric("MCC", round(mcc, 4))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
