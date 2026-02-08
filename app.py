import streamlit as st
import pandas as pd
import joblib
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
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="2025AA05599 ML Assignment 2", layout="wide")

st.title("🏦 Bank Marketing Classification App")
st.markdown("### Compare Multiple Machine Learning Models")
st.info("Upload the dataset once and switch between models to compare performance.")

# -------------------------------------------------
# Load Models (Cached for Performance)
# -------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/logistic_model.pkl"),
        "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
        "KNN": joblib.load("models/knn_model.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "XGBoost": joblib.load("models/xgboost_model.pkl")
    }
    scaler = joblib.load("models/scaler.pkl")
    return models, scaler

models, scaler = load_models()

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
st.markdown("---")
model_choice = st.selectbox(
    "🔽 Select Classification Model",
    list(models.keys())
)

# -------------------------------------------------
# Persistent File Upload
# -------------------------------------------------
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

uploaded_file = st.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    st.session_state.uploaded_data = pd.read_csv(uploaded_file, sep=';')

# -------------------------------------------------
# Run Model if Data Exists
# -------------------------------------------------
if st.session_state.uploaded_data is not None:

    st.markdown("---")
    st.success(f"✅ Model in Use: {model_choice}")

    data = st.session_state.uploaded_data.copy()

    # -------------------------
    # Data Preview
    # -------------------------
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    # -------------------------
    # Preprocessing
    # -------------------------
    data['y'] = data['y'].map({'no': 0, 'yes': 1})

    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop('y', axis=1)
    y = data['y']
    X = scaler.transform(X)

    # -------------------------
    # Model Selection
    # -------------------------
    model = models[model_choice]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # -------------------------
    # Metrics Calculation
    # -------------------------
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    # -------------------------
    # Display Metrics
    # -------------------------
    st.markdown("---")
    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col1.metric("AUC Score", round(auc, 4))

    col2.metric("Precision", round(precision, 4))
    col2.metric("Recall", round(recall, 4))

    col3.metric("F1 Score", round(f1, 4))
    col3.metric("MCC", round(mcc, 4))

    # -------------------------
    # Confusion Matrix
    # -------------------------
    st.markdown("---")
    st.subheader("🔢 Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

    # -------------------------
    # Classification Report
    # -------------------------
    st.markdown("---")
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.warning("⚠️ Please upload the dataset to evaluate models.")
