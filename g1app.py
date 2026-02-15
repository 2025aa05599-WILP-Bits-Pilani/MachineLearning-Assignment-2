import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="2025AA05599 ML Assignment 2",
    layout="wide"
)

st.title("🏦 Bank Marketing Classification Dashboard")
st.markdown("### Advanced Multi-Model Comparison & Analysis")

# ----------------------------
# Sidebar: Data & Config
# ----------------------------
st.sidebar.header("📂 Data Configuration")

DEFAULT_DATA_PATH = "./data/bank-additional-full.csv"

# Load Models (Cached)
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

try:
    models, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Data Loading Logic
if "uploaded_data" not in st.session_state:
    if os.path.exists(DEFAULT_DATA_PATH):
        st.session_state.uploaded_data = pd.read_csv(DEFAULT_DATA_PATH, sep=';')
        st.sidebar.success("✅ Default dataset loaded.")
    else:
        st.session_state.uploaded_data = None
        st.sidebar.warning("⚠️ Default dataset not found.")

# Custom Upload
uploaded_file = st.sidebar.file_uploader("Upload Custom Dataset (Optional)", type=["csv"])

if uploaded_file is not None:
    st.session_state.uploaded_data = pd.read_csv(uploaded_file, sep=';')
    st.sidebar.success("✅ Custom dataset uploaded!")

# ----------------------------
# Main App Logic
# ----------------------------
if st.session_state.uploaded_data is not None:
    data = st.session_state.uploaded_data.copy()

    # Preprocessing
    if "y" not in data.columns:
        st.error("Dataset must contain target column 'y'")
        st.stop()

    data['y'] = data['y'].map({'no': 0, 'yes': 1})

    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop("y", axis=1)
    y = data["y"]

    X_scaled = scaler.transform(X)

    # ----------------------------
    # TABS LAYOUT
    # ----------------------------
    tab1, tab2 = st.tabs(["📊 Model Comparison", "🔍 Individual Analysis"])

    # --- TAB 1: MODEL COMPARISON ---
    with tab1:
        st.subheader("🏆 Model Performance Leaderboard")
        
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y, y_pred),
                "AUC": roc_auc_score(y, y_prob),
                "Precision": precision_score(y, y_pred),
                "Recall": recall_score(y, y_pred),
                "F1": f1_score(y, y_pred),
                "MCC": matthews_corrcoef(y, y_pred)
            })

        results_df = pd.DataFrame(results).set_index("Model")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.caption("Detailed metrics for all models:")
            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
        with col2:
            st.caption("Accuracy vs AUC Comparison:")
            st.bar_chart(results_df[["Accuracy", "AUC"]], height=400)

    # --- TAB 2: INDIVIDUAL ANALYSIS ---
    with tab2:
        st.subheader("🔬 Deep Dive Analysis")
        
        selected_model_name = st.selectbox("Select Model to Analyze", list(models.keys()))
        model = models[selected_model_name]
        
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
        m1.metric("AUC", f"{roc_auc_score(y, y_prob):.4f}")
        m2.metric("Precision", f"{precision_score(y, y_pred):.4f}")
        m2.metric("Recall", f"{recall_score(y, y_pred):.4f}")
        m3.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
        m3.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")
        
        st.markdown("---")
        
        # Graphs Row
        g1, g2 = st.columns(2)
        
        with g1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with g2:
            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y, y_prob)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.4f}")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)
            
        st.markdown("---")
        st.subheader("📄 Classification Report")
        st.code(classification_report(y, y_pred))

else:
    st.info("👈 Please upload a dataset in the sidebar to begin analysis.")
