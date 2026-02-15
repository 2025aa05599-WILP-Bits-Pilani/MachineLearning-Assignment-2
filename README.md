# 🏦 Machine Learning Assignment 2  
## Bank Marketing Classification – End-to-End ML Deployment

---

## 🔗 Live Application

Streamlit App:  
https://2025aa05599machinelearning-assignment-2-zemb2csvshwxvnbkvk8appj.streamlit.app/

---

## 🔗 GitHub Repository

Repository Link:  
https://github.com/2025aa05599-WILP-Bits-Pilani/MachineLearning-Assignment-2.git

---

## 🧠 Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification models on a real-world dataset.  
The models are evaluated using multiple performance metrics and deployed using an interactive Streamlit web application.

The project demonstrates the complete end-to-end machine learning workflow:
- Data preprocessing
- Model implementation
- Performance evaluation
- Model comparison
- Web application deployment

---

## 📊 Dataset Description

**Dataset Used:** Bank Marketing Dataset (UCI Machine Learning Repository)

- Total Instances: 41,188
- Input Features: 20
- Target Variable: `y` (yes/no)
- Problem Type: Binary Classification

### Class Distribution

- No: 88.73%
- Yes: 11.27%

The dataset is imbalanced, making evaluation metrics such as Recall, F1 Score, and MCC important for proper performance analysis.

---

## ⚙️ Data Preprocessing Steps

1. Label encoding applied to categorical variables  
2. Target variable encoded (no → 0, yes → 1)  
3. Stratified train-test split (80-20)  
4. Feature scaling using StandardScaler  
5. Same dataset used across all models  

---

## 🤖 Models Implemented

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## 📈 Evaluation Metrics Used

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 📊 Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9139 | 0.9370 | 0.7001 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.9180 | 0.9334 | 0.6794 | 0.5161 | 0.5866 | 0.5484 |
| KNN | 0.9053 | 0.8616 | 0.6267 | 0.3943 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4023 | 0.6174 | 0.4872 | 0.4189 |
| Random Forest | 0.9196 | 0.9521 | 0.7180 | 0.4719 | 0.5695 | 0.5414 |
| XGBoost | 0.9207 | 0.9543 | 0.6811 | 0.5571 | 0.6129 | 0.5728 |

---

## 🔍 Observations
Table 2: Model Observations
|Model |Observation|
|--------|----------|
|Logistic Regression | High AUC but lower recall for minority
class.|
|Decision Tree| Improved recall compared to Logistic
Regression.|
|KNN |Moderate performance with lower AUC.|
|Naive Bayes |High recall but low precision (more false
positives).|
|Random Forest| Balanced precision and recall.|
|XGBoost |Best overall performance with highest AUC,
F1 and MCC.|

- Logistic Regression achieved high AUC but lower recall for the minority class.
- Decision Tree improved recall compared to Logistic Regression.
- KNN showed moderate performance with lower AUC.
- Naive Bayes achieved high recall but low precision due to more false positives.
- Random Forest improved balance between precision and recall.
- XGBoost achieved the best overall performance with the highest AUC, F1 Score, and MCC, making it the most balanced model for this dataset.

---

## 🚀 Streamlit Web Application Features

The deployed application includes:

- CSV dataset upload option  
- Model selection dropdown  
- Evaluation metrics display  
- Confusion matrix display  
- Classification report display  
- Clean and professional user interface  

The application allows dynamic comparison of model performance on uploaded test data.

---

## 📁 Project Structure

```
MachineLearning-Assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── logistic_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl
```

---

## 🏁 Conclusion

This project successfully demonstrates:

- Implementation of multiple classification models
- Performance comparison using robust evaluation metrics
- Handling of imbalanced dataset
- End-to-end ML deployment using Streamlit
- Public cloud deployment via Streamlit Community Cloud

The assignment fulfills all required components including model implementation, evaluation, UI design, and deployment.

---
