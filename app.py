import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Heart Disease Classification - ML Deployment")

# ------------------------
# Load preprocessors
# -----------------------------
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV - same format as training data)",
    type=["csv"]
)

# -----------------------------
# Model selection
# -----------------------------
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# -----------------------------
# Prediction Pipeline
# -----------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop unused columns
    data = data.drop(columns=["id", "dataset"], errors="ignore")

    # Target
    y_test = data["num"]
    X_test = data.drop("num", axis=1)

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in X_test.columns:
            X_test[col] = encoder.transform(X_test[col])

    # Scale features
    X_test = scaler.transform(X_test)

    # Load model
    model = joblib.load(model_paths[model_name])

    # Predict
    y_pred = model.predict(X_test)

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
