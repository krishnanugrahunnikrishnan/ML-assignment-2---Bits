import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("Heart Disease Classification - ML Deployment")
# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV - same format as training data)",
    type=["csv"]
)
# -----------------------------
# Load saved artifacts
# -----------------------------
label_encoders = joblib.load("models/label_encoders.pkl")

# Preprocessors saved for linear models
model_preprocessors = {
    "Logistic Regression": joblib.load("models/logistic_regression_preprocessor.pkl"),
    "KNN": joblib.load("models/knn_preprocessor.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_preprocessor.pkl"),
    "Decision Tree":joblib.load("models/decision_tree_preprocessor.pkl"),
    "XGBoost":joblib.load("models/xgboost_preprocessor.pkl"),
    "Random Forest":joblib.load("models/random_forest_preprocessor.pkl")
}

# Tree models preprocessor (optional if you saved it)
tree_preprocessor = joblib.load("models/tree_preprocessor.pkl") if "models/tree_preprocessor.pkl" in joblib.os.listdir("models") else None

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
# Prediction
# -----------------------------
if uploaded_file is not None:
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file)
    # Drop unused columns

    for col in label_encoders.keys():
        le =label_encoders[col]
        data[col] = data[col].astype(str)  # ensure string type
        data[col] = le.transform(data[col])

    data = data.drop(columns=["id", "dataset"], errors="ignore")
    # Separate target
    y_test = data["num"]
    X_test = data.drop("num", axis=1)

    # -----------------------------
    # Preprocessing based on algorithm
    # -----------------------------
    linear_models = ["Logistic Regression", "KNN", "Naive Bayes"]
    tree_models = ["Decision Tree", "Random Forest", "XGBoost"]
    preprocessor = model_preprocessors[model_name]
    if model_name in linear_models:
        # Use saved linear preprocessor (imputation + one-hot)
        # Transform features
        X_test_processed = preprocessor.transform(X_test)
        # Scale features
        scaler = joblib.load(f"models/{model_name.lower().replace(' ', '_')}_scaler.pkl")
        X_test_processed = scaler.transform(X_test_processed)

    else:
        X_test_processed = preprocessor.transform(X_test)


    model = joblib.load(model_paths[model_name])
    y_pred = model.predict(X_test_processed)

    try:
        y_prob = model.predict_proba(X_test_processed)[:, 1]
    except:
        y_prob = y_pred

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader(f"{model_name} — Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
