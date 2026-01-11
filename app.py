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
# Load saved artifacts
# -----------------------------
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Preprocessors saved for linear models
linear_preprocessors = {
    "Logistic Regression": joblib.load("models/logistic_regression_preprocessor.pkl"),
    "KNN": joblib.load("models/knn_preprocessor.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_preprocessor.pkl")
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
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV - same format as training data)",
    type=["csv"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop unused columns
    data = data.drop(columns=["id", "dataset"], errors="ignore")

    # Separate target
    y_test = data["num"]
    X_test = data.drop("num", axis=1)

    # -----------------------------
    # Preprocessing based on algorithm
    # -----------------------------
    linear_models = ["Logistic Regression", "KNN", "Naive Bayes"]
    tree_models = ["Decision Tree", "Random Forest", "XGBoost"]

    if model_name in linear_models:
        # Use saved linear preprocessor (imputation + one-hot)
        preprocessor = linear_preprocessors[model_name]

        # Transform features
        X_test_processed = preprocessor.transform(X_test)

        # Scale features
        X_test_processed = scaler.transform(X_test_processed)

    else:
        # Tree models: imputation + label encoding
        # 1️⃣ Fill missing values
        if tree_preprocessor is not None:
            X_test_processed = tree_preprocessor.transform(X_test)
        else:
            # fallback: simple imputation
            num_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = X_test.select_dtypes(include=['object']).columns

            num_imputer = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')

            # Impute numerical
            X_test[num_cols] = num_imputer.fit_transform(X_test[num_cols])
            # Impute categorical
            X_test[cat_cols] = cat_imputer.fit_transform(X_test[cat_cols])
            X_test_processed = X_test.values

        # 2️⃣ Apply label encoders
        for col, le in label_encoders.items():
            if col in X_test.columns:
                X_test_processed[:, X_test.columns.get_loc(col)] = le.transform(X_test[col])

    # -----------------------------
    # Load model
    # -----------------------------
    model = joblib.load(model_paths[model_name])

    # Predict
    y_pred = model.predict(X_test_processed)

    # Some models have predict_proba
    try:
        y_prob = model.predict_proba(X_test_processed)[:, 1]
    except:
        y_prob = y_pred

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
