import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# -----------------------------
# create Models directory
# -------------------------
os.makedirs("models", exist_ok=True)


# -----------------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("ml-assignment-2/heart_disease_uci.csv")
# Binary target conversion
df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
# Drop non-useful feature
df = df.drop(columns=["id", "dataset"])
df = df.dropna()
categorical_cols = df.select_dtypes(include=["object"]).columns

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders for deployment
joblib.dump(label_encoders, "models/label_encoders.pkl")



X = df.drop("num", axis=1)
y = df["num"]

# --------------------------
# Train-Test Split-- 80: 20 split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")


# -----------------------------
# Models Dictionary
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}
#%%

# -----------------------------
# Training, Evaluation & Saving
# -----------------------------
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)}
    

    results.append(metrics)
    # Save model
    file_name = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{file_name}")

    print(metrics)


# -----------------------------
# Summary Table
# -----------------------------
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

