import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
import joblib

# ---------------- MLFLOW ----------------
mlflow.set_experiment("Loan_Default_Prediction")

# ---------------- LOAD DATA ----------------
df = pd.read_csv('data/train.csv', low_memory=False).sample(20000, random_state=42)

# Convert numeric columns
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

X = df.drop('Default', axis=1)
y = df['Default']

# ---------------- COLUMN TYPES ----------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ---------------- PIPELINES ----------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- TRANSFORM ----------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------- HANDLE IMBALANCE ----------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=200,   # change this
    max_depth=8,        # change this
    random_state=42
)

# ---------------- MLFLOW RUN ----------------
with mlflow.start_run():

    print("MLflow run started...")

    # Train
    model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = model.predict(X_test_processed)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---------------- METRICS ----------------
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # ---------------- LOG MODEL ----------------
    mlflow.sklearn.log_model(model, "model")

    # ---------------- SHAP ----------------
    import shap
    import matplotlib.pyplot as plt

    # Take small sample

    X_sample = X_test_processed[:100]

    # Convert to dense (VERY IMPORTANT)
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    # Ensure numeric type
    X_sample = X_sample.astype(float)




    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # Create plot
    feature_names = preprocessor.get_feature_names_out()

    shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
    )

    # Save file
    plt.savefig("shap_summary.png", bbox_inches="tight")
    plt.close()

    # Log artifact
    mlflow.log_artifact("shap_summary.png")

# ---------------- SAVE ----------------
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("\n✅ Run complete. Check MLflow UI → Artifacts → shap_summary.png")