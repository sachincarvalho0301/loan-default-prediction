import pandas as pd
import numpy as np
import mlflow
from scipy.stats import ks_2samp

# ---------------- PSI FUNCTION ----------------
def calculate_psi(expected, actual, bins=10):
    
    expected = np.array(expected)
    actual = np.array(actual)

    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # Avoid division issues
    epsilon = 1e-6
    expected_perc = np.where(expected_perc == 0, epsilon, expected_perc)
    actual_perc = np.where(actual_perc == 0, epsilon, actual_perc)

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))

    return psi


# ---------------- LOAD DATA ----------------
df = pd.read_csv('data/train.csv', low_memory=False)

# ---------------- TRAIN / TEST SPLIT ----------------
train = df.sample(frac=0.7, random_state=42)
test = df.drop(train.index)

# ---------------- SELECT NUMERIC COLUMNS ----------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

psi_values = {}

# ---------------- CALCULATE PSI ----------------
for col in numeric_cols:
    try:
        psi = calculate_psi(
            train[col].dropna(),
            test[col].dropna()
        )
        psi_values[col] = psi
    except:
        psi_values[col] = np.nan

ks_values = {}

for col in numeric_cols:
    try:
        ks_stat, _ = ks_2samp(
            train[col].dropna(),
            test[col].dropna()
        )
        ks_values[col] = ks_stat
    except:
        ks_values[col] = None

# ---------------- CREATE DATAFRAME ----------------
psi_df = pd.DataFrame.from_dict(
    psi_values,
    orient='index',
    columns=['PSI']
)

psi_df = psi_df.sort_values(by='PSI', ascending=False)

print("\nTop Drifted Features:")
print(psi_df.head(10))


# ---------------- MLFLOW LOGGING ----------------
mlflow.set_experiment("Loan_Default_Prediction")

with mlflow.start_run(run_name="drift_analysis"):

    # Log PSI
    for col, psi in psi_values.items():
        if not np.isnan(psi):
            mlflow.log_metric(f"psi_{col}", float(psi))

    # ✅ ADD THIS (KS LOGGING)
    for col, ks in ks_values.items():
        if ks is not None:
            mlflow.log_metric(f"ks_{col}", float(ks))

    # Optional: log full table
    psi_df.to_csv("psi_values.csv")
    mlflow.log_artifact("psi_values.csv")