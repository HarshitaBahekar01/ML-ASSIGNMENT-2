import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Load dataset
if os.path.exists('diabetes_data_upload (1).csv'):
    df = pd.read_csv('diabetes_data_upload (1).csv')
elif os.path.exists('diabetes_data_upload.csv'):
    df = pd.read_csv('diabetes_data_upload.csv')
else:
    df = pd.read_excel('diabetes_data_upload.xls')

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()

# Explicitly map text values to binary numeric values (0 and 1)
# This avoids any LabelEncoder type issues completely
for col in df.columns:
    if col == 'Age':
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # Map common text values to 1 and 0
        mapping = {'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0, 'Male': 1, 'Female': 0, '1': 1, '0': 0}
        if df[col].isin(mapping.keys()).any():
            df[col] = df[col].map(mapping).fillna(0)
        else:
            df[col] = pd.factorize(df[col])[0]

# Ensure all columns are numeric float/int
df = df.apply(pd.to_numeric)

print("Cleaned Data Types:\n", df.dtypes)

# Separate features and target
target_col = df.columns[-1]  # 'class'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Dataset Ready -> Instances: {X.shape[0]}, Features: {X.shape[1]}")

# Split data (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessor artifact inside model folder
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/preprocessor.joblib')

# Save test dataset (features + target) for Streamlit app
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df[target_col] = y_test.values
test_df.to_csv('test_data.csv', index=False)

# 2. Define the 5 required models
models = {
    'logistic_regression': LogisticRegression(random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'naive_bayes': GaussianNB(),
    'svm': SVC(probability=True, random_state=42)
}

# 3. Train and save model artifacts
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'model/{name}.joblib')

print("SUCCESS: All 5 models trained and saved in 'model/' folder, and test_data.csv generated!")