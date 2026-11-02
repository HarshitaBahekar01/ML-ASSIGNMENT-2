import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="Early Stage Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("Early Stage Diabetes Risk Prediction Dashboard")
st.markdown("---")

# Load artifacts safely
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/preprocessor.joblib')
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.joblib'),
        'Decision Tree': joblib.load('model/decision_tree.joblib'),
        'K-Nearest Neighbors (KNN)': joblib.load('model/knn.joblib'),
        'Naive Bayes': joblib.load('model/naive_bayes.joblib'),
        'Support Vector Machine (SVM)': joblib.load('model/svm.joblib')
    }
    return scaler, models

try:
    scaler, models = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure model files exist in the `model/` directory.")
    st.stop()

# Sidebar Navigation
st.sidebar.header("Dashboard Controls")
app_mode = st.sidebar.radio("Select Mode", ["Batch Evaluation (Upload CSV)", "Single Patient Risk Assessment"])

if app_mode == "Batch Evaluation (Upload CSV)":
    st.subheader("Upload Test Dataset & Evaluate Models")
    st.info("please upload test data containing features and the target column.")
    
    uploaded_file = st.file_uploader("Upload Test Data CSV", type=["csv"])
    
    df_test = None
    if uploaded_file is not None:
        try:
            df_test = pd.read_csv(uploaded_file)
            st.success("Uploaded CSV loaded successfully!")
        except Exception as e:
            st.error(f"Error reading the uploaded CSV file: {e}")
    elif os.path.exists('test_data.csv'):
        try:
            df_test = pd.read_csv('test_data.csv')
            st.info("Using default `test_data.csv` for evaluation.")
        except Exception as e:
            st.error(f"Error reading default `test_data.csv`: {e}")
    else:
        st.warning("Please upload a valid test dataset CSV file to proceed.")

    if df_test is not None:
        try:
            # Error Handling: Check if dataset has at least 2 columns (features + target)
            if df_test.shape[1] < 2:
                st.error("The uploaded CSV file must contain multiple feature columns and a target column.")
                st.stop()

            target_col = df_test.columns[-1]
            X_test = df_test.drop(columns=[target_col])
            y_test = df_test[target_col]
            
            # Error Handling: Map categorical text strings to numeric values if present
            for col in X_test.columns:
                if X_test[col].dtype == 'object':
                    mapping = {'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0, 'Male': 1, 'Female': 0}
                    X_test[col] = X_test[col].map(mapping).fillna(0)
            
            # Also handle target column mapping if it contains text strings
            if y_test.dtype == 'object':
                target_mapping = {'Positive': 1, 'Negative': 0, 'Yes': 1, 'No': 0}
                y_test = y_test.map(target_mapping).fillna(0)

            X_test = X_test.astype(float)
            
            # Error Handling: Verify feature dimension matches scaler expectation
            expected_features = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else X_test.shape[1]
            if X_test.shape[1] != expected_features:
                st.error(f"Feature mismatch error: Uploaded dataset has {X_test.shape[1]} features, but the trained model expects {expected_features} features. Please upload the correct test dataset CSV.")
                st.stop()

            # Transform features
            X_test_scaled = scaler.transform(X_test)

            # Model Selection Dropdown
            selected_model_name = st.sidebar.selectbox("Select Model for Detailed Analysis", list(models.keys()))
            selected_model = models[selected_model_name]

            # Overall metrics comparison table across all models
            st.markdown("### Overall Model Comparison")
            metrics_summary = []
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

                metrics_summary.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "AUC": round(roc_auc_score(y_test, y_proba), 4) if len(np.unique(y_test)) > 1 else 1.0,
                    "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                    "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
                })

            metrics_df = pd.DataFrame(metrics_summary)
            st.dataframe(metrics_df, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### 🔍 Detailed Analysis: **{selected_model_name}**")

            # Predictions for selected model
            y_pred_selected = selected_model.predict(X_test_scaled)
            y_proba_selected = selected_model.predict_proba(X_test_scaled)[:, 1] if hasattr(selected_model, "predict_proba") else y_pred_selected

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Evaluation Metrics")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred_selected):.4f}")
                m2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba_selected):.4f}" if len(np.unique(y_test)) > 1 else "N/A")
                
                m3, m4 = st.columns(2)
                m3.metric("Precision", f"{precision_score(y_test, y_pred_selected, zero_division=0):.4f}")
                m4.metric("Recall", f"{recall_score(y_test, y_pred_selected, zero_division=0):.4f}")

                m5, m6 = st.columns(2)
                m5.metric("F1-Score", f"{f1_score(y_test, y_pred_selected, zero_division=0):.4f}")
                m6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred_selected):.4f}")

            with col2:
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_selected)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)

            st.markdown("####  Classification Report")
            report_dict = classification_report(y_test, y_pred_selected, target_names=['Negative (0)', 'Positive (1)'], output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)

        except Exception as e:
            st.error(f"Error processing the dataset: {e}. Please ensure your CSV matches the required dataset format and feature columns.")

elif app_mode == "Single Patient Risk Assessment":
    st.subheader("Single Patient Risk Assessment")
    st.markdown("Input patient clinical symptoms to evaluate risk across models.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=40)
        gender = st.selectbox("Gender", ["Female", "Male"])
        polyuria = st.selectbox("Polyuria (Excessive Urination)", ["No", "Yes"])
        polydipsia = st.selectbox("Polydipsia (Excessive Thirst)", ["No", "Yes"])
        sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
        weakness = st.selectbox("Weakness", ["No", "Yes"])

    with col2:
        polyphagia = st.selectbox("Polyphagia (Excessive Hunger)", ["No", "Yes"])
        genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
        visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
        itching = st.selectbox("Itching", ["No", "Yes"])
        irritability = st.selectbox("Irritability", ["No", "Yes"])
        delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"])

    with col3:
        partial_paresis = st.selectbox("Partial Paresis", ["No", "Yes"])
        muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"])
        alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': 1 if gender == 'Male' else 0,
        'Polyuria': 1 if polyuria == 'Yes' else 0,
        'Polydipsia': 1 if polydipsia == 'Yes' else 0,
        'sudden weight loss': 1 if sudden_weight_loss == 'Yes' else 0,
        'weakness': 1 if weakness == 'Yes' else 0,
        'Polyphagia': 1 if polyphagia == 'Yes' else 0,
        'Genital thrush': 1 if genital_thrush == 'Yes' else 0,
        'visual blurring': 1 if visual_blurring == 'Yes' else 0,
        'Itching': 1 if itching == 'Yes' else 0,
        'Irritability': 1 if irritability == 'Yes' else 0,
        'delayed healing': 1 if delayed_healing == 'Yes' else 0,
        'partial paresis': 1 if partial_paresis == 'Yes' else 0,
        'muscle stiffness': 1 if muscle_stiffness == 'Yes' else 0,
        'Alopecia': 1 if alopecia == 'Yes' else 0,
        'Obesity': 1 if obesity == 'Yes' else 0
    }]).astype(float)

    if st.button("Predict Diabetes Risk", type="primary"):
        try:
            scaled_input = scaler.transform(input_data)
            st.markdown("### Prediction Results Across Models")
            
            results_list = []
            for name, model in models.items():
                pred = model.predict(scaled_input)[0]
                proba = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else float(pred)
                status = "Positive (High Risk)" if pred == 1 else "Negative (Low Risk)"
                results_list.append({
                    "Model": name,
                    "Prediction": status,
                    "Probability / Confidence": f"{proba:.2%}"
                })
            
            st.table(pd.DataFrame(results_list))
        except Exception as e:
            st.error(f"Error during prediction: {e}")