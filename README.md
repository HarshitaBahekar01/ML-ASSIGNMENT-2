# Early Stage Diabetes Risk Prediction

## Problem Statement
The objective of this project is to build and evaluate five machine learning classification models to predict early-stage diabetes risk using clinical and demographic features, and deploy an interactive web application on Streamlit.

## Dataset Description
- **Dataset Name:** Early Stage Diabetes Risk Prediction Dataset
- **Number of Instances:** 520
- **Number of Features:** 17 (16 symptoms/demographic attributes + 1 target class)
- **Target Feature:** `class` (Positive / Negative)
- **Features List:** Age, Gender, Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity.

## Github Repository Link
https://github.com/HarshitaBahekar01/ML-ASSIGNMENT-2

## Models Used

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9423 | 0.9902 | 0.9833 | 0.9219 | 0.9516 | 0.8832 |
| **Decision Tree** | 0.9904 | 0.9922 | 1.0000 | 0.9844 | 0.9921 | 0.9800 |
| **KNN** | 0.9327 | 0.9697 | 0.9831 | 0.9062 | 0.9431 | 0.8653 |
| **Naive Bayes** | 0.9423 | 0.9863 | 0.9677 | 0.9375 | 0.9524 | 0.8800 |
| **Support Vector Machine** | 0.9808 | 0.9973 | 0.9844 | 0.9844 | 0.9844 | 0.9594 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves strong baseline performance with 94.23% accuracy and 98.33% precision, effectively capturing linear symptom relationships. |
| **Decision Tree** | Delivers top overall performance with 99.04% accuracy and 100% precision by effectively splitting hierarchical feature combinations. |
| **KNN** | Yields 93.27% accuracy by grouping similar patient profiles using normalized feature distances. |
| **Naive Bayes** | Obtains 94.23% accuracy and 95.24% F1-score, providing reliable probabilistic predictions despite independence assumptions. |
| **Support Vector Machine** | Achieves 98.08% accuracy and the highest AUC score of 0.9973, showing exceptional margin separation. |
| **Overall, Winner for your dataset?** | **Decision Tree Classifier**, due to achieving highest accuracy (99.04%), perfect precision (1.0000), and top MCC (0.9800). |