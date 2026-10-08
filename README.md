# CreditWise — Loan Approval Prediction System

CreditWise is an end-to-end machine learning application for predicting loan approval using applicant financial, demographic, employment, and credit information.

The project includes data preprocessing, exploratory data analysis, feature engineering, model comparison, and an interactive Streamlit dashboard.

## Overview

CreditWise analyzes loan application data and applies machine learning classification techniques to predict whether a loan application is likely to be approved.

The project follows a complete machine learning workflow:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training
- Model evaluation
- Model comparison
- Loan approval prediction
- Interactive Streamlit dashboard

## Key Features

- Data preprocessing using Pandas and NumPy
- Exploratory Data Analysis and visualization
- Feature engineering
- Binary loan approval classification
- Comparison of multiple machine learning models
- Gaussian Naive Bayes as the final prediction model
- Model evaluation using Accuracy, Precision, Recall, and F1-Score
- Correlation analysis using heatmaps
- Interactive Streamlit dashboard
- Loan approval prediction based on user-provided applicant information

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Gaussian Naive Bayes was selected as the final model for the CreditWise prediction application.

## Dataset

The dataset contains loan application information including:

- Applicant income
- Co-applicant income
- Credit score
- Existing loans
- DTI ratio
- Savings
- Collateral value
- Loan amount
- Loan term
- Employment status
- Education level
- Property area
- Loan purpose
- Other applicant attributes

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Streamlit

## Project Workflow

1. Data Loading
2. Data Inspection and Cleaning
3. Exploratory Data Analysis
4. Data Visualization
5. Feature Engineering
6. Train-Test Split
7. Feature Scaling and Encoding
8. Model Training
9. Model Evaluation
10. Model Comparison
11. Streamlit Application Development

## Project Structure

```text
CreditWise-Loan-Approval-Prediction/
│
├── app.py
├── train_model.py
├── credit_wise.ipynb
├── loan_approval_data.csv
│
├── naive_bayes_model.pkl
├── scaler.pkl
├── onehot_encoder.pkl
├── feature_columns.pkl
├── label_encoder.pkl
│
├── requirements.txt
├── .gitignore
└── README.md