import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# Title of the app
st.title('Credit Card Fraud Detection')

# Introduction
st.markdown("""
This application allows students to learn about different machine learning models and their workings by applying them to a credit card fraud detection dataset.
You can upload your own dataset or use the default one provided to explore the models' performances.
""")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

st.sidebar.header("User Input Features")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset loaded. Shape: ", df.shape)
else:
    df = load_data()
    st.sidebar.write("Default dataset loaded. Shape: ", df.shape)

# Data Preprocessing
features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
X = df[features]
y = df['Class']

# Standardize the feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Model selection
st.sidebar.header("Model Selection")
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'KNN', 'Gradient Boosting']
selected_models = st.sidebar.multiselect("Select models to compare", model_names, default=model_names)

# Model explanations
model_explanations = {
    'Logistic Regression': 'Logistic Regression is a statistical model that predicts the probability of a binary outcome.',
    'Random Forest': 'Random Forest is an ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.',
    'XGBoost': 'XGBoost (Extreme Gradient Boosting) is a scalable machine learning system for tree boosting, which is an ensemble learning method.',
    'SVM': 'Support Vector Machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems.',
    'KNN': 'K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that assigns the class of a sample based on the majority class among its nearest neighbors.',
    'Gradient Boosting': 'Gradient Boosting is a machine learning technique for regression and classification problems, which builds a model in a stage-wise fashion.'
}

# Display model explanations
st.write("## Model Explanations")
for model_name in selected_models:
    st.write(f"### {model_name}")
    st.write(model_explanations[model_name])

if st.sidebar.button("Train Models"):
    with st.spinner('Training models, please wait...'):
        # Show progress bar and estimated time remaining
        training_progress_bar = st.progress(0)
        training_status_text = st.empty()
        evaluation_progress_bar = st.progress(0)
        evaluation_status_text = st.empty()

        # Train models
        def train_models(X_train_res, y_train_res, selected_models):
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'SVM': SVC(probability=True, random_state=42),
                'KNN': KNeighborsClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

            trained_models = {}
            num_models = len(selected_models)
            for idx, model_name in enumerate(selected_models):
                training_status_text.text(f"Training {model_name} ({idx + 1}/{num_models})...")
                model = models[model_name]
                model.fit(X_train_res, y_train_res)
                trained_models[model_name] = model
                training_progress_bar.progress((idx + 1) / num_models)
                time.sleep(0.1)  # Simulate time delay for visualization

            return trained_models

        trained_models = train_models(X_train_res, y_train_res, selected_models)

        # Evaluate models
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm
            }

        metrics = {}
        num_models = len(selected_models)
        for idx, model_name in enumerate(selected_models):
            evaluation_status_text.text(f"Evaluating {model_name} ({idx + 1}/{num_models})...")
            metrics[model_name] = evaluate_model(trained_models[model_name], X_test, y_test)
            evaluation_progress_bar.progress((idx + 1) / num_models)
            time.sleep(0.1)  # Simulate time delay for visualization

        # Display evaluation results in a table
        st.write("## Model Performance Metrics")
        metrics_df = pd.DataFrame(metrics).T.drop(columns='confusion_matrix')
        st.table(metrics_df)

        # Plot ROC curves using Plotly
        roc_fig = go.Figure()
        for model_name in selected_models:
            model = trained_models[model_name]
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {roc_auc:.2f})'))

        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
        roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(roc_fig)

        # Plot confusion matrices using Plotly
        st.write("## Confusion Matrices")
        for model_name in selected_models:
            st.write(f"### {model_name}")
            cm = metrics[model_name]['confusion_matrix']
            cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x='Predicted', y='Actual', color='Count'))
            cm_fig.update_layout(title=f'Confusion Matrix for {model_name}')
            st.plotly_chart(cm_fig)

        # Add an explanation section
        st.write("""
        ## Explanation

        This application demonstrates a machine learning approach to detecting credit card fraud. The dataset used is from Kaggle's Credit Card Fraud Detection competition. It contains transactions made by credit cards in September 2013 by European cardholders.

        ### Features

        - **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
        - **V1-V28**: The result of a PCA Dimensionality reduction to protect user identities and sensitive features.
        - **Amount**: Transaction amount.
        - **Class**: 1 for fraudulent transactions, 0 otherwise.

        ### Steps in the Application

        1. **Data Preprocessing**: Standardizing the feature columns and handling class imbalance using SMOTE.
        2. **Model Training**: Training six different machine learning models - Logistic Regression, Random Forest, XGBoost, SVM, KNN, and Gradient Boosting.
        3. **Model Evaluation**: Evaluating the models using accuracy, precision, recall, F1-score, and ROC-AUC score.
        4. **Visualization**: Visualizing the ROC curves for each model and displaying confusion matrices.

        You can upload your own dataset or use the default one to see how different models perform on the task of fraud detection.
        """)
        st.balloons()
