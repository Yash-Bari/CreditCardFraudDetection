### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Save the script as `fraud_detection_app.py`.

### Running the App

1. Run the Streamlit app:
   ```bash
   streamlit run Creditcard_fraud_detection.py
   ```

### Dataset

- **Default Dataset**: The default dataset used is from Kaggle's Credit Card Fraud Detection competition. It contains transactions made by credit cards in September 2013 by European cardholders.
- **Upload Your Own Dataset**: You can upload your own dataset using the sidebar.

## Application Interface

### Sidebar

- **User Input Features**: Upload your input CSV file.
- **Model Selection**: Select models to compare from Logistic Regression, Random Forest, XGBoost, SVM, KNN, and Gradient Boosting.

### Main Panel

- **Model Explanations**: Displays explanations for the selected models.
- **Model Performance Metrics**: Displays accuracy, precision, recall, F1-score, and ROC-AUC score for each model in a table format.
- **ROC Curves**: Plots ROC curves for each selected model using Plotly.
- **Confusion Matrices**: Displays confusion matrices for each selected model using Plotly.

## Usage

### Model Selection and Training

1. Upload your CSV file or use the default dataset.
2. Select the machine learning models you want to compare.
3. Click the "Train Models" button to start training the selected models.

### Visualization and Metrics

After the models are trained, the app will display the following:
- **Model Performance Metrics**: A table showing accuracy, precision, recall, F1-score, and ROC-AUC score for each selected model.
- **ROC Curves**: Interactive ROC curves for each model.
- **Confusion Matrices**: Interactive confusion matrices for each model.

## Machine Learning Models Used

### Logistic Regression

Logistic Regression is a statistical model that predicts the probability of a binary outcome. It is a linear model that estimates the relationship between the dependent variable and one or more independent variables using a logistic function.

### Random Forest

Random Forest is an ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction. It reduces overfitting by averaging the results of individual trees.

### XGBoost

XGBoost (Extreme Gradient Boosting) is a scalable machine learning system for tree boosting, which is an ensemble learning method. It uses gradient boosting framework to optimize the model performance and handles missing values automatically.

### Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. It finds the hyperplane that best separates the classes in the feature space.

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that assigns the class of a sample based on the majority class among its nearest neighbors. It is easy to implement and understand.

### Gradient Boosting

Gradient Boosting is a machine learning technique for regression and classification problems, which builds a model in a stage-wise fashion. It uses the principle of boosting to combine weak learners into a strong learner.

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

## Acknowledgements

- The dataset used is from [Kaggle's Credit Card Fraud Detection competition](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- This project was developed using [Streamlit](https://streamlit.io/), [Scikit-learn](https://scikit-learn.org/), [Imbalanced-learn](https://imbalanced-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [Plotly](https://plotly.com/).
