Here's a `README.md` file for your project:

```markdown
# Credit Card Fraud Detection

This application allows users to learn about different machine learning models and their workings by applying them to a credit card fraud detection dataset. Users can upload their own dataset or use the default one provided to explore the models' performances.

## Features

- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: The result of a PCA Dimensionality reduction to protect user identities and sensitive features.
- **Amount**: Transaction amount.
- **Class**: 1 for fraudulent transactions, 0 otherwise.

## Steps in the Application

1. **Data Preprocessing**: Standardizing the feature columns and handling class imbalance using SMOTE.
2. **Model Training**: Training six different machine learning models - Logistic Regression, Random Forest, XGBoost, SVM, KNN, and Gradient Boosting.
3. **Model Evaluation**: Evaluating the models using accuracy, precision, recall, F1-score, and ROC-AUC score.
4. **Visualization**: Visualizing the ROC curves for each model and displaying confusion matrices.

## Getting Started

### Prerequisites

Ensure you have the required libraries installed:
```bash
pip install pandas scikit-learn imbalanced-learn xgboost matplotlib streamlit plotly
```

### Running the App

1. Save the script as `fraud_detection_app.py`.
2. Run the Streamlit app:
   ```bash
   streamlit run fraud_detection_app.py
   ```

## Application Interface

### Sidebar

- **User Input Features**: Upload your input CSV file.
- **Model Selection**: Select models to compare from Logistic Regression, Random Forest, XGBoost, SVM, KNN, and Gradient Boosting.

### Main Panel

- **Model Explanations**: Displays explanations for the selected models.
- **Model Performance Metrics**: Displays accuracy, precision, recall, F1-score, and ROC-AUC score for each model in a table format.
- **ROC Curves**: Plots ROC curves for each selected model using Plotly.
- **Confusion Matrices**: Displays confusion matrices for each selected model using Plotly.

## Default Dataset

The default dataset used is from Kaggle's Credit Card Fraud Detection competition. It contains transactions made by credit cards in September 2013 by European cardholders.

## Explanation

This application demonstrates a machine learning approach to detecting credit card fraud. By exploring different models and visualizing their performances, users can gain insights into the strengths and weaknesses of each model in detecting fraudulent transactions.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset used is from [Kaggle's Credit Card Fraud Detection competition](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Contact

For any inquiries, please contact [Your Name](mailto:your.email@example.com).
```

This `README.md` provides a clear overview of the project, including features, steps, prerequisites, running instructions, and explanations for each section of the application. It also includes acknowledgments and contact information.
