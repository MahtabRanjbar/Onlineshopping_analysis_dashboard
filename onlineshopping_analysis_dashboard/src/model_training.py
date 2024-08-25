import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier)
import utils

# Load and preprocess data
def load_and_preprocess_data(filepath):
    """
    Load data, apply preprocessing steps, and return features and target variable.
    
    Parameters:
    - filepath (str): Path to the CSV data file.

    Returns:
    - X (DataFrame): Features.
    - y (Series): Target variable.
    """
    df = pd.read_csv(filepath)
    df = pd.concat([df, pd.get_dummies(df['Month'], prefix='Month')], axis=1).drop(['Month'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['VisitorType'], prefix='VisitorType')], axis=1).drop(['VisitorType'], axis=1)
    y = df['Revenue']
    X = df.drop(['Revenue'], axis=1)
    return X, y

def main():
    # Define file paths
    data_filepath = 'data/online_shoppers_intention.csv'
    report_dir = 'reports/model_reports'
    comparison_dir = 'reports/comparison'
    
    # Create directories for saving results
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(data_filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Data scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Define models
    models = {
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }
    
    # Train and evaluate models
    results = []
    for model_name, model in models.items():
        result = utils.train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name, report_dir)
        results.append(result)
    
    # Save comparison plots and tables
    utils.save_comparison_plot(results, os.path.join(comparison_dir, 'roc_comparison.png'))
    utils.save_comparison_table(results, os.path.join(comparison_dir, 'model_comparison.csv'))

if __name__ == "__main__":
    main()
