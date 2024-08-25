# utils.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, auc, confusion_matrix, 
                             classification_report)


# Function to save model
def save_model(model, name, folder='models'):
    """
    Save a trained model to disk.

    Parameters:
    - model: Trained model object.
    - name (str): Name of the model file.
    - folder (str): Directory to save the model. Default is 'models'.
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f'{name}.pkl')
    joblib.dump(model, filepath)


def evaluate_model(y_val, y_pred):
    """
    Evaluate model performance metrics.

    Parameters:
    - y_val: True labels.
    - y_pred: Predicted labels.

    Returns:
    - tuple: Accuracy, F1 score, Precision, Recall, and Confusion Matrix.
    """
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)
    return accuracy, f1, precision, recall, conf_matrix


def generate_classification_report(y_val, y_pred, target_names=['Class 0', 'Class 1']):
    """
    Generate a classification report.

    Parameters:
    - y_val: True labels.
    - y_pred: Predicted labels.
    - target_names (list): List of class names.

    Returns:
    - str: Classification report.
    """
    return classification_report(y_val, y_pred, target_names=target_names)


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name, report_dir):
    """
    Train and evaluate a model, save performance metrics and plots.

    Parameters:
    - model: Model object.
    - X_train: Training features.
    - y_train: Training labels.
    - X_val: Validation features.
    - y_val: Validation labels.
    - model_name (str): Name for saving the model.
    - report_dir (str): Directory for saving reports.

    Returns:
    - dict: Performance metrics and file paths.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    accuracy, f1, precision, recall, conf_matrix = evaluate_model(y_val, y_pred)
    report = generate_classification_report(y_val, y_pred)
    
    save_model(model, model_name)
    
    # Save confusion matrix plot
    conf_matrix_path = os.path.join(report_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, title=f'{model_name} Confusion Matrix', save_path=conf_matrix_path)
    
    # Save ROC curve data for later comparison
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "conf_matrix_path": conf_matrix_path,
        "classification_report": report,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }


def save_comparison_plot(results, save_path):
    """
    Save a comparison plot of ROC curves.

    Parameters:
    - results (list of dict): List of dictionaries containing model names, FPR, TPR, and AUC.
    - save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    for result in results:
        plt.plot(result['fpr'], result['tpr'], label=f"{result['model_name']} (AUC = {result['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_comparison_table(results, save_path):
    """
    Save a comparison table of model metrics.

    Parameters:
    - results (list of dict): List of dictionaries containing model names and metrics.
    - save_path (str): File path to save the CSV.
    """
    df = pd.DataFrame(results)[['model_name', 'accuracy', 'f1', 'precision', 'recall']]
    df.to_csv(save_path, index=False)


def plot_confusion_matrix(conf_matrix, title="Confusion Matrix", save_path=None):
    """
    Plot and optionally save a confusion matrix.

    Parameters:
    - conf_matrix (array-like): Confusion matrix.
    - title (str): Title of the plot.
    - save_path (str, optional): File path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
