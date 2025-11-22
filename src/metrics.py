import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import mlflow
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculates and returns standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba[:, 1])
    }
    return metrics

def log_confusion_matrix(y_true, y_pred, run_name: str):
    """Generates and logs a confusion matrix plot to MLflow."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({run_name})")

    # Save plot as artifact
    plot_path = f"confusion_matrix_{run_name}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()

def log_metrics_to_mlflow(metrics: dict, y_test, y_pred, y_proba, run_name: str):
    """Logs calculated metrics and confusion matrix to MLflow."""
    mlflow.log_metrics(metrics)
    log_confusion_matrix(y_test, y_pred, run_name)
    
    logger.info("--- Metrics Logged ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("----------------------")