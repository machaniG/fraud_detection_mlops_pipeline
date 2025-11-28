#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient 
import logging

# Import roc_auc_score from scikit-learn if available, otherwise provide a lightweight fallback.
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    # Lightweight fallback implementation for binary ROC AUC using pairwise comparisons.
    # This ensures the script can run in environments without scikit-learn installed.
    def roc_auc_score(y_true, y_score):
        """
        Compute ROC AUC for binary labels using a pairwise comparison method.
        y_true: array-like of 0/1 labels
        y_score: array-like of scores or probabilities; if 2D, uses column 1 as positive-class prob
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # If probabilities for two classes are provided, take the positive class (index 1)
        if y_score.ndim > 1 and y_score.shape[1] > 1:
            y_score = y_score[:, 1]

        # If only one class present, return 0.5 (undefined AUC but commonly treated as chance)
        if len(np.unique(y_true)) == 1:
            return 0.5

        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]

        if pos.size == 0 or neg.size == 0:
            return 0.5

        # Pairwise comparison: count concordant and tied pairs
        n_pos = pos.size
        n_neg = neg.size

        # Use vectorized comparisons where possible
        # This may use more memory for large arrays but is simple and reliable as a fallback.
        # Create a (n_pos, n_neg) comparison matrix implicitly by broadcasting
        comparisons = pos[:, None] - neg[None, :]
        greater = np.sum(comparisons > 0)
        equal = np.sum(comparisons == 0)

        auc = (greater + 0.5 * equal) / (n_pos * n_neg)
        return float(auc)


# Configure logging at the start of the script
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# Add the project root to the system path to enable module imports
# Assumes script is run from project_root/scripts/ or project_root/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Fallback for running from the root itself
sys.path.insert(0, os.path.abspath('.'))


# Import modular components
from src.data_utils import load_data, identify_target, split_data
from src.model_pipeline import build_preprocessor, build_pipeline
from src.metrics import calculate_metrics, log_metrics_to_mlflow

# Configuration
MODEL_NAME = "FraudDetectionXGBoost"
EXPERIMENT_NAME = "Fraud_Detection_Pipeline"
# UPDATED: Using port 5001
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

def initialize_mlflow():
    """Sets up MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def run_training_step(X_train, X_test, y_train, y_test, preprocessor, run_name: str, params: dict):
    """
    Fits the pipeline, calculates metrics, and logs to MLflow.
    
    This function is designed to be called within an active MLflow run context (the parent run), 
    and explicitly sets nested=True to create a child run.
    """
    pipeline = build_pipeline(preprocessor, params)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # FIX: Explicitly set nested=True. Since this function is called inside the 
    # MLOps_Pipeline_Main_Run context, this ensures the run is a child run.
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params(params)
        log_metrics_to_mlflow(metrics, y_test, y_pred, y_proba, run_name)

        # Log the Scikit-learn pipeline (which includes the XGBoost model)
        # Setting registered_model_name here registers the model as a candidate
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(5)
        )
        logger.info(f"Logged run_id: {run.info.run_id} with ROC-AUC: {metrics['roc_auc']:.4f}")
        return run.info.run_id, metrics['roc_auc']

def execute_base_training(X_train, X_test, y_train, y_test, preprocessor):
    """Executes the stable, base model training (V1 candidate)."""
    logger.info("--- Starting Base Training Run (V1 Candidate) ---")
    base_params = {
        'n_estimators': 100, 
        'max_depth': 5, 
        'learning_rate': 0.1, 
        'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1)
    }
    run_id, auc = run_training_step(
        X_train, X_test, y_train, y_test, preprocessor, 
        run_name="Base_Model_V1_Candidate", 
        params=base_params
    )
    return run_id, auc

def execute_hyperparameter_tuning(X_train, X_val, y_train, y_val, X_test, y_test, preprocessor, parent_run_id):
    """
    Executes hyperparameter tuning as nested runs under a main run.
    Uses the validation set (X_val, y_val) for finding the best model, 
    but logs final metrics on the unseen test set (X_test, y_test).
    """
    logger.info("--- Starting Hyperparameter Tuning (HPO) ---")
    
    param_grid = [
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 150},
        {'max_depth': 7, 'learning_rate': 0.15, 'n_estimators': 120},
        {'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 100},
    ]

    best_val_auc = -1
    best_run_id = None
    best_params = {}

    for i, params in enumerate(param_grid):
        
        # Train on X_train, fit preprocessor on X_train, evaluate on X_val
        pipeline = build_pipeline(preprocessor, params)
        pipeline.fit(X_train, y_train)

        y_proba_val = pipeline.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, y_proba_val[:, 1])
        
        logger.info(f"Tuning trial {i+1} completed. Validation ROC-AUC: {val_auc:.4f}")

        # If it's the best performing model on the validation set, 
        # evaluate it on the test set and log the results as a child run.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params.copy()
            logger.info(f"New best model found on validation set! Logging test metrics...")
            
            # --- Log the BEST model's performance on the TEST set ---
            
            # Use the already fit pipeline and evaluate on the test set (X_test)
            y_proba_test = pipeline.predict_proba(X_test)
            y_pred_test = pipeline.predict(X_test)

            metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)
            
            # This starts a nested run for the best candidate
            with mlflow.start_run(run_name=f"HPO_Trial_{i+1}_Best_Candidate", nested=True, tags={"stage": "hpo_best_candidate"}) as run:
                mlflow.log_params(params)
                mlflow.log_metric("validation_auc", val_auc) # Log the validation metric used for selection
                log_metrics_to_mlflow(metrics, y_test, y_pred_test, y_proba_test, "Best_HPO_Candidate")

                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name="model",
                    registered_model_name=MODEL_NAME,
                    input_example=X_test.head(5)
                )
                best_run_id = run.info.run_id
    
    return best_run_id, best_val_auc


def main():
    parser = argparse.ArgumentParser(description="Modular Training/Tuning script for Fraud Detection.")
    parser.add_argument("--data", default="transactions.csv", help="Path to transactions CSV data file.")
    parser.add_argument("--mode", choices=['all', 'base', 'hpo'], default='all',
                        help="Execution mode: 'base' (run base training), 'hpo' (run tuning), 'all' (run both).")
    parser.add_argument("--parent_run_id", type=str, default=None, help="MLflow Parent Run ID for HPO runs.")
    args = parser.parse_args()

    # Initialize MLflow regardless of the mode
    initialize_mlflow()
    
    # Load and Split Data (Required for all training modes)
    if args.mode in ['all', 'base', 'hpo']:
        try:
            df = load_data(args.data)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Critical error during data loading: {e}")
            sys.exit(1)
        
        target_col = identify_target(df)
        # Use 60/20/20 split: Train / Validation / Test
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col)
        preprocessor, _, _ = build_preprocessor(X_train)

    best_base_run_id = None

    if args.mode in ['all', 'base']:
        # 1. BASE TRAINING
        # Start the main parent run for the entire workflow
        with mlflow.start_run(run_name="MLOps_Pipeline_Main_Run", nested=False) as parent_run:
            # execute_base_training will now correctly start a nested run under parent_run
            base_run_id, base_auc = execute_base_training(X_train, X_test, y_train, y_test, preprocessor)
            best_base_run_id = base_run_id
            logger.info(f"Base Training finished. Run ID: {base_run_id}, ROC-AUC: {base_auc:.4f}")
            
            # 2. HPO RUNS
            if args.mode in ['all', 'hpo']:
                hpo_run_id, hpo_auc = execute_hyperparameter_tuning(
                    X_train, X_val, y_train, y_val, X_test, y_test, preprocessor, 
                    parent_run_id=parent_run.info.run_id
                )
                logger.info(f"HPO finished. Best Child Run ID: {hpo_run_id}, Best Validation AUC: {hpo_auc:.4f}")

            # Print the main run ID for Airflow XCom
            print(parent_run.info.run_id) 

    # In 'hpo' mode, the run needs a parent ID, but the logic in main is simplified for 'all' or 'base'.
    # For 'hpo' mode run independently (without 'all' mode logic):
    if args.mode == 'hpo':
        if not args.parent_run_id:
            logger.error("In 'hpo' mode, a --parent_run_id must be provided to nest runs under.")
            sys.exit(1)
            
        with mlflow.start_run(run_id=args.parent_run_id, run_name="HPO_Process_Continuation", nested=False) as parent_run:
             hpo_run_id, hpo_auc = execute_hyperparameter_tuning(
                X_train, X_val, y_train, y_val, X_test, y_test, preprocessor, 
                parent_run_id=parent_run.info.run_id # Pass the parent ID to the execution function
            )
             logger.info(f"HPO finished. Best Child Run ID: {hpo_run_id}, Best Validation AUC: {hpo_auc:.4f}")
             print(parent_run.info.run_id) # Print the parent run ID for XCom

if __name__ == "__main__":
    main()