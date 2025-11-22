#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "FraudDetectionXGBoost"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# FILE: scripts/promote.py

def find_and_promote_best_model(parent_run_id: str, metric: str = "roc_auc"):
    """
    Queries MLflow to find the best run (by metric) among the base run 
    and its nested HPO runs, registers the model, and transitions it to Staging.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
    except Exception as e:
        logger.error(f"Could not initialize MLflow client: {e}")
        sys.exit(1)

    logger.info(f"Searching for the best run relative to parent_run_id: {parent_run_id}")

    # --- FIX START ---
    
    # 1. Get the parent run to extract the Experiment ID
    try:
        parent_run = client.get_run(parent_run_id)
        experiment_id = parent_run.info.experiment_id
    except Exception as e:
        logger.error(f"Could not find parent run ID {parent_run_id} in experiment {EXPERIMENT_NAME}: {e}")
        return

    # 2. Query all nested runs (children) of the parent run
    # This avoids the invalid 'OR' filter and gets all HPO runs.
    filter_string_nested = f"tags.\"mlflow.parentRunId\" = '{parent_run_id}'"
    nested_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string_nested,
        order_by=[f"metrics.{metric} DESC"]
    )
    
    # 3. Combine the parent run and all nested runs for evaluation
    # The parent run (Base Model) must be included as it might be the best.
    runs_to_evaluate = [parent_run] + nested_runs

    # 4. Determine the overall best run based on the metric
    best_run = None
    best_metric_value = -float('inf')

    for run in runs_to_evaluate:
        metric_value = run.data.metrics.get(metric)
        if metric_value is not None and metric_value > best_metric_value:
            best_metric_value = metric_value
            best_run = run
    
    if best_run is None:
        logger.error(f"No runs with the required metric '{metric}' found for promotion.")
        return
        
    best_run_id = best_run.info.run_id
    
    # --- FIX END ---
    
    logger.info(f"Best model found: Run ID {best_run_id} with {metric}={best_metric_value:.4f}")

    # 5. Register the model version from the best run
    model_uri = f"runs:/{best_run_id}/model"
    
    logger.info(f"Registering model from URI: {model_uri}")
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )
    
    # 6. Transition the newly registered version to Staging
    version_number = model_version.version
    target_stage = "Staging"
    
    logger.info(f"Registered model V{version_number}. Transitioning to {target_stage}.")
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version_number,
        stage=target_stage
    )
    logger.info(f"Model {MODEL_NAME} V{version_number} successfully transitioned to {target_stage}.")


def main():
    parser = argparse.ArgumentParser(description="MLflow Model Promotion Script.")
    parser.add_argument("--parent_run_id", type=str, required=True, 
                        help="MLflow Run ID of the parent process (Base Training) to search for best models.")
    args = parser.parse_args()

    find_and_promote_best_model(args.parent_run_id)

if __name__ == "__main__":
    main()