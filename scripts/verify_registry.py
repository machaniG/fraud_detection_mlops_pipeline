import os
import sys
import mlflow
from mlflow.tracking import MlflowClient 
from pprint import pprint

# --- Configuration ---
MODEL_NAME = "FraudDetectionXGBoost"
# Use environment variable. Defaulting to 127.0.0.1 is usually more reliable than localhost.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

def verify_model_versions():
    """Connects to MLflow and prints all versions of the registered model."""
    print(f"Connecting to MLflow Tracking Server at: {MLFLOW_TRACKING_URI}")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Check if the model exists
        try:
            # We are using get_latest_versions because the model is confirmed to exist in the UI
            model_versions = client.get_latest_versions(name=MODEL_NAME)
        except Exception:
            # If the model is not found, the MLflow API raises an exception.
            print(f"\nError: Model '{MODEL_NAME}' not found.")
            print(f"Troubleshooting Tip: Even though the UI shows the model, this script can't find it.")
            print(f"1. Ensure the MLflow server is running.")
            print(f"2. Ensure the connection URI ({MLFLOW_TRACKING_URI}) is correct.")
            return

        print(f"\n--- Registered Versions for Model: {MODEL_NAME} ---")
        if not model_versions:
            print("No model versions found in the registry.")
            return

        for version in model_versions:
            print("-" * 30)
            print(f"Version: {version.version}")
            print(f"Status: {version.status}")
            print(f"Stage: {version.current_stage}")
            print(f"Source Run ID: {version.run_id}")
            
            # Fetch the specific ROC-AUC metric from the source run
            try:
                run_data = client.get_run(version.run_id).data
                auc = run_data.metrics.get('roc_auc', 'N/A')
                print(f"ROC-AUC Metric: {auc}")
            except Exception as run_e:
                print(f"Could not fetch run data for ID {version.run_id}: {run_e}")
        
    except Exception as e:
        print(f"\nAn error occurred while connecting to MLflow: {e}")

if __name__ == "__main__":
    verify_model_versions()