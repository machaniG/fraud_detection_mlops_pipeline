"""
Airflow DAG for AWS EC2 Deployment
This DAG orchestrates fraud detection model training via API calls to your EC2 instance.

Key Changes from Original:
1. Removed KubernetesPodOperator (K8s-specific)
2. Uses SimpleHttpOperator to call FastAPI endpoints
3. Simplified for EC2 deployment
4. Uses AWS S3 for data/model storage
"""

from __future__ import annotations
import pendulum
import json
from airflow.decorators import dag, task
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
import boto3
import logging

logger = logging.getLogger(__name__)

# Configuration - Update these with your actual values
MODEL_API_URL = "http://YOUR_EC2_IP:8000"  # Replace with your EC2 public IP
S3_BUCKET_DATA = "fraud-detection-yourname-data"
S3_BUCKET_MODELS = "fraud-detection-yourname-models"

@dag(
    dag_id = "fraud_detection_aws_pipeline",
    start_date = pendulum.datetime(2023, 1, 1, tz = "UTC"),
    schedule = None,  # Manual trigger only
    catchup = False,
    tags = ["mlops", "mlflow", "fraud", "aws"],
    default_args = {
        "owner": "airflow",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes = 5),
        "execution_timeout": pendulum.duration(hours = 2),
    }
)
def fraud_detection_aws_pipeline():
    """
    Complete MLOps pipeline for fraud detection on AWS.
    
    Flow:
    1. Verify data in S3
    2. Check API health
    3. Trigger training (base + HPO)
    4. Verify model was saved
    5. Promote best model
    6. Test predictions
    """
    
    @task
    def check_s3_data():
        """Verify that training data exists in S3"""
        s3 = boto3.client('s3')
        try:
            s3.head_object(Bucket = S3_BUCKET_DATA, Key='raw/transactions.csv')
            logger.info("✓ Training data found in S3")
            return True
        except Exception as e:
            logger.error(f"Training data not found in S3: {e}")
            raise
    
    @task
    def verify_api_health():
        """Check if the FastAPI service is running"""
        import requests
        try:
            response = requests.get(f"{MODEL_API_URL}/health", timeout=10)
            response.raise_for_status()
            logger.info(f"✓ API is healthy: {response.json()}")
            return response.json()
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            raise
    
    # Task: Trigger model training
    # This calls your FastAPI /train endpoint which runs train.py
    trigger_training = SimpleHttpOperator(
        task_id = 'trigger_model_training',
        http_conn_id = 'fraud_api',  # Configure this in Airflow UI
        endpoint = '/train',
        method = 'POST',
        data = json.dumps({
            "data_path": "transactions.csv",
            "mode": "all"  # Run base training + HPO
        }),
        headers={"Content-Type": "application/json"},
        response_check=lambda response: response.status_code == 200,
        log_response=True,
    )
    
    @task
    def wait_for_training():
        """
        Wait for training to complete.
        In production, you'd poll a status endpoint.
        For now, we wait a fixed time.
        """
        import time
        logger.info("Waiting for training to complete...")
        time.sleep(180)  # Wait 3 minutes - adjust based on your training time
        logger.info("✓ Training should be complete")
        return True
    
    @task
    def verify_model_in_registry():
        """Check that models were registered in MLflow"""
        import requests
        try:
            response = requests.get(f"{MODEL_API_URL}/models/list", timeout = 10)
            response.raise_for_status()
            models = response.json()
            logger.info(f"✓ Models in registry: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to verify models: {e}")
            raise
    
    # Task: Promote best model to staging
    promote_model = SimpleHttpOperator(
        task_id='promote_best_model',
        http_conn_id='fraud_api',
        endpoint='/promote',
        method='POST',
        data=json.dumps({
            "parent_run_id": "{{ task_instance.xcom_pull(task_ids='trigger_model_training') }}"
        }),
        headers={"Content-Type": "application/json"},
        response_check=lambda response: response.status_code == 200,
        log_response=True,
    )
    
    @task
    def test_prediction():
        """Test the promoted model with a sample prediction"""
        import requests
        
        # Sample transaction for testing
        test_data = {
            "user_id": 12345,
            "account_age_days": 365,
            "total_transactions_user": 50,
            "avg_amount_user": 75.5,
            "amount": 150.0,
            "country": "US",
            "bin_country": "US",
            "channel": "online",
            "merchant_category": "retail",
            "promo_used": 0,
            "avs_match": 1,
            "cvv_result": 1,
            "three_ds_flag": 0,
            "shipping_distance_km": 25.5
        }
        
        try:
            response = requests.post(
                f"{MODEL_API_URL}/predict",
                json=test_data,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"✓ Prediction test successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Prediction test failed: {e}")
            raise
    
    @task
    def log_pipeline_success():
        """Log successful pipeline completion"""
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        return {"status": "success"}
    
    # Define task dependencies
    s3_check = check_s3_data()
    api_health = verify_api_health()
    
    # Both checks must pass before training
    [s3_check, api_health] >> trigger_training
    
    # After training, wait and verify
    trigger_training >> wait_for_training() >> verify_model_in_registry()
    
    # Promote and test
    verify_model_in_registry() >> promote_model >> test_prediction()
    
    # Log success
    test_prediction() >> log_pipeline_success()

# Instantiate the DAG
fraud_detection_dag = fraud_detection_aws_pipeline()