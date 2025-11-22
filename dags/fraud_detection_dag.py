from __future__ import annotations

import pendulum
import logging
from airflow.decorators import dag, task
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.python import get_current_context

# Initialize logging for the DAG file
logger = logging.getLogger(__name__)

@dag(
    dag_id="fraud_detection_training_pipeline_modular",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["mlops", "mlflow", "fraud"],
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
        "execution_timeout": pendulum.duration(hours=2),
    }
)
def fraud_detection_training_pipeline():
    """
    Airflow DAG to orchestrate the fraud detection model training workflow,
    split into distinct tasks for better control and visibility.
    """
    
    # Define variables
    DOCKER_IMAGE = "your_dockerhub_username/fraud-detection-model:latest"
    DATA_PATH = "transactions.csv"
    MLFLOW_TRACKING_URI = "http://mlflow-server.default.svc.cluster.local:5001" # K8s Service Name

    def build_kpo_task(task_id: str, script_path: str, args: list = None, **kwargs):
        """Helper to create a KubernetesPodOperator instance."""
        
        return KubernetesPodOperator(
            task_id=task_id,
            name=task_id.replace('_', '-'),
            namespace="airflow",
            image=DOCKER_IMAGE,
            cmds=["python", script_path], 
            arguments=args,
            env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
            do_xcom_push=True, 
            is_delete_operator_pod=True,
            **kwargs
        )

    # 1. Task: Full Training and HPO
    # Runs the base model training AND all HPO trials within a single parent run.
    # The script prints the PARENT RUN ID to stdout, which is captured by XCom.
    full_training_and_hpo = build_kpo_task(
        task_id="full_training_and_hpo",
        script_path="scripts/train.py",
        args=["--data", DATA_PATH, "--mode", "all"],
        container_resources={
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "2000m", "memory": "4Gi"}, # Increased limit for HPO
        }
    )

    # 2. Task: Model Promotion
    # Executes the dedicated promotion script. This script connects to MLflow,
    # queries the runs within the parent run, finds the best metric, 
    # and transitions the corresponding model version to Staging.
    model_promotion = build_kpo_task(
        task_id="model_promotion",
        script_path="scripts/promote.py",
        args=[
            "--parent_run_id", 
            "{{ task_instance.xcom_pull(task_ids='full_training_and_hpo', key='return_value') }}"
        ],
        container_resources={
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        }
    )

    # Define the DAG flow: Training must complete before promotion can occur.
    full_training_and_hpo >> model_promotion 

fraud_detection_pipeline = fraud_detection_training_pipeline()