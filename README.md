# fraud_detection_mlops_pipeline
Airflow, MLflow, Docker, FastAPI, Github Actions, pytest

## 🚀 Production Rollout / Deployment

The pipeline is orchestrated by Apache Airflow using the KubernetesPodOperator (KPO). Deployment requires the following dependencies to be provisioned first.

### Prerequisites (Infrastructure-as-Code)
Before deploying the DAG, the following cloud infrastructure must be provisioned:

1.  **EKS Cluster:** An active AWS Elastic Kubernetes Service (EKS) cluster.
2.  **Airflow Environment:** An Apache Airflow deployment configured to connect to the EKS cluster (e.g., using an EKS worker type or an external connection).
3.  **MLflow Server:** An MLflow Tracking Server deployed within the EKS cluster (or accessible via a stable URL), backed by an **RDS** database (for metadata) and **S3** (for artifacts).
4.  **ECR Image:** The production Docker image must be built (via CI) and pushed to AWS ECR.

## Docker Image

![alt text](image-2.png)

### Final Deployment Step

Once the prerequisites are met, the production DAG (`fraud_detection_dag.py`) should be synced to the Airflow DAGs folder. This will activate the full MLOps workflow, ready for scheduling.



MLflow server: 

mlflow server --host 0.0.0.0 --port 5001 --allowed-hosts host.docker.internal,localhost,127.0.0.1,0.0.0.0,0.0.0.0:5001

# MLflow: Local testing

![alt text](image.png)

![alt text](image-1.png)
