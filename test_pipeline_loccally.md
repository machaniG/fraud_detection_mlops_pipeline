MLOps Fraud Detection Project: Local Testing Workflow

This document provides simple instructions to test the core components of the MLOps pipeline locally before committing to GitHub Actions and AWS deployment.

## 1. Project Setup

Dependencies: Ensure you have Python (3.10+), Docker, and docker-compose installed.

Install Python requirements:
```bash
pip install -r requirements.txt
```

Data: Place your transactions.csv file in the root directory.

## 2. MLflow Local Tracking Server

We will use MLflow for experiment tracking and model registry.

Start MLflow Server: Run the server locally. This will create a local mlruns directory for artifacts and use a local file for the database.
```bash
mlflow server --host 0.0.0.0 --port 5001
```

Access: The MLflow UI will be available at http://localhost:5001. Keep this terminal window open.


## 3. Local Testing (Pytest)

Test your data utility functions and pipeline structure before training.

Run Tests: Execute pytest from the root directory. This uses the files in tests/test_pipeline.py.
```bash
pytest
#or
PYTHONPATH=. pytest
```

## 4. Docker Build and Local Execution

Build the container that holds your model training environment.

Build Docker Image:
```bash
docker build -t fraud-detection-model:local .
```

Run Training: Execute the container. The scripts/train.py will automatically start and connect to the MLflow server running locally on the host machine.

Note: We use --network=host to allow the container to access http://localhost:5001 on the host.

docker run --rm --network=host fraud-detection-model:local --data transactions.csv


Verify Results: Check the MLflow UI (http://localhost:5001). You should see:

An experiment named Fraud_Detection_Pipeline.

Runs for Base_Model_V1 and Tuning_Trial_....

Model artifacts (pipeline object, confusion matrix plot).

A registered model named FraudDetectionXGBoost with versions V1 and V2 (Staging).


## 5. CI/CD Deployment Flow (GitHub Actions)

The .github/workflows/ci_cd.yml file defines the production flow:

Push to GitHub: Commit and push all files to your repository's main branch.

Action Triggered: The GitHub Action will start.

Build and Test: It installs dependencies, runs pytest, and builds the Docker image.

Publish: If tests pass and the code is on main, it will:

Log in to Docker Hub using DOCKER_USERNAME and DOCKER_PASSWORD secrets.

Push the image to your Docker Hub repository.

Log in to AWS ECR using AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY secrets.

Push the image to the specified ECR repository.

This structure allows you to showcase the full MLOps cycle from modular code to containerized orchestration and automated CI/CD deployment.
