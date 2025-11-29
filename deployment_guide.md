# host determine IP
ipconfig getifaddr en0   # typical for Wi‑Fi
# or use ifconfig

### my host IP: 192.168.0.106

# Start mlflow on host (you already do):
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root "$(pwd)/mlartifacts"

# Run container pointing to host IP (replace 192.168.1.10 with your IP)
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://192.168.0.106:5001 \
  frida33876/fraud-detection-model:latest

# Deployment Steps

## terminal 1
### start mlflow server. for production us postgres db

mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts

"""Verify: The server should now be running. You can open your browser and navigate to http://127.0.0.1:5001 to see the MLflow UI. Keep this terminal window open while you execute the next steps.""

## terminal 2

### export mlflow tracking url
export MLFLOW_TRACKING_URI=http://192.168.0.106:5001

### install dependencies
pip install -r requirements.txt

### run training
python scripts/train.py --data transactions.csv --mode all

### Run Promotion

PARENT_RUN_ID="<YOUR_PARENT_RUN_ID_HERE>" # e.g., cd1c867077f04f759f0a8234e48123d4
python scripts/promote.py --parent_run_id $PARENT_RUN_ID

"""" Terminal 2: Run Promotion (from root directory)"""

PARENT_RUN_ID="5d1c867077f04f759f0a8234e48123d4"
python scripts/promote.py --parent_run_id $PARENT_RUN_ID


### Verify Registration
python scripts/verify_registry.py


## Terminal 3

### Step 1: Make Scripts Executable

```bash
chmod +x scripts/deploy_to_aws.sh
chmod +x scripts/cleanup_aws.sh
```

### Step 2: Update Configuration

Edit `scripts/deploy_to_aws.sh` and update these variables:

```bash
KEY_NAME="fraud-detection-key"  # Your key pair name
S3_BUCKET_DATA="fraud-detection-YOURNAME-data"  # Make unique
S3_BUCKET_MODELS="fraud-detection-YOURNAME-models"
DOCKER_IMAGE="YOUR_DOCKERHUB_USERNAME/fraud-detection-model:latest"
```

### build docker image
docker build --no-cache -t frida33876/fraud-detection-model:latest -f dockerfile .
docker build -t fraud-detection-model:local .

### Run container pointing to host IP (replace 192.168.1.10 with your IP)
### Run Docker container, pointing to MLflow server:
### also mount the artifact directory from your host into the container using the -v option in your docker run command:

docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://192.168.0.106:5001 \
  -v /Users/gechemba/Documents/fraud_detection_mlops_pipeline/mlartifacts:/Users/gechemba/Documents/fraud_detection_mlops_pipeline/mlartifacts \
  frida33876/fraud-detection-model:latest


"""How to fix (optional but recommended for production):

Use the exact same Python and package versions in your Docker image as in your training environment.
You can fetch the model’s environment file with:"""
```bash
mlflow.pyfunc.get_model_dependencies(model_uri)
```
"""Then install those dependencies in your Dockerfile."""

### If the API starts and loads the model, test endpoints (e.g., /health, /predict). The warnings are not fatal, but matching environments is best practice.

To test your FastAPI endpoints, you can use:

1. Browser
Open: http://localhost:8000/docs
This shows the interactive Swagger UI where you can test all endpoints.
2. curl (from terminal)

## Health check 

For the /health endpoint, you don’t need to fill in any fields—just click "Try it out" and then "Execute". It’s a simple GET request.


## predict endpoint: provide a json object with the features your model expects
{
  "user_id": 999,
  "account_age_days": 150,
  "total_transactions_user": 50,
  "avg_amount_user": 45.5,
  "amount": 105,
  "country": "US",
  "bin_country": "US",
  "channel": "online",
  "merchant_category": "A",
  "promo_used": 1,
  "avs_match": 1,
  "cvv_result": 1,
  "three_ds_flag": 0,
  "shipping_distance_km": 10.5
}

# push docker image to dockerhub

### Log in to Docker Hub
docker login

### Tag your image (if needed)
docker tag frida33876/fraud-detection-model:latest frida33876/fraud-detection-model:latest

### Push to Docker Hub
docker push frida33876/fraud-detection-model:latest

Before AWS deployment:
1. Commit final code changes so you can lock the exact code version that corresponds to the image you just pushed
```bash
# 1. Check your status and add any new or modified files
git status
git add .

# 2. Commit the changes (including the fixed train.py)
git commit -m "Deployment preparation: Fix MLflow log_model error and finalize local testing"

# 3. Push the code to your GitHub repository
git push
```
2. Configure AWS Credentials (Required for Deployment)

You need to provide your machine with the security credentials to interact with AWS services (like creating resources, pushing to ECR, or running deployments).
```bash

```


#  Deploy to AWS

### Run your deployment script:
```bash
# Run deployment script
./scripts/deploy_to_aws.sh

# This will:
# 1. Create S3 buckets
# 2. Launch EC2 instances
# 3. Deploy services
# 4. Output access URLs

# Wait 5-10 minutes for services to start
```
This will provision EC2, set up S3 buckets, and deploy your services.


### Step 5: Upload Training Data

```bash
# Upload your dataset to S3
aws s3 cp transactions.csv s3://fraud-detection-YOURNAME-data/raw/

ws s3 cp transactions.csv s3://fraud-detection-frida-data/raw/
```

### Step 6: Access Services

After deployment completes, you'll see:

```
FastAPI Docs: http://YOUR_API_IP:8000/docs
FastAPI Health: http://YOUR_API_IP:8000/health
Airflow UI: http://YOUR_AIRFLOW_IP:8080 (admin/admin)
MLflow UI: http://YOUR_AIRFLOW_IP:5001
```

### Step 7: Configure Airflow DAG

1. Access Airflow UI at `http://YOUR_AIRFLOW_IP:8080`
2. Login with `admin` / `admin`
3. Upload `dags/fraud_detection_dag_aws.py`
4. Update the DAG:
   - Change `MODEL_API_URL` to your API IP
   - Save and enable the DAG


### Step 8: Run Training Pipeline

### Run Airflow DAG Via Airflow (Recommended)

1. Go to Airflow UI
2. Find `fraud_detection_aws_pipeline`
3. Click "Trigger DAG" to run the training pipeline.
4. Watch the pipeline execute:
   - Check S3 data ✓
   - Verify API health ✓
   - Trigger training ✓
   - Promote best model ✓
   - Test predictions ✓


### Monitor Metrics & Test Prediction
- Open MLflow UI (URL from deployment output) to view experiment metrics and model registry.
- Use FastAPI docs (http://<your-aws-api-ip>:8000/docs) to test prediction endpoint in production.


### Terminate EC2 & Clean Up

Run the cleanup script:
```bash 
scripts/cleanup_aws.sh

### Verify resources are terminated (EC2, S3, etc.).
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=fraud-detection" \
  --query 'Reservations[].Instances[].State.Name'
  ```