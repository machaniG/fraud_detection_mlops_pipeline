# Complete Explanation of Changes

## Why These Changes Were Needed

Your original code was configured for **Kubernetes deployment** but we're deploying to **AWS EC2** instead. Here's what changed and why:

---

## 1. Airflow DAG Changes

**File**: `dags/fraud_detection_dag_aws.py` (NEW)

### Original Problem
```python
# Your original DAG used KubernetesPodOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

# This requires a Kubernetes cluster running
full_training_and_hpo = KubernetesPodOperator(
    task_id="full_training_and_hpo",
    namespace="airflow",  # K8s namespace
    image=DOCKER_IMAGE,
    # ... K8s-specific config
)
```

**Why it's a problem**: Kubernetes clusters are expensive (~$50-100/month minimum) and complex to set up.

### New Solution
```python
# Uses simple API calls instead
@task
def trigger_training():
    import requests
    response = requests.post(f"{MODEL_API_URL}/train", json=config)
    return response.json()
```

**Benefits**:
- ✅ No Kubernetes needed
- ✅ Works with simple EC2 instances
- ✅ Much cheaper (~$1-2/month)
- ✅ Easier to understand and debug

### What the new DAG does:

1. **Checks S3 for data** - Verifies training data exists
2. **Checks API health** - Ensures FastAPI is running
3. **Triggers training** - Calls your `/train` endpoint
4. **Waits for completion** - Gives time for training
5. **Verifies model** - Checks MLflow registry
6. **Promotes model** - Moves best model to Staging
7. **Tests predictions** - Validates the deployed model

---

## 2. Complete FastAPI Service

**File**: `scripts/api.py` (NEW - replaces your serve.py)

### Why You Need This

Your `serve.py` only had prediction endpoints. We need endpoints for:
- Training models
- Promoting models
- Checking model status
- Making predictions

### Key Endpoints Added

```python
@app.post("/train")  # NEW
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Triggers your train.py script in the background
    Returns immediately so Airflow doesn't timeout
    """
    background_tasks.add_task(run_training_subprocess, command)
    return {"status": "training_started"}
```

**Why background tasks?**: Training takes 5-10 minutes. Without background tasks, the API request would timeout.

```python
@app.post("/promote")  # NEW
async def promote_model(request: PromotionRequest):
    """
    Calls your promote.py script to find and promote best model
    """
    result = subprocess.run([
        "python", "scripts/promote.py",
        "--parent_run_id", request.parent_run_id
    ])
    return {"status": "promotion_successful"}
```

```python
@app.get("/models/list")  # NEW
async def list_models():
    """
    Lists all model versions in MLflow registry
    Useful for verification and debugging
    """
    versions = client.get_latest_versions(MODEL_NAME)
    return {"versions": versions}
```

### S3 Integration

```python
def download_data_from_s3(data_path: str):
    """
    Downloads training data from S3
    Falls back to local file if S3 unavailable
    """
    s3_client.download_file(
        S3_BUCKET_DATA,
        f"raw/{data_path}",
        local_path
    )
```

**Why S3?**: AWS best practice is to store data in S3, not on EC2 instances. Your data persists even if you terminate instances.

---

## 3. Production Dockerfile

**File**: `Dockerfile.production` (NEW)

### Changes from Original

```dockerfile
# OLD: Just runs training
CMD ["python", "scripts/train.py"]

# NEW: Runs complete API server
CMD ["python", "scripts/api.py"]
```

**Why?**: Your EC2 instance needs to run the API server, not just training. Training is triggered via API calls.

```dockerfile
# NEW: Health check for AWS
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

**Why?**: AWS can automatically restart unhealthy containers.

---

## 4. AWS Deployment Script

**File**: `scripts/deploy_to_aws.sh` (NEW)

### What It Does

**Creates S3 Buckets**:
```bash
aws s3 mb s3://fraud-detection-yourname-data
aws s3 mb s3://fraud-detection-yourname-models
aws s3 mb s3://fraud-detection-yourname-mlflow
```
- Stores training data
- Stores trained models
- Stores MLflow artifacts

**Creates Security Group**:
```bash
aws ec2 create-security-group --group-name fraud-detection-sg
# Allows traffic on ports: 22 (SSH), 8000 (FastAPI), 8080 (Airflow), 5001 (MLflow)
```
- Controls what can access your EC2 instances
- Like a firewall

**Launches API Instance** (t2.micro - free tier):
```bash
aws ec2 run-instances \
  --instance-type t2.micro \
  --image-id ami-0c55b159cbfafe1f0 \
  --user-data "install docker, pull your image, run container"
```
- Automatically installs Docker
- Pulls your image from Docker Hub
- Starts FastAPI service

**Launches Airflow Instance** (t2.small):
```bash
aws ec2 run-instances \
  --instance-type t2.small \
  --user-data "install docker-compose, start airflow + mlflow"
```
- Runs Airflow for orchestration
- Runs MLflow for experiment tracking
- Uses docker-compose for easy management

**Outputs Access Info**:
```
FastAPI Docs: http://YOUR_IP:8000/docs
Airflow UI: http://YOUR_IP:8080
MLflow UI: http://YOUR_IP:5001
```

---

## 5. Cleanup Script

**File**: `scripts/cleanup_aws.sh` (NEW)

### What It Does

```bash
# Terminates EC2 instances
aws ec2 terminate-instances --instance-ids $API_INSTANCE_ID $AIRFLOW_INSTANCE_ID

# Waits for confirmation
aws ec2 wait instance-terminated --instance-ids ...

# Keeps S3 buckets (they're free under 5GB)
```

**Why terminate instances?**: After your demo session, you don't need them running. This stops the charges.

**Why keep S3?**: S3 is very cheap (<$1/month for your usage) and contains your models/data for next session.

---

## 6. GitHub Actions Workflow

**File**: `.github/workflows/deploy_aws.yml` (NEW)

### What It Automates

**On every push to main**:
1. Runs tests
2. Builds Docker image
3. Pushes to Docker Hub
4. Uploads models to S3 (if configured)
5. Creates a GitHub issue with deployment instructions

**Why?**: 
- Ensures code quality (tests pass)
- Always have latest image ready
- Reminds you to deploy

---

## 7. Fixes to Your Existing Code

### promote.py Fix

**Original Problem**:
```python
# This OR filter doesn't work in MLflow
filter_string = f"tags.\"mlflow.parentRunId\" = '{parent_run_id}' OR run_id = '{parent_run_id}'"
```

**Fixed**:
```python
# Get parent run first
parent_run = client.get_run(parent_run_id)

# Get all child runs
nested_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.\"mlflow.parentRunId\" = '{parent_run_id}'"
)

# Combine and find best
runs_to_evaluate = [parent_run] + nested_runs
```

**Why it matters**: Your promotion script couldn't find the best model. Now it can.

---

## Complete Workflow

Here's how everything works together:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOCAL DEVELOPMENT                                             │
│    - Write code                                                  │
│    - Test locally                                                │
│    - Push to GitHub                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. GITHUB ACTIONS (Automated)                                    │
│    - Runs tests                                                  │
│    - Builds Docker image                                         │
│    - Pushes to Docker Hub                                        │
│    - Creates deployment reminder                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. MANUAL DEPLOYMENT (When you're ready for demo)               │
│    - Run: ./scripts/deploy_to_aws.sh                            │
│    - Wait 5-10 minutes                                           │
│    - Instances are ready                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. AWS INFRASTRUCTURE                                            │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ EC2: FastAPI     │         │ EC2: Airflow +   │             │
│  │ - Serves model   │◀────────│      MLflow      │             │
│  │ - Trains models  │  calls  │ - Orchestrates   │             │
│  │ - Promotes       │         │ - Tracks         │             │
│  └────────┬─────────┘         └──────────────────┘             │
│           │                                                      │
│           │ reads/writes                                         │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ S3 Buckets       │                                           │
│  │ - Data           │                                           │
│  │ - Models         │                                           │
│  │ - Artifacts      │                                           │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. YOUR DEMO SESSION (2-3 hours)                                │
│    - Access Airflow UI                                          │
│    - Trigger training pipeline                                  │
│    - Watch execution                                            │
│    - Test predictions                                           │
│    - Take screenshots                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. CLEANUP                                                       │
│    - Run: ./scripts/cleanup_aws.sh                              │
│    - Terminates EC2 instances                                   │
│    - S3 buckets remain (models preserved)                       │
│    - Cost: ~$0.12 for this session                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## What You Need to Do

### Before First Deployment

1. **Update Docker Hub username** in all files:
   - `scripts/deploy_to_aws.sh`: Line 19
   - `.github/workflows/deploy_aws.yml`: Line 11

2. **Create AWS resources**:
   ```bash
   # Create key pair
   aws ec2 create-key-pair --key-name fraud-detection-key \
     --query 'KeyMaterial' --output text > fraud-detection-key.pem
   chmod 400 fraud-detection-key.pem
   
   # Create IAM role (via AWS Console)
   # Name: fraud-detection-ec2-role
   # Policies: AmazonS3FullAccess, CloudWatchLogsFullAccess
   ```

3. **Make scripts executable**:
   ```bash
   chmod +x scripts/deploy_to_aws.sh
   chmod +x scripts/cleanup_aws.sh
   ```

4. **Update S3 bucket names** to be unique:
   - In `scripts/deploy_to_aws.sh`
   - In `scripts/api.py`
   - In `dags/fraud_detection_dag_aws.py`

### For Each Demo Session

1. **Deploy** (~5 minutes):
   ```bash
   ./scripts/deploy_to_aws.sh
   ```

2. **Wait** for services to start (~5-10 minutes)

3. **Upload data**:
   ```bash
   aws s3 cp transactions.csv s3://your-bucket-name/raw/
   ```

4. **Access Airflow**: Update DAG with API IP, trigger pipeline

5. **Take screenshots**: Airflow, MLflow, API docs, predictions

6. **Cleanup** after 2-3 hours:
   ```bash
   ./scripts/cleanup_aws.sh
   ```

---

## Key Concepts Explained

### Why Airflow?
- **Orchestrates** the entire ML pipeline
- **Schedules** retraining automatically
- **Monitors** task execution
- **Retries** failed tasks
- Shows you understand production ML

### Why MLflow?
- **Tracks** all experiments (hyperparameters, metrics)
- **Versions** models
- **Manages** model lifecycle (Staging, Production)
- **Compares** different model versions
- Industry standard for ML experiment tracking

### Why FastAPI?
- **Fast** and modern Python web framework
- **Automatic** API documentation (Swagger UI)
- **Type validation** with Pydantic
- **Async** support for better performance
- Perfect for serving ML models

### Why Docker?
- **Consistent** environment (works everywhere)
- **Reproducible** deployments
- **Isolates** dependencies
- **Easy** to scale
- Industry standard for deployment

### Why S3?
- **Cheap** storage (<$0.50/month for your usage)
- **Durable** (99.999999999% durability)
- **Scalable** (store as much as needed)
- **Accessible** from anywhere
- AWS native storage solution

---

## Common Questions

### Q: Why not deploy everything in one EC2 instance?

**A**: We could, but separating concerns is better:
- API instance can be t2.micro (free tier)
- Airflow+MLflow needs t2.small (more memory)
- Shows you understand microservices architecture
- If API crashes, Airflow still runs

### Q: Why not use AWS Lambda?

**A**: Lambda is great for simple tasks, but:
- Training takes >15 minutes (Lambda limit: 15 min)
- Model files might exceed Lambda size limits
- EC2 gives you more control for learning
- Shows you can work with both serverless and servers

### Q: Why not use ECS/EKS (container orchestration)?

**A**: They're better for production, but:
- More expensive (~$50-100/month minimum)
- More complex to learn
- Overkill for a portfolio project
- EC2 is easier to understand and debug

### Q: Do I really need all this for a portfolio project?

**A**: This setup demonstrates:
- ✅ End-to-end ML pipeline
- ✅ Cloud deployment skills
- ✅ MLOps best practices
- ✅ Cost optimization
- ✅ Production-ready thinking

Most candidates show Jupyter notebooks. You'll show a **complete system**.

---

## Estimated Time Investment

- **Initial setup**: 2-3 hours
  - AWS account configuration
  - First deployment
  - Understanding the system

- **Each demo session**: 2-3 hours
  - Deploy: 10 minutes
  - Run pipeline: 30 minutes
  - Screenshots: 30 minutes
  - Testing: 30 minutes
  - Cleanup: 5 minutes
  - Buffer time: 45-75 minutes

- **Documentation**: 2-4 hours
  - README with architecture
  - Screenshots with explanations
  - Video walkthrough (optional)

**Total**: ~10-15 hours for a complete, portfolio-ready project

---

## Final Checklist

- [ ] AWS CLI installed and configured
- [ ] EC2 key pair created
- [ ] IAM role created
- [ ] Docker Hub account created
- [ ] All scripts made executable
- [ ] Configuration variables updated
- [ ] First deployment successful
- [ ] Training pipeline runs
- [ ] Predictions work
- [ ] Screenshots captured
- [ ] Documentation written
- [ ] GitHub repository updated
- [ ] Resume/LinkedIn updated

---

## What Makes This Project Stand Out

1. **Complete System**: Not just a model, but a full ML platform
2. **Production Practices**: Uses industry-standard tools
3. **Cost Conscious**: Optimized for minimal cost
4. **Well Documented**: Clear architecture and setup
5. **Automated**: CI/CD pipeline included
6. **Cloud Native**: AWS deployment experience
7. **MLOps Focus**: Airflow, MLflow, model versioning
8. **API First**: RESTful API for model serving

This demonstrates skills that most data scientists don't have, making you more attractive for **ML Engineer** and **MLOps Engineer** roles.

---

Good luck with your project! 🚀