# AWS Deployment Guide for Fraud Detection Project

## Overview

This guide walks you through deploying your fraud detection ML system to AWS for portfolio demonstrations. The setup costs **~$1-2/month** with the free tier.

## Prerequisites

### 1. AWS Account Setup

```bash
# Install AWS CLI
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Windows
# Download from: https://awscli.amazonaws.com/AWSCLIV2.msi
```

### 2. Configure AWS Credentials

```bash
# Run configuration
aws configure

# Enter your credentials:
AWS Access Key ID: YOUR_ACCESS_KEY
AWS Secret Access Key: YOUR_SECRET_KEY
Default region name: eu-central-1
Default output format: json
```

### 3. Create EC2 Key Pair

```bash
# Create key pair
aws ec2 create-key-pair \
  --key-name fraud-detection-key \
  --query 'KeyMaterial' \
  --output text > fraud-detection-key.pem

# Set permissions
chmod 400 fraud-detection-key.pem

# Save this file securely!
```

### 4. Create IAM Role for EC2

Go to AWS Console → IAM → Roles → Create Role:

1. Select **EC2** as trusted entity
2. Attach policies:
   - `AmazonS3FullAccess`
   - `CloudWatchLogsFullAccess`
3. Name: `fraud-detection-ec2-role`
4. Create role

## Project Structure Updates

Your project should now have these files:

```
fraud-detection/
├── scripts/
│   ├── api.py                    # NEW: Complete FastAPI service
│   ├── train.py                  # Your existing training script
│   ├── promote.py                # Your existing promotion script
│   ├── deploy_to_aws.sh          # NEW: Deployment script
│   └── cleanup_aws.sh            # NEW: Cleanup script
├── dags/
│   └── fraud_detection_dag_aws.py  # NEW: AWS-compatible DAG
├── src/
│   ├── data_utils.py
│   ├── model_pipeline.py
│   └── metrics.py
├── .github/workflows/
│   └── deploy_aws.yml            # NEW: CI/CD for AWS
├── Dockerfile.production         # NEW: Production dockerfile
├── requirements.txt
└── transactions.csv
```

## Deployment Steps

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

### Step 3: Build and Push Docker Image

```bash
# Build production image
docker build -t YOUR_USERNAME/fraud-detection-model:latest -f Dockerfile.production .

# Test locally
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://localhost:5001 \
  YOUR_USERNAME/fraud-detection-model:latest

# Push to Docker Hub
docker login
docker push YOUR_USERNAME/fraud-detection-model:latest
```

### Step 4: Deploy to AWS

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

### Step 5: Upload Training Data

```bash
# Upload your dataset to S3
aws s3 cp transactions.csv s3://fraud-detection-YOURNAME-data/raw/
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

#### Option A: Via Airflow (Recommended)

1. Go to Airflow UI
2. Find `fraud_detection_aws_pipeline`
3. Click "Trigger DAG"
4. Watch the pipeline execute:
   - Check S3 data ✓
   - Verify API health ✓
   - Trigger training ✓
   - Promote best model ✓
   - Test predictions ✓

#### Option B: Via API Directly

```bash
# Get your API IP from deployment-info.txt
API_IP="YOUR_API_IP"

# 1. Check health
curl http://$API_IP:8000/health

# 2. Trigger training
curl -X POST http://$API_IP:8000/train \
  -H "Content-Type: application/json" \
  -d '{"data_path": "transactions.csv", "mode": "all"}'

# 3. Wait for training to complete (check logs)
# Training takes ~5-10 minutes

# 4. List models
curl http://$API_IP:8000/models/list

# 5. Test prediction
curl -X POST http://$API_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Step 9: Take Screenshots for Portfolio

Capture these for your portfolio:

1. **Airflow DAG Graph View**
   - Shows your pipeline structure
   
2. **Airflow Task Logs**
   - Shows successful execution
   
3. **MLflow UI**
   - Model experiments
   - Model versions
   - Metrics comparison
   
4. **FastAPI Swagger Docs**
   - http://YOUR_API_IP:8000/docs
   - Shows all endpoints
   
5. **Successful Prediction**
   - API response with fraud probability
   
6. **S3 Buckets**
   - Show model artifacts stored
   
7. **Architecture Diagram**
   - Create in draw.io or similar

### Step 10: Cleanup (After 2-3 Hours)

```bash
# Terminate EC2 instances
./scripts/cleanup_aws.sh

# Verify cleanup
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=fraud-detection" \
  --query 'Reservations[].Instances[].State.Name'

# Should show: "terminated"
```

## Cost Breakdown

### Per 3-Hour Session

- EC2 t2.micro (API): 3 hrs × $0.0116/hr = **$0.03**
- EC2 t2.small (Airflow): 3 hrs × $0.023/hr = **$0.07**
- S3 storage: **$0.02/month**
- Data transfer: **<$0.01**

**Total per session: ~$0.12**

### Monthly (10 Sessions)

- 10 sessions × $0.12 = **$1.20**
- S3 storage: **$0.50**

**Total: ~$1.70/month**

### Free Tier Benefits

- EC2: 30 hrs/month out of 750 hrs ✅
- S3: ~2GB out of 5GB ✅
- All within free tier limits ✅

## Troubleshooting

### Issue: Can't connect to EC2

```bash
# Check instance status
aws ec2 describe-instances --instance-ids YOUR_INSTANCE_ID

# Check security group
aws ec2 describe-security-groups --group-ids YOUR_SG_ID

# Verify your IP can access
curl -v http://YOUR_API_IP:8000/health
```

### Issue: Docker container not starting

```bash
# SSH to instance
ssh -i fraud-detection-key.pem ec2-user@YOUR_API_IP

# Check Docker status
docker ps -a

# Check logs
docker logs fraud-api

# Restart container
docker restart fraud-api
```

### Issue: Training fails

```bash
# Check API logs
ssh -i fraud-detection-key.pem ec2-user@YOUR_API_IP
docker logs -f fraud-api

# Check if data exists in S3
aws s3 ls s3://fraud-detection-YOURNAME-data/raw/

# Verify MLflow is accessible
curl http://localhost:5001
```

### Issue: Airflow DAG not running

```bash
# SSH to Airflow instance
ssh -i fraud-detection-key.pem ec2-user@YOUR_AIRFLOW_IP

# Check services
docker-compose ps

# Check logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler

# Restart services
docker-compose restart
```

## GitHub Actions Integration

Your CI/CD pipeline automatically:

1. **Tests** code on every push
2. **Builds** Docker image
3. **Pushes** to Docker Hub
4. **Creates** deployment issue with instructions

To deploy new changes:

```bash
# 1. Make changes locally
git add .
git commit -m "Update model"
git push origin main

# 2. GitHub Actions runs automatically

# 3. Check GitHub Actions tab for build status

# 4. When complete, SSH to EC2 and update:
ssh -i fraud-detection-key.pem ec2-user@YOUR_API_IP

docker pull YOUR_USERNAME/fraud-detection-model:latest
docker stop fraud-api && docker rm fraud-api
docker run -d --name fraud-api -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://localhost:5001 \
  -e AWS_DEFAULT_REGION=us-east-1 \
  YOUR_USERNAME/fraud-detection-model:latest
```

## Best Practices

### Security

1. **Never commit AWS credentials** to GitHub
2. **Use IAM roles** for EC2 instances
3. **Restrict security groups** to your IP if possible
4. **Rotate credentials** regularly

### Cost Management

1. **Set billing alerts** at $5, $10, $15
2. **Always run cleanup** after demos
3. **Use free tier calculator** before deploying
4. **Tag all resources** for tracking

### Portfolio Presentation

1. **Document architecture** clearly
2. **Explain trade-offs** (e.g., why EC2 vs Lambda)
3. **Show monitoring** (CloudWatch logs)
4. **Discuss scalability** considerations
5. **Highlight cost optimization**

## Next Steps

Once comfortable with basic deployment:

1. **Add CI/CD**: Automate deployment from GitHub Actions
2. **Add monitoring**: CloudWatch dashboards
3. **Add alerts**: Email notifications for pipeline failures
4. **Add A/B testing**: Deploy multiple model versions
5. **Add data drift detection**: Monitor input distributions

## Support

If you encounter issues:

1. Check deployment-info.txt for details
2. Review AWS CloudWatch logs
3. SSH to instances and check Docker logs
4. Verify security groups allow traffic
5. Ensure IAM role has correct permissions

## Conclusion

You now have a production-like ML system running on AWS that demonstrates:

- ✅ **MLOps pipeline** orchestration with Airflow
- ✅ **Model training** with XGBoost
- ✅ **Experiment tracking** with MLflow
- ✅ **Model serving** with FastAPI
- ✅ **Cloud deployment** on AWS
- ✅ **CI/CD** with GitHub Actions
- ✅ **Cost optimization** (<$2/month)

This is a **complete, portfolio-ready project** that shows employers you can:

- Build end-to-end ML systems
- Deploy to cloud infrastructure
- Automate workflows
- Manage costs effectively
- Follow MLOps best practices

Happy deploying! 🚀