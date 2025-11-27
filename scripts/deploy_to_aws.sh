#!/bin/bash

###############################################################################
# AWS Deployment Script for Fraud Detection Project
# 
# This script deploys your fraud detection system to AWS EC2
# for 2-3 hour demo sessions with minimal cost.
#
# What it does:
# 1. Creates S3 buckets for data and models
# 2. Launches EC2 instance with FastAPI
# 3. Launches EC2 instance with Airflow + MLflow
# 4. Configures security groups
# 5. Deploys Docker containers
#
# Cost per session: ~$0.15
###############################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Fraud Detection AWS Deployment"
echo "=========================================="
echo ""

# Configuration - Update these values
REGION="eu-central-1"  # AWS region
KEY_NAME="fraud-detection-key"  # Your EC2 key pair name
SECURITY_GROUP_NAME="fraud-detection-sg"
IAM_ROLE="fraud-detection-ec2-role"
DOCKER_IMAGE="frida33876/fraud-detection-model:latest"  # Your Docker Hub image
S3_BUCKET_DATA="fraud-detection-frida-data"  # Change 'yourname' to something unique
S3_BUCKET_MODELS="fraud-detection-frida-models"
S3_BUCKET_MLFLOW="fraud-detection-frida-mlflow"

# AMI ID for Amazon Linux 2 (update if needed)
AMI_ID="ami-0c55b159cbfafe1f0"  # Amazon Linux 2 in us-east-1

echo -e "${YELLOW}Configuration:${NC}"
echo "  Region: $REGION"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  S3 Buckets: $S3_BUCKET_DATA, $S3_BUCKET_MODELS"
echo ""

# Function to check if AWS CLI is installed
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI not installed${NC}"
        echo "Install it from: https://aws.amazon.com/cli/"
        exit 1
    fi
    echo -e "${GREEN}✓ AWS CLI found${NC}"
}

# Function to check AWS credentials
check_aws_credentials() {
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}Error: AWS credentials not configured${NC}"
        echo "Run: aws configure"
        exit 1
    fi
    echo -e "${GREEN}✓ AWS credentials configured${NC}"
}

# Function to create S3 buckets
create_s3_buckets() {
    echo -e "${YELLOW}Creating S3 buckets...${NC}"
    
    for bucket in "$S3_BUCKET_DATA" "$S3_BUCKET_MODELS" "$S3_BUCKET_MLFLOW"; do
        if aws s3 ls "s3://$bucket" 2>&1 | grep -q 'NoSuchBucket'; then
            aws s3 mb "s3://$bucket" --region $REGION
            echo -e "${GREEN}✓ Created bucket: $bucket${NC}"
        else
            echo -e "${GREEN}✓ Bucket exists: $bucket${NC}"
        fi
    done
    
    # Create folder structure in data bucket
    aws s3api put-object --bucket $S3_BUCKET_DATA --key raw/ || true
    aws s3api put-object --bucket $S3_BUCKET_DATA --key processed/ || true
    
    # Create folder structure in models bucket
    aws s3api put-object --bucket $S3_BUCKET_MODELS --key v1/ || true
    aws s3api put-object --bucket $S3_BUCKET_MODELS --key v2/ || true
    aws s3api put-object --bucket $S3_BUCKET_MODELS --key production/ || true
    
    echo -e "${GREEN}✓ S3 buckets configured${NC}"
}

# Function to create security group
create_security_group() {
    echo -e "${YELLOW}Creating security group...${NC}"
    
    # Check if security group exists
    if aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --region $REGION &> /dev/null; then
        echo -e "${GREEN}✓ Security group exists${NC}"
        SG_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --region $REGION --query 'SecurityGroups[0].GroupId' --output text)
    else
        # Create security group
        SG_ID=$(aws ec2 create-security-group \
            --group-name $SECURITY_GROUP_NAME \
            --description "Security group for fraud detection demo" \
            --region $REGION \
            --query 'GroupId' \
            --output text)
        
        # Allow SSH (port 22)
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        # Allow FastAPI (port 8000)
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 8000 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        # Allow Airflow UI (port 8080)
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 8080 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        # Allow MLflow UI (port 5001)
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 5001 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        echo -e "${GREEN}✓ Security group created: $SG_ID${NC}"
    fi
    
    export SG_ID
}

# Function to launch FastAPI EC2 instance
launch_api_instance() {
    echo -e "${YELLOW}Launching FastAPI instance (t2.micro)...${NC}"
    
    # User data script
    cat > /tmp/api-userdata.sh << EOF
#!/bin/bash
# Update system
yum update -y

# Install Docker
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Install AWS CLI (already installed on Amazon Linux 2)
# Configure AWS region
mkdir -p /home/ec2-user/.aws
cat > /home/ec2-user/.aws/config << AWSCONFIG
[default]
region = $REGION
AWSCONFIG

# Pull and run Docker container
docker pull $DOCKER_IMAGE

# Run FastAPI container
docker run -d --name fraud-api \\
  --restart unless-stopped \\
  -p 8000:8000 \\
  -e MLFLOW_TRACKING_URI=http://localhost:5001 \\
  -e S3_BUCKET_DATA=$S3_BUCKET_DATA \\
  -e S3_BUCKET_MODELS=$S3_BUCKET_MODELS \\
  -e AWS_DEFAULT_REGION=$REGION \\
  $DOCKER_IMAGE

# Wait for container to start
sleep 10

# Check health
curl -f http://localhost:8000/health || echo "API not ready yet"

echo "FastAPI deployed successfully" > /home/ec2-user/deployment-status.txt
EOF
    
    # Launch instance
    API_INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type t2.micro \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --iam-instance-profile Name=$IAM_ROLE \
        --user-data file:///tmp/api-userdata.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fraud-api},{Key=Project,Value=fraud-detection},{Key=Type,Value=api}]" \
        --region $REGION \
        --query 'Instances[0].InstanceId' \
        --output text \
        2>/dev/null || echo "")
    
    if [ -z "$API_INSTANCE_ID" ]; then
        echo -e "${RED}Error: Could not launch API instance${NC}"
        echo "Check that your key pair '$KEY_NAME' exists"
        echo "Check that IAM role '$IAM_ROLE' exists"
        exit 1
    fi
    
    echo -e "${GREEN}✓ API instance launched: $API_INSTANCE_ID${NC}"
    export API_INSTANCE_ID
}

# Function to launch Airflow+MLflow EC2 instance
launch_airflow_instance() {
    echo -e "${YELLOW}Launching Airflow+MLflow instance (t2.small)...${NC}"
    
    # User data script
    cat > /tmp/airflow-userdata.sh << EOF
#!/bin/bash
# Update system
yum update -y

# Install Docker and docker-compose
yum install -y docker git
service docker start
usermod -a -G docker ec2-user

# Install docker-compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /home/ec2-user/fraud-detection
cd /home/ec2-user/fraud-detection

# Create docker-compose file for Airflow + MLflow
cat > docker-compose.yml << 'COMPOSE_EOF'
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5001:5001"
    command: mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts
    volumes:
      - mlflow-data:/mlflow
    restart: unless-stopped

  airflow-init:
    image: apache/airflow:2.7.3-python3.10
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    command: >
      bash -c "airflow db init &&
               airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin"
    volumes:
      - ./dags:/opt/airflow/dags

  airflow-webserver:
    image: apache/airflow:2.7.3-python3.10
    depends_on:
      - postgres
      - airflow-init
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    ports:
      - "8080:8080"
    command: airflow webserver
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
    restart: unless-stopped

  airflow-scheduler:
    image: apache/airflow:2.7.3-python3.10
    depends_on:
      - postgres
      - airflow-init
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    command: airflow scheduler
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
    restart: unless-stopped

volumes:
  postgres-data:
  mlflow-data:
COMPOSE_EOF

# Create DAGs directory
mkdir -p dags logs

# Start services
docker-compose up -d

# Wait for services
sleep 30

echo "Airflow+MLflow deployed successfully" > /home/ec2-user/deployment-status.txt
EOF
    
    # Launch instance
    AIRFLOW_INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type t2.small \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --iam-instance-profile Name=$IAM_ROLE \
        --user-data file:///tmp/airflow-userdata.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=fraud-airflow},{Key=Project,Value=fraud-detection},{Key=Type,Value=orchestration}]" \
        --region $REGION \
        --query 'Instances[0].InstanceId' \
        --output text \
        2>/dev/null || echo "")
    
    if [ -z "$AIRFLOW_INSTANCE_ID" ]; then
        echo -e "${RED}Error: Could not launch Airflow instance${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Airflow instance launched: $AIRFLOW_INSTANCE_ID${NC}"
    export AIRFLOW_INSTANCE_ID
}

# Function to wait for instances
wait_for_instances() {
    echo -e "${YELLOW}Waiting for instances to be running...${NC}"
    
    aws ec2 wait instance-running \
        --instance-ids $API_INSTANCE_ID $AIRFLOW_INSTANCE_ID \
        --region $REGION
    
    echo -e "${GREEN}✓ Instances are running${NC}"
    
    # Get public IPs
    API_IP=$(aws ec2 describe-instances \
        --instance-ids $API_INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    AIRFLOW_IP=$(aws ec2 describe-instances \
        --instance-ids $AIRFLOW_INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    export API_IP AIRFLOW_IP
}

# Function to save deployment info
save_deployment_info() {
    cat > deployment-info.txt << INFO
========================================
Fraud Detection AWS Deployment Info
========================================
Deployed: $(date)

EC2 Instances:
--------------
API Instance ID: $API_INSTANCE_ID
API Public IP: $API_IP

Airflow Instance ID: $AIRFLOW_INSTANCE_ID
Airflow Public IP: $AIRFLOW_IP

Security Group: $SG_ID

S3 Buckets:
-----------
Data: s3://$S3_BUCKET_DATA
Models: s3://$S3_BUCKET_MODELS
MLflow: s3://$S3_BUCKET_MLFLOW

Access URLs:
------------
FastAPI Docs: http://$API_IP:8000/docs
FastAPI Health: http://$API_IP:8000/health
Airflow UI: http://$AIRFLOW_IP:8080 (admin/admin)
MLflow UI: http://$AIRFLOW_IP:5001

Next Steps:
-----------
1. Wait 5-10 minutes for services to fully start
2. Upload your dataset:
   aws s3 cp transactions.csv s3://$S3_BUCKET_DATA/raw/

3. Test the API:
   curl http://$API_IP:8000/health

4. Access Airflow and update DAG:
   - Update MODEL_API_URL to: http://$API_IP:8000
   - Trigger the DAG manually

IMPORTANT:
----------
Remember to run cleanup script in 2-3 hours!
./scripts/cleanup_aws.sh

Estimated cost: \$0.12 per 3-hour session
INFO

    echo ""
    echo "==========================================" 
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo "=========================================="
    echo ""
    cat deployment-info.txt
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: Run cleanup script in 2-3 hours${NC}"
    echo ""
}

# Main execution
main() {
    echo "Starting deployment..."
    echo ""
    
    check_aws_cli
    check_aws_credentials
    create_s3_buckets
    create_security_group
    launch_api_instance
    launch_airflow_instance
    wait_for_instances
    save_deployment_info
    
    echo -e "${GREEN}✓ Deployment script complete${NC}"
}

# Run main function
main