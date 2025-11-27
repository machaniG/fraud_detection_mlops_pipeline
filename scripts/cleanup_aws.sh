#!/bin/bash

###############################################################################
# AWS Cleanup Script
#
# Run this after your 2-3 hour demo session to terminate EC2 instances
# and avoid charges. S3 buckets are kept (free tier covers storage).
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "AWS Cleanup Script"
echo "=========================================="
echo ""

REGION="us-east-1"

# Try to load deployment info
if [ -f "deployment-info.txt" ]; then
    echo -e "${GREEN}Loading deployment info...${NC}"
    
    API_INSTANCE_ID=$(grep "API Instance ID:" deployment-info.txt | awk '{print $4}')
    AIRFLOW_INSTANCE_ID=$(grep "Airflow Instance ID:" deployment-info.txt | awk '{print $4}')
    
    echo "API Instance: $API_INSTANCE_ID"
    echo "Airflow Instance: $AIRFLOW_INSTANCE_ID"
else
    echo -e "${YELLOW}deployment-info.txt not found${NC}"
    echo "Please enter instance IDs manually:"
    read -p "API Instance ID: " API_INSTANCE_ID
    read -p "Airflow Instance ID: " AIRFLOW_INSTANCE_ID
fi

echo ""
echo -e "${YELLOW}This will terminate the following instances:${NC}"
echo "  - API: $API_INSTANCE_ID"
echo "  - Airflow: $AIRFLOW_INSTANCE_ID"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo -e "${YELLOW}Terminating instances...${NC}"

# Terminate instances
aws ec2 terminate-instances \
    --instance-ids $API_INSTANCE_ID $AIRFLOW_INSTANCE_ID \
    --region $REGION

echo -e "${GREEN}✓ Instances termination initiated${NC}"

echo ""
echo -e "${YELLOW}Waiting for termination to complete...${NC}"

aws ec2 wait instance-terminated \
    --instance-ids $API_INSTANCE_ID $AIRFLOW_INSTANCE_ID \
    --region $REGION

echo -e "${GREEN}✓ Instances terminated successfully${NC}"

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "S3 buckets are still active (free tier):"
echo "  - fraud-detection-yourname-data"
echo "  - fraud-detection-yourname-models"
echo "  - fraud-detection-yourname-mlflow"
echo ""
echo "To check your AWS usage:"
echo "  aws ec2 describe-instances --region $REGION"
echo ""
echo "To delete S3 buckets (if needed):"
echo "  aws s3 rb s3://fraud-detection-yourname-data --force"
echo "  aws s3 rb s3://fraud-detection-yourname-models --force"
echo "  aws s3 rb s3://fraud-detection-yourname-mlflow --force"
echo ""
echo -e "${GREEN}✓ Cleanup complete - no charges should accrue${NC}"