
# AWS Deployment Guide: Pushing from Docker Hub to ECR

This document outlines the steps required to get the container image from **Docker Hub** into **Amazon Elastic Container Registry (ECR)**, which is a prerequisite for deployment on AWS services like Elastic Container Service (ECS) or Elastic Kubernetes Service (EKS).

## Prerequisites

1.  **AWS Account and IAM User:** You must have an AWS account and an IAM user configured with permissions to manage ECR.
2.  **AWS CLI:** The AWS Command Line Interface must be installed and configured (`aws configure`).
3.  **Docker:** The Docker daemon must be running locally.

## Step 1: Create an ECR Repository

Use the AWS CLI to create a new private repository in your target AWS region.

```bash
# Replace <REGION> with your AWS region (e.g., us-east-1)
# Replace <ECR_REPO_NAME> with a name like 'fraud-detection-model'
aws ecr create-repository \
    --repository-name <ECR_REPO_NAME> \
    --region <REGION>
```

This command will return the repository URI (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-model).

## Step 2: Authenticate Docker to AWS ECR

You need to retrieve a temporary authentication token from ECR and pipe it to the Docker client.

```bash
# Replace <REGION> and <AWS_ACCOUNT_ID>
# This command authenticates your Docker client for 12 hours.
aws ecr get-login-password --region <REGION> | \
    docker login \
    --username AWS \
    --password-stdin \
    <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
```

## Step 3: Pull Image from Docker Hub (Optional)

If your local image was deleted, you can pull it back from Docker Hub using the name you pushed in the previous step:

```Bash
docker pull <YOUR_DOCKERHUB_USERNAME>/fraud-detection-model:latest
```

## Step 4: Tag and Push Image to ECR

Now, retag the image so its name matches the destination ECR repository URI, and then push it.

```bash
# 1. Tag the image for ECR
# Replace <ECR_URI> with the full URI from Step 1.
# Example: [123456789012.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-model](https://123456789012.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-model)
docker tag <YOUR_DOCKERHUB_USERNAME>/fraud-detection-model:latest \
    <ECR_URI>:latest

# 2. Push the newly tagged image to ECR
docker push <ECR_URI>:latest
```

## Step 5: AWS Deployment (ECS/EKS)
Once the image is in ECR, it is ready for deployment:

ECS/EKS: Use the ECR image URI (<ECR_URI>:latest) in your Task Definition (for ECS) or your Kubernetes Deployment Manifest (for EKS) to launch the model inference service.

AWS Lambda/SageMaker: The image can also be used as the base for a serverless ML prediction endpoint using AWS Lambda or within an Amazon SageMaker Endpoint.