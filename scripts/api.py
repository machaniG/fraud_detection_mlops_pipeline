#!/usr/bin/env python3
"""
Complete FastAPI service for fraud detection model.

Endpoints:
- GET /health: Health check
- POST /predict: Make fraud predictions
- POST /train: Trigger model training
- POST /promote: Promote model to staging
- GET /models/list: List available models
- GET /metrics: Get model performance metrics

This combines serving (serve.py) with training triggers for a complete API.
"""

import os
import sys
import logging
import subprocess
from typing import Optional, Dict, Any
from datetime import datetime
import json

import uvicorn
import mlflow
import pandas as pd
import boto3
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "FraudDetectionXGBoost"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
TARGET_MODEL_STAGE = "Staging"
S3_BUCKET_DATA = os.getenv("S3_BUCKET_DATA", "fraud-detection-yourname-data")
S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS", "fraud-detection-yourname-models")

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Complete MLOps API for fraud detection model training and serving",
    version="1.0.0"
)

# Global model variable
MODEL = None
s3_client = None

# ============== Pydantic Models ==============

class Transaction(BaseModel):
    """Input schema for fraud prediction"""
    user_id: int
    account_age_days: int
    total_transactions_user: int
    avg_amount_user: float
    amount: float
    country: str
    bin_country: str
    channel: str
    merchant_category: str
    promo_used: int
    avs_match: int
    cvv_result: int
    three_ds_flag: int
    shipping_distance_km: float
    
    class Config:
        extra = 'allow'

class TrainingRequest(BaseModel):
    """Request schema for training"""
    data_path: str = "transactions.csv"
    mode: str = "all"  # 'all', 'base', or 'hpo'

class PromotionRequest(BaseModel):
    """Request schema for model promotion"""
    parent_run_id: str
    metric: str = "roc_auc"

# ============== Helper Functions ==============

def initialize_s3():
    """Initialize S3 client"""
    global s3_client
    try:
        s3_client = boto3.client('s3')
        logger.info("✓ S3 client initialized")
    except Exception as e:
        logger.warning(f"S3 client initialization failed: {e}")
        s3_client = None

def load_model_from_registry():
    """Load the latest staged model from MLflow registry"""
    global MODEL
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{TARGET_MODEL_STAGE}"
        MODEL = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✓ Model loaded from {model_uri}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL = None
        return False

def download_data_from_s3(data_path: str):
    """Download training data from S3 if needed"""
    if s3_client is None:
        logger.info("S3 not available, using local data")
        return data_path
    
    try:
        local_path = f"/tmp/{os.path.basename(data_path)}"
        s3_client.download_file(
            S3_BUCKET_DATA,
            f"raw/{os.path.basename(data_path)}",
            local_path
        )
        logger.info(f"✓ Downloaded data from S3 to {local_path}")
        return local_path
    except Exception as e:
        logger.warning(f"Failed to download from S3: {e}. Using local file.")
        return data_path

# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Fraud Detection API...")
    
    # Initialize S3
    initialize_s3()
    
    # Try to load model (may fail if no model in registry yet)
    load_model_from_registry()
    
    logger.info("✓ API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Fraud Detection API...")

# ============== Health & Info Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "train": "/train",
            "promote": "/promote",
            "models": "/models/list"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mlflow_status = "connected"
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.MlflowClient().list_experiments()
    except Exception as e:
        mlflow_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "model_stage": TARGET_MODEL_STAGE,
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "mlflow_status": mlflow_status,
        "s3_available": s3_client is not None,
        "timestamp": datetime.now().isoformat()
    }

# ============== Prediction Endpoint ==============

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud probability for a transaction
    
    Returns:
        prediction_proba_fraud: Probability of fraud (0-1)
        is_fraud_predicted: Binary prediction (threshold=0.5)
        timestamp: Prediction timestamp
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and promote a model first."
        )
    
    try:
        # Convert transaction to DataFrame
        data_dict = transaction.dict()
        features = {k: [v] for k, v in data_dict.items() 
                   if k not in ['transaction_id', 'transaction_time']}
        input_df = pd.DataFrame(features)
        
        # Make prediction
        prediction_proba = MODEL.predict(input_df)
        
        # Extract fraud probability (class 1)
        fraud_proba = float(prediction_proba[0, 1])
        is_fraud = fraud_proba > 0.5
        
        logger.info(f"Prediction: fraud_prob={fraud_proba:.4f}, is_fraud={is_fraud}")
        
        return {
            "prediction_proba_fraud": fraud_proba,
            "is_fraud_predicted": is_fraud,
            "confidence": max(fraud_proba, 1 - fraud_proba),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============== Training Endpoints ==============

@app.post("/train")
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model training in the background
    
    This calls your train.py script which:
    1. Trains base model (v1)
    2. Runs hyperparameter optimization
    3. Registers models in MLflow
    
    Returns immediately with a job ID
    """
    try:
        # Download data from S3 if available
        local_data_path = download_data_from_s3(request.data_path)
        
        # Prepare training command
        train_script = os.path.join(os.path.dirname(__file__), "train.py")
        command = [
            "python",
            train_script,
            "--data", local_data_path,
            "--mode", request.mode
        ]
        
        # Run training in background
        background_tasks.add_task(run_training_subprocess, command)
        
        logger.info(f"Training triggered with mode={request.mode}")
        
        return {
            "status": "training_started",
            "mode": request.mode,
            "data_path": request.data_path,
            "message": "Training running in background. Check logs for progress.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_training_subprocess(command: list):
    """
    Run training as a subprocess
    This allows training to continue even if API request completes
    """
    try:
        logger.info(f"Starting training subprocess: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✓ Training completed successfully")
            logger.info(f"Training output:\n{result.stdout}")
            
            # Reload model after training
            load_model_from_registry()
        else:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"Error output:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 2 hours")
    except Exception as e:
        logger.exception(f"Training subprocess error: {e}")

@app.post("/promote")
async def promote_model(request: PromotionRequest):
    """
    Promote the best model to Staging
    
    This calls your promote.py script which:
    1. Finds best model from parent run
    2. Transitions it to Staging stage
    3. Makes it available for serving
    """
    try:
        promote_script = os.path.join(os.path.dirname(__file__), "promote.py")
        command = [
            "python",
            promote_script,
            "--parent_run_id", request.parent_run_id
        ]
        
        logger.info(f"Promoting model from run {request.parent_run_id}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("✓ Model promotion successful")
            logger.info(f"Promotion output:\n{result.stdout}")
            
            # Reload the newly promoted model
            load_model_from_registry()
            
            return {
                "status": "promotion_successful",
                "parent_run_id": request.parent_run_id,
                "message": "Model promoted to Staging",
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.error(f"Promotion failed: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Promotion failed: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Promotion timed out")
    except Exception as e:
        logger.exception(f"Promotion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Model Info Endpoints ==============

@app.get("/models/list")
async def list_models():
    """List all model versions in the registry"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        
        versions = client.get_latest_versions(MODEL_NAME)
        
        models_info = []
        for version in versions:
            models_info.append({
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "run_id": version.run_id,
                "creation_timestamp": version.creation_timestamp
            })
        
        return {
            "model_name": MODEL_NAME,
            "versions": models_info,
            "total_versions": len(models_info)
        }
        
    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/current")
async def get_current_model():
    """Get information about the currently loaded model"""
    if MODEL is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        
        versions = client.get_latest_versions(MODEL_NAME, stages=[TARGET_MODEL_STAGE])
        
        if not versions:
            return {"message": "No model in Staging"}
        
        version = versions[0]
        
        # Get metrics from the run
        run = client.get_run(version.run_id)
        metrics = run.data.metrics
        
        return {
            "model_name": MODEL_NAME,
            "version": version.version,
            "stage": version.current_stage,
            "run_id": version.run_id,
            "metrics": metrics,
            "model_loaded": MODEL is not None
        }
        
    except Exception as e:
        logger.exception(f"Failed to get current model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Main ==============

if __name__ == "__main__":
    # Run the API server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )