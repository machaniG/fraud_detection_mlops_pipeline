#!/usr/bin/env python3
import uvicorn
import mlflow
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "FraudDetectionXGBoost"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
TARGET_MODEL_STAGE = "Staging"

# --- MLflow Model Loading ---
MODEL = None
app = FastAPI(title="Fraud Detection Model Server")

class Transaction(BaseModel):
    """
    Defines the expected input schema for a single transaction prediction.
    Corresponds to the features used in training (excluding transaction_id/time).
    """
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

    # Allow compatibility with models that drop transaction_id/time
    class Config:
        extra = 'allow'

def load_model():
    """Loads the model from the MLflow Model Registry (Staging stage)."""
    global MODEL
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Construct the URI to load the model currently in the 'Staging' stage
        model_uri = f"models:/{MODEL_NAME}/{TARGET_MODEL_STAGE}"
        MODEL = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}")
        # In a production environment, you might allow startup to fail if the model isn't ready.
        # For this demo, we'll keep it simple but log the failure.
        MODEL = None
        raise RuntimeError(f"Model load failed. Is a model in '{TARGET_MODEL_STAGE}'?") from e


@app.on_event("startup")
async def startup_event():
    """Load model when the FastAPI application starts."""
    # Ensure the model is loaded before accepting requests
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    return {"status": "ok", "model": MODEL_NAME, "stage": TARGET_MODEL_STAGE}


@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """Predicts the fraud risk for a single transaction."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is loading or failed to load.")

    try:
        # Convert the Pydantic model instance to a dictionary
        data_dict = transaction.dict()
        
        # Prepare data for prediction: must match the training feature set
        # The Model Pipeline expects the same columns as X_train, excluding ID/Time/Target.
        # We manually exclude columns that the model pipeline's preprocessor was designed to drop
        features = {k: [v] for k, v in data_dict.items() if k not in ['transaction_id', 'transaction_time']}
        
        input_df = pd.DataFrame(features)

        # The model loaded by pyfunc handles the preprocessing automatically
        prediction_proba = MODEL.predict(input_df)
        
        # prediction_proba will be an array of shape (1, 2). We want the probability of class 1 (fraud).
        fraud_proba = float(prediction_proba[0, 1])

        return {
            "prediction_proba_fraud": fraud_proba,
            "is_fraud_predicted": fraud_proba > 0.5 # Simple threshold
        }
    
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")


if __name__ == "__main__":
    # If run directly (not via uvicorn in Docker), the script will start the server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "scripts.serve:app", 
        host="0.0.0.0", 
        port=8080, 
        log_level="info", 
        reload=False # Set reload=True for local development
    )