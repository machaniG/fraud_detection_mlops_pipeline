import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Error: Dataset not found at {file_path}")
        raise FileNotFoundError(f"Error: Dataset not found at {file_path}")
    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise

def identify_target(df: pd.DataFrame) -> str:
    """Identifies the target column (is_fraud) or raises an error."""
    possible_targets = ['is_fraud', 'fraud', 'label', 'target']
    for col in possible_targets:
        if col in df.columns:
            logger.info(f"Identified target column: {col}")
            return col
    raise ValueError("Target column ('is_fraud', 'fraud', 'label', or 'target') not found in dataset.")

def split_data(df: pd.DataFrame, target_col: str, random_state: int = 42):
    """Splits data into train, validation, and test sets (60/20/20)."""
    X = df.drop(columns=[target_col, 'transaction_id', 'transaction_time'], errors='ignore')
    y = df[target_col]

    # Split 80% (train+val) and 20% (test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Split train+val into 75% (train) and 25% (val) -> 60/20 overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state, stratify=y_train_val
    )

    logger.info(f"Data split: Train ({len(X_train)}), Validation ({len(X_val)}), Test ({len(X_test)})")
    return X_train, X_val, X_test, y_train, y_val, y_test