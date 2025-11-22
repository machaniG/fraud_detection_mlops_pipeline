import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import modules to test
from src.data_utils import load_data, identify_target, split_data
from src.model_pipeline import build_preprocessor, build_pipeline

# Mock data creation for testing (assuming minimal columns for structure validation)
@pytest.fixture
def mock_data():
    """Returns a small DataFrame simulating the transactions data structure."""
    data = {
        'transaction_id': range(1, 11),
        'user_id': [1, 2, 1, 3, 2, 4, 1, 5, 3, 6],
        'amount': [100.0, 50.5, 120.0, 30.0, 75.0, 200.0, 90.0, 150.0, 40.0, 80.0],
        'country': ['US', 'CA', 'US', 'MX', 'CA', 'US', 'UK', 'MX', 'US', 'CA'],
        'merchant_category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'is_fraud': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'transaction_time': ['2024-01-01T00:00:00Z'] * 10
    }
    return pd.DataFrame(data)

def test_identify_target(mock_data):
    """Test if the target column is correctly identified."""
    target = identify_target(mock_data)
    assert target == 'is_fraud'
    
def test_split_data_shapes(mock_data):
    """Test data splitting into 60/20/20 ratio."""
    target_col = identify_target(mock_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(mock_data, target_col)
    
    total_len = len(mock_data)
    
    # Check for correct number of splits
    assert len(X_train) == 6
    assert len(X_val) == 2
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_val) == 2
    assert len(y_test) == 2
    
    # Check that feature columns were dropped correctly
    expected_cols = ['user_id', 'amount', 'country', 'merchant_category']
    assert list(X_train.columns) == expected_cols

def test_pipeline_builds_and_runs(mock_data):
    """Test if the full ML pipeline can be initialized and fit without error."""
    target_col = identify_target(mock_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(mock_data, target_col)
    
    preprocessor, _, _ = build_preprocessor(X_train)
    pipeline = build_pipeline(preprocessor, params={'n_estimators': 1, 'max_depth': 1})
    
    # Check pipeline components
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps['preprocessor'], ColumnTransformer)

    # Simple fit to ensure no immediate structural errors
    pipeline.fit(X_train, y_train)
    
    # Simple predict
    y_pred = pipeline.predict(X_test)
    assert y_pred.shape[0] == len(X_test)
    assert y_pred.dtype == int