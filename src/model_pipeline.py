from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

def build_preprocessor(X_train):
    """
    Builds the ColumnTransformer for preprocessing.
    Infers numeric and categorical columns based on data types.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Handle categories that might be missing in test/prod
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor, numeric_features, categorical_features

def build_pipeline(preprocessor, params: dict) -> Pipeline:
    """Builds the complete ML Pipeline including preprocessing and XGBoost."""
    xgb_classifier = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        **params
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])
    return pipeline