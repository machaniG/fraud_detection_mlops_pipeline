#!/usr/bin/env python3
"""
Train an XGBoost classifier inside a scikit-learn Pipeline with preprocessing.
Logs experiment & artifacts to MLflow, uses child runs for hyperparameter tuning,
and registers models in the MLflow Model Registry.

Usage:
  - Inspect/help:
      python scripts/train_xgboost_mlflow.py --help
  - Dry-run (validate dataset & pipeline without expensive training):
      python scripts/train_xgboost_mlflow.py --dry-run
  - Full run (will do base training, hyperparameter tuning, and registry operations):
      python scripts/train_xgboost_mlflow.py --run

This script attempts to infer the target column automatically (common names like
isFraud, fraud, label, target). If it can't find one, it will raise an error.

Requirements: see requirements.txt
"""
import argparse
import os
import tempfile
from pprint import pformat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.metrics import ConfusionMatrixDisplay

import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import shap

MLFLOW_EXPERIMENT_NAME = "fraud_xgb_experiment"
MODEL_NAME = "fraud_detector_xgb"


def infer_target(df: pd.DataFrame):
    candidates = ["isFraud", "fraud", "label", "target", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column if binary
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 2:
        return last_col
    raise ValueError("Could not infer target column. Please add a target with a common name.")


def build_preprocessor(X: pd.DataFrame):
    # exclude id-like columns
    exclude_prefixes = ["id", "transactionid", "txid", "index"]
    candidates = [c for c in X.columns if not any(c.lower().startswith(p) for p in exclude_prefixes)]

    numeric_cols = X[candidates].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in candidates if c not in numeric_cols]

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop",
        n_jobs=1
    )

    return preprocessor, numeric_cols, categorical_cols


def eval_and_log(y_true, y_pred, y_prob, run_id, artifacts_dir):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = None
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else None
    except Exception:
        auc = None

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if auc is not None:
        metrics["auc"] = auc

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)

    mlflow.log_metrics(metrics)
    mlflow.log_artifact(cm_path)

    return metrics


def plot_and_log_shap(pipeline: Pipeline, X_sample: pd.DataFrame, artifacts_dir):
    # produce shap on transformed features
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(X_sample)
    # feature names (OHE expands categories)
    try:
        num_features = preprocessor.transformers_[0][2]
        cat_transformer = preprocessor.transformers_[1][1].named_steps["onehot"]
        cat_feature_names = cat_transformer.get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_features) + list(cat_feature_names)
    except Exception:
        # fallback to indices
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    # shap for xgboost
    try:
        explainer = shap.TreeExplainer(clf.get_booster())
        shap_values = explainer.shap_values(X_trans)
    except Exception:
        # try with sklearn-compatible wrapper
        explainer = shap.Explainer(clf)
        shap_values = explainer(X_trans)

    # summary plot
    shap_path = os.path.join(artifacts_dir, "shap_summary.png")
    fig = plt.figure(figsize=(8, 6))
    try:
        shap.summary_plot(shap_values, features=X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        fig.savefig(shap_path)
        plt.close(fig)
        mlflow.log_artifact(shap_path)
    except Exception as e:
        print("Could not create SHAP plot:", e)


def train_base_and_log(X_train, X_test, y_train, y_test, preprocessor, dry_run=False):
    # base classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    fraud_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb_clf)])

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="base_run") as run:
        mlflow.set_tag("run_type", "base")
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_params(xgb_clf.get_params())

        if dry_run:
            print("Dry run: skipping actual fit")
            return run.info.run_id

        fraud_pipeline.fit(X_train, y_train)

        y_pred = fraud_pipeline.predict(X_test)
        y_prob = fraud_pipeline.predict_proba(X_test)

        # create artifacts dir
        with tempfile.TemporaryDirectory() as artifacts_dir:
            metrics = eval_and_log(y_test, y_pred, y_prob, run.info.run_id, artifacts_dir)
            # SHAP
            try:
                plot_and_log_shap(fraud_pipeline, X_test.sample(min(len(X_test), 200), random_state=1), artifacts_dir)
            except Exception as e:
                print("SHAP generation failed:", e)

            # log and register model
            mlflow.sklearn.log_model(fraud_pipeline, "model")

            # Register model as a new version (will be V1 if none existed)
            client = MlflowClient()
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                # create model registry entry if not exists
                client.get_registered_model(MODEL_NAME)
            except Exception:
                client.create_registered_model(MODEL_NAME)

            mv = mlflow.register_model(model_uri, MODEL_NAME)
            print(f"Registered base model version: {mv.version}")

        return run.info.run_id


def hyperparam_tuning_with_child_runs(X_train, X_val, y_train, y_val, preprocessor, param_grid, max_trials=50):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    parent_run = mlflow.start_run(run_name="hyperparam_tuning")
    parent_run = parent_run.__enter__()
    try:
        mlflow.log_param("param_grid_size", sum(1 for _ in param_grid))

        best_auc = -np.inf
        best_run_id = None
        best_params = None

        client = MlflowClient()

        # iterate param_grid (list of dicts)
        for i, params in enumerate(param_grid):
            if i >= max_trials:
                break

            with mlflow.start_run(run_name=f"trial_{i}", nested=True) as child:
                mlflow.log_params(params)
                clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **params)
                pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_val)
                y_prob = pipeline.predict_proba(X_val)

                metrics = {}
                metrics["accuracy"] = accuracy_score(y_val, y_pred)
                metrics["precision"] = precision_score(y_val, y_pred)
                metrics["recall"] = recall_score(y_val, y_pred)
                metrics["f1"] = f1_score(y_val, y_pred)
                try:
                    metrics["auc"] = roc_auc_score(y_val, y_prob[:, 1])
                except Exception:
                    metrics["auc"] = None

                mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

                # log model for this child run
                mlflow.sklearn.log_model(pipeline, "model")

                # register this child model too (optional). We will pick best
                # Collect best
                if metrics.get("auc") is not None and metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    best_run_id = child.info.run_id
                    best_params = params

        print(f"Best tuning AUC: {best_auc}, run_id: {best_run_id}")
        return best_run_id, best_params
    finally:
        mlflow.end_run()


def register_best_and_transition(best_run_id):
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"

    # register (this will create next version, e.g., V2)
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version
    print(f"Registered tuned model version: {version}")

    # Transition versions: promote the new one to Staging and archive the previous
    # We will archive all other versions and set this to Staging
    existing_versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production", "Archived"]) or []
    for v in existing_versions:
        if int(v.version) == int(version):
            continue
        # archive others
        try:
            client.transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Archived")
        except Exception:
            pass

    # promote this version to Staging
    client.transition_model_version_stage(name=MODEL_NAME, version=version, stage="Staging", archive_existing_versions=False)
    # explicitly archive any remaining previous versions (e.g., V1)
    for v in existing_versions:
        if int(v.version) != int(version):
            try:
                client.transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Archived")
            except Exception:
                pass

    print(f"Version {version} set to Staging, others archived.")


def generate_param_grid():
    # simple grid - expand manually to avoid heavy combinatorics
    grid = []
    learning_rates = [0.01, 0.05, 0.1]
    max_depths = [3, 6]
    n_estimators = [100, 200]

    for lr in learning_rates:
        for md in max_depths:
            for ne in n_estimators:
                grid.append({"learning_rate": lr, "max_depth": md, "n_estimators": ne})
    return grid


def main(args):
    df = pd.read_csv(args.data)
    target = infer_target(df)
    print(f"Inferred target column: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    print("Numeric cols:", num_cols)
    print("Categorical cols:", cat_cols)

    # Step 1: base training and register as V1
    base_run_id = train_base_and_log(X_train, X_test, y_train, y_test, preprocessor, dry_run=args.dry_run)

    # Step 2: hyperparameter tuning with child runs
    param_grid = generate_param_grid()
    best_run_id, best_params = hyperparam_tuning_with_child_runs(X_train, X_val, y_train, y_val, preprocessor, param_grid, max_trials=args.max_trials)

    # Step 3: register best as V2 and transition
    if best_run_id is not None and not args.dry_run:
        register_best_and_transition(best_run_id)

    print("Done. Base run id:", base_run_id, "Best tuning run id:", best_run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="transactions.csv", help="Path to transactions CSV")
    parser.add_argument("--dry-run", action="store_true", help="Do not fit models, only validate pipeline")
    parser.add_argument("--run", dest="run", action="store_true", help="Execute full training flow")
    parser.add_argument("--max-trials", type=int, default=20, help="Maximum tuning trials to run")
    args = parser.parse_args()

    # default behaviour: dry-run unless --run provided
    if not args.run:
        args.dry_run = True

    main(args)
