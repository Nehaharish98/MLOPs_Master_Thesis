# src/flows/retrain_models_flow.py

from pathlib import Path
from typing import Tuple
import os

import mlflow
import mlflow.sklearn  # ensure sklearn flavor is registered
import mlflow.catboost  # ensure catboost flavor is registered

import numpy as np
import pandas as pd
from prefect import flow, task, get_run_logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, accuracy_score


# ---------- Paths ----------

# This file lives in src/flows/, so:
# parents[0] = .../src/flows
# parents[1] = .../src
# parents[2] = .../Thesis-mlops   <-- project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = PROJECT_ROOT / "data" / "prepared"


# ---------- Helpers ----------

def _init_mlflow():
    """
    Initialise MLflow tracking URI once per process/flow.

    Priority:
      1) MLFLOW_TRACKING_URI environment variable
      2) Fallback to local MLflow server on http://localhost:5000
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        tracking_uri = env_uri
        print(f"[retrain] Using MLflow tracking URI from MLFLOW_TRACKING_URI: {tracking_uri}")
    else:
        tracking_uri = "http://localhost:5000"
        print(f"[retrain] MLFLOW_TRACKING_URI not set, defaulting to local MLflow at {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri



def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop target + ['split', 'start'] to get feature matrix X, and target y.
    """
    X = df.drop(columns=[target_col, "split", "start"], errors="ignore")
    y = df[target_col].copy()
    return X, y


def build_regression_pipeline(X: pd.DataFrame) -> Tuple[Pipeline, list, list]:
    """
    Build a sklearn Pipeline for regression tasks with:
      - OneHotEncoder for categorical features
      - SimpleImputer(median) for numeric features
      - RandomForestRegressor as the model

    Returns:
      pipeline, cat_cols, num_cols
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe, cat_cols, num_cols


# ---------- Throughput retraining ----------

@task(name="load-throughput-data")
def load_throughput_data() -> pd.DataFrame:
    logger = get_run_logger()
    path = DATA_BASE / "throughput_supervised.parquet"
    logger.info(f"[throughput] Loading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"[throughput] Loaded {len(df)} rows")
    return df


@task(name="train-throughput-model")
def train_throughput_model(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Train a regression model for throughput and log it to MLflow.
    Returns (run_id, rmse_val).
    """
    logger = get_run_logger()
    mlflow.set_experiment("throughput")

    X, y = split_xy(df, target_col="tput_mean_next")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"[throughput] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    pipe, cat_cols, num_cols = build_regression_pipeline(X_train)

    with mlflow.start_run() as run:
        params = {
            "rf_n_estimators": 200,
            "rf_max_depth": 10,
            "rf_random_state": 42,
            "n_cat_features": len(cat_cols),
            "n_num_features": len(num_cols),
        }

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("rmse_val", rmse)
        mlflow.log_metric("r2_val", r2)

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="throughput_model",
        )

        run_id = run.info.run_id
        logger.info(
            f"✅ Throughput model retrained. RMSE={rmse:.4f}, R2={r2:.4f}, run_id={run_id}"
        )
        return run_id, rmse


# ---------- Latency retraining ----------

@task(name="load-latency-data")
def load_latency_data() -> pd.DataFrame:
    logger = get_run_logger()
    path = DATA_BASE / "latency_supervised.parquet"
    logger.info(f"[latency] Loading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"[latency] Loaded {len(df)} rows")
    return df


@task(name="train-latency-model")
def train_latency_model(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Train a regression model for latency and log it to MLflow.
    Returns (run_id, rmse_val).
    """
    logger = get_run_logger()
    mlflow.set_experiment("latency")

    X, y = split_xy(df, target_col="delay_mean_next")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"[latency] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    pipe, cat_cols, num_cols = build_regression_pipeline(X_train)

    with mlflow.start_run() as run:
        params = {
            "rf_n_estimators": 200,
            "rf_max_depth": 10,
            "rf_random_state": 42,
            "n_cat_features": len(cat_cols),
            "n_num_features": len(num_cols),
        }

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("rmse_val", rmse)
        mlflow.log_metric("r2_val", r2)

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="latency_model",
        )

        run_id = run.info.run_id
        logger.info(
            f"✅ Latency model retrained. RMSE={rmse:.4f}, R2={r2:.4f}, run_id={run_id}"
        )
        return run_id, rmse


# ---------- Classification retraining ----------

@task(name="load-classification-data")
def load_classification_data() -> pd.DataFrame:
    logger = get_run_logger()
    path = DATA_BASE / "classification_supervised.parquet"
    logger.info(f"[classification] Loading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"[classification] Loaded {len(df)} rows")
    return df


@task(name="train-classification-model")
def train_classification_model(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Train a classification model and log it to MLflow.
    Returns (run_id, f1_val).
    """
    logger = get_run_logger()
    mlflow.set_experiment("classification")

    target_col = "label_good"
    X, y = split_xy(df, target_col=target_col)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"[classification] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Treat string columns as categorical
    cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype == "object"]

    train_pool = Pool(X_train, y_train, cat_features=cat_features or None)
    val_pool = Pool(X_val, y_val, cat_features=cat_features or None)

    with mlflow.start_run() as run:
        params = {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 300,
            "loss_function": "Logloss",
            "eval_metric": "F1",
            "random_seed": 42,
            "verbose": False,
        }

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        y_pred_val = model.predict(val_pool)
        y_pred_val = y_pred_val.astype(int)

        f1 = float(f1_score(y_val, y_pred_val))
        acc = float(accuracy_score(y_val, y_pred_val))

        mlflow.log_params(params)
        mlflow.log_metric("f1_val", f1)
        mlflow.log_metric("accuracy_val", acc)

        mlflow.catboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="classification_model",
        )

        run_id = run.info.run_id
        logger.info(
            f"✅ Classification model retrained. F1={f1:.4f}, ACC={acc:.4f}, run_id={run_id}"
        )
        return run_id, f1


# ---------- Main retraining flow ----------

@flow(name="retrain-all-models")
def retrain_all_models_flow():
    """
    Prefect flow to retrain all three models and log them to MLflow.
    """
    logger = get_run_logger()
    tracking_uri = _init_mlflow()
    logger.info(f"[retrain] Flow starting with MLflow URI: {tracking_uri}")
    logger.info(f"[retrain] DATA_BASE = {DATA_BASE}")

    # 1) Throughput
    df_tput = load_throughput_data()
    run_id_tput, rmse_tput = train_throughput_model(df_tput)

    # 2) Latency
    df_lat = load_latency_data()
    run_id_lat, rmse_lat = train_latency_model(df_lat)

    # 3) Classification
    df_cls = load_classification_data()
    run_id_cls, f1_cls = train_classification_model(df_cls)

    summary = {
        "throughput_run_id": run_id_tput,
        "throughput_rmse_val": rmse_tput,
        "latency_run_id": run_id_lat,
        "latency_rmse_val": rmse_lat,
        "classification_run_id": run_id_cls,
        "classification_f1_val": f1_cls,
    }

    logger.info(f"✅ Retraining summary: {summary}")
    print("✅ Retraining summary:", summary)
    return summary


if __name__ == "__main__":
    retrain_all_models_flow()
