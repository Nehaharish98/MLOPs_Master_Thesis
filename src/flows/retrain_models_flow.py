# src/flows/retrain_models_flow.py

from pathlib import Path
from typing import Tuple
import os

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, accuracy_score

# ---------- MLflow setup ----------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

DATA_BASE = Path("data/prepared")


# ---------- Helpers ----------

def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop target + ['split', 'start'] to get feature matrix X, and target y.
    """
    X = df.drop(columns=[target_col, "split", "start"], errors="ignore")
    y = df[target_col].copy()
    return X, y


# ---------- Throughput retraining ----------

@task(name="load-throughput-data")
def load_throughput_data() -> pd.DataFrame:
    path = DATA_BASE / "throughput_supervised.parquet"
    df = pd.read_parquet(path)
    return df


@task(name="train-throughput-model")
def train_throughput_model(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Train a regression model for throughput and log it to MLflow.
    Returns (run_id, rmse_val).
    """
    X, y = split_xy(df, target_col="tput_mean_next")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("throughput_supervised")

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("rmse_val", rmse)
        mlflow.log_metric("r2_val", r2)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="throughput_model",
        )

        run_id = run.info.run_id
        print(f"✅ Throughput model trained. RMSE={rmse:.4f}, R2={r2:.4f}, run_id={run_id}")
        return run_id, rmse


# ---------- Latency retraining ----------

@task(name="load-latency-data")
def load_latency_data() -> pd.DataFrame:
    path = DATA_BASE / "latency_supervised.parquet"
    df = pd.read_parquet(path)
    return df


@task(name="train-latency-model")
def train_latency_model(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Train a regression model for latency and log it to MLflow.
    Returns (run_id, rmse_val).
    """
    X, y = split_xy(df, target_col="delay_mean_next")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("latency_supervised")

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("rmse_val", rmse)
        mlflow.log_metric("r2_val", r2)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="latency_model",
        )

        run_id = run.info.run_id
        print(f"✅ Latency model trained. RMSE={rmse:.4f}, R2={r2:.4f}, run_id={run_id}")
        return run_id, rmse


# ---------- Classification retraining ----------

@task(name="load-classification-data")
def load_classification_data() -> pd.DataFrame:
    path = DATA_BASE / "classification_supervised.parquet"
    df = pd.read_parquet(path)
    return df


@task(name="train-classification-model")
def train_classification_model(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Train a classification model and log it to MLflow.
    Returns (run_id, f1_val).
    """
    target_col = "label_good"
    X, y = split_xy(df, target_col=target_col)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("classification_supervised")

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
        print(f"✅ Classification model trained. F1={f1:.4f}, ACC={acc:.4f}, run_id={run_id}")
        return run_id, f1


# ---------- Main retraining flow ----------

@flow(name="retrain-all-models")
def retrain_all_models_flow():
    """
    Prefect flow to retrain all three models and log them to MLflow.
    """
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

    print("✅ Retraining summary:", summary)
    return summary


if __name__ == "__main__":
    retrain_all_models_flow()
