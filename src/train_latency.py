# train_latency.py
"""
Latency forecasting task.

Same design as throughput:

- One supervised dataset per task with precomputed time-ordered splits:
    splits/latency/train|val|test.parquet
- Target: delay_mean_next.
- Categoricals stay as strings/category (provider, regions, proto, sizes, etc.).
- CatBoost uses native categoricals (no OHE).
- Other models use an OHE + numeric ColumnTransformer inside a Pipeline.

This ensures:
- One dataset per task.
- Same splits for all models.
- Fair comparisons on R2/MAE/MAPE.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.catboost
from .mlflow_setup import setup_mlflow

from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pandas.api.types import (
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_bool_dtype,
)
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------------------
# Project paths (still used for local models, but data now lives in S3)
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = DATA_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TARGET_COL = "delay_mean_next"

# -------------------------------------------------------------------------
# S3 configuration for cloud-based data access (latency splits)
# -------------------------------------------------------------------------
S3_BUCKET = "thesis-netperf-mlops-2525"
S3_LATENCY_SPLITS_PREFIX = "splits/latency"

S3_TRAIN_PATH = f"s3://{S3_BUCKET}/{S3_LATENCY_SPLITS_PREFIX}/train.parquet"
S3_VAL_PATH = f"s3://{S3_BUCKET}/{S3_LATENCY_SPLITS_PREFIX}/val.parquet"
S3_TEST_PATH = f"s3://{S3_BUCKET}/{S3_LATENCY_SPLITS_PREFIX}/test.parquet"


def load_splits():
    """
    Load precomputed time-respecting splits for the latency task from S3.

    Expected layout:
      s3://thesis-netperf-mlops-2525/splits/latency/train.parquet
      s3://thesis-netperf-mlops-2525/splits/latency/val.parquet
      s3://thesis-netperf-mlops-2525/splits/latency/test.parquet
    """
    train = pd.read_parquet(S3_TRAIN_PATH)
    val = pd.read_parquet(S3_VAL_PATH)
    test = pd.read_parquet(S3_TEST_PATH)
    return train, val, test


def prepare_features(df: pd.DataFrame):
    """
    Prepare X, y and identify categorical/numeric columns (no pre-encoding).
    """
    drop_cols = [c for c in [TARGET_COL, "split"] if c in df.columns]
    if "start" in df.columns:
        drop_cols.append("start")

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL].astype(float)

    # Identify categorical vs numeric columns robustly
    cat_cols = [
        c
        for c in X.columns
        if (
            is_object_dtype(X[c])
            or is_categorical_dtype(X[c])
            or is_string_dtype(X[c])
            or is_bool_dtype(X[c])
        )
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Normalize categoricals: pure strings, no NAType
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    return X, y, cat_cols, num_cols


def make_sklearn_model(model_type: str, cat_cols, num_cols, args):
    """
    Build a sklearn model with OHE in a ColumnTransformer for non-CatBoost models.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "xgb":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown sklearn model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def make_catboost_model():
    """CatBoostRegressor that uses native categoricals."""
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        random_seed=42,
        verbose=100,
    )


def evaluate_regression(y_true, y_pred, split_name: str):
    """Compute R2, MAE, RMSE, MAPE for a given split and print + return them."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)

    # Safer MAPE: ignore targets that are zero or extremely small
    mask = np.abs(y_true) > 1e-3
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        mape = np.nan  # or 0.0 if you prefer

    print(f"\n[{split_name}] MAE:  {mae:.4f}")
    print(f"[{split_name}] RMSE: {rmse:.4f}")
    print(f"[{split_name}] R^2:  {r2:.4f}")
    print(f"[{split_name}] MAPE: {mape:.2f}%")

    return {
        f"{split_name}_mae": mae,
        f"{split_name}_rmse": rmse,
        f"{split_name}_r2": r2,
        f"{split_name}_mape": mape,
    }


def run_experiment(model_type: str, args):
    """
    Run one MLflow-tracked experiment for latency with a chosen model type:
    - 'catboost' : CatBoostRegressor with native categoricals
    - 'rf'       : RandomForestRegressor with OHE pipeline
    - 'linear'   : LinearRegression with OHE pipeline
    - 'xgb'      : XGBRegressor with OHE pipeline
    """
    print("Setting up MLflow for latency...")
    tracking_uri = setup_mlflow(task_name="latency")
    print("MLflow tracking URI:", tracking_uri)

    train, val, test = load_splits()
    X_train, y_train, cat_cols, num_cols = prepare_features(train)
    X_val, y_val, _, _ = prepare_features(val)
    X_test, y_test, _, _ = prepare_features(test)

    print(f"Train size: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Features: {len(cat_cols)} categorical, {len(num_cols)} numeric")

    with mlflow.start_run():
        mlflow.set_tag("task", "latency")
        mlflow.set_tag("model_type", model_type)

        # --- Dataset metadata (S3 URIs) ---
        mlflow.log_param("data_bucket", S3_BUCKET)
        mlflow.log_param("train_path", S3_TRAIN_PATH)
        mlflow.log_param("val_path", S3_VAL_PATH)
        mlflow.log_param("test_path", S3_TEST_PATH)

        if model_type == "catboost":
            # ---- CatBoost path: use same dataset, just mark categorical columns ----
            # Make copies so we can safely modify categoricals for CatBoost
            X_train_cb = X_train.copy()
            X_val_cb = X_val.copy()
            X_test_cb = X_test.copy()

            # CatBoost doesn't like pandas.NA in categoricals.
            # Convert categorical columns to string and fill missing with a placeholder.
            for c in cat_cols:
                X_train_cb[c] = X_train_cb[c].astype("string").fillna("__MISSING__")
                X_val_cb[c] = X_val_cb[c].astype("string").fillna("__MISSING__")
                X_test_cb[c] = X_test_cb[c].astype("string").fillna("__MISSING__")

            cat_feature_indices = [X_train_cb.columns.get_loc(c) for c in cat_cols]

            train_pool = Pool(X_train_cb, y_train, cat_features=cat_feature_indices)
            val_pool = Pool(X_val_cb, y_val, cat_features=cat_feature_indices)
            test_pool = Pool(X_test_cb, y_test, cat_features=cat_feature_indices)

            model = make_catboost_model()

            mlflow.log_params(
                {
                    "depth": model.get_param("depth"),
                    "learning_rate": model.get_param("learning_rate"),
                    "n_estimators": model.get_param("n_estimators"),
                }
            )

            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            y_pred_train = model.predict(train_pool)
            y_pred_val = model.predict(val_pool)
            y_pred_test = model.predict(test_pool)

            mlflow.catboost.log_model(model, "model")

        else:
            model = make_sklearn_model(model_type, cat_cols, num_cols, args)

            if model_type == "rf":
                mlflow.log_params(
                    {
                        "n_estimators": args.rf_n_estimators,
                        "max_depth": args.rf_max_depth,
                        "min_samples_split": args.rf_min_samples_split,
                        "min_samples_leaf": args.rf_min_samples_leaf,
                    }
                )
            elif model_type == "linear":
                mlflow.log_params({})
            elif model_type == "xgb":
                mlflow.log_params(
                    {
                        "n_estimators": 300,
                        "max_depth": 6,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    }
                )

            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            mlflow.sklearn.log_model(model, "model")

        metrics = {}
        metrics.update(evaluate_regression(y_train, y_pred_train, "train"))
        metrics.update(evaluate_regression(y_val, y_pred_val, "val"))
        metrics.update(evaluate_regression(y_test, y_pred_test, "test"))

        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(X_train.columns), "features.txt")

        print("\nMetrics logged to MLflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Train latency models with MLflow logging."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["catboost", "rf", "linear", "xgb"],
        required=True,
        help="Which model to train.",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=10)
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)

    args = parser.parse_args()
    run_experiment(model_type=args.model, args=args)


if __name__ == "__main__":
    main()
