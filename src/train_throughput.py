# train_throughput.py
"""
Throughput regression task.

Core idea (matches thesis design):

- Use ONE supervised dataset for throughput:
  - Splits: splits/throughput/train|val|test.parquet
  - Target: tput_mean_next
  - Categorical features stay as strings/category:
    provider, region_snd_base, region_rcv_base, proto,
    size_snd, size_rcv, (optionally az_snd, az_rcv, snd_continent, rcv_continent)
  - Numeric features: lag features + summary stats (tput_mean_lag1..3, delay stats,
    loss stats, duration, etc.).

- For CatBoost:
  - Feed the same dataset directly.
  - Tell CatBoost which columns are categorical via column indices.
  - No one-hot encoding; CatBoost handles categoricals natively.

- For LinearRegression / RandomForest (and any other sklearn model later):
  - Use a ColumnTransformer with OneHotEncoder for categoricals and passthrough
    for numerics.
  - Keep encoding INSIDE the pipeline so we don’t need separate “encoded datasets”
    and the encoder is saved together with the model.

- Splits are already time-ordered per (provider, region_snd_base, region_rcv_base),
  so all models train/evaluate on the exact same runs = fair comparison.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.catboost  # requires mlflow[extras] or mlflow[catboost]
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

TARGET_COL = "tput_mean_next"

# -------------------------------------------------------------------------
# S3 configuration for cloud-based data access
# -------------------------------------------------------------------------
S3_BUCKET = "thesis-netperf-mlops-2525"
S3_THROUGHPUT_SPLITS_PREFIX = "splits/throughput"

S3_TRAIN_PATH = f"s3://{S3_BUCKET}/{S3_THROUGHPUT_SPLITS_PREFIX}/train.parquet"
S3_VAL_PATH = f"s3://{S3_BUCKET}/{S3_THROUGHPUT_SPLITS_PREFIX}/val.parquet"
S3_TEST_PATH = f"s3://{S3_BUCKET}/{S3_THROUGHPUT_SPLITS_PREFIX}/test.parquet"


def load_splits():
    """
    Load precomputed time-respecting splits for the throughput task from S3.

    This assumes the following layout in your bucket:
      s3://thesis-netperf-mlops-2525/splits/throughput/train.parquet
      s3://thesis-netperf-mlops-2525/splits/throughput/val.parquet
      s3://thesis-netperf-mlops-2525/splits/throughput/test.parquet
    """
    train = pd.read_parquet(S3_TRAIN_PATH)
    val = pd.read_parquet(S3_VAL_PATH)
    test = pd.read_parquet(S3_TEST_PATH)
    return train, val, test


def prepare_features(df: pd.DataFrame):
    """
    Prepare X, y and identify categorical/numeric columns.

    We do NOT pre-encode anything here. We just:
    - drop target + split marker + raw timestamp
    - keep categoricals as strings/category/bool
    - return cat_cols & num_cols so:
        * CatBoost can get cat indices
        * sklearn models can build an OHE+numeric pipeline
    """
    drop_cols = [c for c in [TARGET_COL, "split"] if c in df.columns]
    if "start" in df.columns:
        drop_cols.append("start")

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL].astype(float)

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


def make_sklearn_model(model_type: str, cat_cols, num_cols):
    """
    Build a sklearn-compatible model with OHE inside a ColumnTransformer.

    This is used for models that DO need numeric input:
    - LinearRegression
    - RandomForestRegressor
    - XGBRegressor

    We never create a separate encoded dataset on disk; encoding stays inside the pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
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
    """CatBoostRegressor that can consume categorical features directly."""
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


def run_experiment(model_type: str):
    """
    Run one MLflow-tracked experiment for the throughput task with a chosen model type:
    - 'catboost' : CatBoostRegressor with native categoricals
    - 'rf'       : RandomForestRegressor with OHE pipeline
    - 'linear'   : LinearRegression with OHE pipeline
    - 'xgb'      : XGBRegressor with OHE pipeline
    """
    print("Setting up MLflow for throughput...")
    tracking_uri = setup_mlflow(task_name="throughput")
    print("MLflow tracking URI:", tracking_uri)

    train, val, test = load_splits()
    X_train, y_train, cat_cols, num_cols = prepare_features(train)
    X_val, y_val, _, _ = prepare_features(val)
    X_test, y_test, _, _ = prepare_features(test)

    print(f"Train size: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Features: {len(cat_cols)} categorical, {len(num_cols)} numeric")

    with mlflow.start_run():
        mlflow.set_tag("task", "throughput")
        mlflow.set_tag("model_type", model_type)

        # --- Dataset metadata (now S3 URIs) ---
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
            # ---- sklearn path: build OHE + model pipeline ----
            model = make_sklearn_model(model_type, cat_cols, num_cols)

            if model_type == "rf":
                mlflow.log_params(
                    {
                        "n_estimators": 300,
                        "max_depth": 10,
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

        # Evaluate and log metrics
        metrics = {}
        metrics.update(evaluate_regression(y_train, y_pred_train, "train"))
        metrics.update(evaluate_regression(y_val, y_pred_val, "val"))
        metrics.update(evaluate_regression(y_test, y_pred_test, "test"))

        mlflow.log_metrics(metrics)

        # Log feature names as artifact (for reproducibility)
        mlflow.log_text("\n".join(X_train.columns), "features.txt")

        print("\nMetrics logged to MLflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Train throughput models with MLflow logging."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["catboost", "rf", "linear", "xgb"],
        required=True,
        help="Which model to train.",
    )
    args = parser.parse_args()
    run_experiment(model_type=args.model)


if __name__ == "__main__":
    main()
