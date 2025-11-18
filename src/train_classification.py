# train_classification.py
"""
Performance classification (good/bad) task.

Dataset:
- Splits: splits/classification/train|val|test.parquet
- Target: label_good (1 if tput >= pair median AND delay <= pair median).
- Features include lagged throughput/latency plus static attributes.

Design is consistent with regression tasks:

- One dataset per task.
- Categorical features stored as strings/category.
- CatBoostClassifier consumes categoricals directly (no OHE).
- RandomForest/LogisticRegression use an OHE+numeric ColumnTransformer in a Pipeline.
- Same time-respecting splits for all models.

Metrics:
- Accuracy, F1, ROC-AUC per split.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.catboost
from .mlflow_setup import setup_mlflow

from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pandas.api.types import (
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_bool_dtype,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_ROOT / "splits" / "classification"
MODEL_DIR = DATA_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TARGET_COL = "label_good"


def load_splits():
    """Load precomputed splits for the classification task."""
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    return train, val, test


def prepare_features(df: pd.DataFrame):
    """
    Prepare X, y and identify categorical/numeric columns (no pre-encoding).
    """
    drop_cols = [c for c in [TARGET_COL, "split"] if c in df.columns]
    if "start" in df.columns:
        drop_cols.append("start")

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET_COL].astype(int)

    # Identify categorical vs numeric columns robustly
    cat_cols = [
        c for c in X.columns
        if (
            is_object_dtype(X[c])
            or is_categorical_dtype(X[c])
            or is_string_dtype(X[c])
            or is_bool_dtype(X[c])
        )
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 2) Normalize categoricals: pure strings, no NAType
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    return X, y, cat_cols, num_cols

def make_sklearn_model(model_type: str, cat_cols, num_cols):
    """
    Build a sklearn classification model with OHE in a ColumnTransformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
        )    
    else:
        raise ValueError(f"Unknown sklearn model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def make_catboost_model(args):
    """CatBoostClassifier that uses native categoricals."""
    return CatBoostClassifier(
        depth=args.cb_depth,
        learning_rate=args.cb_learning_rate,
        iterations=args.cb_iterations,
        l2_leaf_reg=args.cb_l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=100,
    )


def evaluate_classification(y_true, y_proba, split_name: str):
    """
    Compute Accuracy, F1, ROC-AUC for a given split and print + return them.
    """
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"\n[{split_name}] Accuracy: {acc:.4f}")
    print(f"[{split_name}] F1:       {f1:.4f}")
    print(f"[{split_name}] ROC-AUC:  {auc:.4f}")
    print(f"[{split_name}] Classification report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        f"{split_name}_accuracy": acc,
        f"{split_name}_f1": f1,
        f"{split_name}_roc_auc": auc,
    }


def run_experiment(model_type: str, args):
    """
    Run one MLflow-tracked experiment for classification with chosen model type:
    - 'catboost' : CatBoostClassifier with native categoricals
    - 'rf'       : RandomForestClassifier with OHE pipeline
    - 'logreg'   : LogisticRegression with OHE pipeline
    """
    print("Setting up MLflow for classification...")
    tracking_uri = setup_mlflow(task_name="classification")
    print("MLflow tracking URI:", tracking_uri)

    train, val, test = load_splits()
    X_train, y_train, cat_cols, num_cols = prepare_features(train)
    X_val, y_val, _, _ = prepare_features(val)
    X_test, y_test, _, _ = prepare_features(test)

    print(f"Train size: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Features: {len(cat_cols)} categorical, {len(num_cols)} numeric")

    with mlflow.start_run():
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("model_type", model_type)

        # --- Dataset metadata ---
        mlflow.log_param("train_path", str(SPLIT_DIR / "train.parquet"))
        mlflow.log_param("val_path", str(SPLIT_DIR / "val.parquet"))
        mlflow.log_param("test_path", str(SPLIT_DIR / "test.parquet"))

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

            cat_feature_indices = [
                X_train_cb.columns.get_loc(c) for c in cat_cols
            ]

            train_pool = Pool(X_train_cb, y_train, cat_features=cat_feature_indices)
            val_pool = Pool(X_val_cb, y_val, cat_features=cat_feature_indices)
            test_pool = Pool(X_test_cb, y_test, cat_features=cat_feature_indices)

            model = make_catboost_model(args)

            mlflow.log_params({
                "depth": args.cb_depth,
                "learning_rate": args.cb_learning_rate,
                "iterations": args.cb_iterations,
                "l2_leaf_reg": args.cb_l2_leaf_reg,
            })

            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            proba_train = model.predict_proba(train_pool)[:, 1]
            proba_val = model.predict_proba(val_pool)[:, 1]
            proba_test = model.predict_proba(test_pool)[:, 1]

            mlflow.catboost.log_model(model, "model")

        else:
            model = make_sklearn_model(model_type, cat_cols, num_cols, args)

            if model_type == "rf":
                mlflow.log_params({
                    "n_estimators": 300,
                    "max_depth": 10,
                })
            elif model_type == "logreg":
                mlflow.log_params({
                    "max_iter": 1000,
                })
            elif model_type == "xgb":
                mlflow.log_params({
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                })    

            model.fit(X_train, y_train)

            proba_train = model.predict_proba(X_train)[:, 1]
            proba_val = model.predict_proba(X_val)[:, 1]
            proba_test = model.predict_proba(X_test)[:, 1]

            mlflow.sklearn.log_model(model, "model")

        metrics = {}
        metrics.update(evaluate_classification(y_train, proba_train, "train"))
        metrics.update(evaluate_classification(y_val, proba_val, "val"))
        metrics.update(evaluate_classification(y_test, proba_test, "test"))

        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(X_train.columns), "features.txt")

        print("\nMetrics logged to MLflow.")


def main():
    parser = argparse.ArgumentParser(description="Train classification models with MLflow logging.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["catboost", "rf", "logreg", "xgb"],
        required=True,
        help="Which model to train.",
    )
    parser.add_argument("--cb-depth", type=int, default=8)
    parser.add_argument("--cb-learning-rate", type=float, default=0.05)
    parser.add_argument("--cb-iterations", type=int, default=1000)
    parser.add_argument("--cb-l2-leaf-reg", type=float, default=3.0)

    args = parser.parse_args()
    run_experiment(model_type=args.model, args=args)

if __name__ == "__main__":
    main()
