# src/service/app.py

import os
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pandas.api.types import is_categorical_dtype
from sklearn.ensemble import IsolationForest
from mlflow.tracking import MlflowClient

# -------------------------------------------------------------------
# MLflow setup (cloud-based)
# -------------------------------------------------------------------

# Default to your EC2 MLflow server; can be overridden via env
DEFAULT_TRACKING_URI = "http://ec2-98-92-24-18.compute-1.amazonaws.com:5000/"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_client = MlflowClient()

# Registry model names (from register_best_models.py)
REGISTRY_MODEL_NAMES = {
    "throughput": os.getenv("THROUGHPUT_MODEL_NAME", "throughput_model"),
    "latency": os.getenv("LATENCY_MODEL_NAME", "latency_model"),
    "classification": os.getenv("CLASSIFICATION_MODEL_NAME", "classification_model"),
}

# Which stage to load; for now we use "None" (latest version regardless of stage)
MODEL_STAGE = os.getenv("MODEL_STAGE", "None")  # or "Production" later


def _get_latest_model_uri(model_name: str) -> str:
    """
    Return a model URI pointing to the latest *active* version in the desired stage.

    - Ignores deleted versions (lifecycle_stage == "deleted")
    - If MODEL_STAGE != "None", prefers that stage (e.g. "Production")
    - Otherwise picks the highest active version regardless of stage.
    """
    versions = _client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    # 1) drop deleted versions
    active_versions = [
        v for v in versions
        if getattr(v, "lifecycle_stage", "active") != "deleted"
    ]
    if not active_versions:
        raise RuntimeError(f"No active versions found for model '{model_name}'")

    # 2) filter by stage if requested
    if MODEL_STAGE != "None":
        candidates = [v for v in active_versions if v.current_stage == MODEL_STAGE]
        if not candidates:
            raise RuntimeError(
                f"No versions in stage '{MODEL_STAGE}' for model '{model_name}'"
            )
    else:
        candidates = active_versions

    # 3) pick highest numeric version
    candidates.sort(key=lambda v: int(v.version), reverse=True)
    best = candidates[0]

    print(
        f"[MLflow] Using model '{model_name}' "
        f"version {best.version} (stage='{best.current_stage}', "
        f"lifecycle='{best.lifecycle_stage}')"
    )

    return f"models:/{model_name}/{best.version}"


print("Loading models from MLflow Registry...")

# For now, explicitly load Version 1 for each model.
# You can later switch to /Production or a smarter helper if needed.
THROUGHPUT_MODEL_URI = f"models:/{REGISTRY_MODEL_NAMES['throughput']}/1"
LATENCY_MODEL_URI = f"models:/{REGISTRY_MODEL_NAMES['latency']}/1"
CLASSIF_MODEL_URI = f"models:/{REGISTRY_MODEL_NAMES['classification']}/1"

throughput_model = mlflow.pyfunc.load_model(THROUGHPUT_MODEL_URI)
latency_model = mlflow.pyfunc.load_model(LATENCY_MODEL_URI)
classification_model = mlflow.pyfunc.load_model(CLASSIF_MODEL_URI)
anomaly_model: Optional[Any] = None  # not used, kept for clarity

print("Models loaded from registry.")
print("Anomaly model: local IsolationForest (see below).")


# -------------------------------------------------------------------
# Data sources: S3 for templates and anomaly matrix
# -------------------------------------------------------------------

S3_BUCKET = "thesis-netperf-mlops-2525"
S3_PREPARED_PREFIX = "prepared"

THROUGHPUT_SUPERVISED_PATH = f"s3://{S3_BUCKET}/{S3_PREPARED_PREFIX}/throughput_supervised.parquet"
LATENCY_SUPERVISED_PATH = f"s3://{S3_BUCKET}/{S3_PREPARED_PREFIX}/latency_supervised.parquet"
CLASSIF_SUPERVISED_PATH = f"s3://{S3_BUCKET}/{S3_PREPARED_PREFIX}/classification_supervised.parquet"
ANOMALY_MATRIX_PATH = f"s3://{S3_BUCKET}/{S3_PREPARED_PREFIX}/anomaly_matrix.parquet"


def _load_template(path: str, target_cols: list[str]) -> pd.DataFrame:
    """
    Load a single-row feature template from S3:
    drop targets, 'split' and 'start' timestamp.
    """
    df = pd.read_parquet(path)
    drop_cols = [c for c in target_cols + ["split", "start"] if c in df.columns]
    template = df.drop(columns=drop_cols).iloc[[0]].copy()
    return template


THROUGHPUT_TEMPLATE = _load_template(
    THROUGHPUT_SUPERVISED_PATH,
    target_cols=["tput_mean_next"],
)
LATENCY_TEMPLATE = _load_template(
    LATENCY_SUPERVISED_PATH,
    target_cols=["delay_mean_next"],
)
CLASSIF_TEMPLATE = _load_template(
    CLASSIF_SUPERVISED_PATH,
    target_cols=["label_good"],
)

# -------------------------------------------------------------------
# Anomaly model (local IsolationForest trained on S3 data)
# -------------------------------------------------------------------

try:
    anomaly_df = pd.read_parquet(ANOMALY_MATRIX_PATH)
    # Use only numeric columns for IsolationForest
    ANOMALY_FEATURES = anomaly_df.select_dtypes(include="number").columns.tolist()
    if not ANOMALY_FEATURES:
        raise ValueError("No numeric columns found for anomaly detection")

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.01,  # adjust if you used something else
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(anomaly_df[ANOMALY_FEATURES])

    # Template row: first numeric row
    ANOMALY_TEMPLATE = anomaly_df[ANOMALY_FEATURES].iloc[[0]].copy()

    print(
        f"Anomaly model trained with {len(anomaly_df)} rows "
        f"and {len(ANOMALY_FEATURES)} numeric features."
    )
except Exception as e:
    iso_forest = None
    ANOMALY_TEMPLATE = None
    ANOMALY_FEATURES = []
    print(f"⚠️ Could not initialize anomaly model: {e}")


def build_row_from_payload(payload: Dict[str, Any], template: pd.DataFrame) -> pd.DataFrame:
    """
    Start from a 1-row template with all feature columns,
    overwrite any columns that appear in payload, ignore unknown keys.

    If a column is categorical and the incoming value is a new category,
    extend the categories first to avoid pandas TypeError.
    """
    row = template.copy()
    idx = row.index[0]

    for key, value in payload.items():
        if key in row.columns:
            col = row[key]

            # If this is a pandas Categorical, allow new category values
            if is_categorical_dtype(col):
                if value not in col.cat.categories:
                    col = col.cat.add_categories([value])
                    row[key] = col

            row.at[idx, key] = value

    return row


def build_anomaly_row(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame for anomaly scoring:
    start from ANOMALY_TEMPLATE and overwrite keys present in payload.
    Only numeric ANOMALY_FEATURES are used by the model.
    """
    if ANOMALY_TEMPLATE is None:
        raise RuntimeError("Anomaly template not initialized")

    row = ANOMALY_TEMPLATE.copy()
    idx = row.index[0]
    for key, value in payload.items():
        if key in row.columns:
            row.at[idx, key] = value
    return row


# -------------------------------------------------------------------
# FastAPI app & Prometheus instrumentation
# -------------------------------------------------------------------

app = FastAPI(
    title="Inter-DC Network Performance Service",
    version="0.1.0",
    description="Predict throughput, latency, good/bad label, and anomaly score for inter-datacenter runs.",
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "models": {
            "throughput": THROUGHPUT_MODEL_URI,
            "latency": LATENCY_MODEL_URI,
            "classification": CLASSIF_MODEL_URI,
        },
    }


# -------------------------------------------------------------------
# Debug endpoints (now reading from S3)
# -------------------------------------------------------------------


@app.get("/debug/throughput/{idx}")
def debug_throughput(idx: int = 0):
    df = pd.read_parquet(THROUGHPUT_SUPERVISED_PATH)
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=400, detail=f"idx out of range, got {idx}")

    row = df.iloc[idx]
    X = row.drop(labels=["tput_mean_next", "split", "start"], errors="ignore").to_frame().T
    y_pred = throughput_model.predict(X)[0]

    return {
        "idx": idx,
        "tput_true": float(row.get("tput_mean_next", float("nan"))),
        "tput_pred": float(y_pred),
    }


@app.get("/debug/latency/{idx}")
def debug_latency(idx: int = 0):
    df = pd.read_parquet(LATENCY_SUPERVISED_PATH)
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=400, detail=f"idx out of range, got {idx}")

    row = df.iloc[idx]
    X = row.drop(labels=["delay_mean_next", "split", "start"], errors="ignore").to_frame().T
    y_pred = latency_model.predict(X)[0]

    return {
        "idx": idx,
        "delay_true": float(row.get("delay_mean_next", float("nan"))),
        "delay_pred": float(y_pred),
    }


@app.get("/debug/classification/{idx}")
def debug_classification(idx: int = 0):
    df = pd.read_parquet(CLASSIF_SUPERVISED_PATH)
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=400, detail=f"idx out of range, got {idx}")

    row = df.iloc[idx]
    X = row.drop(labels=["label_good", "split", "start"], errors="ignore").to_frame().T

    pred = classification_model.predict(X)
    # handle both shape (1,) or (1,1)
    if hasattr(pred, "__len__"):
        y = pred[0]
        if isinstance(y, (list, tuple)) or getattr(y, "shape", None) and len(
            getattr(y, "shape", [])
        ) > 0:
            y = y[0]
    else:
        y = pred
    label_pred = int(y)

    return {
        "idx": idx,
        "label_true": int(row.get("label_good", -1)),
        "label_pred": label_pred,
    }


# -------------------------------------------------------------------
# Real prediction endpoints (template-based)
# -------------------------------------------------------------------


@app.post("/predict/throughput")
def predict_throughput(payload: Dict[str, Any] = Body(...)):
    """
    Accepts a JSON dict with any subset of feature columns.
    Missing ones are taken from a template row (so all expected columns exist).
    """
    X = build_row_from_payload(payload, THROUGHPUT_TEMPLATE)
    try:
        y_pred = throughput_model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {e}")
    return {"tput_mean_next_pred": float(y_pred)}


@app.post("/predict/latency")
def predict_latency(payload: Dict[str, Any] = Body(...)):
    X = build_row_from_payload(payload, LATENCY_TEMPLATE)
    try:
        y_pred = latency_model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {e}")
    return {"delay_mean_next_pred": float(y_pred)}


@app.post("/classify/performance")
def classify_performance(payload: Dict[str, Any] = Body(...)):
    X = build_row_from_payload(payload, CLASSIF_TEMPLATE)
    try:
        pred = classification_model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {e}")

    if hasattr(pred, "__len__"):
        y = pred[0]
        if isinstance(y, (list, tuple)) or getattr(y, "shape", None) and len(
            getattr(y, "shape", [])
        ) > 0:
            y = y[0]
    else:
        y = pred

    label = int(y)

    return {
        "label_good": label
        # you can add probabilities later if your logged model exposes them
    }


@app.post("/score/anomaly")
def score_anomaly(payload: Dict[str, Any] = Body(...)):
    """
    Compute anomaly score using an IsolationForest trained on anomaly_matrix.parquet.
    Returns:
      - anomaly_score (higher = more normal in sklearn's decision_function)
      - anomaly_flag (1 = anomaly, 0 = normal)
    """
    if iso_forest is None or ANOMALY_TEMPLATE is None or not ANOMALY_FEATURES:
        raise HTTPException(
            status_code=503,
            detail="Anomaly model not available (initialization failed).",
        )

    row = build_anomaly_row(payload)
    X = row[ANOMALY_FEATURES]

    try:
        # IsolationForest:
        # - predict() returns -1 for anomaly, 1 for normal
        # - decision_function() returns a score (higher is less anomalous)
        score = iso_forest.decision_function(X)[0]
        pred = iso_forest.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Anomaly scoring failed: {e}")

    anomaly_flag = 1 if pred == -1 else 0

    return {
        "anomaly_score": float(score),
        "anomaly_flag": anomaly_flag,
    }
