# src/service/app.py

import os
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from sklearn.ensemble import IsolationForest

# --------- MLflow setup ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# --------- Load models from registry ----------
# Use specific versions or "Production" depending on how you manage them
THROUGHPUT_MODEL_URI = "models:/throughput_model/2"
LATENCY_MODEL_URI = "models:/latency_model/2"
CLASSIF_MODEL_URI = "models:/classification_model/2"

print("Loading models from MLflow Registry...")
throughput_model = mlflow.pyfunc.load_model(THROUGHPUT_MODEL_URI)
latency_model = mlflow.pyfunc.load_model(LATENCY_MODEL_URI)
classification_model = mlflow.pyfunc.load_model(CLASSIF_MODEL_URI)
anomaly_model: Optional[Any] = None  # not used, kept for clarity
print("Models loaded.")
print("Models loaded (anomaly model disabled via MLflow; local IF used instead).")

# --------- Load feature templates from parquet ----------
DATA_BASE = "data/prepared"


def _load_template(path: str, target_cols: list[str]) -> pd.DataFrame:
    """
    Load a single-row feature template:
    drop targets, 'split' and 'start' timestamp.
    """
    df = pd.read_parquet(path)
    drop_cols = [c for c in target_cols + ["split", "start"] if c in df.columns]
    template = df.drop(columns=drop_cols).iloc[[0]].copy()
    return template


THROUGHPUT_TEMPLATE = _load_template(
    os.path.join(DATA_BASE, "throughput_supervised.parquet"),
    target_cols=["tput_mean_next"],
)
LATENCY_TEMPLATE = _load_template(
    os.path.join(DATA_BASE, "latency_supervised.parquet"),
    target_cols=["delay_mean_next"],
)
CLASSIF_TEMPLATE = _load_template(
    os.path.join(DATA_BASE, "classification_supervised.parquet"),
    target_cols=["label_good"],
)

# --------- Anomaly model (local IsolationForest) ----------

ANOMALY_MATRIX_PATH = os.path.join(DATA_BASE, "anomaly_matrix.parquet")

try:
    anomaly_df = pd.read_parquet(ANOMALY_MATRIX_PATH)
    # Use only numeric columns for IsolationForest
    ANOMALY_FEATURES = anomaly_df.select_dtypes(include="number").columns.tolist()
    if not ANOMALY_FEATURES:
        raise ValueError("No numeric columns found for anomaly detection")

    # Train IsolationForest once at startup
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
    """
    row = template.copy()
    idx = row.index[0]
    for key, value in payload.items():
        if key in row.columns:
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


# --------- FastAPI app ----------
app = FastAPI(
    title="Inter-DC Network Performance Service",
    version="0.1.0",
    description="Predict throughput, latency, good/bad label, and anomaly score for inter-datacenter runs.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


# --------- Debug endpoints (use REAL data to test) ----------


@app.get("/debug/throughput/{idx}")
def debug_throughput(idx: int = 0):
    df = pd.read_parquet(os.path.join(DATA_BASE, "throughput_supervised.parquet"))
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
    df = pd.read_parquet(os.path.join(DATA_BASE, "latency_supervised.parquet"))
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
    df = pd.read_parquet(os.path.join(DATA_BASE, "classification_supervised.parquet"))
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=400, detail=f"idx out of range, got {idx}")

    row = df.iloc[idx]
    # Drop label, split AND start (like in training)
    X = row.drop(labels=["label_good", "split", "start"], errors="ignore").to_frame().T

    # CatBoost via MLflow pyfunc -> use predict() only
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


# --------- Real prediction endpoints (template-based) ----------


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

    # Same handling as in debug: extract scalar
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
        # you can add "proba_good": None or compute it later if needed
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

    # Build a full feature row from template + payload
    row = build_anomaly_row(payload)

    # Use only the numeric features we trained on
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
