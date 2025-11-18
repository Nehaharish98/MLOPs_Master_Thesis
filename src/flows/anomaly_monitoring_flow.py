# src/flows/anomaly_monitoring_flow.py

from pathlib import Path
from datetime import datetime

import pandas as pd
from prefect import flow, task
from sklearn.ensemble import IsolationForest

DATA_BASE = Path("data/prepared")
ANOMALY_MATRIX_PATH = DATA_BASE / "anomaly_matrix.parquet"

OUT_DIR = Path("reports")
OUT_CSV = OUT_DIR / "anomaly_monitoring_summary.csv"


@task(name="load-anomaly-data")
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ANOMALY_MATRIX_PATH)
    return df


@task(name="compute-anomaly-stats")
def compute_stats(df: pd.DataFrame) -> dict:
    # Use numeric columns only
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found for anomaly monitoring")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(df[numeric_cols])

    scores = model.decision_function(df[numeric_cols])
    preds = model.predict(df[numeric_cols])  # -1 = anomaly, 1 = normal

    anomaly_flags = (preds == -1).astype(int)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(df),
        "n_anomalies": int(anomaly_flags.sum()),
        "fraction_anomalies": float(anomaly_flags.mean()),
        "score_mean": float(scores.mean()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }

    print(f"✅ Anomaly monitoring summary: {summary}")
    return summary


@task(name="save-anomaly-summary")
def save_summary(summary: dict) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Append to CSV (create if doesn’t exist)
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        new_row = pd.DataFrame([summary])
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(OUT_CSV, index=False)
    else:
        pd.DataFrame([summary]).to_csv(OUT_CSV, index=False)

    print(f"✅ Anomaly monitoring summary written to: {OUT_CSV.resolve()}")
    return str(OUT_CSV.resolve())


@flow(name="anomaly-monitoring")
def anomaly_monitoring_flow():
    df = load_data()
    summary = compute_stats(df)
    out_path = save_summary(summary)
    return out_path


if __name__ == "__main__":
    anomaly_monitoring_flow()
