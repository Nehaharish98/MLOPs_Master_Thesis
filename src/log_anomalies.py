# log_anomalies.py
"""
Anomaly detection logging.

This script assumes anomaly detection has been run offline already and the
following files exist:

- prepared/anomaly_matrix.parquet       : feature matrix used for IsoForest / pairZ.
- prepared/anomaly_scored.parquet       : matrix with anomaly scores and flags per run.
- prepared/anomaly_top10_per_pair.parquet : top-k anomalies per (provider, region_snd_base, region_rcv_base).

The goal here is to log unsupervised anomaly detection results into MLflow,
so that the anomaly task appears alongside the other three tasks in the
experiment tracking UI (DagsHub now, AWS later).

Since anomaly detection is unsupervised, we don’t report accuracy/F1/etc.,
but we do log:

- fraction of runs flagged as anomalous;
- basic statistics of anomaly scores;
- small samples of the scored / top10 data as CSV artifacts.
"""

from pathlib import Path
import mlflow
import pandas as pd
import numpy as np
from .mlflow_setup import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
ANOMALY_MATRIX = DATA_ROOT / "prepared" / "anomaly_matrix.parquet"
ANOMALY_SCORED = DATA_ROOT / "prepared" / "anomaly_scored.parquet"
ANOMALY_TOP10 = DATA_ROOT / "prepared" / "anomaly_top10_per_pair.parquet"


def main():
    print("Setting up MLflow for anomaly task...")
    tracking_uri = setup_mlflow(task_name="anomaly_detection")
    print("MLflow tracking URI:", tracking_uri)

    # Load data
    df_matrix = pd.read_parquet(ANOMALY_MATRIX)
    df_scored = pd.read_parquet(ANOMALY_SCORED)
    df_top10 = pd.read_parquet(ANOMALY_TOP10)

    # Try to detect anomaly flag & score columns automatically
    flag_cols = [
        c for c in df_scored.columns
        if "anomaly" in c.lower() or "flag" in c.lower() or "outlier" in c.lower()
    ]
    score_cols = [c for c in df_scored.columns if "score" in c.lower()]

    flag_col = flag_cols[0] if flag_cols else None
    score_col = score_cols[0] if score_cols else None

    print("Detected flag column:", flag_col)
    print("Detected score column:", score_col)

    with mlflow.start_run():
        mlflow.set_tag("task", "anomaly_detection")

        mlflow.log_param("anomaly_matrix_path", str(ANOMALY_MATRIX))
        mlflow.log_param("anomaly_scored_path", str(ANOMALY_SCORED))
        mlflow.log_param("anomaly_top10_path", str(ANOMALY_TOP10))


        # Log basic dataset sizes
        mlflow.log_metric("rows_matrix", len(df_matrix))
        mlflow.log_metric("rows_scored", len(df_scored))
        mlflow.log_metric("rows_top10", len(df_top10))

        # Fraction of anomalies (if flag column found)
        if flag_col is not None:
            n_rows = len(df_scored)
            n_anom = int(df_scored[flag_col].sum())
            frac_anom = n_anom / n_rows if n_rows > 0 else 0.0
            mlflow.log_metric("n_anomalies", n_anom)
            mlflow.log_metric("frac_anomalies", frac_anom)

                # Score distribution statistics (if score column found)
        if score_col is not None:
            desc = df_scored[score_col].describe()

            stat_map = {
                "mean": "mean",
                "std": "std",
                "min": "min",
                "25%": "q25",
                "50%": "q50",
                "75%": "q75",
                "max": "max",
            }

            for src_name, suffix in stat_map.items():
                if src_name not in desc:
                    continue

                raw = desc[src_name]

                # Try to convert to float
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    # Not a numeric value, skip logging this stat
                    continue

                # Skip NaN / inf because MLflow/DagsHub reject them
                if not np.isfinite(value):
                    continue

                metric_name = f"score_{suffix}"
                mlflow.log_metric(metric_name, value)

        # Log small samples as CSV artifacts for inspection
        # (We don’t log the full data to avoid huge artifacts)
        mlflow.log_text(
            df_top10.head(200).to_csv(index=False),
            artifact_file="anomaly_top10_sample.csv",
        )

        mlflow.log_text(
            df_scored.head(2000).to_csv(index=False),
            artifact_file="anomaly_scored_sample.csv",
        )

        # Log feature column names used for anomaly_matrix (for reproducibility)
        mlflow.log_text(
            "\n".join(df_matrix.columns),
            artifact_file="anomaly_features.txt",
        )

        print("Anomaly summary logged to MLflow.")


if __name__ == "__main__":
    main()
