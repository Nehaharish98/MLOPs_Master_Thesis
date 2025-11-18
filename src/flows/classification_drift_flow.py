# src/flows/classification_drift_flow.py

from pathlib import Path

import pandas as pd
from prefect import flow, task
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA_BASE = Path("data/prepared")
REFERENCE_PATH = DATA_BASE / "classification_supervised.parquet"
CURRENT_PATH = DATA_BASE / "classification_supervised.parquet"

OUT_DIR = Path("reports")
OUT_HTML = OUT_DIR / "classification_drift_report.html"


@task(name="load-classification-data")
def load_data():
    df = pd.read_parquet(REFERENCE_PATH)

    # Reference: full data
    reference = df.copy()

    # Current: 30% of data (simulate)
    current = df.sample(frac=0.3, random_state=42).reset_index(drop=True)

    # Drop label for pure feature drift
    if "label_good" in reference.columns:
        reference = reference.drop(columns=["label_good"])
        current = current.drop(columns=["label_good"])

    return reference, current


@task(name="build-classification-drift-report")
def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_HTML.open("w", encoding="utf-8") as f:
        report.save_html(f)


    print(f"✅ Classification drift report saved to: {OUT_HTML.resolve()}")
    return str(OUT_HTML.resolve())


@flow(name="classification-drift-monitoring")
def classification_drift_monitoring_flow():
    reference, current = load_data()
    report_path = build_drift_report(reference, current)
    return report_path


if __name__ == "__main__":
    classification_drift_monitoring_flow()
