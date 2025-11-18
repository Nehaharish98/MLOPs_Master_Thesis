# src/flows/latency_drift_flow.py

from pathlib import Path

import pandas as pd
from prefect import flow, task
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA_BASE = Path("data/prepared")
REFERENCE_PATH = DATA_BASE / "latency_supervised.parquet"
CURRENT_PATH = DATA_BASE / "latency_supervised.parquet"

OUT_DIR = Path("reports")
OUT_HTML = OUT_DIR / "latency_drift_report.html"


@task(name="load-latency-data")
def load_data():
    reference = pd.read_parquet(REFERENCE_PATH)
    current = reference.sample(frac=0.3, random_state=42).reset_index(drop=True)
    return reference, current


@task(name="build-latency-drift-report")
def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_HTML.open("w", encoding="utf-8") as f:
        report.save_html(f)


    print(f"✅ Latency drift report saved to: {OUT_HTML.resolve()}")
    return str(OUT_HTML.resolve())


@flow(name="latency-drift-monitoring")
def latency_drift_monitoring_flow():
    reference, current = load_data()
    report_path = build_drift_report(reference, current)
    return report_path


if __name__ == "__main__":
    latency_drift_monitoring_flow()
