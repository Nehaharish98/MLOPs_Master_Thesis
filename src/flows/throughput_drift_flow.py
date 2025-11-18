# src/flows/throughput_drift_flow.py

from pathlib import Path

import pandas as pd
from prefect import flow, task
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA_BASE = Path("data/prepared")
REFERENCE_PATH = DATA_BASE / "throughput_supervised.parquet"
CURRENT_PATH = DATA_BASE / "throughput_supervised.parquet"  # placeholder for now

OUT_DIR = Path("reports")
OUT_HTML = OUT_DIR / "throughput_drift_report.html"


@task(name="load-throughput-data")
def load_data():
    """
    Load reference + current data for throughput drift analysis.
    """
    reference = pd.read_parquet(REFERENCE_PATH)

    # Simulate "current" as 30% of reference for now.
    current = reference.sample(frac=0.3, random_state=42).reset_index(drop=True)

    return reference, current


@task(name="build-throughput-drift-report")
def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> str:
    """
    Build an Evidently data drift report and save it as HTML.
    NOTE: Evidently expects a file-like object, so we open the file ourselves.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 🔧 FIX: open the file and pass the handle to save_html
    with OUT_HTML.open("w", encoding="utf-8") as f:
        report.save_html(f)

    print(f"✅ Throughput drift report saved to: {OUT_HTML.resolve()}")
    return str(OUT_HTML.resolve())


@flow(name="throughput-drift-monitoring")
def throughput_drift_monitoring_flow():
    """
    Prefect flow for throughput data drift monitoring.
    """
    reference, current = load_data()
    report_path = build_drift_report(reference, current)
    return report_path


if __name__ == "__main__":
    throughput_drift_monitoring_flow()
