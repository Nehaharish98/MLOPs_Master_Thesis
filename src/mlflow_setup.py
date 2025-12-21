# src/mlflow_setup.py
"""
Shared MLflow setup for this thesis project.

This file centralizes how we connect to the tracking backend.

Training scripts call `setup_mlflow(task_name=...)` to:

- Configure the MLflow tracking URI (either from MLFLOW_TRACKING_URI or a
  default EC2 URL).
- Set the experiment name to the current task:
    "throughput", "latency", "classification", "anomaly_detection".

Usage:

    # Option 1: explicit env var (recommended for flexibility)
    export MLFLOW_TRACKING_URI="http://ec2-44-200-25-155.compute-1.amazonaws.com:5000"

    # Option 2: no env var set
    #   -> setup_mlflow() will fall back to the default EC2 URL below.
"""

import os
import mlflow

# Default tracking URI for the EC2-hosted MLflow server
DEFAULT_TRACKING_URI = "http://ec2-44-200-25-155.compute-1.amazonaws.com:5000"


def setup_mlflow(task_name: str) -> str:
    """
    Configure MLflow tracking and experiment name.

    Parameters
    ----------
    task_name : str
        Name of the MLflow experiment / logical task
        (e.g. "throughput", "latency", "classification", "anomaly_detection").

    Returns
    -------
    tracking_uri : str
        The MLflow tracking URI being used.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(task_name)

    return tracking_uri
