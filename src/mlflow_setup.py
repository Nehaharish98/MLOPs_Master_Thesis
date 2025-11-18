# mlflow_setup.py
"""
Shared MLflow setup for this thesis project.

This file centralizes how we connect to the tracking backend (DagsHub now,
AWS later). Training scripts call `setup_mlflow(task_name=...)` to:

- Use the tracking URI from the environment variable MLFLOW_TRACKING_URI.
- Set the experiment name to the current task:
    "throughput", "latency", "classification", "anomaly_detection"

For DagsHub, set in your shell:

    export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
    export MLFLOW_TRACKING_USERNAME="<your_username>"
    export MLFLOW_TRACKING_PASSWORD="<your_token>"

For AWS later, you’ll change MLFLOW_TRACKING_URI to your EC2 MLflow server URL.
"""

import os
import mlflow


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
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set.\n"
            "For DagsHub, set for example:\n"
            "  export MLFLOW_TRACKING_URI=''\n"
            "  export MLFLOW_TRACKING_USERNAME=''\n"
            "  export MLFLOW_TRACKING_PASSWORD=''"
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(task_name)
    return tracking_uri
