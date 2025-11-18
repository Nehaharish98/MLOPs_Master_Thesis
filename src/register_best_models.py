# src/register_best_models.py

import mlflow
from mlflow.tracking import MlflowClient

# Map experiment -> registry model name
REGISTER_CONFIG = {
    "throughput": {
        "model_name": "throughput_model",
    },
    "latency": {
        "model_name": "latency_model",
    },
    "classification": {
        "model_name": "classification_model",
    },
    "anomaly_detection": {
        "model_name": "anomaly_detection_model",
    },
}


def find_best_run_id(exp_name: str, client: MlflowClient):
    """Return run_id of the run tagged as is_best=true for this experiment."""
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"Experiment '{exp_name}' not found.")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.is_best = 'true'",
        max_results=1,
    )

    if runs.empty:
        print(f"No run tagged is_best=true in experiment '{exp_name}'.")
        return None

    return runs.iloc[0]["run_id"]


def register_best_models():
    client = MlflowClient()

    for exp_name, cfg in REGISTER_CONFIG.items():
        model_name = cfg["model_name"]
        print(f"\n=== Registering best model for experiment: {exp_name} ===")

        best_run_id = find_best_run_id(exp_name, client)
        if best_run_id is None:
            print(f"Skipping '{exp_name}' (no best run).")
            continue

        model_uri = f"runs:/{best_run_id}/model"
        print(f"Best run id: {best_run_id}")
        print(f"Model URI:   {model_uri}")
        print(f"Registry name: {model_name}")

        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )
        except Exception as e:
            print(f"Failed to register model for '{exp_name}': {e}")
            continue

        print(f"Registered model version: name={result.name}, version={result.version}")


if __name__ == "__main__":
    # Make sure MLFLOW_TRACKING_URI / USER / PASSWORD are set in env
    print("Registering best models into MLflow Model Registry...")
    register_best_models()
    print("\nDone.")
