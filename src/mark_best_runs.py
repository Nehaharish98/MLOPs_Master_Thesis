# src/mark_best_runs.py

import mlflow
from mlflow.tracking import MlflowClient

# Which experiment to optimize on which metric
EXPERIMENTS = {
    "throughput": {
        "metric": "test_r2",      # maximize
        "maximize": True,
    },
    "latency": {
        "metric": "test_r2",      # or "test_rmse" with maximize=False if you prefer
        "maximize": True,
    },
    "classification": {
        "metric": "test_roc_auc", # maximize
        "maximize": True,
    },
    "anomaly_detection": {
        "metric": "rows_scored",  # or "rows_matrix" or "n_anomalies", something simple
        "maximize": True,
    },
}


def mark_best_runs():
    client = MlflowClient()

    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n=== Processing experiment: {exp_name} ===")
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"Experiment '{exp_name}' not found, skipping.")
            continue

        metric = cfg["metric"]
        maximize = cfg["maximize"]

        # Order runs by metric directly in MLflow
        order = "DESC" if maximize else "ASC"
        order_by = [f"metrics.{metric} {order}"]

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=order_by,
            max_results=1,
        )

        if runs.empty:
            print(f"No runs found in experiment '{exp_name}', skipping.")
            continue

        best = runs.iloc[0]
        best_run_id = best["run_id"]
        best_value = best[f"metrics.{metric}"]

        print(f"Best run for '{exp_name}': {best_run_id}")
        print(f"  {metric} = {best_value:.4f}")

        # Tag this run as the champion
        client.set_tag(best_run_id, "is_best", "true")
        client.set_tag(best_run_id, "best_metric", metric)
        client.set_tag(best_run_id, "stage", "champion")
        client.set_tag(best_run_id, "selected_by", "mark_best_runs.py")


if __name__ == "__main__":
    # Make sure MLFLOW_TRACKING_URI / USER / PASSWORD are set in env
    print("Marking best runs per experiment based on test metrics...")
    mark_best_runs()
    print("\nDone.")