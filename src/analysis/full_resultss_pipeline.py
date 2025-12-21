"""
Full results pipeline for thesis – ALL TASKS

What this script does:

1. Throughput regression
   - Loads local splits: data/splits/throughput/train|test.parquet
   - Target: tput_mean_next
   - Baselines:
       * per-path mean (baseline_mean)
       * per-path last observed value (baseline_last)
   - ML model: loaded from MLflow Model Registry
   - Outputs:
       * reports/results/throughput_global_metrics.csv
       * reports/results/throughput_per_region_metrics.csv
       * reports/figures/throughput_global_mae.png
       * reports/figures/throughput_top10_region_improvement.png

2. Latency regression
   - Same structure as throughput:
       * data/splits/latency/train|test.parquet
       * Target: delay_mean_next
   - Same baselines + ML model from registry
   - Outputs:
       * latency_global_metrics.csv / latency_per_region_metrics.csv
       * latency_global_mae.png / latency_top10_region_improvement.png

3. Classification (good/bad performance)
   - data/splits/classification/train|test.parquet
   - Target: label_good (0/1)
   - Baselines:
       * global majority class
       * per-path majority class
   - ML model: loaded from MLflow Model Registry
   - Metrics: Accuracy, F1
   - Outputs:
       * classification_global_metrics.csv
       * classification_per_region_metrics.csv
       * classification_global_accuracy.png
       * classification_top10_region_improvement.png

4. Anomaly detection (unsupervised)
   - data/prepared/anomaly_scored.parquet
   - Summaries:
       * global anomaly rate, score distribution
       * per-region-pair anomaly rate
   - Outputs:
       * anomaly_global_summary.csv
       * anomaly_per_region_summary.csv
       * anomaly_score_hist.png
       * anomaly_top10_region_anomaly_rate.png

Before running, make sure:
- MLFLOW_TRACKING_URI is set (localhost tunnel to EC2 or direct EC2 URL)
- Model URIs below match your registry model names/versions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
)
from pandas.api.types import (
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_bool_dtype,
)

from src.mlflow_setup import setup_mlflow


# ============================================================
# ====================== PATH CONFIG =========================
# ============================================================

# This file is src/analysis/full_resultss_pipeline.py
# parents[0] = .../src/analysis
# parents[1] = .../src
# parents[2] = .../Thesis-mlops (repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
SPLITS_DIR = DATA_ROOT / "splits"
PREPARED_DIR = DATA_ROOT / "prepared"

RESULTS_DIR = REPO_ROOT / "reports" / "results"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REGION_COLS: List[str] = ["region_snd_base", "region_rcv_base"]
TIME_COL: str = "start"

# Model URIs – override via env vars if needed
THROUGHPUT_MODEL_URI = os.getenv("THROUGHPUT_MODEL_URI", "models:/throughput_model/2")
LATENCY_MODEL_URI = os.getenv("LATENCY_MODEL_URI", "models:/latency_model/2")
CLASSIF_MODEL_URI = os.getenv("CLASSIF_MODEL_URI", "models:/classification_model/2")

TARGET_THROUGHPUT = "tput_mean_next"
TARGET_LATENCY = "delay_mean_next"
TARGET_CLASSIF = "label_good"


# ============================================================
# =================== SHARED UTILITIES =======================
# ============================================================

def ensure_mlflow_tracking() -> str:
    """Configure MLflow tracking using same logic as training scripts."""
    try:
        tracking_uri = setup_mlflow(task_name="throughput")
    except RuntimeError as e:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Example:\n"
            "  export MLFLOW_TRACKING_URI='http://localhost:5000'\n"
        ) from e

    print(f"[MLflow] Using tracking URI: {tracking_uri}")
    return tracking_uri


def add_region_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'region_pair' column like 'eu-west-1->us-east-1'."""
    df = df.copy()
    df["region_pair"] = df[REGION_COLS].astype(str).agg("->".join, axis=1)
    return df


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def regression_metrics_by_region(
    df: pd.DataFrame,
    target_col: str,
    pred_cols: Dict[str, str],
) -> pd.DataFrame:
    df = add_region_pair(df)
    rows = []
    for pair, g in df.groupby("region_pair"):
        y_true = g[target_col].values
        for name, col in pred_cols.items():
            y_pred = g[col].values
            m = compute_regression_metrics(y_true, y_pred)
            m.update({"region_pair": pair, "model": name, "n_samples": len(g)})
            rows.append(m)
    return pd.DataFrame(rows)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"accuracy": acc, "f1": f1}


def classification_metrics_by_region(
    df: pd.DataFrame,
    target_col: str,
    pred_cols: Dict[str, str],
) -> pd.DataFrame:
    df = add_region_pair(df)
    rows = []
    for pair, g in df.groupby("region_pair"):
        y_true = g[target_col].values
        for name, col in pred_cols.items():
            y_pred = g[col].values
            m = compute_classification_metrics(y_true, y_pred)
            m.update({"region_pair": pair, "model": name, "n_samples": len(g)})
            rows.append(m)
    return pd.DataFrame(rows)


def make_feature_matrix(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop target + split + start; keep everything else as features."""
    drop_cols = [target_col, "split", "start"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


# ============================================================
# ============== THROUGHPUT & LATENCY BASELINES ==============
# ============================================================

def per_path_mean_baseline(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.Series:
    train = add_region_pair(train)
    test = add_region_pair(test)
    per_path_mean = train.groupby("region_pair")[target_col].mean().rename("baseline_mean")
    test = test.join(per_path_mean, on="region_pair")
    return test["baseline_mean"]


def per_path_last_value_baseline(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.Series:
    train = add_region_pair(train).sort_values(["region_pair", TIME_COL])
    last_vals = train.groupby("region_pair")[target_col].last().rename("baseline_last")
    test = add_region_pair(test)
    test = test.join(last_vals, on="region_pair")
    return test["baseline_last"]


# ============================================================
# ================== CLASSIFICATION BASELINES =================
# ============================================================

def classification_global_majority_baseline(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.Series:
    majority_label = int(round(train[target_col].mean()))
    return pd.Series(majority_label, index=test.index, name="baseline_global_majority")


def classification_per_path_majority_baseline(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.Series:
    train = add_region_pair(train)
    test = add_region_pair(test)

    per_pair_mean = train.groupby("region_pair")[target_col].mean()
    per_pair_majority = (per_pair_mean >= 0.5).astype(int).rename("baseline_per_path_majority")

    test = test.join(per_pair_majority, on="region_pair")
    return test["baseline_per_path_majority"]


def convert_pyfunc_classification_output_to_labels(pred) -> np.ndarray:
    arr = np.asarray(pred)

    if arr.ndim == 2 and arr.shape[1] == 2:
        proba_1 = arr[:, 1]
        return (proba_1 >= 0.5).astype(int)

    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]

    if arr.ndim == 1:
        if np.issubdtype(arr.dtype, np.floating) and np.all((arr >= 0.0) & (arr <= 1.0)):
            return (arr >= 0.5).astype(int)
        return arr.astype(int)

    arr = arr.ravel()
    return (arr >= 0.5).astype(int)


# ============================================================
# ================= THROUGHPUT EVALUATION ====================
# ============================================================

def evaluate_throughput():
    task_name = "throughput"
    target_col = TARGET_THROUGHPUT

    train_path = SPLITS_DIR / "throughput" / "train.parquet"
    test_path = SPLITS_DIR / "throughput" / "test.parquet"

    print(f"\n===== Evaluating regression task: {task_name} =====")
    print(f"Reading train from: {train_path}")
    print(f"Reading test  from: {test_path}")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    print("Computing baselines (per-path mean, per-path last value)...")
    test["baseline_mean"] = per_path_mean_baseline(train, test, target_col)
    test["baseline_last"] = per_path_last_value_baseline(train, test, target_col)

    print(f"Loading MLflow model: {THROUGHPUT_MODEL_URI}")
    model = mlflow.pyfunc.load_model(THROUGHPUT_MODEL_URI)
    X_test = make_feature_matrix(test, target_col)
    print(f"X_test shape: {X_test.shape}")

    test["pred_ml"] = model.predict(X_test)

    y_true = test[target_col].values

    rows = []
    for name, col in {
        "baseline_mean": "baseline_mean",
        "baseline_last": "baseline_last",
        "ml_model": "pred_ml",
    }.items():
        m = compute_regression_metrics(y_true, test[col].values)
        m.update({"model": name, "task": task_name})
        rows.append(m)

    global_df = pd.DataFrame(rows)
    global_path = RESULTS_DIR / f"{task_name}_global_metrics.csv"
    global_df.to_csv(global_path, index=False)
    print(f"✅ Saved global metrics to {global_path}")

    per_region = regression_metrics_by_region(
        test,
        target_col=target_col,
        pred_cols={
            "baseline_mean": "baseline_mean",
            "baseline_last": "baseline_last",
            "ml_model": "pred_ml",
        },
    )
    per_region["task"] = task_name
    per_region_path = RESULTS_DIR / f"{task_name}_per_region_metrics.csv"
    per_region.to_csv(per_region_path, index=False)
    print(f"✅ Saved per-region metrics to {per_region_path}")

    fig, ax = plt.subplots()
    ax.bar(global_df["model"], global_df["mae"])
    ax.set_title(f"{task_name.capitalize()} – Global MAE (lower is better)")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"{task_name}_global_mae.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved figure: {fig_path}")

    per_region_pivot = per_region.pivot(index="region_pair", columns="model", values="mae")
    per_region_pivot = per_region_pivot.dropna(subset=["baseline_mean", "ml_model"])

    per_region_pivot["relative_improvement_ml_vs_mean"] = (
        (per_region_pivot["baseline_mean"] - per_region_pivot["ml_model"])
        / per_region_pivot["baseline_mean"]
    )

    top10 = (
        per_region_pivot["relative_improvement_ml_vs_mean"]
        .sort_values(ascending=False)
        .head(10)
    )

    if len(top10) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        top10[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("Relative MAE improvement vs per-path mean")
        ax.set_title(f"{task_name.capitalize()} – Top 10 region pairs (ML vs baseline)")
        fig.tight_layout()
        fig_path = FIGURES_DIR / f"{task_name}_top10_region_improvement.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"✅ Saved figure: {fig_path}")
    else:
        print("⚠️ No valid region-pair rows for relative improvement plot.")


# ============================================================
# ================== LATENCY EVALUATION ======================
# ============================================================

def evaluate_latency():
    task_name = "latency"
    target_col = TARGET_LATENCY

    train_path = SPLITS_DIR / "latency" / "train.parquet"
    test_path = SPLITS_DIR / "latency" / "test.parquet"

    print(f"\n===== Evaluating regression task: {task_name} =====")
    print(f"Reading train from: {train_path}")
    print(f"Reading test  from: {test_path}")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    print("Computing baselines (per-path mean, per-path last value)...")
    test["baseline_mean"] = per_path_mean_baseline(train, test, target_col)
    test["baseline_last"] = per_path_last_value_baseline(train, test, target_col)

    print(f"Loading MLflow model: {LATENCY_MODEL_URI}")
    model = mlflow.pyfunc.load_model(LATENCY_MODEL_URI)
    X_test = make_feature_matrix(test, target_col)
    print(f"X_test shape: {X_test.shape}")

    test["pred_ml"] = model.predict(X_test)

    y_true = test[target_col].values

    rows = []
    for name, col in {
        "baseline_mean": "baseline_mean",
        "baseline_last": "baseline_last",
        "ml_model": "pred_ml",
    }.items():
        m = compute_regression_metrics(y_true, test[col].values)
        m.update({"model": name, "task": task_name})
        rows.append(m)

    global_df = pd.DataFrame(rows)
    global_path = RESULTS_DIR / f"{task_name}_global_metrics.csv"
    global_df.to_csv(global_path, index=False)
    print(f"✅ Saved global metrics to {global_path}")

    per_region = regression_metrics_by_region(
        test,
        target_col=target_col,
        pred_cols={
            "baseline_mean": "baseline_mean",
            "baseline_last": "baseline_last",
            "ml_model": "pred_ml",
        },
    )
    per_region["task"] = task_name
    per_region_path = RESULTS_DIR / f"{task_name}_per_region_metrics.csv"
    per_region.to_csv(per_region_path, index=False)
    print(f"✅ Saved per-region metrics to {per_region_path}")

    fig, ax = plt.subplots()
    ax.bar(global_df["model"], global_df["mae"])
    ax.set_title(f"{task_name.capitalize()} – Global MAE (lower is better)")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"{task_name}_global_mae.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved figure: {fig_path}")

    per_region_pivot = per_region.pivot(index="region_pair", columns="model", values="mae")
    per_region_pivot = per_region_pivot.dropna(subset=["baseline_mean", "ml_model"])

    per_region_pivot["relative_improvement_ml_vs_mean"] = (
        (per_region_pivot["baseline_mean"] - per_region_pivot["ml_model"])
        / per_region_pivot["baseline_mean"]
    )

    top10 = (
        per_region_pivot["relative_improvement_ml_vs_mean"]
        .sort_values(ascending=False)
        .head(10)
    )

    if len(top10) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        top10[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("Relative MAE improvement vs per-path mean")
        ax.set_title(f"{task_name.capitalize()} – Top 10 region pairs (ML vs baseline)")
        fig.tight_layout()
        fig_path = FIGURES_DIR / f"{task_name}_top10_region_improvement.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"✅ Saved figure: {fig_path}")
    else:
        print("⚠️ No valid region-pair rows for relative improvement plot.")


# ============================================================
# ================ CLASSIFICATION EVALUATION =================
# ============================================================

def evaluate_classification():
    task_name = "classification"
    target_col = TARGET_CLASSIF

    train_path = SPLITS_DIR / "classification" / "train.parquet"
    test_path = SPLITS_DIR / "classification" / "test.parquet"

    print(f"\n===== Evaluating classification task: {task_name} =====")
    print(f"Reading train from: {train_path}")
    print(f"Reading test  from: {test_path}")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    y_true = test[target_col].values

    print("Computing baselines (global majority, per-path majority)...")
    test["baseline_global_majority"] = classification_global_majority_baseline(
        train, test, target_col
    )
    test["baseline_per_path_majority"] = classification_per_path_majority_baseline(
        train, test, target_col
    )

    print(f"Loading MLflow model: {CLASSIF_MODEL_URI}")
    model = mlflow.pyfunc.load_model(CLASSIF_MODEL_URI)

    # Build feature matrix (drop label_good, split, start)
    X_test = make_feature_matrix(test, target_col)
    print(f"X_test shape: {X_test.shape}")

    # --- IMPORTANT: clean categoricals exactly like in train_classification.py ---
    cat_cols = [
        c for c in X_test.columns
        if (
            is_object_dtype(X_test[c])
            or is_categorical_dtype(X_test[c])
            or is_string_dtype(X_test[c])
            or is_bool_dtype(X_test[c])
        )
    ]

    for c in cat_cols:
        X_test[c] = X_test[c].astype("string").fillna("__MISSING__")

    # (Optional but safe) fill numeric NaNs with column medians
    num_cols = [c for c in X_test.columns if c not in cat_cols]
    if num_cols:
        X_test[num_cols] = X_test[num_cols].fillna(X_test[num_cols].median())

    # Now call the CatBoost model
    raw_pred = model.predict(X_test)
    test["pred_ml"] = convert_pyfunc_classification_output_to_labels(raw_pred)


    rows = []
    for name, col in {
        "baseline_global_majority": "baseline_global_majority",
        "baseline_per_path_majority": "baseline_per_path_majority",
        "ml_model": "pred_ml",
    }.items():
        m = compute_classification_metrics(y_true, test[col].values)
        m.update({"model": name, "task": task_name})
        rows.append(m)

    global_df = pd.DataFrame(rows)
    global_path = RESULTS_DIR / f"{task_name}_global_metrics.csv"
    global_df.to_csv(global_path, index=False)
    print(f"✅ Saved global metrics to {global_path}")

    per_region = classification_metrics_by_region(
        test,
        target_col=target_col,
        pred_cols={
            "baseline_global_majority": "baseline_global_majority",
            "baseline_per_path_majority": "baseline_per_path_majority",
            "ml_model": "pred_ml",
        },
    )
    per_region["task"] = task_name
    per_region_path = RESULTS_DIR / f"{task_name}_per_region_metrics.csv"
    per_region.to_csv(per_region_path, index=False)
    print(f"✅ Saved per-region metrics to {per_region_path}")

    fig, ax = plt.subplots()
    ax.bar(global_df["model"], global_df["accuracy"])
    ax.set_title(f"{task_name.capitalize()} – Global Accuracy")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"{task_name}_global_accuracy.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved figure: {fig_path}")

    per_region_pivot = per_region.pivot(
        index="region_pair", columns="model", values="accuracy"
    )
    per_region_pivot = per_region_pivot.dropna(
        subset=["baseline_per_path_majority", "ml_model"]
    )

    per_region_pivot["accuracy_gain_ml_vs_per_path"] = (
        per_region_pivot["ml_model"] - per_region_pivot["baseline_per_path_majority"]
    )

    top10 = (
        per_region_pivot["accuracy_gain_ml_vs_per_path"]
        .sort_values(ascending=False)
        .head(10)
    )

    if len(top10) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        top10[::-1].plot(kind="barh", ax=ax)
        ax.set_xlabel("Accuracy gain vs per-path majority")
        ax.set_title(f"{task_name.capitalize()} – Top 10 region pairs (ML vs baseline)")
        fig.tight_layout()
        fig_path = FIGURES_DIR / f"{task_name}_top10_region_improvement.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"✅ Saved figure: {fig_path}")
    else:
        print("⚠️ No valid region-pair rows for accuracy gain plot.")


# ============================================================
# ================== ANOMALY SUMMARY =========================
# ============================================================

def summarize_anomaly():
    task_name = "anomaly_detection"

    scored_path = PREPARED_DIR / "anomaly_scored.parquet"
    top10_path = PREPARED_DIR / "anomaly_top10_per_pair.parquet"

    print(f"\n===== Summarising anomaly detection task: {task_name} =====")
    print(f"Reading scored data from: {scored_path}")

    df_scored = pd.read_parquet(scored_path)

    if top10_path.exists():
        df_top10 = pd.read_parquet(top10_path)
        print(f"Reading top-10 anomalies from: {top10_path}")
    else:
        df_top10 = None

    flag_cols = [
        c for c in df_scored.columns
        if "anomaly" in c.lower() or "flag" in c.lower() or "outlier" in c.lower()
    ]
    score_cols = [c for c in df_scored.columns if "score" in c.lower()]

    flag_col = flag_cols[0] if flag_cols else None
    score_col = score_cols[0] if score_cols else None

    print(f"Detected flag column: {flag_col}")
    print(f"Detected score column: {score_col}")

    rows_global = []
    rows_global.append({"metric": "rows_scored", "value": float(len(df_scored))})
    if df_top10 is not None:
        rows_global.append({"metric": "rows_top10", "value": float(len(df_top10))})

    if flag_col is not None:
        n_anom = int(df_scored[flag_col].sum())
        frac_anom = n_anom / len(df_scored) if len(df_scored) > 0 else 0.0
        rows_global.append({"metric": "n_anomalies", "value": float(n_anom)})
        rows_global.append({"metric": "frac_anomalies", "value": float(frac_anom)})

    if score_col is not None:
        desc = df_scored[score_col].describe()
        stat_map = {
            "mean": "score_mean",
            "std": "score_std",
            "min": "score_min",
            "25%": "score_q25",
            "50%": "score_q50",
            "75%": "score_q75",
            "max": "score_max",
        }
        for src_name, metric_name in stat_map.items():
            if src_name in desc:
                rows_global.append(
                    {"metric": metric_name, "value": float(desc[src_name])}
                )

    global_df = pd.DataFrame(rows_global)
    global_path = RESULTS_DIR / "anomaly_global_summary.csv"
    global_df.to_csv(global_path, index=False)
    print(f"✅ Saved global anomaly summary to {global_path}")

    if all(c in df_scored.columns for c in REGION_COLS):
        df_scored = add_region_pair(df_scored)
        rows_region = []
        for pair, g in df_scored.groupby("region_pair"):
            row = {
                "region_pair": pair,
                "n_samples": len(g),
            }
            if flag_col is not None:
                n_anom = int(g[flag_col].sum())
                frac_anom = n_anom / len(g) if len(g) > 0 else 0.0
                row["n_anomalies"] = n_anom
                row["frac_anomalies"] = frac_anom
            if score_col is not None:
                row["score_mean"] = float(g[score_col].mean())
            rows_region.append(row)

        per_region_df = pd.DataFrame(rows_region)
        per_region_path = RESULTS_DIR / "anomaly_per_region_summary.csv"
        per_region_df.to_csv(per_region_path, index=False)
        print(f"✅ Saved per-region anomaly summary to {per_region_path}")

        if score_col is not None:
            fig, ax = plt.subplots()
            g = df_scored[score_col].dropna()
            ax.hist(g, bins=30)
            ax.set_title("Anomaly scores distribution")
            ax.set_xlabel(score_col)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig_path = FIGURES_DIR / "anomaly_score_hist.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"✅ Saved figure: {fig_path}")

        if flag_col is not None and "frac_anomalies" in per_region_df.columns:
            top10 = (
                per_region_df.sort_values("frac_anomalies", ascending=False)
                .head(10)
                .set_index("region_pair")["frac_anomalies"]
            )
            if len(top10) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                top10[::-1].plot(kind="barh", ax=ax)
                ax.set_xlabel("Anomaly rate")
                ax.set_title("Top 10 region pairs by anomaly rate")
                fig.tight_layout()
                fig_path = FIGURES_DIR / "anomaly_top10_region_anomaly_rate.png"
                fig.savefig(fig_path, dpi=200)
                plt.close(fig)
                print(f"✅ Saved figure: {fig_path}")
    else:
        print("⚠️ Region columns not found in anomaly_scored; skipping per-region summary.")


# ============================================================
# ========================= MAIN =============================
# ============================================================

def main():
    print(
        f"Running full results pipeline. "
        f"Output tables: {RESULTS_DIR}, figures: {FIGURES_DIR}"
    )

    ensure_mlflow_tracking()

    evaluate_throughput()
    evaluate_latency()
    evaluate_classification()
    summarize_anomaly()

    print("\nAll done ✅ – check 'reports/results' for CSVs and 'reports/figures' for PNGs.")


if __name__ == "__main__":
    main()
