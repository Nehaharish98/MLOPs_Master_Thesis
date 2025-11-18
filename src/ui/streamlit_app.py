import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

# -----------------------------
# Basic configuration
# -----------------------------
st.set_page_config(
    page_title="Inter-DC Network MLOps Demo",
    page_icon="🌐",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(f"**API base URL:** `{API_BASE_URL}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Monitoring links (update as needed)")
st.sidebar.markdown("- Grafana: [http://localhost:3000](http://localhost:3000)")
st.sidebar.markdown("- Prometheus: [http://localhost:9090](http://localhost:9090)")
st.sidebar.markdown(
    "- Throughput drift report: "
    "[reports/throughput_drift_report.html](reports/throughput_drift_report.html)"
)
st.sidebar.markdown(
    "- Latency drift report: "
    "[reports/latency_drift_report.html](reports/latency_drift_report.html)"
)
st.sidebar.markdown(
    "- Classification drift report: "
    "[reports/classification_drift_report.html](reports/classification_drift_report.html)"
)

# -----------------------------
# Helpers
# -----------------------------
def load_optional_parquet(path: Path):
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            st.warning(f"Could not load {path}: {e}")
            return None
    return None


def build_features_from_row(df: pd.DataFrame, idx: int):
    """
    Convert one row of a prepared dataframe into a `features` dict.
    Drops obvious non-feature columns like targets, split, timestamps.
    """
    row = df.iloc[idx].copy()

    drop_cols = [
        "split",
        "start",
        "tput_mean_next",
        "delay_mean_next",
        "label_good",
        "anomaly_flag_iso",
        "anomaly_score_iso",
    ]
    for c in drop_cols:
        if c in row.index:
            row = row.drop(c)

    features = {}
    for k, v in row.items():
        if pd.isna(v):
            continue
        if hasattr(v, "item"):
            v = v.item()
        features[str(k)] = v

    return features


def call_api(path: str, payload: dict):
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None, None

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    return resp.status_code, data


def pretty_response(status_code, data):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("HTTP status", status_code)
    with col2:
        st.write("Response body:")
    st.json(data)


def plot_history_with_future_point(
    df: pd.DataFrame,
    time_col: str,
    actual_col: str,
    pred_value: float,
    minutes_ahead: int = 5,
    title: str = "",
):
    """
    Plot last N experiments as a single line, where:
    - History is a solid line
    - The predicted next point is a dotted continuation
      (line from last actual point -> future predicted point)
    """
    if df is None:
        st.info("No data available to plot.")
        return

    if pred_value is None:
        st.info("No numeric prediction available for plotting.")
        return

    if time_col not in df.columns or actual_col not in df.columns:
        st.warning(
            f"Cannot plot: missing '{time_col}' or '{actual_col}' in dataframe."
        )
        return

    # Take last 10 points as history
    df_sorted = df.copy()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
    window = df_sorted.sort_values(time_col).tail(10).copy()

    if window.empty:
        st.info("Not enough data to build a history window.")
        return

    # Solid history line
    history_df = pd.DataFrame(
        {
            "time": window[time_col],
            "value": window[actual_col],
            "segment": "Actual",
        }
    )

    # Build dotted "extension" segment: last actual point + future predicted point
    last_row = history_df.iloc[-1]
    future_time = last_row["time"] + pd.Timedelta(minutes=minutes_ahead)

    extension_df = pd.DataFrame(
        {
            "time": [last_row["time"], future_time],
            "value": [last_row["value"], pred_value],
            "segment": ["Predicted", "Predicted"],
        }
    )

    plot_df = pd.concat([history_df, extension_df], ignore_index=True)
    plot_df["time"] = pd.to_datetime(plot_df["time"])

    st.markdown(
        f"#### {title} – recent history + next {minutes_ahead} min prediction"
    )

    base = (
        alt.Chart(plot_df)
        .encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("value:Q", title=title),
        )
    )

    # Solid line for all 'Actual' points
    actual_line = base.transform_filter(
        alt.datum.segment == "Actual"
    ).mark_line(strokeWidth=2)

    # Dotted line from last actual point to future predicted point
    future_line = base.transform_filter(
        alt.datum.segment == "Predicted"
    ).mark_line(strokeDash=[4, 4], strokeWidth=2)

    chart = (actual_line + future_line).properties(
        height=300,
        title=f"{title} – last points (solid) + predicted next (dotted)",
    )

    st.altair_chart(chart, width="stretch")


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = np.abs(y_true) > 1e-9
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return dict(R2=r2, MAE=mae, RMSE=rmse, MAPE=mape)


def slice_by_pair(df: pd.DataFrame, key_prefix: str):
    """
    Interactive filters for provider / regions / VM sizes.
    Returns the filtered slice.

    key_prefix ensures all widgets have unique keys across tabs.
    """
    required_cols = [
        "provider",
        "region_snd_base",
        "region_rcv_base",
        "size_snd",
        "size_rcv",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in dataframe: {missing}")
        return None

    provider = st.selectbox(
        "Provider",
        options=sorted(df["provider"].dropna().unique()),
        key=f"{key_prefix}_provider",
    )

    df_p = df[df["provider"] == provider]

    region_snd = st.selectbox(
        "Sender region (base)",
        options=sorted(df_p["region_snd_base"].dropna().unique()),
        key=f"{key_prefix}_region_snd",
    )

    df_ps = df_p[df_p["region_snd_base"] == region_snd]

    region_rcv = st.selectbox(
        "Receiver region (base)",
        options=sorted(df_ps["region_rcv_base"].dropna().unique()),
        key=f"{key_prefix}_region_rcv",
    )

    df_pair = df_ps[df_ps["region_rcv_base"] == region_rcv]

    size_snd = st.selectbox(
        "Sender VM size",
        options=sorted(df_pair["size_snd"].dropna().unique()),
        key=f"{key_prefix}_size_snd",
    )

    df_pair = df_pair[df_pair["size_snd"] == size_snd]

    size_rcv = st.selectbox(
        "Receiver VM size",
        options=sorted(df_pair["size_rcv"].dropna().unique()),
        key=f"{key_prefix}_size_rcv",
    )

    df_pair = df_pair[df_pair["size_rcv"] == size_rcv]

    st.markdown("**Selected slice:**")
    st.code(
        f"provider = {provider}\n"
        f"region_snd_base = {region_snd}\n"
        f"region_rcv_base = {region_rcv}\n"
        f"size_snd = {size_snd}\n"
        f"size_rcv = {size_rcv}"
    )

    if df_pair.empty:
        st.warning("No rows match this combination.")
        return None

    return df_pair


def batch_predict_regression(
    df_slice: pd.DataFrame, endpoint: str, target_col: str, max_rows: int = 50
):
    """
    For a slice of data, call the regression API endpoint row by row
    and return a dataframe with actual + predicted columns for plotting.
    """
    if df_slice is None or df_slice.empty:
        return None

    df_slice = df_slice.sort_values("start").tail(max_rows).reset_index(drop=True)

    preds = []
    progress = st.progress(0.0, text="Calling API for each run...")

    for i in range(len(df_slice)):
        features = build_features_from_row(df_slice, i)
        payload = {"features": features}
        status, data = call_api(endpoint, payload)

        pred_value = None
        if status == 200 and isinstance(data, dict):
            for key, v in data.items():
                if isinstance(v, (int, float)):
                    pred_value = float(v)
                    break

        preds.append(pred_value)
        progress.progress((i + 1) / len(df_slice))

    df_out = df_slice.copy()
    df_out["prediction"] = preds
    return df_out


# -----------------------------
# Load local prepared data (if available)
# -----------------------------
BASE_PATH = Path("data/prepared")

df_tput = load_optional_parquet(BASE_PATH / "throughput_supervised.parquet")
df_lat = load_optional_parquet(BASE_PATH / "latency_supervised.parquet")
df_cls = load_optional_parquet(BASE_PATH / "classification_supervised.parquet")
df_anom = load_optional_parquet(BASE_PATH / "anomaly_matrix.parquet")
df_anom_scored = load_optional_parquet(BASE_PATH / "anomaly_scored.parquet")

# -----------------------------
# Layout (tabs)
# -----------------------------
st.title("🌐 Inter-DC Network MLOps Demo UI")
st.markdown(
    """
This app talks to your **FastAPI service** and shows predictions for:

- 📈 Throughput (`/predict/throughput`)
- ⏱️ Latency (`/predict/latency`)
- ✅ Performance classification (`/classify/performance`)
- 🚨 Anomaly score (`/score/anomaly`)

You can:
- Use sample rows from your prepared parquet files, **or**
- Paste your own JSON payload, **and**
- Explore **per-pair model behaviour** (actual vs predicted, classification performance, anomalies).
"""
)

tab_overview, tab_tput, tab_lat, tab_cls, tab_anom = st.tabs(
    ["📊 Overview", "📈 Throughput", "⏱️ Latency", "✅ Classification", "🚨 Anomaly"]
)

# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    st.subheader("📊 System overview")

    col1, col2, col3, col4 = st.columns(4)

    # Simple KPIs
    if df_tput is not None and "tput_mean_next" in df_tput.columns:
        avg_tput = df_tput["tput_mean_next"].mean()
        col1.metric("Avg throughput (Mbps)", f"{avg_tput:.1f}")
    else:
        col1.metric("Avg throughput (Mbps)", "n/a")

    if df_lat is not None and "delay_mean_next" in df_lat.columns:
        avg_lat = df_lat["delay_mean_next"].mean()
        col2.metric("Avg latency (ms)", f"{avg_lat:.1f}")
    else:
        col2.metric("Avg latency (ms)", "n/a")

    if df_cls is not None and "label_good" in df_cls.columns:
        pct_good = 100.0 * df_cls["label_good"].mean()
        col3.metric("Good performance (%)", f"{pct_good:.1f}")
    else:
        col3.metric("Good performance (%)", "n/a")

    if df_anom_scored is not None and "anomaly_flag_iso" in df_anom_scored.columns:
        pct_anom = 100.0 * df_anom_scored["anomaly_flag_iso"].mean()
        col4.metric("Anomaly rate (%)", f"{pct_anom:.1f}")
    else:
        col4.metric("Anomaly rate (%)", "n/a")

    st.markdown("---")

    # Simple time-series view
    st.subheader("📈 Throughput over time (sample)")
    if df_tput is not None and {"start", "tput_mean_next"}.issubset(df_tput.columns):
        df_plot = df_tput[["start", "tput_mean_next"]].dropna().copy()
        df_plot["start"] = pd.to_datetime(df_plot["start"])
        df_plot = df_plot.sort_values("start").tail(500)

        chart = (
            alt.Chart(df_plot)
            .mark_line()
            .encode(
                x="start:T",
                y=alt.Y("tput_mean_next:Q", title="Throughput (Mbps)"),
                tooltip=["start:T", "tput_mean_next:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("throughput_supervised.parquet not available or missing columns.")

# -----------------------------
# Throughput tab
# -----------------------------
with tab_tput:
    st.subheader("📈 Throughput – per-path forecast & evaluation")

    if df_tput is None:
        st.info("throughput_supervised.parquet not available.")
    else:
        # 1) Pick a single provider / region pair / VM sizes
        df_slice = slice_by_pair(df_tput, key_prefix="tput_pair")
        if df_slice is not None:
            df_slice = df_slice.sort_values("start").reset_index(drop=True)
            st.caption(f"Slice size for this path: {len(df_slice)} rows")

            # ============================
            # A) One-step-ahead forecast
            # ============================
            st.markdown("### 🔮 One-step-ahead forecast for this path")

            horizon = st.slider(
                "Forecast horizon (minutes)",
                min_value=5,
                max_value=60,
                step=5,
                value=5,
                key="tput_horizon",
            )

            if st.button(
                "🚀 Predict next throughput for this path", key="tput_forecast_btn"
            ):
                last_idx = len(df_slice) - 1

                st.caption("Current context (last measurement for this path):")
                st.dataframe(df_slice.iloc[[last_idx]].iloc[:, :10])

                # Build features from the LAST row of this path only
                features = build_features_from_row(df_slice, last_idx)
                payload = {"features": features}

                st.markdown("**Payload sent to API:**")
                st.json(payload)

                status, data = call_api("/predict/throughput", payload)
                if status is not None:
                    pretty_response(status, data)

                    # Extract numeric prediction
                    pred_value = None
                    if isinstance(data, dict):
                        # Prefer keys that look like throughput
                        for key, v in data.items():
                            if key.lower().startswith("tput") or key.lower().startswith(
                                "throughput"
                            ):
                                if isinstance(v, (int, float)):
                                    pred_value = float(v)
                                    break
                        # Fallback: first numeric value
                        if pred_value is None:
                            for v in data.values():
                                if isinstance(v, (int, float)):
                                    pred_value = float(v)
                                    break

                    if pred_value is not None:
                        # IMPORTANT: history is now ONLY this path
                        plot_history_with_future_point(
                            df=df_slice,
                            time_col="start",
                            actual_col="tput_mean_next",
                            pred_value=pred_value,
                            minutes_ahead=horizon,
                            title="Throughput (Mbps)",
                        )
                    else:
                        st.info("Could not extract a numeric prediction to plot.")

            # ============================
            # B) Per-path evaluation
            # ============================
            st.markdown("---")
            st.markdown("### 📊 Per-path evaluation on historical data")

            max_rows = st.slider(
                "Max runs to evaluate",
                min_value=10,
                max_value=min(100, len(df_slice)),
                step=10,
                value=min(50, len(df_slice)),
                key="tput_max_rows",
            )

            if st.button("📈 Evaluate model on this path", key="tput_eval_btn"):
                df_eval = batch_predict_regression(
                    df_slice,
                    "/predict/throughput",
                    target_col="tput_mean_next",
                    max_rows=max_rows,
                )

                if df_eval is not None and not df_eval["prediction"].isna().all():
                    df_eval = df_eval.dropna(subset=["prediction"])
                    df_eval = df_eval.sort_values("start")

                    metrics = compute_regression_metrics(
                        df_eval["tput_mean_next"].values,
                        df_eval["prediction"].values,
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R²", f"{metrics['R2']:.3f}")
                    c2.metric("MAE", f"{metrics['MAE']:.2f}")
                    c3.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    if not np.isnan(metrics["MAPE"]):
                        c4.metric("MAPE (%)", f"{metrics['MAPE']:.1f}")
                    else:
                        c4.metric("MAPE (%)", "n/a")

                    chart_df = df_eval[
                        ["start", "tput_mean_next", "prediction"]
                    ].copy()
                    chart_df["start"] = pd.to_datetime(chart_df["start"])
                    chart_df = chart_df.rename(
                        columns={
                            "tput_mean_next": "Actual throughput",
                            "prediction": "Predicted throughput",
                        }
                    )

                    chart = (
                        alt.Chart(chart_df)
                        .transform_fold(
                            ["Actual throughput", "Predicted throughput"],
                            as_=["Series", "value"],
                        )
                        .mark_line()
                        .encode(
                            x="start:T",
                            y="value:Q",
                            color="Series:N",
                            tooltip=["start:T", "Series:N", "value:Q"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.warning("No predictions returned from API for this path.")


# -----------------------------
# Latency tab
# -----------------------------
with tab_lat:
    st.subheader("⏱️ Predict latency")

    mode = st.radio(
        "Input mode",
        ["Use example from latency_supervised.parquet", "Manual JSON"],
        key="lat_mode",
    )

    payload = {"features": {}}
    idx = 0

    if mode.startswith("Use example"):
        if df_lat is None:
            st.error(
                "latency_supervised.parquet not found under data/prepared/. "
                "Switch to Manual JSON mode or check your paths."
            )
        else:
            max_idx = len(df_lat) - 1
            idx = st.slider(
                "Row index", 0, max_idx, min(10, max_idx), key="lat_idx"
            )
            st.caption("Selected row preview (first 10 columns):")
            st.dataframe(df_lat.iloc[[idx]].iloc[:, :10])

            features = build_features_from_row(df_lat, idx)
            payload["features"] = features

            st.markdown("**Payload preview (features dict):**")
            st.json(payload)
    else:
        default_json = json.dumps(
            {
                "features": {
                    "example_feature_1": 0.5,
                    "example_feature_2": 10.0,
                }
            },
            indent=2,
        )
        txt = st.text_area(
            "Edit JSON payload (must be valid JSON matching the model schema):",
            value=default_json,
            height=200,
            key="lat_json",
        )
        try:
            payload = json.loads(txt)
        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")
            payload = None

    if st.button("🚀 Predict latency", type="primary"):
        if payload is None:
            st.error("Fix JSON errors before sending.")
        else:
            status, data = call_api("/predict/latency", payload)
            if status is not None:
                pretty_response(status, data)

                pred_value = None
                if isinstance(data, dict):
                    for key, v in data.items():
                        if key.lower().startswith("delay") or key.lower().startswith(
                            "latency"
                        ):
                            if isinstance(v, (int, float)):
                                pred_value = float(v)
                                break
                    if pred_value is None:
                        for v in data.values():
                            if isinstance(v, (int, float)):
                                pred_value = float(v)
                                break

                if pred_value is not None and df_lat is not None:
                    plot_history_with_future_point(
                        df=df_lat,
                        time_col="start",
                        actual_col="delay_mean_next",
                        pred_value=pred_value,
                        minutes_ahead=5,
                        title="Latency (ms)",
                    )
                else:
                    st.info(
                        "Could not build latency timeseries plot "
                        "(no numeric prediction or no data)."
                    )

    st.markdown("---")
    st.subheader("🔎 Per-pair evaluation: actual vs predicted (test data)")

    if df_lat is None:
        st.info("latency_supervised.parquet not available.")
    else:
        df_slice = slice_by_pair(df_lat, key_prefix="lat_pair")
        if df_slice is not None:
            st.caption(f"Slice size: {len(df_slice)} rows (before limiting)")
            max_rows = st.slider(
                "Max runs to evaluate", 10, 100, 50, step=10, key="lat_max_rows"
            )
            if st.button("📊 Evaluate model on this slice", key="lat_eval_btn"):
                df_pred = batch_predict_regression(
                    df_slice,
                    "/predict/latency",
                    target_col="delay_mean_next",
                    max_rows=max_rows,
                )
                if df_pred is not None and not df_pred["prediction"].isna().all():
                    df_pred = df_pred.dropna(subset=["prediction"])
                    df_pred = df_pred.sort_values("start")

                    metrics = compute_regression_metrics(
                        df_pred["delay_mean_next"].values,
                        df_pred["prediction"].values,
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R²", f"{metrics['R2']:.3f}")
                    c2.metric("MAE", f"{metrics['MAE']:.2f}")
                    c3.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    if not np.isnan(metrics["MAPE"]):
                        c4.metric("MAPE (%)", f"{metrics['MAPE']:.1f}")
                    else:
                        c4.metric("MAPE (%)", "n/a")

                    chart_df = df_pred[
                        ["start", "delay_mean_next", "prediction"]
                    ].copy()
                    chart_df["start"] = pd.to_datetime(chart_df["start"])
                    chart_df = chart_df.rename(
                        columns={
                            "delay_mean_next": "Actual latency",
                            "prediction": "Predicted latency",
                        }
                    )

                    chart = (
                        alt.Chart(chart_df)
                        .transform_fold(
                            ["Actual latency", "Predicted latency"],
                            as_=["Series", "value"],
                        )
                        .mark_line()
                        .encode(
                            x="start:T",
                            y="value:Q",
                            color="Series:N",
                            tooltip=["start:T", "Series:N", "value:Q"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.warning("No predictions returned from API for this slice.")

# -----------------------------
# Classification tab
# -----------------------------
with tab_cls:
    st.subheader("✅ Classify performance (good / bad)")

    mode = st.radio(
        "Input mode",
        ["Use example from classification_supervised.parquet", "Manual JSON"],
        key="cls_mode",
    )

    payload = {"features": {}}

    if mode.startswith("Use example"):
        if df_cls is None:
            st.error(
                "classification_supervised.parquet not found under data/prepared/. "
                "Switch to Manual JSON mode or check your paths."
            )
        else:
            max_idx = len(df_cls) - 1
            idx = st.slider(
                "Row index", 0, max_idx, min(10, max_idx), key="cls_idx"
            )
            st.caption("Selected row preview (first 10 columns):")
            st.dataframe(df_cls.iloc[[idx]].iloc[:, :10])

            features = build_features_from_row(df_cls, idx)
            payload["features"] = features

            st.markdown("**Payload preview (features dict):**")
            st.json(payload)
    else:
        default_json = json.dumps(
            {
                "features": {
                    "example_feature_1": 0.5,
                    "example_feature_2": 10.0,
                }
            },
            indent=2,
        )
        txt = st.text_area(
            "Edit JSON payload (must be valid JSON matching the model schema):",
            value=default_json,
            height=200,
            key="cls_json",
        )
        try:
            payload = json.loads(txt)
        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")
            payload = None

    if st.button("🚀 Classify performance", type="primary"):
        if payload is None:
            st.error("Fix JSON errors before sending.")
        else:
            status, data = call_api("/classify/performance", payload)
            if status is not None:
                pretty_response(status, data)

    st.markdown("---")
    st.subheader("🔎 Per-pair classification performance (test data)")

    if df_cls is None:
        st.info("classification_supervised.parquet not available.")
    else:
        df_slice = slice_by_pair(df_cls, key_prefix="cls_pair")
        if df_slice is not None:
            st.caption(f"Slice size: {len(df_slice)} rows")
            if st.button("📊 Evaluate classifier on this slice", key="cls_eval_btn"):
                y_true = df_slice["label_good"].values
                y_pred = []
                progress = st.progress(
                    0.0, text="Calling classification API for each run..."
                )

                for i in range(len(df_slice)):
                    features = build_features_from_row(df_slice, i)
                    payload = {"features": features}
                    status, data = call_api("/classify/performance", payload)

                    pred_label = None
                    if status == 200 and isinstance(data, dict):
                        if "label_good" in data and isinstance(
                            data["label_good"], (int, float)
                        ):
                            pred_label = int(data["label_good"])
                        else:
                            for v in data.values():
                                if v in (0, 1):
                                    pred_label = int(v)
                                    break

                    if pred_label is None:
                        pred_label = 0
                    y_pred.append(pred_label)
                    progress.progress((i + 1) / len(df_slice))

                y_pred = np.asarray(y_pred)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                c1, c2 = st.columns(2)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("F1-score", f"{f1:.3f}")

                st.caption("First 20 predictions in this slice:")
                preview_df = df_slice[["start", "label_good"]].copy()
                preview_df["pred_label"] = y_pred
                st.dataframe(preview_df.head(20))

# -----------------------------
# Anomaly tab
# -----------------------------
with tab_anom:
    st.subheader("🚨 Anomaly score")

    mode = st.radio(
        "Input mode",
        ["Use example from anomaly_matrix.parquet", "Manual JSON"],
        key="anom_mode",
    )

    payload = {"features": {}}

    if mode.startswith("Use example"):
        if df_anom is None:
            st.error(
                "anomaly_matrix.parquet not found under data/prepared/. "
                "Switch to Manual JSON mode or check your paths."
            )
        else:
            max_idx = len(df_anom) - 1
            idx = st.slider(
                "Row index", 0, max_idx, min(10, max_idx), key="anom_idx"
            )
            st.caption("Selected row preview (first 10 columns):")
            st.dataframe(df_anom.iloc[[idx]].iloc[:, :10])

            features = build_features_from_row(df_anom, idx)
            payload["features"] = features

            st.markdown("**Payload preview (features dict):**")
            st.json(payload)
    else:
        default_json = json.dumps(
            {
                "features": {
                    "example_feature_1": 0.5,
                    "example_feature_2": 10.0,
                }
            },
            indent=2,
        )
        txt = st.text_area(
            "Edit JSON payload (must be valid JSON matching the model schema):",
            value=default_json,
            height=200,
            key="anom_json",
        )
        try:
            payload = json.loads(txt)
        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")
            payload = None

    if st.button("🚀 Score anomaly", type="primary"):
        if payload is None:
            st.error("Fix JSON errors before sending.")
        else:
            status, data = call_api("/score/anomaly", payload)
            if status is not None:
                pretty_response(status, data)

    st.markdown("---")
    st.subheader("🔎 Anomalies per pair (scatter plot)")

    if df_anom_scored is None:
        st.info(
            "anomaly_scored.parquet not found under data/prepared/. "
            "Scatter plot per pair will not be available."
        )
    else:
        df_slice = slice_by_pair(df_anom_scored, key_prefix="anom_pair")
        if df_slice is not None:
            st.caption(f"Slice size: {len(df_slice)} rows")

            if (
                "anomaly_score_iso" not in df_slice.columns
                and "anomaly_flag_iso" not in df_slice.columns
            ):
                st.warning(
                    "anomaly_scored.parquet does not contain 'anomaly_score_iso' "
                    "or 'anomaly_flag_iso'. Cannot plot anomalies."
                )
            else:
                x_col = "tput_mean" if "tput_mean" in df_slice.columns else None
                y_col = "delay_mean" if "delay_mean" in df_slice.columns else None

                if x_col is None or y_col is None:
                    st.warning(
                        "Expected 'tput_mean' and 'delay_mean' columns not found. "
                        "Showing only anomaly score over time."
                    )
                    if "start" in df_slice.columns and "anomaly_score_iso" in df_slice.columns:
                        df_plot = df_slice.copy()
                        df_plot["start"] = pd.to_datetime(df_plot["start"])
                        chart = (
                            alt.Chart(df_plot)
                            .mark_circle()
                            .encode(
                                x="start:T",
                                y="anomaly_score_iso:Q",
                                color=(
                                    "anomaly_flag_iso:N"
                                    if "anomaly_flag_iso" in df_plot.columns
                                    else alt.value("steelblue")
                                ),
                                tooltip=list(df_plot.columns),
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(chart, width="stretch")
                    else:
                        st.info("Not enough columns to plot anomaly scatter.")
                else:
                    st.caption(
                        "Scatter of throughput vs delay, colored by anomaly flag / score"
                    )
                    df_plot = df_slice.copy()
                    if "anomaly_flag_iso" in df_plot.columns:
                        color_enc = "anomaly_flag_iso:N"
                    elif "anomaly_score_iso" in df_plot.columns:
                        color_enc = "anomaly_score_iso:Q"
                    else:
                        color_enc = alt.value("steelblue")

                    chart = (
                        alt.Chart(df_plot)
                        .mark_circle(size=60)
                        .encode(
                            x=f"{x_col}:Q",
                            y=f"{y_col}:Q",
                            color=color_enc,
                            tooltip=list(df_plot.columns),
                        )
                        .interactive()
                        .properties(height=300)
                    )
                    st.altair_chart(chart, width="stretch")
