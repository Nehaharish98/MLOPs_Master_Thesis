
# Monitoring & Observability (FastAPI + Prometheus + Grafana)

This part of the project provides **runtime monitoring** for the Inter-DC Network
Performance service using:

- **FastAPI** – model serving (4 tasks)
- **Prometheus** – metrics collection
- **Grafana** – dashboards


## 1. Service overview

The FastAPI service exposes the following endpoints:

- **Health**
  - `GET /health`

- **Prediction tasks**
  - **Throughput**: `POST /predict/throughput`
  - **Latency**: `POST /predict/latency`
  - **Classification**: `POST /classify/performance`
  - **Anomaly**: `POST /score/anomaly`

All endpoints are automatically instrumented with Prometheus metrics via
`prometheus_fastapi_instrumentator` and exposed at:

- **Metrics endpoint**: `GET /metrics`

---

## 2. Repository files (monitoring-related)

Key files for the monitoring stack:

- `src/service/app.py`  
  FastAPI app with Prometheus instrumentation and the 4 task endpoints.

- `prometheus.yml`  
  Prometheus configuration, scrapes the FastAPI service.

- `docker-compose.yml`  
  Starts:
  - `api` – FastAPI service
  - `prometheus` – metrics collector
  - `grafana` – dashboards

You should have something like:

```text
Thesis-mlops/
├── src/
│   └── service/
│       └── app.py
├── prometheus.yml
├── docker-compose.yml
└── ...
````

---

## 3. Prerequisites

* Docker
* Docker Compose
* Python environment for local dev (optional, only needed if you want to run the app without Docker)
* Valid MLflow tracking configuration in `.env`, e.g.:

```bash
MLFLOW_TRACKING_URI=...
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

---

## 4. Running the monitoring stack

From the project root:

```bash
docker-compose up
```

This will:

* build / start the `api` container on port **8000**
* start Prometheus on port **9090**
* start Grafana on port **3000**

To stop everything:

```bash
docker-compose down
```

---

## 5. Accessing the service & metrics

Once `docker-compose` is running:

### FastAPI

* Docs (if enabled in your app):
  `http://localhost:8000/docs`

* Health check:

  ```bash
  curl http://localhost:8000/health
  ```

* Metrics:

  ```bash
  curl http://localhost:8000/metrics
  ```

You should see Prometheus-style text output when hitting `/metrics`.

### Example requests for the 4 tasks

All endpoints support an empty JSON body (using feature templates inside the service):

```bash
# Throughput prediction
curl -X POST "http://localhost:8000/predict/throughput" \
  -H "Content-Type: application/json" \
  -d '{}'

# Latency prediction
curl -X POST "http://localhost:8000/predict/latency" \
  -H "Content-Type: application/json" \
  -d '{}'

# Classification (good / bad)
curl -X POST "http://localhost:8000/classify/performance" \
  -H "Content-Type: application/json" \
  -d '{}'

# Anomaly score
curl -X POST "http://localhost:8000/score/anomaly" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## 6. Prometheus UI

Prometheus is available at:

* `http://localhost:9090`

Useful test queries:

```promql
http_requests_total
label_values(http_requests_total, handler)
```

The second query shows the list of HTTP handlers that are being tracked, for example:

* `/health`
* `/predict/throughput`
* `/predict/latency`
* `/classify/performance`
* `/score/anomaly`

These label values are used in the Grafana panels below.

---

## 7. Grafana setup

Grafana is available at:

* `http://localhost:3000`

### 7.1 Login

Default credentials:

* **User**: `admin`
* **Password**: `admin` (you will be asked to change this on first login)

### 7.2 Add Prometheus as a data source

1. In Grafana, go to **Connections → Data sources**.

2. Click **“Add data source”** and choose **Prometheus**.

3. Set the **URL** to:

   ```text
   http://prometheus:9090
   ```

   > Note: use `prometheus`, not `localhost`, because Grafana and Prometheus run in the same Docker network.

4. Click **“Save & Test”** – you should see “Data source is working”.

---

## 8. Dashboards for the 4 tasks

Create a new dashboard and add panels as described below.
This makes the setup reproducible for anyone.

### 8.1 Requests per task (Requests/sec)

For each panel:

1. Dashboard → **Add panel**
2. Data source: `prometheus`
3. Panel type: **Time series**
4. Use one of the queries below
5. Set a title like “Throughput – Requests/sec”
6. Unit: “requests/sec” (or any “per second” unit)

**Throughput**

```promql
sum(
  rate(http_requests_total{handler="/predict/throughput"}[5m])
)
```

**Latency**

```promql
sum(
  rate(http_requests_total{handler="/predict/latency"}[5m])
)
```

**Classification**

```promql
sum(
  rate(http_requests_total{handler="/classify/performance"}[5m])
)
```

**Anomaly**

```promql
sum(
  rate(http_requests_total{handler="/score/anomaly"}[5m])
)
```

> If your `handler` label values differ, adjust the string to match what you see from `label_values(http_requests_total, handler)` in Prometheus.

### 8.2 P95 latency per task

To monitor latency for each task, add one panel per endpoint:

1. Add panel → data source: Prometheus
2. Panel type: Time series
3. Unit: **seconds**
4. Use one of the queries below
5. Title e.g. `Throughput – P95 latency`

**Throughput**

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{handler="/predict/throughput"}[5m])
  )
)
```

**Latency**

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{handler="/predict/latency"}[5m])
  )
)
```

**Classification**

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{handler="/classify/performance"}[5m])
  )
)
```

**Anomaly**

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{handler="/score/anomaly"}[5m])
  )
)
```

### 8.3 Global panels (optional but recommended)

You can also add global panels:

* **Total requests per endpoint**

  ```promql
  sum by (handler) (
    rate(http_requests_total[5m])
  )
  ```

* **Error rate (%)**

  ```promql
  (
    sum(rate(http_requests_total{status!="2xx"}[5m]))
  /
    sum(rate(http_requests_total[5m]))
  ) * 100
  ```

  Use unit **Percent (0–100)**.

---

## 9. Generating traffic for the dashboard

To see the panels update:

1. Ensure `docker-compose up` is running.
2. Send requests to each endpoint, e.g.:

```bash
# Run each of these a few times
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/predict/throughput" \
  -H "Content-Type: application/json" \
  -d '{}'

curl -X POST "http://localhost:8000/predict/latency" \
  -H "Content-Type: application/json" \
  -d '{}'

curl -X POST "http://localhost:8000/classify/performance" \
  -H "Content-Type: application/json" \
  -d '{}'

curl -X POST "http://localhost:8000/score/anomaly" \
  -H "Content-Type: application/json" \
  -d '{}'
```

3. In Grafana, set the time range to **“Last 5 minutes”** and click **Refresh**.

You should see lines for all four tasks.

---

## 10. Summary (for thesis documentation)

* The FastAPI service exposes 4 ML tasks as HTTP endpoints.
* All requests are instrumented with Prometheus metrics via
  `prometheus_fastapi_instrumentator` and exposed at `/metrics`.
* Prometheus runs as a container and scrapes `api:8000/metrics`.
* Grafana runs as a container and visualizes metrics from Prometheus.
* Dashboards include:

  * Requests/sec per task
  * P95 latency per task
  * Global traffic and error rate

This stack is fully reproducible using `docker-compose` and can be easily ported
to a cloud VM (e.g. AWS EC2) by running the same commands on the server.

```

If you want, I can next draft a similar README section for **Evidently + Prefect** once we wire those in, so your whole MLOps story is documented in the same style.
::contentReference[oaicite:0]{index=0}
```
