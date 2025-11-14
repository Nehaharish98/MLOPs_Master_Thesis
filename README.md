````markdown
# Thesis – Cloud Inter-Datacenter Network Performance (MLOps)

This repository contains the code and experiment setup for my master’s thesis on **cloud inter-datacenter network performance**.

The project uses an inter-DC measurement dataset (AWS & Azure) to:

- predict **throughput** for the next 5-minute run  
- forecast **latency** for the next 5-minute run  
- classify runs as **good / bad performance**  
- detect **anomalous** runs using unsupervised models  

All experiments are tracked with **MLflow** and stored remotely on **DagsHub**, and the final goal is to integrate these models into a cloud-based MLOps pipeline (AWS).

---

## 1. Dataset

### 1.1 Source & granularity

The raw dataset consists of active measurements between cloud regions (AWS & Azure). For each *experiment run*:

- A sender region and receiver region exchange traffic for about **5 minutes**.
- Detailed per-second measurements (throughput, delay, loss, jitter) are collected.
- Summary statistics are computed per run.

In this project, we work on **per-run summary rows** only (no per-second arrays).

### 1.2 Prepared base table

The main cleaned base file is:

```text
data/prepared/base_clean.parquet
````

Key properties:

* Rows: ~9,500 runs (after filtering to ~5-min durations)
* Time column: `start` (datetime)
* Pair keys (directional):

  * `provider` (e.g. aws / azure)
  * `region_snd_base` (e.g. eu-west-1)
  * `region_rcv_base` (e.g. us-east-1)
* Duration policy: keep runs with `duration` in **[290, 310] seconds**
* Region normalization: AZs collapsed into base regions

  * example: `eu-west-1a`, `eu-west-1b` → `eu-west-1`
  * original AZs kept as optional categoricals: `az_snd`, `az_rcv`

#### Main columns

* **Categorical**

  * `provider`
  * `region_snd_base`, `region_rcv_base`
  * `proto` (e.g. tcp / udp)
  * `size_snd`, `size_rcv`
  * `az_snd`, `az_rcv`
  * `snd_continent`, `rcv_continent`
* **Numeric examples**

  * Throughput stats: `tput_mean`, `tput_std`, `tput_min`, `tput_max`, `tput_pctl5/25/50/75/95`
  * Delay stats: `delay_mean`
  * Loss stats: `loss_mean`, `loss_std`, `loss_min`, `loss_max`, `loss_pctl5/25/50/75/95`
  * `duration` (seconds)
* **Time**

  * `start` (experiment start timestamp)

This base table is then transformed into **task-specific supervised datasets** with lags and labels.

---

## 2. Supervised tasks

All tasks are defined at the **run level** (one row = one 5-minute experiment).

Splits are **time-ordered per pair** (`provider, region_snd_base, region_rcv_base`) to avoid leakage from future to past.

### 2.1 Throughput regression

**Goal:** Predict the *next run’s* mean throughput for a given directional pair.

* File:

  ```text
  data/prepared/throughput_supervised.parquet
  ```
* Target:

  * `tput_mean_next` – throughput mean of the next run for that pair
* Lags:

  * `tput_mean_lag1`, `tput_mean_lag2`, `tput_mean_lag3`
    (previous 1–3 runs’ mean throughput per pair)
* Splits:

  ```text
  data/splits/throughput/train.parquet
  data/splits/throughput/val.parquet
  data/splits/throughput/test.parquet
  ```

### 2.2 Latency regression

**Goal:** Predict the *next run’s* mean delay.

* File:

  ```text
  data/prepared/latency_supervised.parquet
  ```
* Target:

  * `delay_mean_next`
* Lags:

  * `delay_mean_lag1`, `delay_mean_lag2`, `delay_mean_lag3`
* Splits:

  ```text
  data/splits/latency/train.parquet
  data/splits/latency/val.parquet
  data/splits/latency/test.parquet
  ```

### 2.3 Performance classification (good / bad)

**Goal:** Classify each run as **good** or **bad** network performance.

* File:

  ```text
  data/prepared/classification_supervised.parquet
  ```
* Target:

  * `label_good` (1 = good performance, 0 = bad)
  * A run is labeled “good” if:

    * throughput ≥ pair-wise median, **and**
    * delay ≤ pair-wise median
* Features include:

  * summary stats,
  * lagged throughput/latency (`tput_mean_lag1`, `delay_mean_lag1`),
  * static attributes (provider, regions, proto, sizes, etc.)
* Splits:

  ```text
  data/splits/classification/train.parquet
  data/splits/classification/val.parquet
  data/splits/classification/test.parquet
  ```

### 2.4 Anomaly detection (unsupervised)

**Goal:** Highlight runs that are unusual compared to typical behaviour, per pair and globally.

* Feature matrix:

  ```text
  data/prepared/anomaly_matrix.parquet
  ```
* Scored output:

  ```text
  data/prepared/anomaly_scored.parquet
  ```
* Top anomalies per pair:

  ```text
  data/prepared/anomaly_top10_per_pair.parquet
  ```
* Methods:

  * **IsolationForest** (global anomaly score)
  * **Pair-relative z-score** (interpretable deviation from per-pair mean)

Anomaly results are summarised and logged into MLflow using `src/log_anomalies.py`.

---

## 3. Models & metrics

Each supervised task is benchmarked with three model families:

* **LinearRegression** (with one-hot encoding)
* **RandomForest** (with one-hot encoding)
* **CatBoost** (native categorical handling)

### 3.1 Feature handling

All tasks use **one dataset per task**, no separate “encoded” copies:

* Categorical features:

  * detected via pandas dtypes and converted to `string`
  * missing values filled as `"__MISSING__"`
* For **CatBoost**:

  * same dataset is passed along with categorical column indices
  * no one-hot encoding
* For **LinearRegression / RandomForest**:

  * `ColumnTransformer` with:

    * `OneHotEncoder(handle_unknown="ignore")` for cats
    * `SimpleImputer(strategy="median")` for numeric columns
  * Wrapped in a `Pipeline` so encoding and imputation are part of the model object

### 3.2 Metrics

**Regression tasks (throughput, latency)**

* `R²`
* `MAE` – Mean Absolute Error
* `RMSE` – Root Mean Squared Error
* `MAPE` – Mean Absolute Percentage Error (used cautiously due to near-zero targets)

**Classification task (good/bad)**

* `Accuracy`
* `F1` score (for the positive class)
* `ROC-AUC`
* Full classification report (precision/recall/F1 per class)

High-level findings (as of current experiments):

* **Throughput:**

  * LinearRegression generalises best (highest test R², lowest RMSE/MAE).
* **Latency:**

  * RandomForest slightly outperforms CatBoost, with very high R² and low error.
* **Classification:**

  * CatBoostClassifier achieves near-perfect ROC-AUC and F1, clearly ahead of RF and LogisticRegression.

These choices feed into the “final model per task” for later pipeline/deployment work.

---

## 4. Repository structure

Expected layout:

```text
.
├── data/
│   ├── prepared/
│   │   ├── base_clean.parquet
│   │   ├── throughput_supervised.parquet
│   │   ├── latency_supervised.parquet
│   │   ├── classification_supervised.parquet
│   │   ├── anomaly_matrix.parquet
│   │   ├── anomaly_scored.parquet
│   │   └── anomaly_top10_per_pair.parquet
│   └── splits/
│       ├── throughput/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── test.parquet
│       ├── latency/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── test.parquet
│       └── classification/
│           ├── train.parquet
│           ├── val.parquet
│           └── test.parquet
├── src/
│   ├── __init__.py
│   ├── mlflow_setup.py
│   ├── train_throughput.py
│   ├── train_latency.py
│   ├── train_classification.py
│   └── log_anomalies.py
├── models/          # saved models (optional; can also be in MLflow artifacts)
└── README.md
```

### Key scripts

* `src/mlflow_setup.py`

  * Central MLflow configuration.
  * Reads `MLFLOW_TRACKING_URI` and credentials from environment variables, and sets the experiment name per task.

* `src/train_throughput.py`

  * Trains and evaluates throughput models.
  * CLI: `--model catboost | rf | linear`
  * Logs params, metrics, and model to MLflow (DagsHub).

* `src/train_latency.py`

  * Trains and evaluates latency models.
  * Same interface as throughput script.

* `src/train_classification.py`

  * Trains and evaluates classification models.
  * CLI: `--model catboost | rf | logreg`.

* `src/log_anomalies.py`

  * Logs anomaly detection summaries to MLflow.
  * Metrics: number/fraction of anomalies, score distribution stats, sample artifacts.

---

## 5. Experiment tracking with MLflow & DagsHub

Experiments are tracked remotely on DagsHub using MLflow’s REST backend.

### 5.1 Environment variables

Before running any training script, set:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
export MLFLOW_TRACKING_USERNAME="<your_username_or_token>"
export MLFLOW_TRACKING_PASSWORD="<your_password_or_token_if_used>"
```

Example:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/Nehaharish98/Thesis-mlops.mlflow"
export MLFLOW_TRACKING_USERNAME="Nehaharish98"
export MLFLOW_TRACKING_PASSWORD="<personal_access_token>"
```

### 5.2 Installing dependencies

Create a virtual environment and install:

```bash
pip install \
  mlflow \
  scikit-learn \
  catboost \
  pandas \
  pyarrow
```

(You can turn this into a `requirements.txt` later.)

---

## 6. Running experiments

From the project root:

### 6.1 Throughput regression

```bash
python -m src.train_throughput --model catboost
python -m src.train_throughput --model rf
python -m src.train_throughput --model linear
```

### 6.2 Latency regression

```bash
python -m src.train_latency --model catboost
python -m src.train_latency --model rf
python -m src.train_latency --model linear
```

### 6.3 Performance classification

```bash
python -m src.train_classification --model catboost
python -m src.train_classification --model rf
python -m src.train_classification --model logreg
```

### 6.4 Anomaly detection summary

```bash
python -m src.log_anomalies
```

Each run:

* loads the appropriate `data/splits/<task>/train|val|test.parquet`,
* trains the selected model,
* evaluates on train/val/test,
* logs metrics & parameters (and the model artifact) to MLflow on DagsHub.

Experiments appear in DagsHub under:

* `throughput`
* `latency`
* `classification`
* `anomaly_detection`

---

## 7. MLOps / future work

This repo currently focuses on:

* **data preparation** and task definition,
* **model benchmarking** with MLflow tracking,
* **remote experiment management** via DagsHub.

Planned / possible next steps for the thesis:

* Integrate with **DVC** for data versioning.
* Orchestrate training & evaluation with **Prefect** or **Airflow**.
* Containerise training/inference with **Docker**.
* Deploy models and tracking backend on **AWS** (e.g. EC2 + S3).
* Add monitoring dashboards (e.g. Grafana) for production metrics.

These pieces together will form the final **cloud-based MLOps pipeline** for inter-datacenter network performance monitoring.

---

## 8. License

*(Add your license here, e.g. MIT, or “All rights reserved” depending on university policy.)*

```

If you tell me your uni name / program, I can add a small line at the top like “Master’s Thesis – <Program>, <University>” so it looks even more official.
::contentReference[oaicite:0]{index=0}
```


# Throughput
python -m src.train_throughput --model catboost
python -m src.train_throughput --model rf
python -m src.train_throughput --model linear

# Latency
python -m src.train_latency --model catboost
python -m src.train_latency --model rf
python -m src.train_latency --model linear

# Classification
python -m src.train_classification --model catboost
python -m src.train_classification --model rf
python -m src.train_classification --model logreg

# Anomaly logging
python -m src.log_anomalies
