# Thesis-mlops

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
