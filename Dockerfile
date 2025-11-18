FROM python:3.11-slim

# 1) System setup
WORKDIR /app

# Ensure Python prints straight to logs
ENV PYTHONUNBUFFERED=1

# 2) Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your code + data structure
COPY src ./src
COPY data ./data

# 4) Environment for MLflow (you'll override these in docker run / Kubernetes)
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""

# 5) Expose the API port
EXPOSE 8000

# 6) Start the FastAPI service
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
