FROM python:3.11-slim

# 1) System setup
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# (Optional but helpful) basic build tools if any wheel needs compiling
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 2) Install dependencies for the SERVICE only
COPY requirements-service.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-service.txt

# 3) Copy your code
COPY src ./src

# 4) Environment for MLflow (can be overridden at docker run)
ENV MLFLOW_TRACKING_URI="http://ec2-44-200-25-155.compute-1.amazonaws.com:5000"

# 5) Expose the API port
EXPOSE 8000

# 6) Start the FastAPI service
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
