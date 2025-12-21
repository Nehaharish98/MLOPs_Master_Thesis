from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.generic.compute import Rack
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.monitoring import Prometheus, Grafana

# One PNG will be generated: mlops_architecture_clarity.png
with Diagram(
    "MLOps Architecture",
    filename="mlops_architecture_clarity",
    outformat="png",
    show=False,
    direction="TB",   # Top -> Bottom : Client -> Serving -> MLOps
):

    # ------------------------ Client Plane ------------------------ #
    with Cluster("Client Plane (Interface)"):
        user = User("User (Browser)")
        streamlit = Rack("Streamlit App")

        # User interacts with the Streamlit UI
        user >> Edge(label="HTTP/S") >> streamlit

    # ------------------------ Serving Plane ----------------------- #
    with Cluster("Serving Plane (Backend)"):
        fastapi = Rack("FastAPI Inference Server")
        data_store = PostgreSQL("Data Store\n(Postgres / S3)")
        prometheus = Prometheus("Prometheus")

        # UI calls the API
        streamlit >> Edge(label="HTTP/JSON\nrequest") >> fastapi

        # Logging & metrics
        fastapi >> Edge(label="Log input +\nprediction") >> data_store
        fastapi >> Edge(label="Expose /metrics") >> prometheus

    # ------------------------ MLOps Control Plane ----------------- #
    with Cluster("MLOps Control Plane"):
        # ---- Continuous Training (CT) loop ---- #
        with Cluster("Continuous Training (CT)"):
            prefect = Rack("Prefect\n(Orchestrator)")
            dagshub = Rack("DagsHub\n(Git + DVC + Artifacts)")
            mlflow = Rack("MLflow\nTracking & Registry")

            # Prefect orchestrates training
            prefect >> Edge(label="Trigger\ntraining flow") >> dagshub
            prefect >> Edge(label="Trigger\ntraining flow") >> mlflow

            # Code + data for training, tracked in MLflow
            dagshub >> Edge(label="Code + data\nfor training") >> mlflow

        # ---- Continuous Monitoring (CM) loop ---- #
        with Cluster("Continuous Monitoring (CM)"):
            evidently = Rack("Evidently\n(Drift Analyzer)")
            monitoring_db = PostgreSQL("Monitoring DB")
            grafana = Grafana("Grafana Dashboards")

            # Data logging bridge: Serving -> Data Store -> Evidently
            data_store >> Edge(label="Prediction logs") >> evidently
            dagshub >> Edge(label="Reference\n(training) data") >> evidently
            evidently >> Edge(label="Drift metrics") >> monitoring_db

            # Monitoring bridge: Grafana as cockpit
            grafana << Edge(label="Infra metrics") >> prometheus
            grafana << Edge(label="Drift / quality") >> monitoring_db

    # ------------------------ Cross-plane bridges ----------------- #

    # Deployment Bridge: MLflow Registry -> FastAPI
    mlflow >> Edge(label="Load\nProduction model") >> fastapi
