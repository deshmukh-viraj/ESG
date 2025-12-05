import os
import mlflow
print("MLFLOW TRACKING:", mlflow.get_tracking_uri())
print("MLFLOW REGISTRY:", mlflow.get_registry_uri())
print("CWD:", os.getcwd())
