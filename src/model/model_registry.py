import os
import json
import mlflow
from mlflow.tracking import MlflowClient

try:
    from src.logger import logging
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_params(params_path: str = "params_model_eval.yaml") -> dict:
    """load params from yaml"""
    import yaml
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logging.info("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logging.error("Failed to load parameters: %s", e)
        raise


def setup_dagshub(params: dict):
    """Initialize Dagshub MLflow tracking"""
    dags_param = params.get("dagshub", {})
    username = dags_param.get("username")
    repo_name = dags_param.get("repo_name")

    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
    
    # Use direct MLflow tracking URI instead of dagshub.init()
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    tracking_uri = f'https://dagshub.com/{username}/{repo_name}.mlflow'
    mlflow.set_tracking_uri(tracking_uri)
    logging.info(f"MLflow tracking URI set to: {tracking_uri}")


def load_model_info(file_path: str) -> dict:
    """Load run_id and model path from the experiment info JSON."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.info(f"Model info loaded from {file_path}")
        logging.info(f"Run ID: {model_info.get('run_id')}")
        logging.info(f"Model Path: {model_info.get('model_path')}")
        return model_info
    except FileNotFoundError:
        logging.error("Model info file not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Failed to load model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model and transition it to 'Staging' in mlflow registry"""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Registering model from URI: {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(f"Model registered successfully. Version: {model_version.version}")

        # Try to transition stage
        logging.info("Attempting to transition model to Staging...")
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )
        logging.info(f"Model '{model_name}' transitioned to 'Staging' | version: {model_version.version}")
    
    except Exception as e:
        logging.error("Model Registration Failed: %s", e)
        import traceback
        logging.error(traceback.format_exc())
        raise


def main():
    try:
        params = load_params("params_model_eval.yaml")
        setup_dagshub(params)

        model_info_path = "C:\\ESG\\reports\\experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "esg-catboost-model"
        register_model(model_name, model_info)

        logging.info("Model registration successful")
    
    except Exception as e:
        logging.error("Model registration failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()