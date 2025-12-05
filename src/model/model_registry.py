import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from mlflow.artifacts import download_artifacts


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


def setup_dagshub(params: dict, run_id: str):
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

    client_remote = MlflowClient(tracking_uri=tracking_uri)
    run_data = client_remote.get_run(run_id=run_id)
    metrics = run_data.data.metrics
    params_run = run_data.data.params

    logging.info(f"MLflow tracking URI set to: {tracking_uri}")
    logging.info(f"Fetched {len(metrics)} metrics from Dagshub run {run_id}")
    logging.info(f"Fetched {len(params_run)} parameters from Dagshub run {run_id}")


    # Local registry URI (to avoid Dagshub unsupported endpoint errors)
    # registry_uri = "sqlite:///C:/ESG/mlflow.db"
    # mlflow.set_registry_uri(registry_uri)
    # logging.info(f"Mlflow registry URI set to: {registry_uri}")
    
    return metrics, params_run



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
    


def windows_path_to_uri(win_path: str) -> str:
    """convert windows path to proper file:// uri"""

    abs_path =Path(win_path).absolute()
    uri = abs_path.as_uri()
    return uri



def register_model(model_name: str, model_info: dict, metrics: dict, params_run: dict):
    """
    Download model from Dagshub and re-logs it locally.
    Register the model by downloading artifacts from Dagshub and storing locally
    """
    try:

        dagshub_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Downloading model from Dagshub URI: {dagshub_uri}")

        downloaded_path = download_artifacts(dagshub_uri)
        logging.info(f"Model downloaded locally to: {downloaded_path}")

        downloaded_path = Path(downloaded_path)
        if not downloaded_path.exists():
            raise FileNotFoundError("Downloaded model path does not exist")
        

        base_dir = Path("C:/ESG")
        base_dir.mkdir(parents=True, exist_ok=True)

        mlruns_dir = base_dir / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)

        db_path = base_dir / "mlflow.db"                
        local_uri = f"sqlite:///{db_path.as_posix()}"
        artifact_root = f"file:///{mlruns_dir.as_posix()}"

        logging.info(f"Local tracking URI: {local_uri}")
        logging.info(f"Artifact root: {artifact_root}")
        
        mlflow.set_tracking_uri(local_uri)
        mlflow.set_registry_uri(local_uri)
        logging.info(f"Switched to local Mlfow: {local_uri}")

        #create a new run to log the model locally
        with mlflow.start_run(run_name=f"local_deployment_{model_name}") as run:
            mlflow.log_artifact(downloaded_path, artifact_path="model")


            mlflow.log_metrics(metrics)
            logging.info(f"Logged {len(metrics)} metrics locally")

            mlflow.log_params(params_run)
            logging.info(f"logged {len(params_run)} parameters locally")

            local_ru_id = run.info.run_id
            logging.info(f"Model logged locally with run ID: {local_ru_id}")

        client = MlflowClient(registry_uri=local_uri)
        model_uri = f"runs:/{local_ru_id}/model"

        logging.info(f"Registering model from Local URI: {model_uri}")


        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=local_ru_id
            
        )
        logging.info(f"Model registered successfully. Version: {model_version.version}")

        for metric_name, metric_value in metrics.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=f"metric_{metric_name}",
                value=str(metric_value)
            )

            logging.info(f"Tagged metric: {metric_name} = {metric_value}")

        for param_name, param_value in params_run.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=f"param_{param_name}",
                value=str(param_value)
            )
            logging.info(f"Tagged parameter: {param_name} = {param_value}")
        
        
        # Try to transition stage
        logging.info("Attempting to transition model to Staging...")
        
        client.set_registered_model_alias(
            name=model_name,
            version=model_version.version,
            alias='staging'
        )
        logging.info(f"Model '{model_name}' transitioned to 'Staging' | version: {model_version.version}")
        return model_version.version
    

    except Exception as e:
        error_msf = str(e)
        if "unsupported endpoint" in error_msf or "INTERNAL ERROR" in error_msf:
            logging.warning("Dagshub does not support model registry API. Model logged, but not registered.")
        else:
            logging.error("Model Registration Failed: %s", e)
            import traceback
            logging.error(traceback.format_exc())
            raise


def main():
    try:
        params = load_params("params_model_eval.yaml")
        
        model_info_path = "C:\\ESG\\reports\\experiment_info.json"
        model_info = load_model_info(model_info_path)

        metrics, params_run = setup_dagshub(params, model_info["run_id"])
        model_name = "my-esg-catboost-model"

        version = register_model(model_name, model_info, metrics, params_run)

        logging.info(f"Model registration successful. version: {version}")
        logging.info(f"Model artifacts are now stored locally in C:/ESG/mlflow.db")
        logging.info(f"now load the model using version '{version}'")
        logging.info(f"Update your settings.py: MODEL_VERSION = '{version}'")
    
    except Exception as e:
        logging.error("Model registration failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()