import os
import json
import joblib
import mlflow
import dagshub
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import yaml
from dotenv import load_dotenv

load_dotenv()


try:
    from src.logger import logging
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)


def load_params(params_path: str ="params_model_eval.yaml") -> dict:
    """load parameters from the yaml file"""
    
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load params from {params_path}: {e}")
        raise


def load_model(model_path: str):
    """load train catboost model"""
    
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error Loading Model: {e}")
        raise


def load_data(test_path: str) -> pd.DataFrame:
    """load test dataset"""
    
    try:
        df = pd.read_csv(test_path)
        logging.info(f"Test data loaded from {test_path}, shape={df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise


def evaluate_model(model, X_test, y_test) -> dict:
    """evalute model performance"""
    
    try:
        y_pred = model.predict(X_test)
        # Fixed: Use mean_squared_error instead of mean_absolute_error for RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        metrics = {"RMSE": rmse, "R2": r2}
        logging.info(f"Evaluation complete | RMSE={rmse:.4f}, R2={r2:.4f}")
        return metrics
    except Exception as e:
        logging.error(f"Error during Evaluation: {e}")
        raise


def main():
    """main evaluation pipeline with mlflow and dagshub tracking"""
    
    params = load_params()

    
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
    if not dagshub_username:
        raise EnvironmentError("DAGSHUB_USERNAME environment variable is not set")
    

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    logging.info(f"Authenticating with DagsHub as user: {dagshub_username}")

    #initialize DagsHub
    dagshub.init(
        repo_owner=params["dagshub"]["username"],
        repo_name=params["dagshub"]["repo_name"],
        mlflow=True
    )

    dagshub_url = "https://dagshub.com"
    tracking_uri = f"{dagshub_url}/{params['dagshub']['username']}/{params['dagshub']['repo_name']}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    
    logging.info(f"MLflow tracking URI set to: {tracking_uri}")

    model_path = params["paths"]["model_path"]
    test_data_path = params["paths"]["test_data"]
    target_col = params["paths"]["target_col"]

    df = load_data(test_data_path)
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)

    
    with mlflow.start_run() as run:

        mlflow.log_metrics(metrics)
        mlflow.log_param("model_path", model_path)

        #log model hyperparameters
        try:
            model_params = model.get_params()
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            logging.info("Model hyperparameters logged to MLflow")
        except Exception as e:
            logging.warning(f"Could not log model hyperparameters: {e}")

        #log model artifact
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir:
            save_path = os.path.join(model_dir, "catboost_model.cbm")
            model.save_model(save_path)
            mlflow.log_artifact(save_path, artifact_path="model")
            logging.info(f"Model artifact logged to MLflow from {save_path}")

        #save experiment info for model registry
        experiment_info_path = os.path.join("reports", "experiment_info.json")
        model_info = {
            "run_id": run.info.run_id,
            "model_path": "model/catboost_model.cbm"   
        }

        os.makedirs(os.path.dirname(experiment_info_path), exist_ok=True)
        with open(experiment_info_path, "w") as f:
            json.dump(model_info, f, indent=4)

        logging.info(f"Model experiment info saved to {experiment_info_path}")

        #save metrics
        metrics_path = params["paths"]["metrics_output"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(metrics_path)
        
    logging.info(f"Metrics saved and logged to MLflow: {metrics}")


if __name__=="__main__":
    main()