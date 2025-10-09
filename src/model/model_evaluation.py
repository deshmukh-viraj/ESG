import os
import json
import joblib
import mlflow
import dagshub
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import yaml

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
        rmse = np.sqrt(mean_absolute_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        metrics = {"RMSE": rmse, "R2": r2}
        logging.info(f"Evauation complete | RMSE={rmse:.4f}, R2={r2:.4f}")
        return metrics
    except Exception as e:
        logging.error(f"Error during Evaluation: {e}")
        raise


def main():
    """Main evaluation pipeline with MLflow and Dagshub tracking"""
    params = load_params()

    os.environ["MLFLOW_TRACKING_USERNAME"] = params["dagshub"]["username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = params["dagshub"]["token"]

    dagshub.init(
        repo_owner=params["dagshub"]["username"],
        repo_name=params["dagshub"]["repo_name"],
        mlflow=True
    )

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{params['dagshub']['username']}/{params['dagshub']['repo_name']}.mlflow"
    )
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    model_path = params["paths"]["model_path"]
    test_data_path = params["paths"]["test_data"]
    target_col = params["paths"]["target_col"]

    df = load_data(test_data_path)
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)

    #log metrics
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_path", model_path)
        mlflow.sklearn.log_model(model, "catboost_model")

        metrics_path = params["paths"]["metrics_output"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(metrics_path)
    logging.info(f"Metrics saved and logged to mlflow: {metrics}")


if __name__=="__main__":
    main()