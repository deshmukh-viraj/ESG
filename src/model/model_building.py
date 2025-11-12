import os
import pandas as pd
import numpy as np
import json
import pickle
import joblib
from typing import Dict, Any, Optional, Tuple

try:
    from src.logger import logging
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging = logging.getLogger(__name__)

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

try:
    from catboost import CatBoostRegressor
except Exception as e:
    logging.error("Catbost is required for this script. Install using pip install catboost")
    raise


def load_params(params_path: str="params.yaml") -> Dict[str, Any]:
    """load parameters from yaml, return empty dict on failure"""
    
    try:
        import yaml
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = yaml.safe_load(f) or {}
            logging.info("Loaded params from %s", params_path)
            return params
        logging.info("No params file found at %s - using defaults", params_path)
        return {}
    except Exception as e:
        logging.warnning("Failed to laod params.yama: %s", e)
        return {}
    

def load_data(file_path: str, target: str, test_size: float, random_state: int=42):
    """load single dataset and split into train/test"""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at{file_path}")
    df = pd.read_csv(file_path)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset")
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
    logging.info(f"Data loaded and split: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def create_objective(X,y, random_seed: int=42):
    """objective for optuna study"""
    
    def objective(trial):
        params ={
            "iterations": trial.suggest_int("iterations", 200, 1200),
            "depth": trial.suggest_int("depth", 4,10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_int("random_strength", 0.1, 5.0),
            "od_type": "Iter",
            "od_wait": 40,
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": random_seed      
        }  
        model = CatBoostRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        scores = cross_val_score(model, X,y, cv=cv, scoring="neg_root_mean_squared_error")
        return -scores.mean()
    return objective


def run_optuna(X_train, y_train, n_trials: int, timeout: Optional[int], random_seed: int):
    """run optuna tuning"""
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(create_objective(X_train, y_train, random_seed), n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    logging.info(f"Optuna completed. Best Value: {study.best_value}")
    return study.best_params


def train_and_evaluate(best_params: Dict[str, Any], X_train, y_train, X_test, y_test, random_seed: int = 42):
    """train and evaluate catboost model"""
    
    best_params.update({"random_seed": random_seed, "verbose": 0})
    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_absolute_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    metrics = {"rmse": rmse, "r2": r2}
    logging.info(f"Evaluation Complete. RMSE={rmse:.4f}, R2={r2:.4f}")
    return model, metrics


def save_artifacts(model, metrics, output_dir:str, model_name:str, metrics_name: str):
    """save model and metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    metrics_path = os.path.join(output_dir, metrics_name)
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved model to {model_path} and metrics to {metrics_path}")
    return model_path, metrics_path


def main(params_path: str = "params.yaml"):
    params = load_params(params_path)
    cfg = params.get("model_building", {})

    file_path = cfg.get("input_path", "C:\ESG\src\\features\\featured_esg_dataset.csv")
    target = cfg.get("target", "ESG_Overall")
    test_size = cfg.get("split", {}).get("test_size", 0.2)
    random_seed = cfg.get("catboost", {}).get("random_seed", 42)
    n_trials = cfg.get("optuna", {}).get("n_trials", 20)
    timeout = cfg.get("optuna", {}).get("timeout", None)

    output_dir = cfg.get("output", {}).get("model_dir", "C:\\ESG\\models")
    model_name = cfg.get("output", {}).get("model_name", "catboost_final_model.pkl")
    metrics_name = cfg.get("output", {}).get("metrics_name", "final_model_metrics.json")

    X_train, y_train, X_test, y_test = load_data(file_path, target, test_size, random_seed)
    
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")

    X_train.assign(**{target: y_train}).to_csv(train_path, index=False)
    X_test.assign(**{target: y_test}).to_csv(test_path, index=False)
    logging.info(f"Train and Test splits saved: {train_path}, {test_path}")

    best_params = run_optuna(X_train, y_train, n_trials, timeout, random_seed)
    model, metrics = train_and_evaluate(best_params, X_train, y_train, X_test, y_test, random_seed)
    save_artifacts(model, metrics, output_dir, model_name, metrics_name)



if __name__ == "__main__":
    main()