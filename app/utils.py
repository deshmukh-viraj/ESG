import logging
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
from typing import Dict, Any, Optional
from settings import MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_VERSION
import joblib
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def load_model_from_registry(model_name: str = MODEL_NAME, model_version: Optional[str] = MODEL_VERSION):
#     """
#     load the latest version of the model from mlflow model registry
#     """
#     # If specific version is provided, load it
#     if model_version:
#         model_uri = f"models:/{model_name}/{model_version}"
#         logger.info(f"Loading model {model_name} version {model_version}")
#         try:
#             return mlflow.pyfunc.load_model(model_uri)
#         except Exception as e:
#             logger.error(f"Failed to load model version {model_version}:{e}")

            
#     # Try to load from staging alias first
#     try:
#         staging_version = client.get_model_version_by_alias(model_name, "staging")
#         model_uri = f"models:/{model_name}/{staging_version.version}"
#         logger.info(f"Loading model from staging: {model_uri}")
#         return mlflow.pyfunc.load_model(model_uri)
#     except Exception as e:
#         logger.warning(f"No staging alias found: {e}. Loading latest version instead...")
    
#     # Fallback: Load the latest version
   
#     versions = client.search_model_versions(f"name='{model_name}'")
#     if not versions:
#         raise RuntimeError(f"No versions found for model {model_name}")
        
#     versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
#     latest_version = versions[0].version
#     model_uri = f"models:/{model_name}/{latest_version}"

#     logger.info(f"Loading latest model version: {model_uri}")
#     return mlflow.pyfunc.load_model(model_uri)


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(registry_uri=MLFLOW_TRACKING_URI, tracking_uri=MLFLOW_TRACKING_URI) 


def load_model_from_registry(model_name: str = MODEL_NAME, model_version: Optional[str] = MODEL_VERSION):
    """
    Loads the CatBoost model using the native loader by manually downloading the artifact.
    This bypasses the incompatible mlflow.pyfunc.load_model method.
    """
    
    # 1. Resolve the specific version
    if not model_version:
        # Fallback logic to find the latest version (simplified from your original code)
        versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number DESC"])
        if not versions:
            raise RuntimeError(f"No versions found for model {model_name}")
        model_version = versions[0].version
        logger.info(f"Loading latest model version: {model_version}")


    try:
        # Get the Model Version object to find the Run ID
        mv = client.get_model_version(model_name, model_version)
        run_id = mv.run_id
        
        # 2. Define the exact artifact path where the file was logged
        # This MUST match the path used in model_building.py: artifact_path="model"
        artifact_file_path = "model/catboost_model.cbm"
        
        logger.info(f"Downloading artifact: runs:/{run_id}/{artifact_file_path}")

        # 3. Download the specific file locally
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path=artifact_file_path
        )
        
        logger.info(f"Model artifact downloaded to: {local_path}")

        # 4. Load using CatBoost's NATIVE loader
        model = CatBoostRegressor() 
        model.load_model(local_path)
        
        logger.info("CatBoost model loaded successfully using native loader.")
        return model

    except Exception as e:
        logger.error(f"Failed to load model version {model_version}: {e}")
        # Re-raise the exception to stop the application startup
        raise



def _add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if {"Revenue", "CarbonEmissions"}.issubset(df.columns):
        df["Revenue_per_carbon"] = df["Revenue"] / (df["CarbonEmissions"] + 1e-3)
    if {"Revenue", "WaterUsage"}.issubset(df.columns):
        df["Revenue_per_Water"] = df["Revenue"] / (df["WaterUsage"] + 1e-3)
    if {"Revenue", "EnergyConsumption"}.issubset(df.columns):
        df["Revenue_per_Energy"] = df["Revenue"] / (df["EnergyConsumption"] + 1e-3)
    if {"ProfitMargin", "ESG_Overall"}.issubset(df.columns):
        df["ProfitMargin_x_ESG"] = df["ProfitMargin"] * df["ESG_Overall"]
    logging.info("Added financial ratio features")
    return df

def _add_esg_pillar_scores(df: pd.DataFrame) -> pd.DataFrame:
    if "ESG_Overall" in df.columns:
        if "ESG_Environmental" not in df.columns:
            df["ESG_Environmental"] = df["ESG_Overall"] * np.random.uniform(0.85, 1.15, size=len(df))
        if "ESG_Governance" not in df.columns:
            df["ESG_Governance"] = df["ESG_Overall"] * np.random.uniform(0.85, 1.15, size=len(df))

    pillars = ["ESG_Environmental", "ESG_Social", "ESG_Governance"]
    if set(pillars).issubset(df.columns):
        df["ESG_Pillar_Mean"] = df[pillars].mean(axis=1)
        df["ESG_Pillar_Std"] = df[pillars].std(axis=1)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(int)
        df["Year_Since_2015"] = df["Year"] - 2015
        # avoid divide-by-zero for single-row; normalize with global min/max if available
        ymin = df["Year"].min()
        ymax = df["Year"].max() if df["Year"].max() != ymin else ymin + 1
        df["Year_Normalised"] = (df["Year"] - ymin) / (ymax - ymin)
    return df


def _add_lag_feature_from_input(df: pd.DataFrame) -> pd.DataFrame:
    if {"CompanyID", "Year", "ESG_Overall"}.issubset(df.columns):
        df = df.sort_values(["CompanyID", "Year"])
        df["ESG_Overall_Lag1"] = df.groupby("CompanyID")["ESG_Overall"].shift(1)
        df["ESG_Overall_Lag1"] = df["ESG_Overall_Lag1"].fillna(df["ESG_Overall"])
    return df


def feature_engineering_for_inference(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Re-create features used by model during training for inference"""

    df = df.copy()
    df = _add_financial_ratios(df)
    df = _add_esg_pillar_scores(df)
    df = _add_temporal_features(df)
    df = _add_lag_feature_from_input(df)
    logging.info("Feature engineering for inference complete")
    return df