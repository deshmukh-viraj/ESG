import logging
import os
import numpy as np
import yaml
import pandas as pd
from typing import Dict, Any

try:
    from src.logger import logging
except Exception:
    logging.basicConfig(level=logging.INFO)


def load_params(params_path: str="params.yaml") -> Dict[str, Any]:
    """Load params from the yaml file, returns empty dict on failure"""
    try:
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = yaml.safe_load(f) or {}
            logging.info(f"Paramenters loaded from {params_path}")
            return params
        logging.info(f"No params.yaml found at {params_path}")
        return {}
    except Exception as e:
        logging.info(f"Failed to load params from {params_path}: {e}")
        return {}


def _add_financial_ratios(df:pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """revenue efficiency on environmental metrics"""

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


def _add_esg_pillar_scores(df:pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """create ESG pillar score if missing"""

    if not cfg.get("esg_pillars", True):
        return df
    if "ESG_Overall" in df.columns:
        if "ESG_Environmental" not in df.columns:
            df["ESG_Environmental"] = df["ESG_Overall"] * np.random.uniform(0.85, 1.15, size=len(df))
        if "ESG_Governance" not in df.columns:
            df["ESG_Governance"] = df["ESG_Overall"] * np.random.uniform(0.85, 1.15, size=len(df))

    pillars = ["ESG_Environmental", "ESG_Social", "ESG_Governance"]
    if set(pillars).issubset(df.columns):
        df["ESG_Pillar_Mean"] = df[pillars].mean(axis=1)
        df["ESG_Pillar_Std"] = df[pillars].std(axis=1)

    logging.info("Added ESG pillar features")
    return df


def _add_temporal_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add year-based temporal features."""

    if not cfg.get("temporal", True):
        return df

    date_cols = cfg.get("date_columns", [])
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(int)
        df["Year_Since_2015"] = df["Year"] - 2015
        df["Year_Normalised"] = (df["Year"] - df["Year"].min()) / (df["Year"].max() - df["Year"].min())

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month

    logging.info("Added temporal features")
    return df


def _add_lag_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add 1-year lagged ESG metrics per compan"""

    if not cfg.get("lag_features", True):
        return df

    if {"CompanyID", "Year", "ESG_Overall"}.issubset(df.columns):
        df = df.sort_values(["CompanyID", "Year"])
        df["ESG_Overall_Lag1"] = df.groupby("CompanyID")["ESG_Overall"].shift(1)
        df["ESG_Overall_Lag1"] = df["ESG_Overall_Lag1"].fillna(df["ESG_Overall"])

    logging.info("Added lag features")
    return df


def _add_rolling_stats(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Rolling mean/std of ESG scores (3-year window)"""

    if not cfg.get("rolling_stats", True):
        return df

    if {"CompanyID", "Year", "ESG_Overall"}.issubset(df.columns):
        window = cfg.get("rolling_window", 3)
        df = df.sort_values(["CompanyID", "Year"])
        df["ESG_Overall_RollingMean"] = (
            df.groupby("CompanyID")["ESG_Overall"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df["ESG_Overall_RollingStd"] = (
            df.groupby("CompanyID")["ESG_Overall"]
            .transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        )

    logging.info("Added rolling stats features")
    return df



def feature_engineering(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Apply all requested feature engineering step"""

    cfg = params.get("feature_engineering", {})
    if not cfg.get("enabled", True):
        logging.info("Feature engineering disabled in params")
        return df

    logging.info("Starting feature engineering")
    df = df.copy()

    df = _add_financial_ratios(df, cfg)
    df = _add_esg_pillar_scores(df, cfg)
    df = _add_temporal_features(df, cfg)
    df = _add_lag_features(df, cfg)
    df = _add_rolling_stats(df, cfg)

    logging.info("Feature engineering complete. New shape: %s", df.shape)
    return df


def save_processed(df: pd.DataFrame, output_dir: str, file_name: str = "processed.csv") -> str:
    """Save processed dataframe to CSV and return the file path."""

    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, file_name)
        df.to_csv(out_path, index=False)
        logging.info("Saved processed data to %s", out_path)
        return out_path
    except Exception as e:
        logging.error("Failed to save processed data: %s", e)
        raise


def main(params_path: str = "params.yaml") -> None:
    """Run feature engineering stand-alone"""
    params = load_params(params_path)
    input_path = params.get("feature_engineering", {}).get("input_path",
                    params.get("data_preprocessing", {}).get("output_dir", "data/interim")
                    + "/" + params.get("data_preprocessing", {}).get("output_file_name", "processed_esg_dataset.csv"))
    output_dir = params.get("feature_engineering", {}).get("output_dir",
                     params.get("storage", {}).get("featured_data_dir", "src/features"))
    output_name = params.get("feature_engineering", {}).get("output_file_name", "featured_esg_dataset.csv")

    df = pd.read_csv(input_path)
    df = feature_engineering(df, params)
    save_processed(df, output_dir, output_name)


if __name__ == "__main__":
    main()      