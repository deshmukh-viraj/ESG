#settings.py
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")


REPO_OWNER = "virajdeshmukh080818"
REPO_NAME = "ESG"

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT.joinpath('mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_URI = f"file:///{PROJECT_ROOT.joinpath('artifacts').as_posix()}"
MODEL_NAME = "my-esg-catboost-model"
MODEL_VERSION = "1"

TRAIN_FEAT_ORDER = [
    'CompanyID', 'CompanyName', 'Industry', 'Region', 'Year',
    'Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
    'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
    'CarbonEmissions', 'WaterUsage', 'EnergyConsumption',
    'missing_count', 'Revenue_per_carbon', 'Revenue_per_Water',
    'Revenue_per_Energy', 'ProfitMargin_x_ESG', 'ESG_Pillar_Mean',
    'ESG_Pillar_Std', 'Year_Since_2015', 'Year_Normalised',
    'ESG_Overall_Lag1', 'ESG_Overall_RollingMean',
    'ESG_Overall_RollingStd'
]