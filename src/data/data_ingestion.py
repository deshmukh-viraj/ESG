import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections.s3_connection import s3_operations
from src.connections import s3_connection
import os
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

def load_params(params_path: str) -> dict:
    """Load parameters from a ymal file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters from {params_path}: {e}")
        raise

def load_data_S3(params: dict) -> pd.DataFrame:
    """Load data from a S3"""
    try:
        if all(k in params["data_ingestion"] for k in ("s3_bucket", "aws_access_key", "aws_secret_key", "s3_file_key")):
            logging.info("Loading data from AWS S3....")
            s3 = s3_operations(
                bucket_name=params["data_ingestion"]["s3_bucket"],
                aws_access_key=params["data_ingestion"]["aws_access_key"],
                aws_secret_key=params["data_ingestion"]["aws_secret_key"],
                region_name=params["data_ingestion"].get("region_name", "us-east-1")

            )
            return s3.fetch_file_from_s3(params['data_ingestion']['s3_file_key'])
        
        data_url = params['data_ingestion'].get("data_url")
        if data_url and data_url.startswith("http"):
            logging.info(f"loading data from URL: {data_url}")
            return pd.read_csv(data_url)

        local_path = params['data_ingestion'].get("local_path")
        if local_path and os.path.exists(local_path):
            logging.info("Loading data from local file")
            return pd.read_csv(local_path)
        raise ValueError("No valid data source found in pramas.yaml")
    except Exception as e:
        logging.error(f"Failed to laod data: {e}")
        raise

# def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Preprocess the data according to params.yaml setting"""
    try:
        drop_cols = params["data_ingestion"].get("drop_columns", [])
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True, errors='ignore')
            logging.info(f"Dropped columns: {drop_cols}")

            missing_strategy = params["data_ingestion"].get("handle_missing", "drop")
            if missing_strategy == "drop":
                df.dropna(inplace=True)
                logging.info("Dropped rows with missing values")
            elif missing_strategy == "mean":
                df.fillna(df.mean(), inplace=True)
                logging.info("Filled missing values with ")
            elif missing_strategy == "median":
                df.fillna(df.median(), inplace=True)
                logging.info("Filled missing values with median")
            elif missing_strategy == 'mode':
                df.fillna(df.mode().iloc[0], inplace=True)
                logging.info("Filled missing values with columns modes")
            logging.info(f"preprocessing complete")
            return df
        
    except Exception as e:
        logging.error(f"Error During preprocessimg: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, params: dict) -> None:
    """Save train and test datasets"""
    try:
        raw_data_path = params['storage']['raw_data_dir']
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.info(f"Train and test data saved to {raw_data_path}")
    
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main():
    try:
        params = load_params('params.yaml')
        test_size = params["data_ingestion"]["test_size"]
        random_state = params["data_ingestion"]["random_state"]
        # test_size=0.24
        # random_state=42
        BUCKET_NAME = "esg.csv"
        AWS_ACCESS_KEY = os.getenv("aws_access_key")
        AWS_SECRET_KEY = os.getenv("aws_secret_key")
        

        # df = load_data(file_path="C:\ESG\data\\raw\company_esg_financial_dataset.csv")
        s3 = s3_connection.s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
        df = s3.fetch_file_from_s3("company_esg_financial_dataset.csv")

        final_df = preprocess_data(df, params)
        # final_df = df
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=random_state)
        save_data(train_data, test_data, params)
    
    except Exception as e:
        logging.error(f"Data ingestion process failed: {e}")

    
if __name__=="__main__":
    main()
