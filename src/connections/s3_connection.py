import boto3
import pandas as pd
import logging
from io import StringIO
from src.logger import configure_logger

configure_logger()
logger = logging.getLogger(__name__)

class s3_operations:
    def __init__(self, bucket_name:str, aws_access_key: str, aws_secret_key: str, region_name: str ="us-east-1"):
        """
        Initialize the s3_operations class with AWS credentials"""
        self.bucket_name = bucket_name
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                region_name=region_name
            )
            logger.info(f"S3 connection initialize for bucket {bucket_name}")
        except Exception as e:
            logger.exception(f"Failed to initialize S3 connection: {e}")
            raise
    
    def fetch_file_from_s3(self, file_key: str) -> pd.DataFrame:
        """
        Fetches a CSV file from the s3 bucket and return it as a pandas Dataframe
        :param file_key: S3 file path (e.g. 'data/data.csv')
        :return Pandas Dataframe
        """
        try:
            logger.info(f"Fetching file '{file_key}' from s3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logger.info(f"Successfully fetched '{file_key}' from S3 with {len(df)} records.")
            return df
        except Exception as e:
            logger.exception(f"Failed to fetch '{file_key}' from s3: {e}")
            raise