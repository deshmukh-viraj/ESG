import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Optional, List, Any

try:
    from src.logger import logging
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available; scaling will be skipped")



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
    

def read_data(file_path: str) -> pd.DataFrame:
    """Read CSV file into a Dataframe"""
    try:
        df = pd.read_csv(file_path)
        logging.info("Loaded data from %s (shape=%s)", file_path, df.shape)
        return df
    except Exception as e:
        logging.info("Failed to load data from %s: %s", file_path, e)
        raise


def identify_column_types(df: pd.DataFrame, params: Dict[str,Any]) -> Dict[str,List[str]]:
    """
    Identify numerical, categorical and datetime columns.
    Allows overriding via params (e.g., params['data_preprocessing']['date_columns']).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","bool",'category']).columns.tolist()
    date_cols = params.get("data_preprocessing", {}).get("date_columns", [])
    if not date_cols:
        inferred_dates = []
        for col in df.columns:
            if col in num_cols or col in cat_cols:
                continue
            sample = df[col].dropna().astype(str).head(5)
            if sample.apply(lambda x: any(c.isalpha() for c in x)).sum() == 0 and any("-" in x or "/" in x for x in sample):
                inferred_dates.append(col)
        date_cols = inferred_dates
    logging.info("Identified columns - numerical: %s, categorica: %s, date-like: %s", num_cols, cat_cols, date_cols)
    return {"numerical": num_cols, "categorical": cat_cols, "date-like": date_cols}


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and lof how many were removed"""
    try:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        after = len(df)
        logging.info("Removed %s duplicate rows", before-after)
        return df
    except Exception as e:
        logging.error("Error while removing duplicates: %s", e)
        raise


def convert_dtypes(df: pd.DataFrame, dtype_map: Dict[str, str]) -> pd.DataFrame:
    """convert column types according to dtype map"""
    try:
        if not dtype_map:
            return df
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    logging.info("Converted column %s to %s", col, dtype)
                except Exception as e:
                    logging.warning("Could not convert column %s to %s: %s", col, dtype, e)
        return df
    except Exception as e:
        logging.error("Error in converting dtypes: %s",e)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "auto",
    numeric_strategy: Optional[str] = None,
    categorical_strategy: Optional[str] = None,
    custom_fill_map: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Handle missing values with several strategies:
    - strategy='drop' => drop rows with any NA
    - strategy='auto' => numeric -> median, categorical -> mode
    - strategy='mean'/'median' => fill numeric with mean/median, categorical with mode
    - strategy='ffill'/'bfill' => forward/backward fill
    - custom_fill_map => dict of column->value to fill explicitly
    """
    try:
        if custom_fill_map:
            df = df.fillna(value=custom_fill_map)
            logging.info("Applied custom fill map for missing values.")
            return df

        if strategy == "drop":
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            logging.info("Dropped rows with missing values (%d removed)", before - len(df))
            return df

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if strategy in ("auto", None):
            
            for col in num_cols:
                median = df[col].median()
                df[col].fillna(median, inplace=True)
            for col in cat_cols:
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)
            logging.info("Filled missing values using auto strategy (median for numeric, mode for categorical)")
            return df

        if strategy in ("mean", "median"):
            for col in num_cols:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            for col in cat_cols:
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)
            logging.info("Filled missing values using %s strategy", strategy)
            return df

        if strategy in ("ffill", "bfill"):
            df.fillna(method=strategy, inplace=True)
            logging.info("Filled missing values using %s", strategy)
            return df

        logging.warning("Unknown missing value strategy '%s'. No filling applied.", strategy)
        return df

    except Exception as e:
        logging.error("Error handling missing values: %s", e)
        raise


def handle_outliers_iqr(df: pd.DataFrame, cols: Optional[List[str]] = None, method: str = "cap", factor: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers using the IQR rule.
    - method='cap' => cap values to the lower/upper whiskers
    - method='remove' => remove rows with outliers
    """
    try:
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            if method == "cap":
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])
            elif method == "remove":
                before = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)
                logging.info("Removed %d rows due to outliers on %s", before - len(df), col)
        if method == "cap":
            logging.info("Capped outliers using IQR method (factor=%s).", factor)
        return df
    except Exception as e:
        logging.error("Error handling outliers: %s", e)
        raise


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    strategy: str = "onehot",
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Encode categorical columns.
    - strategy='onehot' => pd.get_dummies (fast, no sklearn dependency)
    - strategy='ordinal' => pd.factorize per column (stable mapping)
    """
    try:
        if cat_cols is None:
            cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if len(cat_cols) == 0:
            logging.info("No categorical columns to encode.")
            return df

        if strategy == "onehot":
            logging.info("Applying one-hot encoding to categorical columns: %s", cat_cols)
            df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
            return df

        if strategy == "ordinal":
            logging.info("Applying ordinal encoding (factorize) to categorical columns: %s", cat_cols)
            for col in cat_cols:
                df[col], _ = pd.factorize(df[col], sort=True)
            return df

        logging.warning("Unknown encoding strategy '%s'. No encoding applied.", strategy)
        return df
    except Exception as e:
        logging.error("Error encoding categorical variables: %s", e)
        raise


def scale_numerical(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    scaler_name: str = "standard",
) -> pd.DataFrame:
    """Scale numerical columns using sklearn if available; otherwise skip."""
    try:
        if not SKLEARN_AVAILABLE:
            logging.warning("scikit-learn not available; skipping scaling step.")
            return df

        if num_cols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            logging.info("No numerical columns to scale.")
            return df

        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        else:
            logging.warning("Unknown scaler '%s'; skipping scaling.", scaler_name)
            return df

        df[num_cols] = scaler.fit_transform(df[num_cols])
        logging.info("Applied %s scaling to numerical columns: %s", scaler_name, num_cols)
        return df
    except Exception as e:
        logging.error("Error scaling numerical features: %s", e)
        raise


def feature_engineering(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Basic feature engineering:
    - parse date columns and extract year/month/day if present
    - add missing_count feature (number of missing values per row)
    - placeholder for additional feature transformations from params
    """
    try:
        date_cols = params.get("data_preprocessing", {}).get("date_columns", [])
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    logging.info("Parsed date column %s and extracted year/month/day.", col)
                except Exception as e:
                    logging.warning("Failed to parse date column %s: %s", col, e)

        
        df["missing_count"] = df.isnull().sum(axis=1)
        logging.info("Added 'missing_count' feature.")
        return df
    except Exception as e:
        logging.error("Error during feature engineering: %s", e)
        raise


def preprocess_dataframe(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    End-to-end preprocessing orchestrator.
    The order of operations is chosen conservatively for safety and reproducibility.
    """
    try:
       
        df = df.copy()
        logging.info("Starting preprocessing pipeline.")

        # remove duplicates first
        df = remove_duplicates(df)

        # convert dtypes if specified
        dtype_map = params.get("data_preprocessing", {}).get("dtypes", {})
        df = convert_dtypes(df, dtype_map)

        # identify column types
        col_types = identify_column_types(df, params)
        num_cols = col_types["numerical"]
        cat_cols = col_types["categorical"]

        # missing values
        missing_strategy = params.get("data_preprocessing", {}).get("handle_missing", "auto")
        custom_fill_map = params.get("data_preprocessing", {}).get("custom_fill_map", None)
        df = handle_missing_values(df, strategy=missing_strategy, custom_fill_map=custom_fill_map)

        # outliers
        outlier_cfg = params.get("data_preprocessing", {}).get("outliers", {})
        outlier_method = outlier_cfg.get("method", "cap")
        outlier_cols = outlier_cfg.get("columns", num_cols)
        outlier_factor = float(outlier_cfg.get("factor", 1.5))
        df = handle_outliers_iqr(df, cols=outlier_cols, method=outlier_method, factor=outlier_factor)

        # feature engineering
        df = feature_engineering(df, params)

        # encode categorical variables
        encoding_strategy = params.get("data_preprocessing", {}).get("encoding", "onehot")
        df = encode_categoricals(df, cat_cols, strategy=encoding_strategy)

        # scale numerical variables 
        col_types_after = identify_column_types(df, params)
        num_cols_after = col_types_after["numerical"]
        scaling = params.get("data_preprocessing", {}).get("scaling", {}).get("method", None)
        if scaling:
            df = scale_numerical(df, num_cols_after, scaler_name=scaling)

        logging.info("Preprocessing pipeline completed successfully. Resulting shape: %s", df.shape)
        return df
    except Exception as e:
        logging.error("Error in preprocess_dataframe: %s", e)
        raise


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
    """
    Entrypoint for running preprocessing as a script.
    Example usage:
        python -m src.data_preprocessing
    """
    try:
        params = load_params(params_path)
        input_path = params.get("data_preprocessing", {}).get("input_path", params.get("data_ingestion", {}).get("local_path", "C:\ESG\data\\raw\company_esg_financial_dataset.csv"))
        output_dir = params.get("data_preprocessing", {}).get("output_dir", params.get("storage", {}).get("processed_data_dir", "data/processed"))
        file_name = params.get("data_preprocessing", {}).get("output_file_name", "processed_esg_dataset.csv")

        df = read_data(input_path)
        processed_df = preprocess_dataframe(df, params)
        save_processed(processed_df, output_dir, file_name)

    except Exception as e:
        logging.error("Data preprocessing failed: %s", e)


if __name__ == "__main__":
    main()

