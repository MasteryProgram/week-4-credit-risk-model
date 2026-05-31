# src/data_processing.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_raw_data(data_path: str = "data/raw/xente_transaction_data.csv") -> pd.DataFrame:
    """Load raw transaction data."""
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: convert timestamp, handle basic issues."""
    df = df.copy()
    
    # Convert TransactionStartTime to datetime
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Create a few basic derived columns for EDA
    if 'TransactionStartTime' in df.columns:
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
    
    print("Basic cleaning completed.")
    return df

def get_data_summary(df: pd.DataFrame):
    """Print overview statistics."""
    print("=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values (%):\n{(df.isnull().sum() / len(df) * 100).round(3)}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    print("\n=== Numerical Features Summary ===")
    print(df.describe().round(2))
    
    print("\n=== Categorical Features ===")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head(5))
    
    return df

# For future RFM (we'll expand this in Task 4)
def calculate_rfm(df: pd.DataFrame, snapshot_date: str = "2019-01-01") -> pd.DataFrame:
    """Calculate RFM metrics per customer (placeholder for now)."""
    snapshot = pd.to_datetime(snapshot_date)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot - x.max()).days,  # Recency
        'TransactionId': 'count',                                     # Frequency
        'Value': 'sum'                                                # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary'
    })
    return rf