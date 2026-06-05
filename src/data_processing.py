import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TARGET_COL = 'is_high_risk'

CATEGORICAL_COLS = [
    'CurrencyCode', 'ProviderId', 'ProductId',
    'ProductCategory', 'ChannelId', 'PricingStrategy',
]

NUMERICAL_COLS = [
    'total_amount', 'avg_amount', 'std_amount', 'transaction_count',
    'total_value', 'recency_days', 'fraud_count', 'txn_hour_mean',
    'txn_day_mean', 'txns_per_day', 'value_per_day', 'weekend_ratio',
    'unique_products', 'unique_categories', 'unique_providers',
    'amount_cv', 'avg_ticket_size',
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    logger.info(f'Loading raw data from: {filepath}')
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    logger.info(f'Loaded {len(df)} transactions, {df["CustomerId"].nunique()} unique customers.')
    return df


def compute_customer_features(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    logger.info(f'Computing customer features with snapshot date: {snapshot_date}')
    df = df.copy()
    df['txn_hour'] = df['TransactionStartTime'].dt.hour
    df['txn_day'] = df['TransactionStartTime'].dt.day
    df['dayofweek'] = df['TransactionStartTime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    customer_features = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        std_amount=('Amount', 'std'),
        transaction_count=('TransactionId', 'count'),
        total_value=('Value', 'sum'),
        fraud_count=('FraudResult', 'sum'),
        first_txn=('TransactionStartTime', 'min'),
        last_txn=('TransactionStartTime', 'max'),
        txn_hour_mean=('txn_hour', 'mean'),
        txn_day_mean=('txn_day', 'mean'),
        ProductCategory=('ProductCategory', lambda x: x.mode()[0]),
        ChannelId=('ChannelId', lambda x: x.mode()[0]),
        ProviderId=('ProviderId', lambda x: x.mode()[0]),
        ProductId=('ProductId', lambda x: x.mode()[0]),
        CurrencyCode=('CurrencyCode', lambda x: x.mode()[0]),
        PricingStrategy=('PricingStrategy', lambda x: str(x.mode()[0])),
    ).reset_index()

    customer_features['std_amount'] = customer_features['std_amount'].fillna(0)
    customer_features['recency_days'] = (
        snapshot_date - customer_features['last_txn']
    ).dt.days
    customer_features['customer_age_days'] = (
        snapshot_date - customer_features['first_txn']
    ).dt.days.clip(lower=1)
    customer_features['txns_per_day'] = (
        customer_features['transaction_count'] / customer_features['customer_age_days']
    )
    customer_features['value_per_day'] = (
        customer_features['total_value'] / customer_features['customer_age_days']
    )
    customer_features['avg_ticket_size'] = (
        customer_features['total_value'] / customer_features['transaction_count'].replace(0, 1)
    )
    customer_features['amount_cv'] = (
        customer_features['std_amount'] / customer_features['avg_amount'].abs().clip(lower=1)
    )

    preferred_hour = (
        df.groupby('CustomerId')['txn_hour']
        .agg(lambda x: x.mode().iloc[0])
        .rename('preferred_hour')
    )
    weekend_ratio = (
        df.groupby('CustomerId')['is_weekend']
        .mean()
        .rename('weekend_ratio')
    )
    diversity = (
        df.groupby('CustomerId').agg(
            unique_products=('ProductId', 'nunique'),
            unique_categories=('ProductCategory', 'nunique'),
            unique_providers=('ProviderId', 'nunique'),
        ).reset_index()
    )

    customer_features = customer_features.merge(preferred_hour, on='CustomerId', how='left')
    customer_features = customer_features.merge(weekend_ratio, on='CustomerId', how='left')
    customer_features = customer_features.merge(diversity, on='CustomerId', how='left')

    channel_counts = pd.crosstab(df['CustomerId'], df['ChannelId'])
    if not channel_counts.empty:
        channel_counts = channel_counts.reset_index()
        customer_features = customer_features.merge(channel_counts, on='CustomerId', how='left')
        channel_cols = [c for c in customer_features.columns if c.startswith('channel_')]
        customer_features[channel_cols] = customer_features[channel_cols].fillna(0)

    customer_features['preferred_hour'] = customer_features['preferred_hour'].fillna(-1).astype(int)
    customer_features['weekend_ratio'] = customer_features['weekend_ratio'].fillna(0)
    customer_features['unique_products'] = customer_features['unique_products'].fillna(0).astype(int)
    customer_features['unique_categories'] = customer_features['unique_categories'].fillna(0).astype(int)
    customer_features['unique_providers'] = customer_features['unique_providers'].fillna(0).astype(int)

    logger.info('Customer feature assembly complete: %s customers', len(customer_features))
    return customer_features


def build_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_high_risk'] = (df['fraud_count'] >= 1).astype(int)

    if df['is_high_risk'].sum() == 0:
        logger.warning('No fraud-positive customers found; using risk proxy thresholds instead.')
        non_fraud = df
    else:
        non_fraud = df[df['is_high_risk'] == 0]

    if not non_fraud.empty:
        total_99 = non_fraud['total_amount'].quantile(0.99)
        avg_99 = non_fraud['avg_amount'].quantile(0.99)
        proxy_mask = (
            (df['is_high_risk'] == 0)
            & (df['total_amount'] >= total_99)
            & (df['avg_amount'] >= avg_99)
        )
        df.loc[proxy_mask, 'is_high_risk'] = 1
        logger.info('Added %d proxy high-risk customers from extreme thresholds.', int(proxy_mask.sum()))

    risk_counts = df['is_high_risk'].value_counts()
    logger.info('Final is_high_risk distribution:\n%s', risk_counts)
    logger.info('High-risk percentage: %.3f%%', df['is_high_risk'].mean() * 100)
    return df


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None, num_strategy: str = 'median', cat_strategy: str = 'most_frequent'):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])
        all_cols = list(X.columns)
        self.numerical_cols_ = [c for c in (self.numerical_cols or all_cols) if c in all_cols and pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in (self.categorical_cols or all_cols) if c in all_cols and not pd.api.types.is_numeric_dtype(X[c])]
        self.num_imputer_ = SimpleImputer(strategy=self.num_strategy)
        self.cat_imputer_ = SimpleImputer(strategy=self.cat_strategy, fill_value='missing')
        if self.numerical_cols_:
            self.num_imputer_.fit(X[self.numerical_cols_])
        if self.categorical_cols_:
            self.cat_imputer_.fit(X[self.categorical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            X[self.numerical_cols_] = self.num_imputer_.transform(X[self.numerical_cols_])
        if self.categorical_cols_:
            X[self.categorical_cols_] = self.cat_imputer_.transform(X[self.categorical_cols_])
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5):
        self.cols = cols
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = self.cols or [c for c in NUMERICAL_COLS if c in X.columns]
        self.lower_, self.upper_ = {}, {}
        for col in self.cols_:
            if col not in X.columns:
                continue
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[col] = q1 - self.iqr_multiplier * iqr
            self.upper_[col] = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols_:
            if col in X.columns and col in self.lower_:
                X[col] = X[col].clip(lower=self.lower_[col], upper=self.upper_[col])
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.numerical_cols_ = self.numerical_cols or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in [TARGET_COL, 'CustomerId']]
        self.scaler_ = StandardScaler()
        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            X[self.numerical_cols_] = self.scaler_.transform(X[self.numerical_cols_])
        return X


class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols: Optional[List[str]] = None, sparse_output: bool = False, handle_unknown: str = 'ignore'):
        self.categorical_cols = categorical_cols
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y=None):
        present = self.categorical_cols or [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in [TARGET_COL, 'CustomerId']]
        self.categorical_cols_ = [c for c in present if c in X.columns]
        self.encoder_ = OneHotEncoder(sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)
        self.encoder_.fit(X[self.categorical_cols_])
        self.feature_names_ = self.encoder_.get_feature_names_out(self.categorical_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded = self.encoder_.transform(X[self.categorical_cols_])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        X = X.drop(columns=self.categorical_cols_)
        return pd.concat([X, encoded_df], axis=1)


def build_feature_pipeline(numerical_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None) -> Pipeline:
    return Pipeline([
        ('imputer', DataFrameImputer(numerical_cols=numerical_cols, categorical_cols=categorical_cols)),
        ('capper', OutlierCapper(cols=numerical_cols)),
        ('scaler', FeatureScaler(numerical_cols=numerical_cols)),
        ('one_hot', DataFrameOneHotEncoder(categorical_cols=categorical_cols)),
    ])


def split_data(df: pd.DataFrame, target: str = TARGET_COL, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    X = df.drop(columns=[target, 'CustomerId'], errors='ignore')
    y = df[target]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp)
    logger.info(f'Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline(raw_filepath: str, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_data(raw_filepath)
    features_df = compute_customer_features(raw_df)
    labeled_df = build_risk_label(features_df)
    feature_cols = [c for c in labeled_df.columns if c not in [TARGET_COL, 'CustomerId', 'first_txn', 'last_txn']]
    pipeline = build_feature_pipeline(numerical_cols=NUMERICAL_COLS, categorical_cols=CATEGORICAL_COLS)
    X_processed = pipeline.fit_transform(labeled_df[feature_cols])
    processed_df = X_processed if isinstance(X_processed, pd.DataFrame) else pd.DataFrame(X_processed, index=labeled_df.index)
    processed_df[TARGET_COL] = labeled_df[TARGET_COL].values
    processed_df['CustomerId'] = labeled_df['CustomerId'].values
    out_path = os.path.join(output_dir, 'customer_features.csv')
    processed_df.to_csv(out_path, index=False)
    logger.info(f'Processed data saved to: {out_path}')
    return {'df': processed_df, 'pipeline': pipeline, 'processed_path': out_path, 'n_customers': len(processed_df), 'n_features': processed_df.shape[1] - 2, 'high_risk_rate': processed_df[TARGET_COL].mean()}


if __name__ == '__main__':
    run_pipeline('data/raw/alternate_data.csv', 'data/processed')
