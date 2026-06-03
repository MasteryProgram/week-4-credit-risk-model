import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Column constants ──────────────────────────────────────────────────────────
TARGET_COL = "is_high_risk"

CATEGORICAL_COLS = [
    "CurrencyCode", "ProviderId", "ProductId",
    "ProductCategory", "ChannelId", "PricingStrategy",
]

NUMERICAL_COLS = [
    "total_amount", "avg_amount", "std_amount", "transaction_count",
    "total_value", "recency_days", "fraud_count",
    "txn_hour_mean", "txn_day_mean",
]


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def load_raw_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)
    logger.info(f"Loaded {len(df)} transactions, {df['CustomerId'].nunique()} unique customers.")
    return df


# ── 2. RFM Feature Engineering ────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Aggregate transaction-level rows into one row per customer with RFM features.
    """
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    
    logger.info(f"Computing RFM with snapshot date: {snapshot_date}")

    # Time features per transaction (before groupby)
    df = df.copy()
    df["txn_hour"] = df["TransactionStartTime"].dt.hour
    df["txn_day"]  = df["TransactionStartTime"].dt.day

    agg = df.groupby("CustomerId").agg(
        recency_days        = ("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        transaction_count   = ("TransactionId", "count"),
        total_amount        = ("Amount", "sum"),
        avg_amount          = ("Amount", "mean"),
        std_amount          = ("Amount", "std"),
        total_value         = ("Value", "sum"),
        fraud_count         = ("FraudResult", "sum"),
        txn_hour_mean       = ("txn_hour", "mean"),
        txn_day_mean        = ("txn_day", "mean"),
        # Most-frequent categorical per customer
        ProductCategory     = ("ProductCategory", lambda x: x.mode()[0]),
        ChannelId           = ("ChannelId", lambda x: x.mode()[0]),
        ProviderId          = ("ProviderId", lambda x: x.mode()[0]),
        ProductId           = ("ProductId", lambda x: x.mode()[0]),
        CurrencyCode        = ("CurrencyCode", lambda x: x.mode()[0]),
        PricingStrategy     = ("PricingStrategy", lambda x: str(x.mode()[0])),
    ).reset_index()

    agg["std_amount"] = agg["std_amount"].fillna(0)
    logger.info(f"RFM aggregation complete: {agg.shape[0]} customers, {agg.shape[1]} columns.")
    return agg


# ── 3. Missing Value Handling ─────────────────────────────────────────────────

class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        num_strategy: str = "median",
        cat_strategy: str = "most_frequent",
    ):
        self.numerical_cols  = numerical_cols
        self.categorical_cols = categorical_cols
        self.num_strategy    = num_strategy
        self.cat_strategy    = cat_strategy

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if TARGET_COL in X.columns:
            X = X.drop(columns=[TARGET_COL])

        all_cols = list(X.columns)
        self.numerical_cols_   = [c for c in (self.numerical_cols  or all_cols) if c in all_cols and pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in (self.categorical_cols or all_cols) if c in all_cols and not pd.api.types.is_numeric_dtype(X[c])]

        self.num_imputer_ = SimpleImputer(strategy=self.num_strategy)
        self.cat_imputer_ = SimpleImputer(strategy=self.cat_strategy, fill_value="missing")

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


# ── 4. Outlier Capping ────────────────────────────────────────────────────────

class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, cols: Optional[List[str]] = None, iqr_multiplier: float = 1.5):
        self.cols           = cols
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


# ── 5. Feature Scaling ────────────────────────────────────────────────────────

class FeatureScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.numerical_cols_ = self.numerical_cols or [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and c not in [TARGET_COL, "CustomerId"]
        ]
        self.scaler_ = StandardScaler()
        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            X[self.numerical_cols_] = self.scaler_.transform(X[self.numerical_cols_])
        return X


# ── 6. Categorical Encoding ───────────────────────────────────────────────────

class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        sparse_output: bool = False,
        handle_unknown: str = "ignore",
    ):
        self.categorical_cols = categorical_cols
        self.sparse_output    = sparse_output
        self.handle_unknown   = handle_unknown

    def fit(self, X: pd.DataFrame, y=None):
        present = self.categorical_cols or [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c]) and c not in [TARGET_COL, "CustomerId"]
        ]
        self.categorical_cols_ = [c for c in present if c in X.columns]
        self.encoder_ = OneHotEncoder(
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown,
        )
        self.encoder_.fit(X[self.categorical_cols_])
        self.feature_names_ = self.encoder_.get_feature_names_out(self.categorical_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded = self.encoder_.transform(X[self.categorical_cols_])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        X = X.drop(columns=self.categorical_cols_)
        return pd.concat([X, encoded_df], axis=1)


# ── 7. WoE / IV Utilities ─────────────────────────────────────────────────────

def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET_COL,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, float]:
    total_events     = (df[target] == 1).sum()
    total_non_events = (df[target] == 0).sum()

    stats = (
        df.groupby(feature, observed=False)[target]
        .agg(
            events     = lambda x: (x == 1).sum(),
            non_events = lambda x: (x == 0).sum(),
        )
        .reset_index()
    )
    stats["dist_events"]     = (stats["events"]     + epsilon) / (total_events     + epsilon)
    stats["dist_non_events"] = (stats["non_events"] + epsilon) / (total_non_events + epsilon)
    stats["woe"]             = np.log(stats["dist_events"] / stats["dist_non_events"])
    stats["iv_component"]    = (stats["dist_events"] - stats["dist_non_events"]) * stats["woe"]
    return stats, stats["iv_component"].sum()


def compute_all_iv(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = TARGET_COL,
) -> pd.DataFrame:
    features = features or CATEGORICAL_COLS + NUMERICAL_COLS
    results  = []
    for feat in features:
        if feat not in df.columns or feat == target:
            continue
        temp_df = df.copy()
        if pd.api.types.is_numeric_dtype(df[feat]):
            try:
                temp_df[feat] = pd.qcut(df[feat], q=10, duplicates="drop")
            except Exception:
                temp_df[feat] = pd.cut(df[feat], bins=5)
        _, iv_value = compute_woe_iv(temp_df, feat, target)
        results.append({"feature": feat, "iv": iv_value})

    iv_df = pd.DataFrame(results).sort_values("iv", ascending=False).reset_index(drop=True)
    iv_df["predictive_power"] = iv_df["iv"].apply(
        lambda v: "Useless" if v < 0.02 else
                  "Weak"    if v < 0.1  else
                  "Medium"  if v < 0.3  else
                  "Strong"  if v < 0.5  else "Very Strong"
    )
    return iv_df


# -- woe encoding 
def encode_woe(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target: str = TARGET_COL,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create WoE-encoded versions of categorical columns.

    Returns
    -------
    df_woe : pd.DataFrame
        Original dataframe plus *_woe columns.
    woe_maps : dict
        Mapping dictionary for later inference.
    """

    df_woe = df.copy()
    woe_maps = {}

    total_events = (df[target] == 1).sum()
    total_non_events = (df[target] == 0).sum()

    for col in categorical_cols:

        if col not in df.columns:
            continue

        stats = (
            df.groupby(col)[target]
            .agg(
                events=lambda x: (x == 1).sum(),
                non_events=lambda x: (x == 0).sum(),
            )
        )

        stats["woe"] = np.log(
            (
                (stats["events"] + epsilon)
                / (total_events + epsilon)
            )
            /
            (
                (stats["non_events"] + epsilon)
                / (total_non_events + epsilon)
            )
        )

        mapping = stats["woe"].to_dict()

        df_woe[f"{col}_woe"] = df[col].map(mapping)

        woe_maps[col] = mapping

    return df_woe, woe_maps

# ── 8. RFM Clustering → is_high_risk label ───────────────────────────────────

from sklearn.cluster import KMeans
def build_risk_label(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Improved version - Creates a better balanced high-risk proxy label using RFM.
    """
    df = df.copy()
    
    # Use key RFM features for clustering
    rfm_cols = ['recency_days', 'transaction_count', 'total_amount', 'avg_amount']
    
    # Handle any missing values
    for col in rfm_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[rfm_cols])
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Detailed cluster analysis
    cluster_summary = df.groupby('cluster').agg({
        'recency_days': 'mean',
        'transaction_count': 'mean',
        'total_amount': 'mean',
        'avg_amount': 'mean',
        'CustomerId': 'count'
    }).round(2)
    
    print("=== Cluster Summary ===")
    print(cluster_summary)
    
    # Strategy: Choose the cluster that looks most "disengaged"
    # Usually: Highest recency + Lowest transaction count / amount
    risk_score = (
        cluster_summary['recency_days'] * 0.4 +
        (1 / (cluster_summary['transaction_count'] + 1)) * 100 * 0.4 +
        (1 / (cluster_summary['total_amount'] + 1)) * 100 * 0.2
    )
    
    high_risk_cluster = risk_score.idxmax()
    
    print(f"\nSelected High-Risk Cluster: {high_risk_cluster}")
    print(f"High Risk Customers: {cluster_summary.loc[high_risk_cluster, 'CustomerId']}")
    
    df['is_high_risk'] = (df['cluster'] == high_risk_cluster).astype(int)
    
    # Final distribution
    print(f"\nFinal is_high_risk distribution:\n{df['is_high_risk'].value_counts()}")
    print(f"High-risk percentage: {df['is_high_risk'].mean()*100:.3f}%")
    
    return df


# ── 9. Pipeline ───────────────────────────────────────────────────────────────

def build_feature_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Pipeline:
    return Pipeline([
        ("imputer",  DataFrameImputer(numerical_cols=numerical_cols, categorical_cols=categorical_cols)),
        ("capper",   OutlierCapper(cols=numerical_cols)),
        ("scaler",   FeatureScaler(numerical_cols=numerical_cols)),
        ("one_hot",  DataFrameOneHotEncoder(categorical_cols=categorical_cols)),
    ])


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target, "CustomerId"], errors="ignore")
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )
    logger.info(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 10. End-to-End Runner ─────────────────────────────────────────────────────

def run_pipeline(raw_filepath: str, output_dir: str) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — load raw transactions
    raw_df = load_raw_data(raw_filepath)

    # Step 2 — aggregate to customer level (RFM + categoricals)
    rfm_df = compute_rfm(raw_df)

    # Step 3 — assign risk label via K-Means clustering
    rfm_df = build_risk_label(rfm_df)

    # Step 4 — run sklearn pipeline (impute → cap → scale → encode)
    feature_cols = [c for c in rfm_df.columns if c not in [TARGET_COL, "CustomerId"]]
    pipeline     = build_feature_pipeline()
    X_processed  = pipeline.fit_transform(rfm_df[feature_cols])

    # Re-attach target and CustomerId
    processed_df = X_processed.copy()
    processed_df[TARGET_COL]   = rfm_df[TARGET_COL].values
    processed_df["CustomerId"] = rfm_df["CustomerId"].values

    # Step 5 — save
    out_path = os.path.join(output_dir, "xente_credit_processed.csv")
    processed_df.to_csv(out_path, index=False)
    logger.info(f"Processed data saved to: {out_path}")

    return {
        "df":           processed_df,
        "pipeline":     pipeline,
        "processed_path": out_path,
        "n_customers":  len(processed_df),
        "n_features":   processed_df.shape[1] - 2,   # exclude target + CustomerId
        "high_risk_rate": processed_df[TARGET_COL].mean(),
    }


if __name__ == "__main__":
    run_pipeline(
        raw_filepath="data/raw/xente_data.csv",
        output_dir="data/processed",
    )