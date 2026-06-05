import pandas as pd

from src.data_processing import build_risk_label, compute_customer_features


def test_compute_customer_features_creates_expected_columns():
    raw = pd.DataFrame([
        {
            'TransactionId': 't1',
            'CustomerId': 'c1',
            'Amount': 100,
            'Value': 100,
            'TransactionStartTime': pd.to_datetime('2024-01-01 10:00:00'),
            'FraudResult': 0,
            'ProductCategory': 'airtime',
            'ChannelId': 'channel_1',
            'ProviderId': 'provider_1',
            'ProductId': 'product_1',
            'CurrencyCode': 'UGX',
            'PricingStrategy': 1,
        },
        {
            'TransactionId': 't2',
            'CustomerId': 'c1',
            'Amount': 200,
            'Value': 200,
            'TransactionStartTime': pd.to_datetime('2024-01-02 12:00:00'),
            'FraudResult': 0,
            'ProductCategory': 'airtime',
            'ChannelId': 'channel_1',
            'ProviderId': 'provider_1',
            'ProductId': 'product_2',
            'CurrencyCode': 'UGX',
            'PricingStrategy': 1,
        },
    ])
    features = compute_customer_features(raw)
    expected_columns = {
        'CustomerId',
        'recency_days',
        'transaction_count',
        'total_amount',
        'avg_amount',
        'std_amount',
        'total_value',
        'fraud_count',
        'preferred_hour',
        'weekend_ratio',
        'unique_products',
        'unique_categories',
        'unique_providers',
    }
    assert expected_columns.issubset(set(features.columns))
    assert features.loc[0, 'transaction_count'] == 2


def test_build_risk_label_creates_binary_target():
    raw = pd.DataFrame([
        {
            'TransactionId': 't1',
            'CustomerId': 'c1',
            'Amount': 100,
            'Value': 100,
            'TransactionStartTime': pd.to_datetime('2024-01-01 10:00:00'),
            'FraudResult': 0,
            'ProductCategory': 'airtime',
            'ChannelId': 'channel_1',
            'ProviderId': 'provider_1',
            'ProductId': 'product_1',
            'CurrencyCode': 'UGX',
            'PricingStrategy': 1,
        },
        {
            'TransactionId': 't2',
            'CustomerId': 'c2',
            'Amount': 10,
            'Value': 10,
            'TransactionStartTime': pd.to_datetime('2023-01-01 12:00:00'),
            'FraudResult': 0,
            'ProductCategory': 'airtime',
            'ChannelId': 'channel_1',
            'ProviderId': 'provider_1',
            'ProductId': 'product_1',
            'CurrencyCode': 'UGX',
            'PricingStrategy': 1,
        },
        {
            'TransactionId': 't3',
            'CustomerId': 'c3',
            'Amount': 5,
            'Value': 5,
            'TransactionStartTime': pd.to_datetime('2022-01-01 08:00:00'),
            'FraudResult': 0,
            'ProductCategory': 'airtime',
            'ChannelId': 'channel_1',
            'ProviderId': 'provider_1',
            'ProductId': 'product_1',
            'CurrencyCode': 'UGX',
            'PricingStrategy': 1,
        },
    ])
    features = compute_customer_features(raw)
    labeled = build_risk_label(features)
    assert 'is_high_risk' in labeled.columns
    assert set(labeled['is_high_risk'].unique()) <= {0, 1}
