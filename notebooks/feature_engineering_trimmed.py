# %% [markdown]
# # Feature Engineering for Credit Risk Modeling
#
# This notebook transforms raw Xente transaction data into customer-level
# behavioral features. All logic lives in `src/data_processing.py` — this
# notebook only calls it and inspects the results.

# %%
import sys; sys.path.append("..")
import pandas as pd

from src.data_processing import (
    load_raw_data,
    compute_customer_features,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
)

# %% [markdown]
# ## Load raw transaction data

# %%
df = load_raw_data("../data/raw/alternate_data.csv")
df.head()

# %% [markdown]
# ## Customer-level aggregation
#
# Collapses transaction-level rows into one row per `CustomerId`:
# - **Aggregates**: total/avg/std transaction amount, transaction count, total value
# - **Velocity**: transactions per day, value per day
# - **Temporal**: mean transaction hour/day
# - **Diversity**: unique products/categories/providers
# - **Channel behavior**: transaction counts per channel

# %%
customer_features = compute_customer_features(df)
print(f"Shape: {customer_features.shape}")
customer_features.head()

# %%
customer_features[NUMERICAL_COLS].describe().T