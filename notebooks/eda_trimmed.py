# %% [markdown]
# # Credit Risk EDA — Commerce Data
#
# ## Business Context
# We're building a **proxy credit risk model** for Bati Bank using transaction data
# from the Xente eCommerce platform. There is no direct "default" label — we'll
# engineer one from behavioral patterns (RFM) in Task 4.
#
# This notebook covers:
# 1. Data overview & structure
# 2. Summary statistics
# 3. Numerical & categorical distributions
# 4. Missing values & outlier detection
# 5. Temporal patterns & fraud distribution

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="tab10")

df = pd.read_csv("../data/raw/alternate_data.csv", parse_dates=["TransactionStartTime"])
df.head()

# %%
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nUnique customers: {df['CustomerId'].nunique()}")
print(f"Date range: {df['TransactionStartTime'].min()} → {df['TransactionStartTime'].max()}")

# %% [markdown]
# ## Summary Statistics
# Quick look at central tendency and spread of numerical columns.

# %%
df.describe(include="all").T

# %% [markdown]
# ## Missing Values

# %%
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if missing.empty:
    print("No missing values found.")
else:
    print(missing)
    missing.plot(kind="bar", title="Missing Values per Column")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## Numerical Feature Distributions (Transaction Level)
# Raw transaction amounts before any aggregation. `Amount` can be negative
# (refunds/credits back to customer). `Value` is always positive: the absolute amount.

# %%
num_cols = ["Amount", "Value"]

fig, axes = plt.subplots(len(num_cols), 2, figsize=(14, 5 * len(num_cols)))

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color="steelblue", bins=50)
    axes[i, 0].set_title(f"Distribution: {col}")
    axes[i, 0].set_xlabel(f"{col} (UGX)")

    sns.boxplot(x=df[col], ax=axes[i, 1], color="steelblue")
    axes[i, 1].set_title(f"Boxplot: {col}")
    axes[i, 1].set_xlabel(f"{col} (UGX)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Categorical Feature Distributions
# Top categories for `ProductCategory`, `ChannelId`, `ProviderId`, and `PricingStrategy`.

# %%
cat_cols = ["ProductCategory", "ChannelId", "ProviderId", "PricingStrategy", "CurrencyCode"]

fig, axes = plt.subplots(len(cat_cols), 1, figsize=(12, 4 * len(cat_cols)))

for i, col in enumerate(cat_cols):
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, ax=axes[i], palette="tab10")
    axes[i].set_title(f"Count: {col}")
    axes[i].set_xlabel("Count")

plt.tight_layout(); plt.show()

# %% [markdown]
# ## Transaction Volume Over Time
# Are there any seasonal or temporal spikes in transaction activity?

# %%
df["date"] = df["TransactionStartTime"].dt.date
daily = df.groupby("date")["TransactionId"].count().reset_index()
daily.columns = ["date", "count"]

plt.figure(figsize=(14, 4))
plt.plot(pd.to_datetime(daily["date"]), daily["count"], linewidth=1)
plt.title("Daily Transaction Volume")
plt.xlabel("Date"); plt.ylabel("Transactions")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Transaction Hour & Day Patterns
# When do customers transact most?

# %%
df["hour"] = df["TransactionStartTime"].dt.hour
df["dayofweek"] = df["TransactionStartTime"].dt.day_name()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

df["hour"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Transactions by Hour of Day")
axes[0].set_xlabel("Hour")

dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["dayofweek"].value_counts().reindex(dow_order).plot(kind="bar", ax=axes[1], color="coral")
axes[1].set_title("Transactions by Day of Week")

plt.tight_layout(); plt.show()

# %% [markdown]
# ## Fraud Distribution
# `FraudResult` is highly imbalanced — this is typical in fraud datasets and will
# inform how we treat class imbalance in modeling.

# %%
fraud_counts = df["FraudResult"].value_counts()
print(fraud_counts)
print(f"\nFraud rate: {fraud_counts[1] / len(df):.2%}")

fraud_counts.plot(kind="bar", title="Fraud Result Distribution", color=["steelblue","coral"])
plt.xticks([0, 1], ["Legit", "Fraud"], rotation=0)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Key Insights
#
# 1. **Heavy skew in Amount/Value**: a small number of high-value transactions
#    dominate; outlier capping will be needed before modeling.
# 2. **Temporal patterns**: transaction spikes at certain hours/days suggest
#    behavioral features (hour, day-of-week) could be predictive.
# 3. **Fraud is rare (~<1%)**: severe class imbalance; model evaluation must use
#    AUC-ROC and F1, not accuracy.
# 4. **`Value` is the absolute value of `Amount`**: expect near-perfect correlation
#    between amount-based and value-based aggregates — likely redundant at the
#    customer level, to confirm in Task 3/4.
# 5. **Customers vary widely in transaction count and recency of activity**: this
#    motivates using RFM (Recency, Frequency, Monetary) as the basis for a proxy
#    risk segmentation in Task 4.