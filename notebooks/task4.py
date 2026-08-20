# %% [markdown]
# # Task 4 — Proxy Target Variable Engineering
#
# The raw dataset has no default label. Here we build one by clustering
# customers on RFM (Recency, Frequency, Monetary) behavior and labeling the
# most disengaged cluster as high-risk.
#
# **This label is a modeling assumption, not verified ground truth** — see the
# final report for a discussion of the business risk this introduces.

# %%
import sys; sys.path.append("..")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import (
    load_raw_data,
    compute_customer_features,
    compute_rfm,
    cluster_rfm,
    identify_high_risk_cluster,
    build_risk_label,
    OutlierCapper,
)

sns.set_theme(style="whitegrid", palette="tab10")

# %%
df = load_raw_data("../data/raw/alternate_data.csv")
customer_features = compute_customer_features(df)
customer_features.shape

# %% [markdown]
# ## Build the proxy label
# RFM → scale → KMeans (k=3, random_state=42) → identify most-disengaged cluster.

# %%
labeled = build_risk_label(customer_features)
print(labeled["is_high_risk"].value_counts())
print(f"High-risk rate: {labeled['is_high_risk'].mean():.2%}")
labeled.head()

# %% [markdown]
# ## Why this cluster? Validating the high-risk assignment
#
# Rebuild the RFM/cluster view directly so we can show cluster centers side by
# side — this is the plot that justifies the label to a non-technical reader.

# %%
rfm = compute_rfm(customer_features)
rfm, kmeans, scaler = cluster_rfm(rfm)
high_risk_cluster = identify_high_risk_cluster(rfm)

profile = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()
print(profile)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
    colors = ["coral" if c == high_risk_cluster else "steelblue" for c in profile.index]
    profile[col].plot(kind="bar", ax=ax, color=colors)
    ax.set_title(f"Mean {col} by Cluster")
    ax.set_xlabel("Cluster")
plt.suptitle(f"Cluster {high_risk_cluster} (coral) identified as high-risk", y=1.02)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Customer-Level Distributions by Risk Label
# Compare high-risk vs low-risk customers across RFM features.

# %%
rfm_plot_cols = [
    ("recency_days",      "Days since last transaction"),
    ("transaction_count", "Number of transactions"),
    ("total_amount",      "Total spend (UGX)"),
    ("avg_amount",        "Average transaction amount (UGX)"),
]

fig, axes = plt.subplots(len(rfm_plot_cols), 2, figsize=(14, 5 * len(rfm_plot_cols)))

for i, (col, label) in enumerate(rfm_plot_cols):
    sns.histplot(
        data=labeled, x=col, hue="is_high_risk",
        kde=True, multiple="stack",
        hue_order=[0, 1],
        palette={0: "steelblue", 1: "coral"},
        ax=axes[i, 0], bins=30
    )
    axes[i, 0].set_title(f"Distribution: {label}")
    axes[i, 0].set_xlabel(label)
    axes[i, 0].legend(title="is_high_risk", labels=["High Risk (1)", "Low Risk (0)"])

    sns.boxplot(
        data=labeled, x="is_high_risk", y=col,
        hue="is_high_risk",
        palette={0: "steelblue", 1: "coral"},
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"Boxplot: {label}")
    axes[i, 1].set_xlabel("is_high_risk (0=Low, 1=High)")
    axes[i, 1].set_ylabel(label)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Correlation Analysis — RFM & Related Features

# %%
rfm_num = labeled[["recency_days", "transaction_count", "total_amount",
                    "avg_amount", "std_amount", "total_value", "fraud_count"]]

plt.figure(figsize=(10, 7))
sns.heatmap(rfm_num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix — RFM Features")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Outlier Detection (Pre-Capping)

# %%
cols_to_check = ["recency_days", "transaction_count", "total_amount",
                  "avg_amount", "std_amount", "total_value"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, cols_to_check):
    sns.boxplot(y=labeled[col], ax=ax, color="steelblue")
    ax.set_title(col)

plt.suptitle("Outlier Detection — RFM Features (Pre-Capping)", y=1.02)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Outlier Capping (Winsorization)
# IQR-based capping: values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` are clipped.

# %%
capper = OutlierCapper(cols=cols_to_check)
labeled_capped = capper.fit_transform(labeled)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, cols_to_check):
    sns.boxplot(y=labeled_capped[col], ax=ax, color="coral")
    ax.set_title(f"{col} (capped)")

plt.suptitle("After Outlier Capping", y=1.02)
plt.tight_layout(); plt.show()