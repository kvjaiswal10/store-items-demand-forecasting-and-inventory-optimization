
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# Example summary_df from expiry-aware simulation
# Each row = one item’s result
# ------------------------------------------
# summary_df = pd.DataFrame({
#     "item": [1,2,3,...],
#     "expiry_days": [15, 30, 45, ...],
#     "service_level": [...],
#     "waste_percent": [...]
# })

# (If you already have summary_df from your simulation, skip this block)
np.random.seed(42)
summary_df = pd.DataFrame({
    "item": np.arange(1, 51),
    "expiry_days": np.random.choice([7, 14, 30, 60, 90], size=50),
    "service_level": np.random.uniform(0.7, 0.99, 50),
    "waste_percent": np.random.uniform(0.02, 0.15, 50)
})

# ------------------------------------------
# Step 1 — Aggregate by expiry bucket
# ------------------------------------------
summary_df["expiry_bucket"] = pd.cut(
    summary_df["expiry_days"],
    bins=[0, 15, 30, 60, 90, np.inf],
    labels=["<=15 days", "16–30 days", "31–60 days", "61–90 days", ">90 days"]
)

agg_df = (
    summary_df
    .groupby("expiry_bucket", observed=True)
    .agg({
        "service_level": "mean",
        "waste_percent": "mean"
    })
    .reset_index()
)

# ------------------------------------------
# Step 2 — Plot aggregated trends
# ------------------------------------------
plt.figure(figsize=(9, 6))

# Bar plot for Waste
plt.bar(
    agg_df["expiry_bucket"],
    agg_df["waste_percent"] * 100,
    color="salmon",
    width=0.4,
    label="Avg Waste (%)"
)

# Line plot for Service Level
plt.plot(
    agg_df["expiry_bucket"],
    agg_df["service_level"] * 100,
    color="green",
    marker="o",
    linewidth=3,
    label="Avg Service Level (%)"
)

# ------------------------------------------
# Step 3 — Styling
# ------------------------------------------
plt.title("Impact of Product Expiry on Inventory Performance", fontsize=14, fontweight="bold")
plt.xlabel("Expiry Range (Days)")
plt.ylabel("Percentage")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Annotate bars
for i, (waste, sl) in enumerate(zip(agg_df["waste_percent"], agg_df["service_level"])):
    plt.text(i - 0.15, waste*100 + 0.5, f"{waste*100:.1f}%", fontsize=9, color="red")
    plt.text(i + 0.05, sl*100 + 0.5, f"{sl*100:.1f}%", fontsize=9, color="green")

plt.show()

# ------------------------------------------
# Step 4 — Print table summary
# ------------------------------------------
print("\n=== Summary by Expiry Bucket ===")
print(agg_df.to_string(index=False, formatters={
    "service_level": "{:.3f}".format,
    "waste_percent": "{:.3f}".format
}))
