base_dataset = "BridgePier"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

# Read the tab-separated file
dfSD = pd.read_csv("../Data/BridgePier.tsv", sep="\t")
print(dfSD.shape)
# Show first rows
print(dfSD.head())
dfSD["Source"].value_counts()

selected_Source = ["USGS"]

dfSD_filtered = dfSD[dfSD["Source"].isin(selected_Source)]
print(dfSD_filtered.shape)

X, y = dfSD_filtered.drop(columns=["ysb", "Source"]), dfSD_filtered["ysb"]

# Fit TabPFN
# reg = TabPFNRegressor()
reg = TabPFNRegressor()

reg.fit(X, y)

# Predict full distribution
preds = reg.predict(X, output_type="full")

# Extract quantiles
quantiles = np.array(preds["quantiles"])  # shape: (n_quantiles, n_samples)

# Common quantiles (TabPFN default)
q10 = quantiles[1]   # ~0.1
q50 = quantiles[4]   # ~0.5 (median)
q90 = quantiles[7]   # ~0.9

print(quantiles.shape)

# ===============================
# Probabilistic metrics
# ===============================

# Convert to numpy arrays (safety)
y_true = np.array(y)
lower = np.array(q10)
upper = np.array(q90)

# PICP: proportion of observations inside interval
inside_interval = (y_true >= lower) & (y_true <= upper)
PICP = np.mean(inside_interval)

# MPIW: average width of interval
MPIW = np.mean(upper - lower)

print(f"PICP (90% interval): {PICP:.3f}")
print(f"MPIW: {MPIW:.3f}")

MPIW_norm = MPIW / (y_true.max() - y_true.min())
print(f"Normalized MPIW: {MPIW_norm:.3f}")


######## Probabilistic Prediction

plt.figure(figsize=(4, 4), dpi=300)

# Scatter: median prediction
plt.scatter(y, q50, c="tab:blue", alpha=0.7, label="Median prediction")

# Prediction intervals
plt.vlines(
    y,
    q10,
    q90,
    color="gray",
    alpha=0.4,
    linewidth=1.5,
    label="90% prediction interval"
)

# 1:1 reference line
lims = [min(y.min(), q10.min()), max(y.max(), q90.max())]
plt.plot(lims, lims, "k--", linewidth=1)

plt.xlabel("Observed scour depth $y_s / b$")
plt.ylabel("Predicted scour depth $\\hat{y}_s / b$")
plt.title(f"Probabilistic Prediction", fontsize=10)
plt.legend()
# Tick label font size
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

plt.text(
    0.05, 0.95,
    f"PICP = {PICP:.2f}\nMPIW = {MPIW:.2f}",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"../Figures/Uncertainty_Probabilistic_Prediction_{base_dataset}_TabPFN.png", dpi=300, bbox_inches="tight")
plt.show()


########### Uncertainty vs Flow intensity
# Compute uncertainty width
uncertainty_width = q90 - q10

param_name = "yb"  # "yb" or "VVc"
name_map = {
    "ysb": "$y_s / b$",
    "yb": "$y / b$",
    "VVc": "$V / V_c$",
    "bD50": "$b / d_{50}$",
    "Fr": "Fr",
}

param = X[param_name].values if hasattr(X, "columns") else X[:, 0]

plt.figure(figsize=(4, 4), dpi=300)
plt.scatter(param, uncertainty_width, c="tab:red", alpha=0.7)

# ---------------------------------------------------------
# Add vertical dashed line based on param_name
# ---------------------------------------------------------
if param_name == "VVc":
    plt.axvline(
        x=1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="$V/V_c = 1$",
    )
elif param_name == "yb":
    plt.axvline(
        x=5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="$y/b = 5$",
    )

plt.xlabel(f"{name_map[param_name]}")
plt.ylabel("Prediction interval width $(Q_{0.9} - Q_{0.1})$")
plt.title(
    f"Uncertainty of Scour Prediction vs. {name_map[param_name]}", fontsize=10
)

# Show legend if a threshold line was added
if param_name in ["VVc", "yb"]:
    plt.legend()

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"../Figures/Uncertainty_Prediction_vs_{param_name}_{base_dataset}_TabPFN.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()