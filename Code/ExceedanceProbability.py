base_dataset = "BridgePier"
model = "TabPFN" 
param_name = "yb" #yb|VVc
name_map = {
    "ysb": "$y_s / b$",
    "yb": "$y / b$",
    "VVc": "$V / V_c$",
    "bD50": "$b / d_{50}$",
    "Fr": "Fr",
}
print(f"model: {model}, param: {param_name}")

# ===============================
# Exceedance Probability Analysis
# ===============================

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

# ---- Define critical threshold ----
# Option 1: fixed engineering threshold
# y_critical = 2.5

# Option 2 (alternative): percentile-based threshold
y_critical = np.percentile(y, 90)

reg = TabPFNRegressor()
reg.fit(X, y)

# Predict full distribution
preds = reg.predict(X, output_type="full")



# ---- Extract quantiles ----
q_vals = np.array(preds["quantiles"])           # shape: (n_quantiles, n_samples)
# q_levels = np.array(preds["quantile_levels"])   # e.g., [0.1, ..., 0.9]
# Define quantile levels manually (TabPFN default)
q_levels = np.linspace(0.1, 0.9, q_vals.shape[0])

n_samples = q_vals.shape[1]

# ---- Compute exceedance probability ----
P_exceed = np.zeros(n_samples)

for i in range(n_samples):
    qs = q_vals[:, i]

    # Ensure monotonicity (important for interpolation stability)
    qs_sorted_idx = np.argsort(qs)
    qs_sorted = qs[qs_sorted_idx]
    ql_sorted = q_levels[qs_sorted_idx]

    # Interpolate CDF: P(ds <= d_critical)
    prob_leq = np.interp(
        y_critical,
        qs_sorted,
        ql_sorted,
        left=0.0,
        right=1.0
    )

    # Convert to exceedance probability
    P_exceed[i] = 1 - prob_leq

# ---- Summary statistics ----
print(f"Critical threshold y_critical = {y_critical}")
print(f"Mean exceedance probability: {P_exceed.mean():.3f}")
print(f"Max exceedance probability: {P_exceed.max():.3f}")
print(f"Min exceedance probability: {P_exceed.min():.3f}")


# ===============================
# Plot: Exceedance Probability vs log10(param_name)
# ===============================

param = X[param_name].values if hasattr(X, "columns") else X[:, 0]

# Compute log base 10 transformation
log10_param = np.log10(param)

plt.figure(figsize=(4, 4), dpi=300)

plt.scatter(log10_param, P_exceed, c="purple", alpha=0.7)

# Risk thresholds
plt.axhline(0.1, linestyle="--", color="gray", linewidth=1, label="10% risk")
plt.axhline(0.5, linestyle="--", color="black", linewidth=1, label="50% risk")

# ---------------------------------------------------------
# Add vertical dashed lines based on param_name (in log10 scale)
# ---------------------------------------------------------
if param_name == "VVc":
    # log10(1) = 0
    plt.axvline(x=np.log10(1), color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="$V/V_c = 1$")
elif param_name == "yb":
    # log10(5) ≈ 0.699
    plt.axvline(x=np.log10(5), color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="$y/b = 5$")

# Updated x-label and title with log10 LaTeX formatting
clean_name = name_map[param_name].strip("$")
plt.xlabel(f"$\\log_{{10}}({clean_name})$", fontsize=11)
plt.ylabel("$P(y_s / b > y_{critical})$", fontsize=11)
plt.title(f"Exceedance Probability vs. $\\log_{{10}}({clean_name})$", fontsize=10)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"../Figures/Exceedance_Probability_vs_log10_{param_name}_{base_dataset}_{model}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()