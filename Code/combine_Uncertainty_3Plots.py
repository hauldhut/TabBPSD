base_dataset = "BridgePier"
model = "TabPFN"
print(f"model: {model}, param: VVc & yb")

from PIL import Image
import matplotlib.pyplot as plt


# Load individual SHAP figures
img1 = Image.open(f"../Figures/Uncertainty_Probabilistic_Prediction_{base_dataset}_{model}.png")
img2 = Image.open(f"../Figures/Uncertainty_Prediction_vs_VVc_{base_dataset}_{model}.png")
img3 = Image.open(f"../Figures/Uncertainty_Prediction_vs_yb_{base_dataset}_{model}.png")

gap = 50
# Combine images vertically
width = img1.width + img2.width + gap + img3.width + gap
height = max(img1.height, img2.height, img3.height)


combined = Image.new("RGB", (width, height), "white")
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width+gap, 0))
combined.paste(img3, (img1.width+gap+img2.width+gap, 0))

# Plot combined image using Matplotlib
plt.figure(figsize=(8, 4), dpi=300)
plt.imshow(combined)
plt.axis("off")

# Add panel labels

plt.text(10, 25, "A", fontsize=11, fontweight="bold")#, fontweight="bold"
plt.text(img1.width + 10 + gap, 25, "B", fontsize=11, fontweight="bold")#, fontweight="bold"
plt.text(img1.width + 10 + gap + img2.width + 10 + gap, 25, "C", fontsize=11, fontweight="bold")#, fontweight="bold"

plt.tight_layout()

# ✅ SAVE THE MATPLOTLIB FIGURE (labels included)
plt.savefig(f"../Figures/Uncertainty_Prediction_vs_VVc_yb_{base_dataset}_{model}_Final.png", dpi=300, bbox_inches="tight")
plt.show()
