base_dataset = "BridgePier"

import pandas as pd
from tabpfn import TabPFNClassifier, TabPFNRegressor

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

feature_names = X.columns.to_numpy()
print(feature_names)

# from tabpfn import TabPFNRegressor
# from tabpfn_extensions import interpretability
import shap

feature_names = X.columns.to_numpy()

reg = TabPFNRegressor()

reg.fit(X, y)

explainer = shap.Explainer(
    reg.predict,
    X
)

shap_values = explainer(X)

#Save shap_values
import pickle

with open(f"../Results/shap_values_{base_dataset}_all.pkl", "wb") as f:
    pickle.dump(shap_values, f)
