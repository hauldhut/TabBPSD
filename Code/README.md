# This folder contains all source code used in the manuscript

## Environment Setup
- **Create conda environment**
  - *conda env create -f TabBPSD.yml*: To run DeepDisSNP

## Experiments
The analysis follows this order: Performance assessment → Physical interpretability → Uncertainty quantification → Exceedance probability → Figure Creation

- **1. Performance assessment**
  - *PerformanceAssessment.py*: For performance assessment and comparison between TabPFN and CatBoost, NGBoost, XGBoost, and Random Forest

- **2. Physical interpretability**
  - *Interpretability.py*: For generating SHAP values
  - *Interpretability_drawFigs.py*: For draw SHAP plots from SHAP values

- **3. Uncertainty quantification**
  - *UncertaintyQuantification.py*: For uncertainty quantification. Note: Change the param_name = "yb" or "VVc"

- **4. Exceedance probability**
  - *ExceedanceProbability.py*: For exceedance probability analysis. Note: Change the param_name = "yb" or "VVc"

## Figure Creation
  - *combine_Interpretability_3Plots.py*: To create Figure 3
  - *combine_Uncertainty_3Plots.py*: To create Figure 4
  - *combine_Exceedance_2Plots.py*: To create Figure 5
