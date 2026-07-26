# TabBPSD: Probabilistic Bridge-Pier Scour Prediction Using an In-Context Learning-Based Tabular Foundation Model


- TabPFN enables accurate probabilistic prediction of bridge-pier scour depth.
- TabPFN outperforms established tree-based models, including CatBoost and NGBoost.
- SHAP analysis reveals nonlinear effects of key hydraulic and sediment variables.
- Predictive uncertainty varies substantially across hydraulic conditions.
- Exceedance probabilities support uncertainty-aware, risk-informed scour assessment.

![TabPFN for scour depth prediction](https://github.com/hauldhut/TabBPSD/blob/main/TabPFN-3-Adapted.png)

Performance comparison


| Model | CC | R² | RMSE | MAE | MAPE | MBE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TabPFN** | **0.843<br>(±0.068)** | **0.706<br>(±0.116)** | **0.247<br>(±0.038)** | **0.164<br>(±0.022)** | **0.161<br>(±0.036)** | **0.001<br>(±0.036)** |
| CatBoost | 0.836<br>(±0.066) | 0.693<br>(±0.113) | 0.254<br>(±0.045) | 0.174<br>(±0.027) | 0.167<br>(±0.041) | 0.002<br>(±0.035) |
| NGBoost | 0.812<br>(±0.073) | 0.648<br>(±0.109) | 0.273<br>(±0.032) | 0.201<br>(±0.020) | 0.198<br>(±0.040) | 0.001<br>(±0.031) |
| XGBoost | 0.818<br>(±0.065) | 0.652<br>(±0.130) | 0.271<br>(±0.046) | 0.184<br>(±0.026) | 0.173<br>(±0.033) | -0.002<br>(±0.037) |
| RandomForest | 0.815<br>(±0.067) | 0.658<br>(±0.113) | 0.269<br>(±0.040) | 0.186<br>(±0.024) | 0.178<br>(±0.043) | -0.001<br>(±0.034) |


## Repo structure
- **Data**: Contains all datasets 
- **Code**: Contains all source code to reproduce all the results

## How to run
- Follow instructions in the folders **Data** and **Code** to run
