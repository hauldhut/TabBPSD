# TabBPSD: Probabilistic Bridge-Pier Scour Prediction Using an In-Context Learning-Based Tabular Foundation Model


- TabPFN enables accurate probabilistic prediction of bridge-pier scour depth.
- TabPFN outperforms established tree-based models, including CatBoost and NGBoost.
- SHAP analysis reveals nonlinear effects of key hydraulic and sediment variables.
- Predictive uncertainty varies substantially across hydraulic conditions.
- Exceedance probabilities support uncertainty-aware, risk-informed scour assessment.

![TabPFN for scour depth prediction](https://github.com/hauldhut/TabBPSD/blob/main/TabPFN-3-Adapted.png)

Performance comparison
Model	CC	R2	RMSE	MAE	MAPE	MBE
TabPFN	0.843 (±0.068)	0.706 (±0.116)	0.247 (±0.038)	0.164 (±0.022)	0.161 (±0.036)	0.001 (±0.036)
CatBoost	0.836 (±0.066)	0.693 (±0.113)	0.254 (±0.045)	0.174 (±0.027)	0.167 (±0.041)	0.002 (±0.035)
NGBoost	0.812 (±0.073)	0.648 (±0.109)	0.273 (±0.032)	0.201 (±0.020)	0.198 (±0.040)	0.001 (±0.031)
XGBoost	0.818 (±0.065)	0.652 (±0.130)	0.271 (±0.046)	0.184 (±0.026)	0.173 (±0.033)	-0.002 (±0.037)
RandomForest	0.815 (±0.067)	0.658 (±0.113)	0.269 (±0.040)	0.186 (±0.024)	0.178 (±0.043)	-0.001 (±0.034)
<img width="468" height="171" alt="image" src="https://github.com/user-attachments/assets/7330a2f4-573c-48c7-a499-3c9d934f0d94" />


## Repo structure
- **Data**: Contains all datasets 
- **Code**: Contains all source code to reproduce all the results

## How to run
- Follow instructions in the folders **Data** and **Code** to run
