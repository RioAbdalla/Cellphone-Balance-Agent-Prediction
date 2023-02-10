# Cellphone Balance Agent Prediction
## Summary
The data comprised each Indonesian city's economic values and the administrative category. The data originated from BPS, which is accessible to the public. Multiple columns are both numerical and categorical. Numerical values are unstandardized and categorical data are not encoded.
## Objective
To determine the necessary funds to support cellphone credit agents and small local businesses, the Indonesian People’s Bank needs to predict their growth in the upcoming year.
## Problem Definition
To determine the right amount of loan for cellphone credit agents, the Indonesian People’s Bank is required to know their exact growth in the next year. The right amount of loans will help impact and stimulate the economy efficiently. Initially, the data has a few problems: zero values, missing Values, and inconsistent format, which means it needs to be cleaned accordingly. Furthermore, the numerical data has a different scale and an outlier, which means it needs to be standardized.
## Steps
1. EDA
2. Fill missing values, nan values, and delete duplicate data
3. Split and determine the number of village and sub district of each city for further use
4. Encode categorical values and standarize numerical values
5. Check correlation for feature analysis
6. Vanilla modeling with linear regression
7. Feature evalution with Forward and backward propagation
8. Model evalution with XGBoost
9. Finalize Model and Feature
## Results
Model able to predict cellphone balance agent with available next year data. The accuracy of prediction can reach almost 90% with XGBoost and using specific feature
 
