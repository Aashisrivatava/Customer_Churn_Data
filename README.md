# Customer Churn Prediction

This project analyzes bank customer data to understand and predict customer churn, when a customer leaves the bank. The aim is to help banks identify at-risk customers and take proactive steps to retain them.

## Approach
- EDA and preprocessing: one-hot encoding, standard scaling, stratified 80/20 train-test split
- Handled class imbalance (about 80% stayed / 20% churned) using class weights
- Neural network (TensorFlow/Keras) with dropout and early stopping
- Decision threshold tuned for the churn class instead of the default 0.5
- XGBoost baseline for comparison

## Results (churn class)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Neural network, default setup | 0.74 | 0.47 | 0.58 |
| Neural network, improved | 0.64 | 0.59 | 0.61 |
| XGBoost (scale_pos_weight, 0.5 threshold) | 0.50 | 0.73 | 0.59 |

XGBoost ROC-AUC: 0.861

## Tech stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow, XGBoost, Matplotlib
