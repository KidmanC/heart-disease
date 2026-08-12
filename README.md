# Heart Disease Classification
 
A machine learning project that predicts whether a patient has heart disease based on clinical measurements, using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.
 
> This project was built as part of a machine learning bootcamp (Zero to Mastery). I followed the course's project structure and dataset as a learning exercise, then reworked the analysis, code, and write-up on my own.
 
## Overview
 
- **Problem**: binary classification, predict presence of heart disease from 13 clinical features.
- **Data**: 303 patients, Cleveland Heart Disease dataset (UCI / Kaggle).
- **Models compared**: Logistic Regression, K-Nearest Neighbors, Random Forest.
- **Best model**: tuned Logistic Regression, ~84.5% cross-validated accuracy.
## What's inside
 
- `heart-disease-classification.ipynb`: full notebook covering EDA, model comparison, hyperparameter tuning (RandomizedSearchCV + GridSearchCV), evaluation (confusion matrix, ROC/AUC, classification report, cross-validated metrics), and feature importance.
- `heart-disease.csv`: dataset used in the notebook.
## Key steps
 
1. Exploratory data analysis of clinical features against the target.
2. Baseline comparison of three model families.
3. Hyperparameter tuning, first by hand (KNN), then with `RandomizedSearchCV` and `GridSearchCV`.
4. Evaluation beyond raw accuracy: precision, recall, F1, ROC-AUC, and 5-fold cross-validation.
5. Feature importance via Logistic Regression coefficients.
## Tools
 
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.
 
## Notes
 
Cross-validated accuracy (~84.5%) is the more reliable read on model performance; the single train/test split accuracy is noisier due to the small test set (61 samples).
