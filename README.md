# ML-Project
# Claim Injury Type Classification – NY Workers’ Compensation Board

This project was developed as part of the Machine Learning course in the Master’s in Data Science and Advanced Analytics at NOVA IMS (2024/2025).

Our objective was to assist the **New York Workers’ Compensation Board (WCB)** in automating the classification of workplace injury claims. Using over 590,000 labeled records from 2020 to 2022, we built a supervised learning pipeline to predict the injury type category associated with each claim. The final solution integrates a web application to support decision-making in real-time.

The final model is a **stacked ensemble** (XGBoost + Random Forest), achieving an **F1-macro score of 0.43291** on Kaggle.

---

## Table of Contents
- Dataset
- Methodology
- Models
- Results
- Streamlit Web App
- Limitations
- Future Work

---

## Dataset

- Source: Provided by WCB for claims between 2020–2022
- Train: 593,471 records, 32 variables
- Test: 387,975 records, 30 variables
- Target: `Claim Injury Type` (multiclass classification)

### Key Preprocessing Steps:
- Handling of missing values, outliers, and temporal inconsistencies
- Flagging incoherent values (e.g., age outliers, invalid dates)
- Fuzzy matching for categorical harmonization
- Engineered 20+ new features: date differences, region risk scores, and validation flags

---

## Methodology

- **Target Task**: Multiclass classification (Injury Type)
- **Evaluation Metric**: Macro F1-Score (used for imbalanced classes)
- **Feature Selection**: Correlation analysis, Chi-Square, RFE, Lasso, Random Forest and XGBoost importance
- **Encoding**: Frequency encoding (for high-cardinality categorical variables)
- **Scaling**: Min-max scaling applied post-imputation and feature engineering

---

## Models

We tested several models using consistent pipelines and validation splits:

| Model                 | F1 (Val) | F1 (Kaggle) |
|----------------------|----------|-------------|
| Logistic Regression   | 0.267    | 0.291       |
| KNN                   | 0.321    | 0.306       |
| Random Forest         | 0.388    | 0.366       |
| XGBoost (tuned)       | 0.425    | 0.430       |
| Stacked RF + XGBoost  | 0.416    | **0.433**   |

The stacked ensemble showed the best generalization across validation and Kaggle test sets.

---

## Streamlit Web App

An interactive web application (`app.py`) was built using **Streamlit** to allow non-technical users to input claim data and receive injury type predictions.

### Features:
- Four logical input sections (basic info, timeline, other factors, medical)
- Input validation with dropdowns, checkboxes, and sliders
- Prediction output with probabilities per class

> ⚠️ Preprocessing (encoding & scaling) is not yet integrated into the app interface

---

## Limitations

- Incomplete integration of preprocessing in the web app
- Imbalanced classes limited prediction accuracy for rare injury types
- Hyperparameter tuning was computationally constrained

---

## Future Work

- Integrate full preprocessing pipeline in the web app
- Automate hyperparameter tuning with Optuna or GridSearchCV
- Explore alternate targets (e.g., “Agreement Reached”)
- Test class balancing techniques (SMOTE, focal loss)
- Deploy app using Streamlit Cloud or Docker

---

## Authors

- Ricardo Pereira  
- Benedikt Ruggaber  
- Francisco Pontes  

**NOVA IMS – Master in Data Science and Advanced Analytics 2024/2025**
