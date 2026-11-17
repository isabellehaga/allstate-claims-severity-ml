# Allstate Claims Severity

## Project Overview

This project focuses on predicting insurance claim severity using machine learning techniques.

The goal is to explore the relationship between anonymized insurance claim features and their associated financial losses, developing predictive models to estimate claim severity.

The project is based on the [Allstate Claims Severity Kaggle competition](https://www.kaggle.com/competitions/allstate-claims-severity).

## Objectives

- Explore and understand the dataset.
- Perform exploratory data analysis (EDA).
- Prepare and preprocess categorical and continuous variables.
- Develop and evaluate machine learning regression models.
- Compare model performance using evaluation metrics.
- Generate predictions for unseen data and submit results to Kaggle.

## Dataset

The dataset is provided by the Allstate Claims Severity competition on Kaggle.

It contains anonymized insurance claim information, including:
- 116 categorical features.
- 14 continuous features.
- A target variable (`loss`) representing insurance claim severity.

The training dataset used in this project contains 188,318 records, while the test dataset contains 125,546 records.

Original competition data files are not stored in this repository.

Dataset source: [Allstate Claims Severity | Kaggle](https://www.kaggle.com/competitions/allstate-claims-severity)

## Exploratory Data Analysis

The exploratory analysis covered:
- Dataset structure and feature types.
- Missing value assessment.
- Categorical feature cardinality.
- Continuous variable distributions.
- Target variable distribution and extreme values.

The target variable (`loss`) presented a strongly right-skewed distribution, with a small number of observations showing very high claim values.

## Machine Learning Models

Two regression algorithms were developed and compared:

- **CatBoost Regressor:** trained directly with categorical features.
- **Random Forest Regressor:** categorical features encoded using Ordinal Encoding.

### Model Evaluation

Models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on a validation set.

| Model | MAE | RMSE |
|---|---:|---:|
| CatBoost | **1,176.09** | **1,852.26** |
| Random Forest | 1,225.16 | 1,902.49 |

CatBoost achieved the best results on both metrics and was selected for final training.

## Final Model and Kaggle Submission

The selected CatBoost model was retrained using all 188,318 available training records.

The final model generated predictions for the 125,546 records in the test dataset.

The submission file was validated and processed by Kaggle.

| Kaggle Evaluation | MAE |
|---|---:|
| Public Score | 1,171.27053 |
| Private Score | 1,180.60350 |

**Note:** The submission was processed after the competition deadline and was not eligible for the final leaderboard ranking.

## Project Structure

```text
Allstate-Claims-Severity/
│
├── README.md
├── .gitignore
├── models/
│   └── catboost_final.cbm
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   └── 02_preparacao_modelagem.ipynb
└── submission.csv
```

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- Jupyter Notebook
- Git and GitHub

## Key Learnings

This project provided practical experience in:
- Exploratory data analysis.
- Data preparation and categorical feature handling.
- Regression modeling and model comparison.
- Model evaluation and error analysis.
- Training a final model with the complete training dataset.
- Generating predictions and preparing Kaggle submissions.

## Project Status

**Completed.**

