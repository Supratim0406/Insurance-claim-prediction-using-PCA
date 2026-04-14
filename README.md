#  Insurance Claim Prediction Project

![Project Banner](https://img.shields.io/badge/Status-Complete-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

##  Project Overview

This project aims to develop a robust machine learning model to accurately predict insurance claim amounts based on policyholder demographics, vehicle details, and historical data. By leveraging advanced regression techniques and ensemble methods, this tool helps insurance companies optimize premium pricing and improve risk assessment.

The project follows a complete Data Science lifecycle, from data preprocessing and exploratory data analysis (EDA) to model tuning, evaluation, and deployment via a user-friendly web interface.

##  Goal

The primary goal is to predict the **total claim amount** for a given insurance policy. This involves:
1.  **Data Preprocessing:** Handling missing values, encoding categorical features, and scaling numerical data.
2.  **Feature Engineering:** Utilizing 42 anonymous features alongside demographic data.
3.  **Model Selection:** Comparing Linear Regression, Random Forest, Gradient Boosting, and Ensemble methods.
4.  **Deployment:** Creating an interactive web application for real-time predictions.

##  Dataset

The dataset consists of policyholder information including:
* **Demographics:** Age, Gender, Marital Status, etc.
* **Vehicle Details:** Vehicle Type, Age, Safety Rating, Market Value.
* **Policy Info:** Annual Premium, Policy Deductible, Coverage Type.
* **History:** Number of past claims, traffic violations.
* **Anonymous Features:** 42 normalized features (`feature_1` to `feature_42`) representing various risk factors and customer attributes.

> **Note:** `feature_2` (Income Level) and `feature_1` (Credit Score) are normalized on a scale of 0-100.

##  Technologies Used

* **Python:** Core programming language.
* **Pandas & NumPy:** Data manipulation and numerical operations.
* **Scikit-Learn:** Machine learning modeling, pipeline construction, and evaluation.
* **Matplotlib & Seaborn:** Data visualization.
* **Gradio:** Web interface for model deployment.
* **Joblib:** Model serialization.

##  Model Performance

After extensive training and hyperparameter tuning, the models achieved the following performance on the test set:

| Model | R² Score | MSE |
| :--- | :--- | :--- |
| **Linear Regression** | 0.9999 | 52,442 |
| **Stacking Regressor** | 0.9999 | 52,343 |
| Gradient Boosting  | 0.9997 | 195,348 |
| Random Forest | 0.9994 | 434,786 |
| Decision Tree | 0.9988 | 834,200 |

**Winner:** The **Stacking Regressor** (combining Linear, RF, GB, etc.) and **Linear Regression** models performed exceptionally well.

> **Observation:** The remarkably high $R^2$ score (0.9999) suggests a strong linear relationship in this specific dataset, likely indicating that the target variable (`claim_amount`) is mathematically derived from features like `annual_premium`.

##  Key Features of the Code

* **Pipeline Architecture:** Uses `sklearn.pipeline.Pipeline` to chain preprocessing (StandardScaler, OneHotEncoder) and modeling steps, preventing data leakage.
* **Ensemble Learning:** Implements **Voting Regressor** and **Stacking Regressor** to combine the strengths of multiple models.
* **Hyperparameter Tuning:** Utilizes `GridSearchCV` to optimize the Gradient Boosting model.
* **Interactive UI:** Features a Gradio interface that maps user-friendly inputs (e.g., "Credit Score") to the model's expected feature names (e.g., `feature_1`).

##  Project Structure

```text
├── insurance_model.pkl    # Serialized trained model pipeline
├── insurance_dataset.csv  # Dataset used for training and testing
├── features.txt           # Description of the dataset features
├── README.md              # Project documentation
└── Insurance_Domain.ipynb # Jupyter Notebook with EDA and training logic

