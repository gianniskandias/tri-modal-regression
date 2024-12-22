# Tabular Regression Project

This project implements a regression task using tabular data to predict a target value. The objective is to explore multiple machine learning approaches, including both manual model tuning and automated solutions, to achieve accurate and robust predictions.


# Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Data Preprocessing](#data-preprocessing)
    - [Handling Missing Values](#handling-missing-values)
    - [Target Scaling](#target-scaling)
    - [Feature Selection](#feature-selection)
    - [Duplicates](#duplicates)
4. [Model Development](#model-development)
    - [Approaches](#approaches)
        - [Automated Machine Learning (AutoML)](#automated-machine-learning-automl)
        - [Tree-Based Models](#tree-based-models)
        - [Other Methods](#other-methods)
    - [Optimization](#optimization)
        - [Loss Functions](#loss-functions)
        - [Hyperparameter Tuning](#hyperparameter-tuning)
        - [Cross-Validation (in XGBoost)](#cross-validation)
6. [Results](#results)
    - [Performance Metrics](#performance-metrics)
    - [Key Observations](#key-observations)
7. [Future Improvements](#future-improvements)

## Introduction

This repository presents a regression model designed to predict a target value using tabular data. The project leverages advanced techniques in data preprocessing, model optimization, and evaluation to deliver a robust solution. Key approaches include the use of tree-based models, automated machine learning (AutoML), and hyperparameter tuning with Optuna.

The goal is to ensure scalability and generalizability, allowing users to easily apply the framework to their own datasets. While the data used for training cannot be disclosed, the methodology and codebase provide a clear path for implementing similar solutions on other tabular datasets.

## Features

- **Data Preprocessing**: Handles missing values, scales target variables, and applies minimal preprocessing to preserve data integrity.
- **Model Development**: Implements a range of machine learning models, including tree-based algorithms and AutoML frameworks.
- **Hyperparameter Optimization**: Uses Optuna for efficient hyperparameter tuning to enhance model performance.
- **Cross-Validation**: Ensures model robustness and generalization.

## Data Preprocessing

Minimal preprocessing was applied to preserve the dataset's integrity and avoid assumptions due to the lack of domain knowledge. Key steps include:

- **Handling Missing Values**:
  - Columns with only NaNs were removed.
  - For models capable of handling missing values (e.g., XGBoost, CatBoost), NaNs were left as-is.
  - For other models, NaNs were imputed with zeros, which improved accuracy.
- **Target Scaling**:
  - Applied log and quantile transformations to address the target's strong right skew.
  - The log transformation compacted the value range effectively, improving model performance.
- **Feature Selection**:
  - Columns with >90% NaN values were considered for removal but were retained as their removal led to performance deterioration.
- **Duplicates**:
  - Removed duplicate rows and duplicate features with differing target values.


## Model Development

### Approaches
1. **Automated Machine Learning (AutoML)**:
   - AutoGluon: Provided the best performance and time efficiency.
2. **Tree-Based Models**:
   - XGBoost: Known for high accuracy and configurability.
3. **Other Methods**:
   - Linear Regression and other traditional ML models were tested (lightGBM, CatBoost, Random Forest) but did not match the performance of advanced methods.

### Optimization
- **Loss Functions**:
  - Mean Squared Error (MSE): Standard loss, sensitive to outliers.
  - Huber Loss: Combines MSE and Mean Absolute Error (MAE), robust to outliers.
  - Mean Absolute Error (MAE): Not influenced by outliers.
  - Best Results: Achieved using MSE, closely followed by Huber Loss.
- **Hyperparameter Tuning**:
  - Optuna was used for efficient hyperparameter tuning of XGBoost, yielding improved performance.
- **Cross-Validation (in XGBoost)**:
  - Used to ensure the robustness of model performance, particularly given the target variable's wide value range.

---

## Results

### Performance Metrics
The performance of various approaches is summarized below:

| Criterion      | MAE  | Median Error | MAPE  |
|----------------|------|--------------|-------|
| **AutoGluon**  | 5074 | 1033         | 30.42% |
| **XGBoost + Optuna** | 4933 | 1058         | 31.01% |

- **AutoGluon**: Chosen as the final model due to its balance of performance and efficiency.
- **XGBoost**: Achieved slightly better MAE but required extensive tuning with Optuna.

### Key Observations
- **AutoML vs Manual Tuning**: AutoML solutions like AutoGluon matched the performance of manually tuned XGBoost while saving time and effort.
- **Preprocessing Impact**: Minimal preprocessing preserved dataset integrity and contributed to model robustness.
- **Loss Functions**: MSE and Huber Loss performed best, with Huber offering added resilience against outliers.

---

## Future Improvements

- **Feature Engineering**: Incorporating domain-specific knowledge to create meaningful features.
- **Advanced Models**: Exploring deep learning architectures for further improvements.
- **Scalability**: Adapting the pipeline for larger datasets and more complex use cases.

This project provides a flexible and efficient approach to tabular regression tasks, leveraging both manual tuning and AutoML frameworks for optimal results.