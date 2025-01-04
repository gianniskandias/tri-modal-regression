# Multimodal and Tabular Regression Framework

This repository implements a versatile regression framework for predicting target values using either tabular data alone or a combination of tabular, textual, and image data. The project explores multiple machine learning approaches, including traditional tree-based models, automated machine learning (AutoML), and advanced multimodal fusion techniques, to achieve robust and accurate predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Development](#model-development)
6. [Results](#results)
7. [Observations and insights](#observations-and-insights)
8. [Conclusions](#conclusions)

## Introduction
This repository offers a robust regression framework for diverse use cases:

* **Tabular Regression**: Predicts target values using traditional tabular data, leveraging tree-based models like XGBoost and AutoML frameworks like AutoGluon.
* **Multimodal Regression**: Combines tabular, text, and image data to predict continuous values through innovative fusion strategies and neural network architectures.

The framework is designed to maximize generalizability and scalability, making it suitable for various domains and datasets. While the specific data used for these experiments cannot be disclosed, the modular codebase allows users to easily adapt the solution to their own datasets.

## Features
* **Tabular Models**:
    * Minimal preprocessing to preserve dataset integrity.
    * Advanced optimization techniques such as Optuna for hyperparameter tuning.
    * Support for multiple loss functions, including MSE, MAE, and Huber Loss.
* **Multimodal Models**:
    * Combines tabular, text, and image data with specialized preprocessing.
    * Incorporates state-of-the-art architectures like TabNet, BiLSTMs, and CNNs.
    * Employs advanced fusion strategies such as gated networks and Mixture of Experts (MoE).
* **Performance Evaluation**:
    * Uses robust metrics like Mean Absolute Percentage Error (MAPE) and Mean Absolute Error (MAE).
    * Provides a detailed comparison of tabular-only and multimodal approaches.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gianniskandias/tri-modal-regression.git
   cd multimodal-regression
   ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) Install Jupyter for exploring notebooks:

    ```bash
    pip install notebook
    ```

## Data Preprocessing

### Tabular Data Preprocessing

* **Handling Missing Values**:
    * Columns with only NaNs were removed.
    * NaNs were imputed with zeros or left as-is for models capable of handling missing values (e.g., XGBoost, CatBoost).

* **Target Scaling**:
    * Log and quantile transformations were applied to address right-skewed distributions.
* **Feature Selection**:
    * Columns with >90% NaN values were retained, as their removal worsened model performance.
* **Duplicate Removal**:
    * Duplicates in both rows and features were removed to ensure data consistency.
### Text Data Preprocessing
* Tokenized using a lightweight BERT model for generating embeddings.
* Averaged textual embeddings into dense tensors for downstream tasks.
### Image Data Preprocessing
* Resized images to 128x128 and standardized pixel values.
* Processed through a lightweight CNN architecture for feature extraction.

## Model Development

### Tabular Models
1. **Tree-Based Models**:
    * Implemented XGBoos for its speed and accuracy.
    * Used Optuna for hyperparameter tuning.
2. **AutoML**:
    * Leveraged AutoGluon for rapid model experimentation and optimization.
3. **Loss Functions**:
    * MSE and Huber Loss were preferred for their balance between sensitivity and robustness.


### Multimodal Models
1. **Individual Modalities**:
    * **Tabular Data**: Processed with TabNet for enhanced tabular learning.
    * **Text Data**: Encoded with BiLSTM after tokenization.
    * **Image Data**: Features extracted using a three-layer CNN with global average pooling.
2. **Fusion Strategies**:
    * Weighted Concatenation: Dynamically scales modalities before merging.
    * Gated Networks: Assigns adaptive weights to each modality.
    * Mixture of Experts (MoE): Regularized experts ensure collaborative representation learning.

## Results

### Tabular Results

| Metric      | AutoGluon  | XGBoost + Optuna |
|----------------|------|-------------|
| **MAE**  | 5074 | 4933         |
| **Median Error** | 1033 | 1058         |
| **MAPE** | 30.42% | 31.01%         |

### Multimodal Results

| Metric      | AutoGluon  
|----------------|------|
| **MAE**  | 5689 | 
| **Median Error** | 498 |
| **MAPE** | 18.7% | 

## Observations and Insights
* **Tabular Models**:
    * AutoGluon demonstrated excellent performance with minimal effort, while XGBoost required extensive tuning for slightly better results.
* **Multimodal Models**:
    * Gated networks and MoE fusion strategies significantly improved robustness to outliers.
    * Combining tabular, text, and image data yielded state-of-the-art performance, particularly for datasets with diverse modalities.
* **Cross-Domain Potential**:
    * The tabular and multimodal frameworks could complement each other for datasets with mixed distributions, leveraging tabular models for outlier robustness and multimodal models for better representation of low-value ranges.

## Conclusions

This project showcases a flexible and powerful regression framework, adaptable to both tabular and multimodal data scenarios. Whether you're tackling traditional datasets or integrating diverse modalities, this repository provides a solid foundation for building accurate and scalable predictive models.
