# Multimodal Regression for Value Prediction

This repository contains a multimodal regression framework designed to predict target values by combining information from three distinct data modalities: **tabular**, **text**, and **image** data. The project employs advanced neural network architectures and innovative fusion strategies to achieve state-of-the-art results in a generalized, scalable format.

---

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Further Improvements](#further-improvements)

---

## Description

The project integrates and processes tabular data, textual descriptions, and image data to predict a continuous target variable. Each modality is processed through specialized neural networks, and their latent representations are fused using a combination of **gated networks**, **multimodal attention mechanisms**, and **Mixture of Experts (MoE)** approaches. 

Key highlights include:
- **Multimodal Learning**: Combines tabular, text, and image data for robust predictions.
- **Innovative Fusion Strategies**: Employs weighted concatenation, gated attention, and expert gating mechanisms for optimal latent space utilization.
- **Advanced Preprocessing**: Custom handling of missing values, standardization, tokenization, and transformations to prepare diverse data for seamless integration.

The best-performing model achieves a **Mean Absolute Percentage Error (MAPE) of 18.7%**, with a **Mean Absolute Error (MAE) of 5689** and a **Median Error of 498**.


## Features

- **Tabular Data**:
  - Processed using **TabNet**, a state-of-the-art architecture for tabular data.
  - Handles missing values with mean imputation post-standardization.
- **Text Data**:
  - Embedding using a lightweight BERT model for generalization on unseen data.
  - Semantic extraction using a **BiLSTM** model with bidirectional encoding.
- **Image Data**:
  - Processed through a three-layer **Convolutional Neural Network (CNN)** with global average pooling for computational efficiency.
- **Fusion Strategies**:
  - **Weighted Concatenation**: Scales modalities dynamically before merging.
  - **Gated Network**: Trains modality-specific gates for adaptive weighting.
  - **Mixture of Experts (MoE)**: Enforces expert specialization with regularization techniques.


## Data Preprocessing

### Tabular Data

* Standardized to a mean of 0 and standard deviation of 1.
* Missing values imputed with the mean of each column.

### Text

* Tokenized using a lightweight BERT model for embedding generation.
* Textual embeddings averaged into dense tensors for downstream tasks.

### Image Data

* Resized to a uniform scale of 128x128.
* Converted to RGB format and standardized across the dataset.

## Model Architecture

### Individual Modalities

* **Tabular Data**: Processed using TabNet for feature extraction.
* **Text Data**: Encoded with BiLSTM after BERT-based tokenization.
* **Image Data**: Processed through a three-layer CNN with global average pooling.

### Multimodal Fusion

Two primary approaches were implemented to combine the modalities:

1. **MLP-Based Fusion**:
* **Weighted Concatenation**: Trains modality-specific weights for dynamic scaling.
* **Multimodal Attention**: Uses cross-modality and self-attention mechanisms to refine latent representations.
* **Gated Networks**: Trains modality-specific gates to dynamically weigh contributions.

2. **Mixture of Experts (MoE)**:
* **Plain MoE**: Combines latent spaces with gating networks.
* **Regularized MoE**:
    * L1 regularization for gate sparsity
    * L2 regularization for expert stability (not the biases)
    * High entropy enforced in gating outputs for collaboration among experts.

## Results

The **MLP with gated network** produced the best results:

<table style="width: 100%;">
  <tr>
    <th style="width: 70%;">Metric</th>
    <th style="width: 30%;">Value</th>
  </tr>
  <tr>
    <td style="width: 70%;">Mean Absolute Error (MAE)</td>
    <td style="width: 30%;">5689</td>
  </tr>
  <tr>
    <td style="width: 70%;">Median Error</td>
    <td style="width: 30%;">498</td>
  </tr>
  <tr>
    <td style="width: 70%;">Mean Absolute Percentage Error (MAPE)</td>
    <td style="width: 30%;">18.7%</td>
  </tr>
</table>

### Observations

* Multimodal models demonstrated superior robustness and reduced sensitivity to outliers.
* Regularization techniques improved expert collaboration and reduced dead gates.


### Further Improvements
Future enhancements to the framework could include:

* Incorporating advanced feature engineering techniques for tabular data.
* Exploring transformer-based architectures for image processing.
* Leveraging additional multimodal fusion techniques, such as cross-modal transformers.