# Sema-ChestX-Former: A Parameter-Efficient Hybrid Transformer-CNN for Robust Thoracic Disease Classification with XAI

[![ELITE Research Lab](https://img.shields.io/badge/Research-ELITE%20Research%20Lab-blue)](https://github.com/ELITE-Research-Lab)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Results](#results)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## 📖 About the Paper

**Sema-ChestX-Former** is a novel, parameter-efficient hybrid architecture designed for the automated analysis of Chest X-Rays (CXRs). Developed by researchers at the **ELITE Research Lab**, this model synergistically integrates the local feature extraction capabilities of Convolutional Neural Networks (CNNs) with the global context understanding of Vision Transformers (ViTs).

Deep learning models often face a trade-off: CNNs excel at local details but miss global context, while ViTs capture long-range dependencies but are computationally expensive. Sema-ChestX-Former bridges this gap, achieving state-of-the-art performance with only **1.84 million parameters**.

The model has been rigorously validated on three large-scale public datasets:
1. **Chest X-Ray (Pneumonia)** (Binary classification)
2. **COVID-19 Radiography** (Multi-class classification)
3. **NIH ChestX-ray14** (Multi-label classification)

---

## ✨ Key Features

* **Hybrid Architecture**: Combines a Transformer backbone for semantic spatial feature extraction with specialized CNN-based attention blocks.
* **Parameter Efficiency**: Achieves robust performance with only ~1.84M parameters, making it suitable for clinical deployment.
* **Comprehensive Validation**: Tested on binary, multi-class, and multi-label challenges.
* **Explainability**: Integrated Gradient-weighted Class Activation Mapping (Grad-CAM) to provide visual, human-understandable heatmaps for model predictions.

---

## 🏗️ Architecture

The architecture consists of four distinct stages:
1.  **Semantic Spatial Backbone**: Extracts features using an initial CNN block followed by "Partition and Reconstruct" blocks containing Multi-Head Self-Attention (MHSA).
2.  **Attention CNN**: Refines features using convolutional layers and custom Channel/Spatial attention modules.
3.  **Squeeze and Excitation**: Performs final channel-wise recalibration.
4.  **Classification Head**: Flattens the features for the final prediction.

<img width="1708" height="1000" alt="4  Sema-ChestX-Former" src="https://github.com/user-attachments/assets/06688d34-b0e3-4cb9-b2bd-214049ed25d2" />


---

## 📊 Results

### 1. NIH ChestX-ray14 Dataset (Multi-Label)
Comparison with SOTA Graph and Transformer-based models. Sema-ChestX-Former achieves the highest Mean AUC.

| Model | Card | Emp | Effu | Her | Inf | Mass | Nod | Atel | P1 | P2 | PT | Edem | Fib | Cons | Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MSASG | 0.936 | 0.915 | 0.881 | **0.982** | **0.711** | 0.800 | 0.707 | 0.786 | 0.828 | **0.880** | **0.815** | **0.915** | **0.852** | 0.815 | 0.842 |
| ImageGCN | 0.890 | 0.915 | 0.874 | 0.943 | 0.702 | 0.843 | 0.768 | 0.802 | 0.715 | 0.883 | 0.791 | 0.900 | 0.825 | 0.796 | 0.832 |
| GWSA-LCDS | 0.877 | 0.924 | 0.827 | 0.921 | 0.701 | 0.822 | 0.790 | 0.770 | 0.732 | 0.870 | 0.782 | 0.847 | 0.839 | 0.746 | 0.818 |
| PCAN | 0.899 | 0.921 | 0.837 | 0.943 | 0.706 | 0.834 | 0.786 | 0.785 | 0.730 | 0.871 | 0.791 | 0.854 | 0.817 | 0.763 | 0.824 |
| CheXGAT | 0.879 | **0.944** | 0.837 | 0.931 | 0.699 | 0.839 | **0.793** | 0.786 | 0.741 | 0.879 | 0.794 | 0.851 | 0.842 | 0.754 | 0.826 |
| **Sema-ChestX-Former (Ours)** | 0.913 | 0.935 | **0.904** | 0.864 | 0.671 | **0.885** | 0.789 | **0.829** | **0.904** | 0.811 | 0.808 | 0.911 | 0.786 | **0.818** | **0.846** |


<img width="609" height="604" alt="0  NIH ROC AUC" src="https://github.com/user-attachments/assets/6afd62cb-cd99-4632-9c95-c19c35b30f8e" />


### 2. Pneumonia Dataset (Binary)
Performance comparison on the binary pediatric pneumonia dataset.

| Model / Method | Validation Accuracy | Precision | Recall | F1-Score | ROC AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ensemble (ResNet-50, etc.) | 95.83% | 95.73% | 97.76% | 96.70% | 95.21% |
| EfficientNet-B2 | 96.33% | 98.00% | 97.00% | 97.00% | 99.10% |
| Ensemble (GoogLeNet, etc.) | 98.81% | 98.82% | 98.80% | 98.79% | 98.35% |
| **Sema-ChestX-Former (Ours)** | **99.69%** | **99.63%** | **99.71%** | **99.67%** | **0.9992** |


<img width="1006" height="855" alt="0  ROC Pneumonia" src="https://github.com/user-attachments/assets/1e4aa06f-2273-4e96-a086-5dd8d0f34193" />


### 3. COVID-19 Radiography Dataset (Multi-Class)
Performance on the 4-class COVID-19 Radiography Database.

| Model / Method | Validation Accuracy | Precision | Recall | F1-Score | ROC AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Xception | 97.97% | 99.00% | 92.00% | 95.00% | N/A |
| Modified MobileNet V2 | 98.00% | 97.00% | 98.00% | 97.00% | N/A |
| ResNet-34 | 98.33% | 96.77% | **100.00%** | 98.36% | 0.9836 |
| **Sema-ChestX-Former (Ours)** | **98.34%** | **98.33%** | 98.41% | **98.37%** | **0.9923** |

<img width="797" height="701" alt="0  covid 19 confisuion" src="https://github.com/user-attachments/assets/df7946b0-f666-4a69-87c2-ec29fe4e779d" />

<img width="658" height="561" alt="0  Pneumonia Confusion" src="https://github.com/user-attachments/assets/cc594848-18a2-432c-a133-da782a1f7780" />



<img width="1790" height="1090" alt="0  ROC AUC Loss function" src="https://github.com/user-attachments/assets/59ce050b-5dad-4fc5-96a3-e6cdd1681548" />

---

## 🔍 Explainable AI (XAI)

To address the "black box" nature of deep learning, we utilize **Grad-CAM** (Gradient-weighted Class Activation Mapping). This provides visual heatmaps highlighting the specific image regions driving the model's decisions, ensuring transparency and clinical interpretability.

<img width="2319" height="1965" alt="0  Pneumonia Grad Cam" src="https://github.com/user-attachments/assets/ed806223-bc22-435a-b010-ff2f1e44fa2c" />

<img width="2722" height="1398" alt="0  NIH XAI 1" src="https://github.com/user-attachments/assets/8cb14c3b-b251-4f3f-a0db-88ad1b067671" />

<img width="2742" height="1049" alt="0  NIH XAI 3" src="https://github.com/user-attachments/assets/5e7869fd-7f71-4f28-a9c8-7febbceeeb99" />

---

## 👥 Acknowledgements

This research was conducted by the **ELITE Research Lab**.
