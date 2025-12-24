# Hybrid Approach for Skin Lesion Classification Combining Handcrafted and Deep Learning Features

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

## 📄 Abstract

This repository contains the PyTorch implementation of our **ICSPIS 2025** submission for binary melanoma detection on the **HAM10000** dataset. We address key challenges in dermoscopy-based skin cancer diagnosis: class imbalance, limited data, and the need for lightweight, interpretable models.

Our **hybrid framework** fuses **33 handcrafted dermatological descriptors** (shape, color, texture) with **6 supervised LDA-reduced deep features** extracted from a lightly fine-tuned **ResNet-50**. Using classical ML classifiers such as **LightGBM**, we achieve a **ROC-AUC of 0.958** and **accuracy of 0.944**, outperforming individual feature sets while remaining computationally efficient for clinical deployment on consumer-grade hardware.

---

This is the official code for the ICSPIS 2025 paper:  
**A Hybrid Approach for Skin Lesion Classification Combining Handcrafted and Deep Learning Features**  
Authors: Soud Asaad, Mohamed Deriche (Ajman University)  
[Download Paper (PDF)](paper/Asaad_Deriche_Paper.pdf)

## Features
- Preprocessing: Lesion masking, augmentation (rotations, flips, shifts), standardization.
- Handcrafted Features: Asymmetry, border irregularity, GLCM, LBP, RGB statistics.
- Deep Features: ResNet-50 fine-tuned (5 epochs, LR=1e-5), penultimate activations reduced via LDA to 6 components.
- Fusion: 33 + 6 = 39-D vector.
- Classifiers: SVM, KNN, RF, XGBoost, LightGBM (with grid search CV).
- Evaluation: Accuracy, ROC-AUC, precision/recall/F1 on stratified splits.

## Installation
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/hybrid-skin-lesion-classification.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place in `/data/` (or symlink).

## Usage
Run the main notebook:


## 📂 Repository Structure

```text
├── src/                      # Source code
│   ├── preprocessing.py
│   ├── handcrafted_features.py
│   ├── deep_features.py
│   ├── fusion_and_classification.py
│   └── utils.py
├── notebooks/                # Jupyter notebooks
│   ├── main_pipeline.ipynb
│   ├── experiments_E1_E4.ipynb
│   └── visualization.ipynb
├── data/                     # Dataset instructions (HAM10000 not included)
├── models/                   # Saved models (e.g., LightGBM pickles)
├── results/                  # Figures, ROC curves, CSVs
├── paper/                    # Conference paper PDF
├── README.md
└── requirements.txt
