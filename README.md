# A Hybrid Approach for Skin Lesion ClassificationCombining Handcrafted and Deep Learning Features

# Hybrid Approach for Skin Lesion Classification Combining Handcrafted and Deep Learning Features

**License:** MIT  
**Python:** 3.10+  
**Framework:** PyTorch  

---

## 📄 Abstract

This repository contains the PyTorch implementation of our **ICSPIS 2025** submission for binary melanoma detection on the **HAM10000** dataset. We address key challenges in dermoscopy-based skin cancer diagnosis: class imbalance, limited data, and the need for lightweight, interpretable models.

Our **hybrid framework** fuses **33 handcrafted dermatological descriptors** (shape, color, texture) with **6 supervised LDA-reduced deep features** extracted from a lightly fine-tuned **ResNet-50**. Using classical ML classifiers such as **LightGBM**, we achieve a **ROC-AUC of 0.958** and **accuracy of 0.944**, outperforming individual feature sets while remaining computationally efficient for clinical deployment on consumer-grade hardware.

---

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
