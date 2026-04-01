#  Credit Card Fraud Detection: A Comparative Machine Learning Research Study (2023 Dataset)
https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

##  Project Overview
This project is a deep-dive research study into the effectiveness of various machine learning architectures for detecting fraudulent credit card transactions. Using the **Credit Card Fraud Detection Dataset 2023**, we compare traditional statistical models, ensemble methods, and deep learning approaches to determine which is most effective for high-stakes financial security.

###  Research Focus
- **Model Benchmarking:** Comparing six different algorithms (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, and Neural Networks).
- **Data Integrity:** Investigating and preventing "Data Leakage" caused by synthetic index columns (`id`, `Unnamed: 0`).
- **Metric Prioritization:** Evaluating performance based on **Recall** (catching fraudsters) and **AUPRC** (the gold standard for fraud research) rather than just simple Accuracy.

---

##  Dataset Specifications
The dataset used is the **2023 version** by Nelgiriye Withana (Kaggle).
- **Size:** 568,630 total transactions.
- **Features:** 28 anonymized PCA-transformed variables (`V1`-`V28`) and a transaction `Amount`.
- **Target:** `Class` (0: Legitimate, 1: Fraud).
- **Balance:** This dataset is **pre-balanced (50/50 split)**, providing a controlled environment for testing model architectures without the bias issues found in the original 2017 imbalanced dataset.

---

##  Methodology & Research Integrity

### 1. Feature Engineering
- **Standard Scaling:** The `Amount` feature was normalized using `StandardScaler` to ensure it doesn't dominate model gradients.
- **Leakage Prevention:** A critical part of this research involved identifying and **dropping the `id` and `Unnamed: 0` columns**. In synthetic data, these columns often contain patterns that lead to "artificially perfect" (1.0) scores. By removing them, we forced the models to learn from transaction behavior alone.

### 2. Model Architectures Tested
1. **Logistic Regression:** A linear baseline for classification.
2. **Random Forest:** An ensemble method for capturing non-linear relationships.
3. **Linear SVM:** A Support Vector Machine optimized for high-dimensional separation.
4. **XGBoost:** A state-of-the-art Gradient Boosting framework.
5. **LightGBM:** A high-speed, leaf-wise growth boosting algorithm.
6. **Neural Network (MLP):** A Multi-Layer Perceptron for deep pattern recognition.

---

##  Research Results & Comparison


| Model | Accuracy | Precision | Recall | F1-Score | AUPRC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.9997** | **0.9994** | **1.0000** | **0.9997** | **0.9999** |
| **LightGBM** | 0.9993 | 0.9987 | 0.9999 | 0.9993 | 0.9998 |
| **Neural Network** | 0.9982 | 0.9969 | 0.9995 | 0.9982 | 0.9997 |
| **Random Forest** | 0.9857 | 0.9987 | 0.9726 | 0.9855 | 0.9994 |
| **Logistic Regression**| 0.9652 | 0.9773 | 0.9526 | 0.9648 | 0.9943 |
| **SVM (Linear)** | 0.9639 | 0.9782 | 0.9492 | 0.9635 | 0.9943 |
*(Note: These results highlight the high separability of classes within this specific synthetic 2023 dataset.)*

---

##  Key Insights & Conclusions
1. **Gradient Boosting Supremacy:** XGBoost and LightGBM consistently achieved the highest AUPRC scores, proving to be the most robust architectures for this tabular data.
2. **Computational Efficiency:** On a MacBook Pro M3 (8GB RAM), **LightGBM** was significantly faster to train than the Neural Network (MLP), suggesting that for large-scale tabular datasets, Tree-based models are more hardware-efficient.
3. **The 1.0 AUPRC Discussion:** While XGBoost achieved a near-perfect score, our research concludes this is a result of the highly distinct patterns in the 2023 synthetic data generation, rather than model "cheating," provided the index columns are removed.

---

##  Tech Stack & Libraries
- **Language:** Python 3.12
- **Data Handling:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Visualization:** `matplotlib`, `seaborn`
- **Hardware Platform:** Apple Silicon M3 (8GB Unified RAM)

---

##  How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-2023.git
