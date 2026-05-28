# 🛡️ FraudShield AI: Real-Time Credit Card Fraud Detection

An advanced, feature-rich machine learning suite and glassmorphic Streamlit dashboard designed to detect, analyze, and inspect credit card fraud. Equipped with six state-of-the-art classifier models trained on a balanced dataset of credit card transactions.It shows when we balanced the full dataset first and then split:
the result is that model indirectly sees patterns from test data as well as metrics become unrealistically perfect.

Live Web Application: **[Deployed on Streamlit Community Cloud](https://credit-card-fraud-detection-balanced-data-ekft7frbmnbtpbmhe8hj.streamlit.app/)**

---

## ✨ Features

* **🤖 Six Classifier ML Suite:** Fully trained, evaluated, and saved pipeline binaries (`.joblib`) for:
  * **XGBoost Classifier** (Best Overall Performance)
  * **LightGBM Classifier** (Highly Efficient & Fast)
  * **Multi-Layer Perceptron** (Neural Network)
  * **Linear Support Vector Machine** (Linear SVC)
  * **Logistic Regression** (Interpretable Baseline)
  * **Decision Tree Classifier**
* **📈 Performance Curves Dashboard:** View comparative metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC, PR AUC) side-by-side with pre-computed charts (ROC Curves, Precision-Recall Curves, and Confusion Matrices) for all models.
* **🎯 Single-Transaction Predictor:** A real-time transaction integrity calculator.
  * **Quick scenario loaders:** Instant buttons to pull real safe/fraudulent transactions from the dataset to test predictions instantly without manually typing 28 components.
  * **Confidence gauges:** Visual indicators displaying security integrity probability.
* **📁 Batch CSV Processing:** Upload bulk transaction files, inspect prediction summaries, preview formatted results highlighting fraudulent rows, and download complete annotated result sheets.
* **💎 Ultra-Premium UI:** Styled with a modern glassmorphic theme using custom dark backgrounds,Outfit & Space Grotesk typography, responsive layouts, and soft indicator alerts.

---

## 📊 Model Evaluation Summary

The classifiers were trained on a balanced dataset (568,630 rows) with the following final test validation results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | PR AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **99.958%** | **99.916%** | **100.00%** | **99.958%** | **1.00000** | **1.00000** |
| **LightGBM** | 99.970% | 99.942% | 100.00% | 99.970% | 0.99992 | 0.99980 |
| **Decision Tree** | 98.605% | 98.568% | 98.644% | 98.606% | 0.99678 | 0.98760 |
| **Neural Network** | 96.400% | 98.340% | 94.390% | 96.330% | 0.99480 | 0.99520 |
| **Logistic Regression** | 96.498% | 97.717% | 95.220% | 96.452% | 0.99350 | 0.99446 |
| **SVM (Linear)** | 96.333% | 98.015% | 94.582% | 96.268% | 0.99331 | 0.99433 |

---

## 📂 Directory Structure

```directory
├── app.py                      # Main Streamlit web application & UI
├── train_models.py             # Model training, saving, and plotting pipeline
├── creditcard_samples.csv      # Pre-extracted authentic transactions for testing (56KB)
├── requirements.txt            # Python dependencies for local & cloud runs
├── model_outputs/              # Generated model binaries & performance plots
│   ├── xgboost_model.joblib
│   ├── lightgbm_model.joblib
│   ├── metrics_summary.csv
│   ├── xgboost_confusion_matrix.png
│   └── ... (evaluation figures)
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
