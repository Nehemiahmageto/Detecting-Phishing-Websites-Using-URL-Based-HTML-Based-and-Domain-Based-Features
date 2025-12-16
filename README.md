# Detecting Phishing Websites Using URL-Based, HTML-Based, and Domain-Based Features

This repository demonstrates a workflow for detecting phishing URLs by combining extensive feature engineering, exploratory data analysis, and machine learning modeling. The notebook `Model.ipynb` implements the end‑to‑end process, and this README mirrors its structure.

---

## Dataset Overview

- **Rows:** 11,430
- **Features:** 88 (numeric) + 1 textual column `url`
- The original `status` column contained only missing values and was removed.

The feature set includes:
- URL structural metrics (e.g., `length_url`, `nb_dots`, `nb_hyphens`)
- Domain‑level attributes (`domain_age`, `domain_registration_length`)
- HTML/JavaScript signals (`onmouseover`, `sfh`, `popup_window`)
- Link ratios (`ratio_intHyperlinks`, `ratio_extHyperlinks`)
- Search‑engine visibility (`google_index`)
- Statistical phishing indicators (`statistical_report`, `suspecious_tld`)

---

## Data Cleaning

1. Dropped the `status` column due to 100 % missing values.
2. Confirmed no remaining NaNs across the dataset.
3. Ensured all feature columns are numeric (except `url`).
4. Applied `StandardScaler` to numeric features for downstream PCA and clustering.

---

## Exploratory Analysis

### Principal Component Analysis (PCA)
- Reduced dimensionality to visualize data structure.
- Most points cluster within **PC1: –3 to 10** and **PC2: –3 to 5**.
- No clear separation between legitimate and phishing URLs, which is expected for high‑dimensional security data.

### K‑Means Clustering (Unsupervised)
- Performed clustering on the PCA‑reduced components.
- Identified distinct clusters reflecting structural differences in URLs, not class labels.

---

## Reconstructing the Target Variable

Because the original label was unavailable, a rule‑based heuristic was created to generate a **silver‑label** target:
- `0` = Legitimate
- `1` = Phishing

Heuristics include:
- Very short domain age
- Excessive special characters in the URL
- Presence of raw IP addresses
- Suspicious TLDs (`suspecious_tld`)
- High values in `statistical_report`
- Fake HTTPS tokens
- URLs not indexed by Google

These rules produce a reliable proxy label for supervised learning.

---

## Supervised Machine Learning (Next Steps)

With the reconstructed target, the notebook proceeds to:
1. Train/test split
2. Model training using:
   - Random Forest
   - Logistic Regression
   - XGBoost
   - Support Vector Machine
3. Evaluation with accuracy, ROC‑AUC, confusion matrix, and feature importance visualizations.
4. (Optional) Deploy the best model as a Flask API for real‑time prediction.

---

## Project Status
- Dataset loaded and cleaned
- PCA and K‑Means completed
- Rule‑based target variable generated
- Ready to begin supervised classification modeling

---

## Next Steps
- Train multiple ML models and compare performance
- Plot ROC curves, confusion matrices, and feature importance
- Optionally package the model into a prediction API (Flask) for deployment

---

*All code and analysis are available in `Model.ipynb`.*

