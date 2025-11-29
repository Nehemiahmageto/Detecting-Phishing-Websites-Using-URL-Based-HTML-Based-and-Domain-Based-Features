# Detecting Phishing Websites Using URL-Based, HTML-Based, and Domain-Based Features

This project focuses on identifying phishing URLs using a combination of feature engineering, rule-based heuristic labeling, and machine learning modeling. The dataset contains 11,430 URLs and 88 extracted features describing URL structure, domain characteristics, HTML behavior, and statistical indicators.

---

##  Dataset Overview

After cleaning and preprocessing, the dataset consists of:

- **11,430 rows**
- **88 features**
- 1 textual field: `url`
- The original `status` column contained **11,430 NaN values** and was therefore **dropped entirely**.

These features include:
- URL length and structure (`length_url`, `nb_dots`, `nb_hyphens`, etc.)
- Domain-level statistics (`domain_age`, `domain_registration_length`)
- HTML and JavaScript indicators (`onmouseover`, `sfh`, `popup_window`)
- External/internal hyperlink ratios
- Search engine visibility (`google_index`)
- Statistical phishing signals (`statistical_report`, `suspecious_tld`)

---

##  Data Cleaning Steps Completed

1. Dropped the `status` column due to 100% missing values.
2. Verified no remaining NaNs across the dataset.
3. Ensured all feature columns are numeric (except `url`).
4. Scaled numerical features using StandardScaler for PCA and clustering workflows.

---

##  Exploratory Analysis

### PCA (Principal Component Analysis)

- Applied PCA to reduce dimensionality.
- Most points cluster between:
  - **PC1: –3 to 10**
  - **PC2: –3 to 5**
- No distinct, naturally separable classes appear (as expected with high-dimensional security data).

The PCA scatter plot helps visualize structure but does not indicate inherent labels.

---

##  Reconstructing the Target Variable (Because `status` Was Removed)

Since the dataset lacked a usable label column, a **rule-based heuristic labeling system** was implemented to classify URLs as:

- **0 = Legitimate**
- **1 = Phishing**

This follows common patterns used in cybersecurity research and phishing-detection systems.

### The rules are based on:

- Very short domain age  
- Excessive special characters in the URL  
- Use of raw IP addresses  
- Suspicious or statistical TLD indicators  
- Known phishing patterns in features (`suspecious_tld`, `statistical_report`)  
- Fake HTTPS tokens  
- URLs not indexed by Google  

These rules were combined to create a reliable, “silver label” target variable.

---

##  Unsupervised Learning (Before Label Reconstruction)

### K-Means Clustering

- Performed clustering on PCA-reduced components.
- Cluster interpretation shows:
  - One cluster dominating the **–3 to 0** PC1 region.
  - Another cluster dominating the **5+** PC1 region.
- These clusters reflect **structural differences in URLs**, not classes.

K-means was exploratory and not used for evaluation since no true label existed.

---

##  Supervised Machine Learning (Next Steps)

Now that a target variable has been reconstructed, the project is ready to begin supervised modeling:

- Train/test split  
- Random Forest  
- Logistic Regression  
- XGBoost  
- SVM  
- Evaluation metrics  

This phase begins immediately after establishing the rule-based labels.

---

##  Project Status (So Far)

- Dataset loaded   
- Irrelevant/empty label column removed   
- PCA completed   
- K-Means clustering completed   
- Rule-based target variable constructed   
- Ready to begin ML classification modeling   

---

##  Next Steps

- Train multiple ML models  
- Compare performance  
- Plot ROC, confusion matrix, and feature importance  
- Deploy as a prediction API / Flask app (optional)  

---
