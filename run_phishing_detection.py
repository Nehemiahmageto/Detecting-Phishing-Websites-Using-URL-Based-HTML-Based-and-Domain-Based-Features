#!/usr/bin/env python3
"""Phishing detection pipeline

This script performs the full workflow:
1. Load the original training CSV (dataset_phishing.csv).
2. Engineer the exact feature set used during training.
3. Fit a preprocessing ColumnTransformer (StandardScaler + OneHotEncoder).
4. Train a RandomForest classifier (you can replace it with any model).
5. Save the trained model (model.pkl) and the fitted transformer
   (preprocess_transformer.pkl).
6. Load a new PhishTank CSV, engineer the same features, transform them
   with the saved transformer, generate predictions, and write the results
   to a CSV file.

Adjust the file‑paths at the top of the script if your data lives in a
different location.
"""

import os
import pandas as pd
import numpy as np
import re
import urllib.parse as urlparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------------------------------------------------
# CONFIGURATION – change these paths if needed
# ------------------------------------------------------------------
PROJECT_ROOT = r"C:/Users/levyc/OneDrive/PUGATORY/Nehemiah Mageto git projects/Detecting-Phishing-Websites-Using-URL-Based-HTML-Based-and-Domain-Based-Features"
TRAIN_CSV = os.path.join(PROJECT_ROOT, "dataset_phishing.csv")
NEW_CSV   = os.path.join(PROJECT_ROOT, "phishtank-com-2025-12-02.csv")
MODEL_PKL = os.path.join(PROJECT_ROOT, "model.pkl")
TRANSFORMER_PKL = os.path.join(PROJECT_ROOT, "preprocess_transformer.pkl")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "predictions_phishtank_2025-12-02.csv")

# ------------------------------------------------------------------
# Helper functions – identical to those used in the notebook
# ------------------------------------------------------------------
def extract_url_parts(url: str):
    p = urlparse.urlparse(url)
    return p.hostname or "", p.path or "", p.query or ""

def is_ip(host: str) -> int:
    return int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host)))

def count_char(s: str, ch: str) -> int:
    return s.count(ch)

def ratio_digits(s: str) -> float:
    if not s:
        return 0.0
    digits = sum(c.isdigit() for c in s)
    return digits / len(s)

def word_stats(s: str):
    words = re.findall(r"\w+", s)
    if not words:
        return 0, 0, 0.0, 0
    lengths = [len(w) for w in words]
    return min(lengths), max(lengths), float(np.mean(lengths)), len(words)

# ------------------------------------------------------------------
# Feature engineering – works for both training and new data
# ------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure URL column is named 'url'
    if 'url' not in df.columns:
        # PhishTank uses 'Phish URL'
        df = df.rename(columns={df.columns[0]: 'url'})

    df['hostname'], df['path'], df['query'] = zip(*df['url'].apply(extract_url_parts))

    # Numerical
    df['length_url']        = df['url'].apply(len)
    df['length_hostname']   = df['hostname'].apply(len)
    df['nb_hyphens']        = df['url'].apply(lambda x: count_char(x, '-'))
    df['nb_percent']        = df['url'].apply(lambda x: count_char(x, '%'))
    df['nb_slash']          = df['url'].apply(lambda x: count_char(x, '/'))
    df['ratio_digits_url']  = df['url'].apply(ratio_digits)
    df['ratio_digits_host'] = df['hostname'].apply(ratio_digits)
    short, long, avg_len, n_words = zip(*df['url'].apply(word_stats))
    df['shortest_words_raw'] = short
    df['longest_words_raw']  = long
    df['avg_words_raw']      = avg_len
    df['length_words_raw']   = n_words

    # Binary flags
    df['ip']          = df['hostname'].apply(is_ip)
    df['nb_tilde']    = df['url'].apply(lambda x: int('~' in x))
    df['nb_star']     = df['url'].apply(lambda x: int('*' in x))
    df['nb_dslash']   = df['url'].apply(lambda x: int('//' in x))
    df['https_token'] = df['url'].apply(lambda x: int('https' in x.lower()))
    df['punycode']    = df['hostname'].apply(lambda x: int('xn--' in x.lower()))
    df['port']        = df['hostname'].apply(lambda x: int(':' in x and x.split(':')[-1].isdigit()))

    # Categorical counts
    df['nb_dots']        = df['hostname'].apply(lambda x: count_char(x, '.'))
    df['nb_at']          = df['url'].apply(lambda x: count_char(x, '@'))
    df['nb_qm']          = df['url'].apply(lambda x: count_char(x, '?'))
    df['nb_and']         = df['url'].apply(lambda x: count_char(x, '&'))
    df['nb_or']          = df['url'].apply(lambda x: count_char(x, '|'))
    df['nb_eq']          = df['url'].apply(lambda x: count_char(x, '='))
    df['nb_underscore']  = df['url'].apply(lambda x: count_char(x, '_'))
    df['nb_colon']       = df['url'].apply(lambda x: count_char(x, ':'))
    df['nb_comma']       = df['url'].apply(lambda x: count_char(x, ','))
    df['nb_semicolumn']  = df['url'].apply(lambda x: count_char(x, ';'))

    return df

# ------------------------------------------------------------------
# Column groups – must match the lists used during training
# ------------------------------------------------------------------
NUMERICAL_COLS = [
    'length_url','length_hostname','nb_hyphens','nb_percent','nb_slash',
    'ratio_digits_url','ratio_digits_host','length_words_raw','shortest_words_raw',
    'longest_words_raw','avg_words_raw','nb_hyperlinks','ratio_intHyperlinks',
    # add any extra numeric columns you created (e.g., domain_registration_length, web_traffic)
]

BINARY_COLS = [
    'ip','nb_tilde','nb_star','nb_dslash','https_token','punycode','port',
    # add any other binary columns you defined
]

CATEGORICAL_COLS = [
    'nb_dots','nb_at','nb_qm','nb_and','nb_or','nb_eq',
    'nb_underscore','nb_colon','nb_comma','nb_semicolumn'
]

# ------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------
def main():
    # ---------- Train ----------
    print("Loading training data …")
    df_train = pd.read_csv(TRAIN_CSV)
    df_train = engineer_features(df_train)

    # Build preprocessing pipeline
    numeric_tf = Pipeline([('scaler', StandardScaler())])
    binary_tf  = 'passthrough'
    cat_tf = Pipeline([
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])
    preprocess = ColumnTransformer([
        ('num', numeric_tf, NUMERICAL_COLS),
        ('bin', binary_tf, BINARY_COLS),
        ('cat', cat_tf, CATEGORICAL_COLS)
    ], remainder='drop')

    X_train = df_train[NUMERICAL_COLS + BINARY_COLS + CATEGORICAL_COLS]
    y_train = df_train['status'].map({'legitimate': 0, 'phishing': 1}).astype(int)

    print("Fitting preprocessing pipeline …")
    preprocess.fit(X_train)
    joblib.dump(preprocess, TRANSFORMER_PKL)
    print(f"Transformer saved to {TRANSFORMER_PKL}")

    X_train_prepared = preprocess.transform(X_train)
    print("Training RandomForest model …")
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train_prepared, y_train)
    joblib.dump(model, MODEL_PKL)
    print(f"Model saved to {MODEL_PKL}")

    # ---------- Predict on new data ----------
    print("\nLoading new PhishTank data …")
    df_new = pd.read_csv(NEW_CSV)
    df_new = engineer_features(df_new)

    X_new = df_new[NUMERICAL_COLS + BINARY_COLS + CATEGORICAL_COLS]
    X_new_prepared = preprocess.transform(X_new)
    preds = model.predict(X_new_prepared)
    df_new['predicted_label'] = preds
    df_new.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions written to {PREDICTIONS_CSV}")

    # Optional evaluation if ground truth column exists
    if 'Valid?' in df_new.columns:
        mapping = {'VALID PHISH': 1, 'ONLINE': 1, '': 0}
        y_true = df_new['Valid?'].map(mapping).astype(int)
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        acc = accuracy_score(y_true, preds)
        print("\nEvaluation on PhishTank (if ground truth present):")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_true, preds))
        print("Confusion matrix:\n", confusion_matrix(y_true, preds))

if __name__ == "__main__":
    main()
