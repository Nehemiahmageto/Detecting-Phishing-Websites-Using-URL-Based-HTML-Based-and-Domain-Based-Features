"""
Phishing Website Detection - Complete ML Pipeline
Building from scratch with proper feature engineering and model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, roc_auc_score, 
                            roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("PHISHING WEBSITE DETECTION - ML PIPELINE")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n[1/7] Loading and Exploring Data...")

# Load dataset
df = pd.read_csv('dataset_phishing.csv')

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")  # Excluding target
print(f"Samples: {df.shape[0]}")

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"Missing values: {missing_values}")

# Check class distribution
print("\nClass Distribution:")
print(df['status'].value_counts())
print(df['status'].value_counts(normalize=True))

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================
print("\n[2/7] Preparing Data...")

# Encode target variable (legitimate=0, phishing=1)
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# Separate features and target
# Drop 'url' column as it's not useful for ML (it's just text)
X = df.drop(columns=['url', 'status'])
y = df['status']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Feature types
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical features: {len(numerical_features)}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/7] Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y  # Maintain class distribution
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training set class distribution:\n{y_train.value_counts()}")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n[4/7] Scaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n[5/7] Training Multiple Models...")

# Dictionary to store models and their results
models = {}
results = {}

# Model 1: Logistic Regression
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr_model

# Model 2: Random Forest
print("  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model

# Model 3: Gradient Boosting
print("  Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)
gb_model.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb_model

# Model 4: Support Vector Machine
print("  Training SVM...")
svm_model = SVC(
    kernel='rbf',
    random_state=RANDOM_STATE,
    probability=True
)
svm_model.fit(X_train_scaled, y_train)
models['SVM'] = svm_model

# Model 5: K-Nearest Neighbors
print("  Training KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
models['KNN'] = knn_model

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n[6/7] Evaluating Models...")
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

for model_name, model in models.items():
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
    
    if y_pred_proba is not None:
        results[model_name]['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if 'auc_roc' in results[model_name]:
        print(f"  AUC-ROC:   {results[model_name]['auc_roc']:.4f}")

# ============================================================================
# 7. DETAILED ANALYSIS OF BEST MODEL
# ============================================================================
print("\n[7/7] Detailed Analysis...")

# Find best model based on F1-score
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)
print("\nInterpretation:")
print(f"  True Negatives (Legitimate correctly classified): {cm[0,0]}")
print(f"  False Positives (Legitimate classified as Phishing): {cm[0,1]}")
print(f"  False Negatives (Phishing classified as Legitimate): {cm[1,0]}")
print(f"  True Positives (Phishing correctly classified): {cm[1,1]}")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, best_predictions, 
                          target_names=['Legitimate', 'Phishing']))

# Cross-validation score
print("\nCross-Validation Performance (5-fold):")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"  CV F1-Scores: {cv_scores}")
print(f"  Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 8. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Test F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"Test Precision: {results[best_model_name]['precision']:.4f}")
print(f"Test Recall: {results[best_model_name]['recall']:.4f}")

# Check for overfitting
train_pred = best_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = results[best_model_name]['accuracy']
print(f"\nOverfitting Check:")
print(f"  Training Accuracy: {train_accuracy:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Difference: {abs(train_accuracy - test_accuracy):.4f}")
if abs(train_accuracy - test_accuracy) < 0.05:
    print("  Status: ✓ Good generalization (no significant overfitting)")
else:
    print("  Status: ⚠ Possible overfitting detected")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)
