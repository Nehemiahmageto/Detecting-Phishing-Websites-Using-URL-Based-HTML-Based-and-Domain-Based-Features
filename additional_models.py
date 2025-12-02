"""
Additional ML Models for Phishing Detection
Implements Decision Tree, Lasso (L1), and Ridge (L2) Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath='dataset_phishing.csv', test_size=0.2, random_state=42):
    """
    Load and prepare the dataset for modeling
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file
    test_size : float
        Proportion of dataset to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Encode target if needed
    if df['status'].dtype == 'object':
        df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})
    
    # Separate features and target
    X = df.drop(columns=['status', 'url'], errors='ignore')
    y = df['status']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train, tune_hyperparameters=False):
    """
    Train a Decision Tree classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    model : DecisionTreeClassifier
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING DECISION TREE CLASSIFIER")
    print("="*60)
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Use reasonable default parameters
        model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='gini',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model


def train_lasso_logistic(X_train, y_train, tune_hyperparameters=False):
    """
    Train Logistic Regression with L1 regularization (Lasso)
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    model : LogisticRegression
        Trained model
    scaler : StandardScaler
        Fitted scaler (Lasso requires feature scaling)
    """
    print("\n" + "="*60)
    print("TRAINING LASSO (L1) LOGISTIC REGRESSION")
    print("="*60)
    
    # Scale features (required for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        }
        
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            lasso, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Use default parameter
        model = LogisticRegression(
            penalty='l1',
            C=1.0,
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Show feature selection (Lasso can zero out coefficients)
    non_zero_features = np.sum(model.coef_ != 0)
    print(f"Number of non-zero features: {non_zero_features} out of {X_train.shape[1]}")
    
    return model, scaler


def train_ridge_logistic(X_train, y_train, tune_hyperparameters=False):
    """
    Train Logistic Regression with L2 regularization (Ridge)
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    model : LogisticRegression
        Trained model
    scaler : StandardScaler
        Fitted scaler (Ridge requires feature scaling)
    """
    print("\n" + "="*60)
    print("TRAINING RIDGE (L2) LOGISTIC REGRESSION")
    print("="*60)
    
    # Scale features (required for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        }
        
        ridge = LogisticRegression(penalty='l2', solver='lbfgs', random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            ridge, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Use default parameter
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, scaler


def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """
    Evaluate model performance with comprehensive metrics
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name of the model for display
    scaler : StandardScaler, optional
        Scaler for models that require it
        
    Returns:
    --------
    results : dict
        Dictionary containing all evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*60}")
    
    # Scale test data if scaler is provided
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Store results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def compare_models(results_list):
    """
    Compare multiple models side by side
    
    Parameters:
    -----------
    results_list : list of dict
        List of results dictionaries from evaluate_model
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score'],
            'ROC-AUC': r['roc_auc']
        }
        for r in results_list
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Find best model for each metric
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        best_idx = comparison_df[metric].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_score = comparison_df.loc[best_idx, metric]
        print(f"{metric:12s}: {best_model:30s} ({best_score:.4f})")
    
    return comparison_df


def save_models(models_dict, filepath_prefix='model'):
    """
    Save trained models to disk
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and (model, scaler) tuples as values
    filepath_prefix : str
        Prefix for saved model files
    """
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    for name, model_data in models_dict.items():
        if isinstance(model_data, tuple):
            model, scaler = model_data
            filename = f"{filepath_prefix}_{name.lower().replace(' ', '_')}.pkl"
            joblib.dump({'model': model, 'scaler': scaler}, filename)
        else:
            model = model_data
            filename = f"{filepath_prefix}_{name.lower().replace(' ', '_')}.pkl"
            joblib.dump({'model': model}, filename)
        
        print(f"Saved {name} to {filename}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PHISHING DETECTION - ADDITIONAL MODELS")
    print("="*60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data('dataset_phishing.csv')
    
    # Train models
    dt_model = train_decision_tree(X_train, y_train, tune_hyperparameters=False)
    lasso_model, lasso_scaler = train_lasso_logistic(X_train, y_train, tune_hyperparameters=False)
    ridge_model, ridge_scaler = train_ridge_logistic(X_train, y_train, tune_hyperparameters=False)
    
    # Evaluate models
    dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    lasso_results = evaluate_model(lasso_model, X_test, y_test, "Lasso (L1) Logistic Regression", lasso_scaler)
    ridge_results = evaluate_model(ridge_model, X_test, y_test, "Ridge (L2) Logistic Regression", ridge_scaler)
    
    # Compare models
    results_list = [dt_results, lasso_results, ridge_results]
    comparison_df = compare_models(results_list)
    
    # Save models
    models_dict = {
        'Decision_Tree': dt_model,
        'Lasso_L1': (lasso_model, lasso_scaler),
        'Ridge_L2': (ridge_model, ridge_scaler)
    }
    save_models(models_dict)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
