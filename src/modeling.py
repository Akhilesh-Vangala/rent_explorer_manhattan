"""
Model training and evaluation functions for SmartRent Manhattan.

This module provides functions for training machine learning models,
evaluating their performance, and comparing results.
"""

import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from src.config.constants import (
    DATA_CONSTANTS,
    MODEL_CONSTANTS
)


def train_test_split_df(df: pd.DataFrame, target_column: str=DATA_CONSTANTS.TARGET_COLUMN) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column. If None, uses default from config.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"\n[STEP 2] Train/Validation/Test Split ({100 - (MODEL_CONSTANTS.TEST_SIZE + MODEL_CONSTANTS.VAL_SIZE) * 100}%/{MODEL_CONSTANTS.VAL_SIZE * 100}%/{MODEL_CONSTANTS.TEST_SIZE * 100}%)...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONSTANTS.TEST_SIZE,
        random_state=MODEL_CONSTANTS.RANDOM_STATE
    )

    val_size = MODEL_CONSTANTS.VAL_SIZE / (MODEL_CONSTANTS.TEST_SIZE + MODEL_CONSTANTS.VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=MODEL_CONSTANTS.RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model_name, X_train, y_train):
    """
    Train a model.
    
    Args:
        model_name: Name of the model
        X_train: Training features
        y_train: Training target
    """
    start_time = time.time()
    if model_name == "linear_regression":
        print("\n   [3a] Linear Regression...")
        model = LinearRegression(**MODEL_CONSTANTS.LINEAR_REGRESSION_PARAMS)
    elif model_name == "random_forest":
        print("\n   [3b] Random Forest...")
        model = RandomForestRegressor(**MODEL_CONSTANTS.RF_PARAMS)
    elif model_name == "xgboost":
        print("\n   [3c] XGBoost (with early stopping)...")
        model = XGBRegressor(**MODEL_CONSTANTS.XGB_PARAMS)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"      Model training time: {training_time:.4f} seconds")
    return model, training_time


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary containing evaluation metrics (rmse, mae, r2)
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def cross_validate_model(model, X_train, y_train, cv=5, scoring='r2'):
    """
    Perform k-fold cross-validation.

    Parameters:
        model: Model to evaluate
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv (int): Number of folds (default 5)
        scoring (str): Scoring metric (default 'r2')

    Returns:
        np.array: Array of scores for each fold
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    return scores


def get_cv_metrics(model, X_train, y_train, cv=5):
    """
    Get comprehensive cross-validation metrics.

    Computes mean and standard deviation for R², RMSE, and MAE
    across k folds.

    Parameters:
        model: Model to evaluate
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv (int): Number of folds (default 5)

    Returns:
        dict: Dictionary with mean and std for each metric
    """
    print(f"\n[STEP 4] Cross-Validation ({cv}-fold)...")
    r2_scores = cross_validate_model(model, X_train, y_train, cv=cv, scoring='r2')

    rmse_scores = -cross_validate_model(
        model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error'
    )

    mae_scores = -cross_validate_model(
        model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error'
    )

    return {
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores)
    }


def model_comparison_table(results):
    """
    Create a comparison table from model evaluation results.
    
    Args:
        results: Dictionary mapping model names to evaluation metrics
    
    Returns:
        DataFrame with model comparison results
    """
    df = pd.DataFrame(results)
    df = df.T
    return df


def print_model_metrics(metrics):
    """
    Print model metrics.
    
    Args:
        metrics: Dictionary containing model metrics
    """
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    test_metrics = metrics['test']
    print(f"      Training time: {metrics['time']:.4f} seconds")
    print(f"      Train R2: {train_metrics['r2']:.6f}, RMSE: ${train_metrics['rmse']:.2f}")
    print(f"      Val R2: {val_metrics['r2']:.6f}, RMSE: ${val_metrics['rmse']:.2f}")
    print(f"      Test R2: {test_metrics['r2']:.6f}, RMSE: ${test_metrics['rmse']:.2f}")