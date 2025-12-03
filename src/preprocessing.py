"""
Data preprocessing functions for SmartRent Manhattan.

This module provides functions for cleaning, transforming, and engineering
features from raw rental data.
"""

import pandas as pd
import numpy as np
from src.config.constants import (
    DATA_CONSTANTS,
    PREPROCESSING_CONSTANTS,
    FEATURE_CONSTANTS
)


def load_raw_data(path):
    """
    Load raw data from CSV file.
    
    Args:
        path: Path to the CSV file
    
    Returns:
        DataFrame containing raw data
    """
    return pd.read_csv(path)


def drop_columns(df):
    """
    Drop non-predictive columns from the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with columns dropped
    """
    return df.drop(columns=DATA_CONSTANTS.COLUMNS_TO_DROP)


def cast_types(df):
    """
    Cast columns to appropriate data types.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with corrected data types
    """
    for col in PREPROCESSING_CONSTANTS.INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def handle_outliers(df):
    """
    Clip extreme values to handle outliers.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with outliers clipped
    """
    df['size_sqft'] = df['size_sqft'].clip(upper=PREPROCESSING_CONSTANTS.MAX_SIZE_SQFT)
    df['rent'] = df['rent'].clip(upper=PREPROCESSING_CONSTANTS.MAX_RENT)
    return df


def encode_categoricals(df):
    """
    One-hot encode categorical variables.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with categorical variables encoded
    """
    df = pd.get_dummies(df, columns=DATA_CONSTANTS.CATEGORICAL_COLUMNS, drop_first=False)
    return df


def feature_engineering(df):
    """
    Create derived features from existing columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    # Rent per square foot
    df[FEATURE_CONSTANTS.RENT_PER_SQFT] = df['rent'] / df['size_sqft']
    
    # Log transformations
    df[FEATURE_CONSTANTS.LOG_RENT] = np.log1p(df['rent'])
    df[FEATURE_CONSTANTS.LOG_SIZE_SQFT] = np.log1p(df['size_sqft'])
    
    return df


def preprocess(df, train: bool = False):
    """
    Complete preprocessing pipeline.
    
    This function applies all preprocessing steps in sequence:
    1. Drop non-predictive columns
    2. Cast data types
    3. Handle outliers
    4. Encode categorical variables
    5. Engineer features
    
    Args:
        df: Raw input DataFrame
    
    Returns:
        Preprocessed DataFrame ready for modeling
    """
    print("\n[STEP 1] Loading and preprocessing data...")
    df = drop_columns(df)
    df = cast_types(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    if not train:
        df = feature_engineering(df)
    return df
