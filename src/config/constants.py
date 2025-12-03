"""
Constants used throughout the SmartRent Manhattan application.

This module contains all constant values including column names, thresholds,
model parameters, and feature engineering constants.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DataConstants:
    """Constants related to data structure and columns."""
    
    # Target column
    TARGET_COLUMN: str = 'rent'
    
    # Columns to drop during preprocessing
    COLUMNS_TO_DROP: List[str] = None
    
    # Categorical columns for encoding
    CATEGORICAL_COLUMNS: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.COLUMNS_TO_DROP is None:
            object.__setattr__(self, 'COLUMNS_TO_DROP', ['rental_id', 'borough'])
        if self.CATEGORICAL_COLUMNS is None:
            object.__setattr__(self, 'CATEGORICAL_COLUMNS', ['neighborhood'])


@dataclass(frozen=True)
class PreprocessingConstants:
    """Constants for data preprocessing."""
    
    # Outlier handling thresholds
    MAX_SIZE_SQFT: float = 3500.0
    MAX_RENT: float = 20000.0
    
    # Type casting
    INTEGER_COLUMNS: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.INTEGER_COLUMNS is None:
            object.__setattr__(self, 'INTEGER_COLUMNS', ['bedrooms', 'floor'])


@dataclass(frozen=True)
class FeatureConstants:
    """Constants for feature engineering."""
    
    # Feature names
    RENT_PER_SQFT: str = 'rent_per_sqft'
    LOG_RENT: str = 'log_rent'
    LOG_SIZE_SQFT: str = 'log_size_sqft'
    
    # Feature engineering operations
    FEATURE_NAMES: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.FEATURE_NAMES is None:
            object.__setattr__(self, 'FEATURE_NAMES', [
                self.RENT_PER_SQFT,
                self.LOG_RENT,
                self.LOG_SIZE_SQFT
            ])


@dataclass(frozen=True)
class ModelConstants:
    """Constants for model training and evaluation."""
    
    # Train/test split
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    
    # Linear Regression (no hyperparameters)
    LINEAR_REGRESSION_PARAMS: dict = None
    
    # Random Forest parameters
    RF_N_ESTIMATORS: int = 200
    RF_RANDOM_STATE: int = 42
    RF_PARAMS: dict = None
    
    # XGBoost parameters - Moderate regularization for small dataset (3,539 samples)
    XGB_N_ESTIMATORS: int = 200  # Reduced from 300 to reduce complexity
    XGB_LEARNING_RATE: float = 0.03  # Reduced from 0.05 for more conservative learning
    XGB_MAX_DEPTH: int = 5  # Reduced from 6 to prevent overfitting
    XGB_SUBSAMPLE: float = 0.8  # Added 20% row subsampling for randomness
    XGB_COLSAMPLE_BYTREE: float = 0.8  # Added 20% feature subsampling for randomness
    XGB_REG_ALPHA: float = 0.5  # L1 regularization to reduce overfitting
    XGB_REG_LAMBDA: float = 2.0  # L2 regularization to reduce overfitting
    XGB_MIN_CHILD_WEIGHT: int = 3  # Minimum samples per leaf to reduce overfitting
    XGB_GAMMA: float = 0.1  # Minimum loss reduction for split (pruning)
    XGB_OBJECTIVE: str = 'reg:squarederror'
    XGB_RANDOM_STATE: int = 42
    XGB_PARAMS: dict = None
    
    # Evaluation metrics
    METRICS: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for dictionaries and lists."""
        if self.LINEAR_REGRESSION_PARAMS is None:
            object.__setattr__(self, 'LINEAR_REGRESSION_PARAMS', {})
        
        if self.RF_PARAMS is None:
            object.__setattr__(self, 'RF_PARAMS', {
                'n_estimators': self.RF_N_ESTIMATORS,
                'random_state': self.RF_RANDOM_STATE
            })
        
        if self.XGB_PARAMS is None:
            object.__setattr__(self, 'XGB_PARAMS', {
                'n_estimators': self.XGB_N_ESTIMATORS,
                'learning_rate': self.XGB_LEARNING_RATE,
                'max_depth': self.XGB_MAX_DEPTH,
                'subsample': self.XGB_SUBSAMPLE,
                'colsample_bytree': self.XGB_COLSAMPLE_BYTREE,
                'reg_alpha': self.XGB_REG_ALPHA,  # L1 regularization
                'reg_lambda': self.XGB_REG_LAMBDA,  # L2 regularization
                'min_child_weight': self.XGB_MIN_CHILD_WEIGHT,  # Minimum samples per leaf
                'gamma': self.XGB_GAMMA,  # Minimum loss reduction for split
                'random_state': self.XGB_RANDOM_STATE,
                'objective': self.XGB_OBJECTIVE
            })
        
        if self.METRICS is None:
            object.__setattr__(self, 'METRICS', ['rmse', 'mae', 'r2'])


# Create singleton instances
DATA_CONSTANTS = DataConstants()
PREPROCESSING_CONSTANTS = PreprocessingConstants()
FEATURE_CONSTANTS = FeatureConstants()
MODEL_CONSTANTS = ModelConstants()
