from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataConstants:
    TARGET_COLUMN: str = 'rent'
    COLUMNS_TO_DROP: List[str] = None
    CATEGORICAL_COLUMNS: List[str] = None

    def __post_init__(self):
        if self.COLUMNS_TO_DROP is None:
            object.__setattr__(self, 'COLUMNS_TO_DROP', ['rental_id', 'borough'])
        if self.CATEGORICAL_COLUMNS is None:
            object.__setattr__(self, 'CATEGORICAL_COLUMNS', ['neighborhood'])

@dataclass(frozen=True)
class PreprocessingConstants:
    MAX_SIZE_SQFT: float = 3500.0
    MAX_RENT: float = 20000.0
    INTEGER_COLUMNS: List[str] = None

    def __post_init__(self):
        if self.INTEGER_COLUMNS is None:
            object.__setattr__(self, 'INTEGER_COLUMNS', ['bedrooms', 'floor'])

@dataclass(frozen=True)
class FeatureConstants:
    RENT_PER_SQFT: str = 'rent_per_sqft'
    LOG_RENT: str = 'log_rent'
    LOG_SIZE_SQFT: str = 'log_size_sqft'
    FEATURE_NAMES: List[str] = None

    def __post_init__(self):
        if self.FEATURE_NAMES is None:
            object.__setattr__(self, 'FEATURE_NAMES', [self.RENT_PER_SQFT, self.LOG_RENT, self.LOG_SIZE_SQFT])

@dataclass(frozen=True)
class ModelConstants:
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    LINEAR_REGRESSION_PARAMS: dict = None
    RF_N_ESTIMATORS: int = 200
    RF_RANDOM_STATE: int = 42
    RF_PARAMS: dict = None
    XGB_N_ESTIMATORS: int = 200
    XGB_LEARNING_RATE: float = 0.03
    XGB_MAX_DEPTH: int = 5
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_REG_ALPHA: float = 0.5
    XGB_REG_LAMBDA: float = 2.0
    XGB_MIN_CHILD_WEIGHT: int = 3
    XGB_GAMMA: float = 0.1
    XGB_OBJECTIVE: str = 'reg:squarederror'
    XGB_RANDOM_STATE: int = 42
    XGB_PARAMS: dict = None
    METRICS: List[str] = None

    def __post_init__(self):
        if self.LINEAR_REGRESSION_PARAMS is None:
            object.__setattr__(self, 'LINEAR_REGRESSION_PARAMS', {})
        if self.RF_PARAMS is None:
            object.__setattr__(self, 'RF_PARAMS', {'n_estimators': self.RF_N_ESTIMATORS, 'random_state': self.RF_RANDOM_STATE})
        if self.XGB_PARAMS is None:
            object.__setattr__(self, 'XGB_PARAMS', {'n_estimators': self.XGB_N_ESTIMATORS, 'learning_rate': self.XGB_LEARNING_RATE, 'max_depth': self.XGB_MAX_DEPTH, 'subsample': self.XGB_SUBSAMPLE, 'colsample_bytree': self.XGB_COLSAMPLE_BYTREE, 'reg_alpha': self.XGB_REG_ALPHA, 'reg_lambda': self.XGB_REG_LAMBDA, 'min_child_weight': self.XGB_MIN_CHILD_WEIGHT, 'gamma': self.XGB_GAMMA, 'random_state': self.XGB_RANDOM_STATE, 'objective': self.XGB_OBJECTIVE})
        if self.METRICS is None:
            object.__setattr__(self, 'METRICS', ['rmse', 'mae', 'r2'])
DATA_CONSTANTS = DataConstants()
PREPROCESSING_CONSTANTS = PreprocessingConstants()
FEATURE_CONSTANTS = FeatureConstants()
MODEL_CONSTANTS = ModelConstants()
