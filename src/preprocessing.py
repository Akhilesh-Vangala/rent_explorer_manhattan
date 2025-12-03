import pandas as pd
import numpy as np
from src.config.constants import DATA_CONSTANTS, PREPROCESSING_CONSTANTS, FEATURE_CONSTANTS

def load_raw_data(path):
    return pd.read_csv(path)

def drop_columns(df):
    return df.drop(columns=DATA_CONSTANTS.COLUMNS_TO_DROP)

def cast_types(df):
    for col in PREPROCESSING_CONSTANTS.INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

def handle_outliers(df):
    df['size_sqft'] = df['size_sqft'].clip(upper=PREPROCESSING_CONSTANTS.MAX_SIZE_SQFT)
    df['rent'] = df['rent'].clip(upper=PREPROCESSING_CONSTANTS.MAX_RENT)
    return df

def encode_categoricals(df):
    df = pd.get_dummies(df, columns=DATA_CONSTANTS.CATEGORICAL_COLUMNS, drop_first=False)
    return df

def feature_engineering(df):
    df[FEATURE_CONSTANTS.RENT_PER_SQFT] = df['rent'] / df['size_sqft']
    df[FEATURE_CONSTANTS.LOG_RENT] = np.log1p(df['rent'])
    df[FEATURE_CONSTANTS.LOG_SIZE_SQFT] = np.log1p(df['size_sqft'])
    return df

def preprocess(df, train: bool=False):
    print('\n[STEP 1] Loading and preprocessing data...')
    df = drop_columns(df)
    df = cast_types(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    df = feature_engineering(df)
    if train:
        df = df.drop(columns=[FEATURE_CONSTANTS.LOG_RENT, FEATURE_CONSTANTS.RENT_PER_SQFT], errors='ignore')
    return df
