import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import joblib
import streamlit as st
from src.config.config import Config
from src.preprocessing import preprocess

def get_base_dir() -> Path:
    current_file = Path(__file__).resolve()
    base_dir = current_file.parent.parent
    return base_dir

def ensure_base_dir_in_path(base_dir: Optional[Path]=None):
    if base_dir is None:
        base_dir = get_base_dir()
    base_dir_str = str(base_dir)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)

@st.cache_data
def load_data(data_path: Optional[Path]=None, preprocess_data: bool=True, train: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data_path is None:
        data_path = Config.RAW_DATA_FILE
    if isinstance(data_path, str):
        data_path = Path(data_path)
    df_raw = pd.read_csv(data_path)
    if preprocess_data:
        df_preprocessed = preprocess(df_raw, train=train)
        return (df_raw, df_preprocessed)
    else:
        return (df_raw, df_raw)

@st.cache_resource
def load_model(model_path: Optional[Path]=None):
    if model_path is None:
        model_path = Config.BEST_MODEL_FILE
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found at {model_path}. Please ensure the model has been trained and saved.')
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f'Failed to load model from {model_path}: {e}')

def get_neighborhoods(df_preprocessed: pd.DataFrame) -> list:
    neighborhood_cols = [col for col in df_preprocessed.columns if col.startswith('neighborhood_')]
    neighborhoods = [col.replace('neighborhood_', '') for col in neighborhood_cols]
    return neighborhoods

def get_neighborhood_data(df_preprocessed: pd.DataFrame, neighborhood: str) -> pd.DataFrame:
    neighborhood_col = f'neighborhood_{neighborhood}'
    if neighborhood_col not in df_preprocessed.columns:
        raise ValueError(f"Neighborhood '{neighborhood}' not found in data")
    return df_preprocessed[df_preprocessed[neighborhood_col] == 1]
