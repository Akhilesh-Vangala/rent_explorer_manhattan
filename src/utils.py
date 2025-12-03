"""
Utility functions for SmartRent Manhattan application.

This module provides reusable utility functions for data loading, model loading,
and path management used across the Streamlit application.

All Streamlit pages should import functions from this module directly.
"""

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
    """
    Get the base directory of the project.
    
    This function calculates the base directory relative to the current file,
    handling both direct execution and module import scenarios.
    
    Returns:
        Path to the project base directory
    """
    # Get the directory of this file (src/utils.py)
    current_file = Path(__file__).resolve()
    # Go up two levels: src/utils.py -> src/ -> project_root/
    base_dir = current_file.parent.parent
    return base_dir


def ensure_base_dir_in_path(base_dir: Optional[Path] = None):
    """
    Ensure the base directory is in sys.path for imports.
    
    Args:
        base_dir: Base directory path. If None, calculates automatically.
    """
    if base_dir is None:
        base_dir = get_base_dir()
    
    base_dir_str = str(base_dir)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)


@st.cache_data
def load_data(
    data_path: Optional[Path] = None,
    preprocess_data: bool = True,
    train: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and optionally preprocess data with Streamlit caching.
    
    This function loads raw data from CSV and optionally preprocesses it.
    Results are cached by Streamlit to avoid reloading on every rerun.
    
    Args:
        data_path: Path to data file. If None, uses default from config.
        preprocess_data: If True, preprocesses the data. Default True.
    
    Returns:
        Tuple of (raw_dataframe, preprocessed_dataframe)
        If preprocess_data=False, returns (raw_dataframe, raw_dataframe)
    """
    if data_path is None:
        data_path = Config.RAW_DATA_FILE
    
    # Ensure path is a Path object
    if isinstance(data_path, str):
        data_path = Path(data_path)
    
    # Load raw data
    df_raw = pd.read_csv(data_path)
    
    if preprocess_data:
        df_preprocessed = preprocess(df_raw, train=train)
        return df_raw, df_preprocessed
    else:
        return df_raw, df_raw


@st.cache_resource
def load_model(model_path: Optional[Path] = None):
    """
    Load trained model with Streamlit caching.
    
    This function loads a saved model from disk. Results are cached by Streamlit
    to avoid reloading on every rerun.
    
    Args:
        model_path: Path to model file. If None, uses default from config.
    
    Returns:
        Loaded model object
    """
    if model_path is None:
        model_path = Config.BEST_MODEL_FILE
    
    # Ensure path is a Path object
    if isinstance(model_path, str):
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure the model has been trained and saved."
        )
    
    return joblib.load(model_path)


def get_neighborhoods(df_preprocessed: pd.DataFrame) -> list:
    """
    Extract neighborhood names from preprocessed dataframe.
    
    Args:
        df_preprocessed: Preprocessed dataframe with one-hot encoded neighborhoods
    
    Returns:
        List of neighborhood names
    """
    neighborhood_cols = [
        col for col in df_preprocessed.columns 
        if col.startswith('neighborhood_')
    ]
    neighborhoods = [
        col.replace('neighborhood_', '') 
        for col in neighborhood_cols
    ]
    return neighborhoods


def get_neighborhood_data(
    df_preprocessed: pd.DataFrame,
    neighborhood: str
) -> pd.DataFrame:
    """
    Get data filtered for a specific neighborhood.
    
    Args:
        df_preprocessed: Preprocessed dataframe
        neighborhood: Name of the neighborhood
    
    Returns:
        Filtered dataframe for the specified neighborhood
    """
    neighborhood_col = f'neighborhood_{neighborhood}'
    if neighborhood_col not in df_preprocessed.columns:
        raise ValueError(f"Neighborhood '{neighborhood}' not found in data")
    
    return df_preprocessed[df_preprocessed[neighborhood_col] == 1]
