"""
Configuration settings for paths and directories.

This module manages all file paths and directory structures used throughout
the application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default.
    
    Returns:
        Dictionary containing configuration values
    
    Raises:
        ImportError: If PyYAML is not installed
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        )
    
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


class Config:
    """Configuration class for managing paths and directories."""
    
    # Base directory - project root
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    
    # Data paths
    DATA_DIR: Path = BASE_DIR / 'data'
    RAW_DATA_DIR: Path = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR: Path = DATA_DIR / 'processed'
    
    # Data files
    RAW_DATA_FILE: Path = RAW_DATA_DIR / 'manhattan.csv'
    
    # Output paths
    OUTPUTS_DIR: Path = BASE_DIR / 'outputs'
    MODELS_DIR: Path = OUTPUTS_DIR / 'models'
    SHAP_DIR: Path = OUTPUTS_DIR / 'shap'
    METRICS_DIR: Path = OUTPUTS_DIR / 'metrics'
    
    # Model files
    BEST_MODEL_FILE: Path = MODELS_DIR / 'best_model.pkl'
    FEATURE_NAMES_FILE: Path = MODELS_DIR / 'feature_names.txt'
    MODEL_METRICS_FILE: Path = MODELS_DIR / 'model_metrics.txt'
    CV_METRICS_FILE: Path = MODELS_DIR / 'cv_metrics.txt'
    
    @classmethod
    def get_data_path(cls, filename: Optional[str] = None) -> Path:
        """
        Get path to data file.
        
        Args:
            filename: Optional filename. If None, returns raw data file path.
        
        Returns:
            Path to data file
        """
        if filename:
            return cls.RAW_DATA_DIR / filename
        return cls.RAW_DATA_DIR / 'manhattan.csv'
    
    @classmethod
    def get_model_path(cls, model_name: Optional[str] = None) -> Path:
        """
        Get path to model file.
        
        Args:
            model_name: Optional model filename. If None, returns best model path.
        
        Returns:
            Path to model file
        """
        if model_name:
            return cls.MODELS_DIR / model_name
        return cls.BEST_MODEL_FILE
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.MODELS_DIR,
            cls.SHAP_DIR,
            cls.METRICS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_base_dir(cls) -> Path:
        """Get the base directory of the project."""
        return cls.BASE_DIR
