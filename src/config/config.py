import os
from pathlib import Path
from typing import Optional, Dict, Any
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def load_yaml_config(config_path: Optional[Path]=None) -> Dict[str, Any]:
    if not YAML_AVAILABLE:
        raise ImportError('PyYAML is required to load YAML config files. Install it with: pip install pyyaml')
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}

class Config:
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / 'data'
    RAW_DATA_DIR: Path = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR: Path = DATA_DIR / 'processed'
    RAW_DATA_FILE: Path = RAW_DATA_DIR / 'manhattan.csv'
    OUTPUTS_DIR: Path = BASE_DIR / 'outputs'
    MODELS_DIR: Path = OUTPUTS_DIR / 'models'
    SHAP_DIR: Path = OUTPUTS_DIR / 'shap'
    METRICS_DIR: Path = OUTPUTS_DIR / 'metrics'
    BEST_MODEL_FILE: Path = MODELS_DIR / 'best_model.pkl'
    FEATURE_NAMES_FILE: Path = MODELS_DIR / 'feature_names.txt'
    MODEL_METRICS_FILE: Path = MODELS_DIR / 'model_metrics.txt'
    CV_METRICS_FILE: Path = MODELS_DIR / 'cv_metrics.txt'

    @classmethod
    def get_data_path(cls, filename: Optional[str]=None) -> Path:
        if filename:
            return cls.RAW_DATA_DIR / filename
        return cls.RAW_DATA_DIR / 'manhattan.csv'

    @classmethod
    def get_model_path(cls, model_name: Optional[str]=None) -> Path:
        if model_name:
            return cls.MODELS_DIR / model_name
        return cls.BEST_MODEL_FILE

    @classmethod
    def ensure_directories(cls):
        directories = [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.OUTPUTS_DIR, cls.MODELS_DIR, cls.SHAP_DIR, cls.METRICS_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_base_dir(cls) -> Path:
        return cls.BASE_DIR
