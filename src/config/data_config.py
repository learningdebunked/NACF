"""Data paths and preprocessing configuration."""

from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path


@dataclass
class DataPaths:
    """Data directory paths."""
    RAW_DATA_DIR: Path = Path("data/raw")
    PROCESSED_DATA_DIR: Path = Path("data/processed")
    SYNTHETIC_DATA_DIR: Path = Path("data/synthetic")
    
    DATASET_PATHS: Dict[str, str] = field(default_factory=lambda: {
        "retailrocket": "data/raw/ecommerce/retailrocket/",
        "uci_retail": "data/raw/ecommerce/uci_online_retail/",
        "mit_gaze": "data/raw/cognitive/mit_gazecapture/",
        "deap": "data/raw/cognitive/deap/",
        "ascertain": "data/raw/cognitive/ascertain/",
        "openneuro_adhd": "data/raw/neurodivergent/openneuro_adhd/",
        "kaggle_asd": "data/raw/neurodivergent/kaggle_asd/"
    })


@dataclass
class PreprocessingConfig:
    """Preprocessing parameters."""
    sequence_length: int = 50
    stride: int = 10
    min_session_length: int = 10
    max_session_length: int = 200
    normalize: bool = True
    remove_outliers: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    extract_t_delta: bool = True
    extract_dwell_time: bool = True
    extract_scroll_oscillation: bool = True
    extract_navigation_entropy: bool = True
    extract_hesitation_bursts: bool = True


# Global config instances
DATA_PATHS = DataPaths()
PREPROCESSING_CONFIG = PreprocessingConfig()
FEATURE_CONFIG = FeatureConfig()
