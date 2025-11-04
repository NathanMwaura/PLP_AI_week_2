"""
AQI Climate Action Package
Air Quality Index Prediction for SDG 13: Climate Action
"""

__version__ = "1.0.0"
__author__ = "Nathan Mwaura"
__email__ = "nathanwaweru75@gmail.com"

# Import main components for easy access
from .utils import (
    categorize_aqi,
    get_aqi_color,
    get_health_advisory,
    generate_synthetic_aqi_data,
    save_model,
    load_model,
    calculate_metrics,
)

from .data_processing import AQIDataPreprocessor, preprocess_pipeline

__all__ = [
    "categorize_aqi",
    "get_aqi_color",
    "get_health_advisory",
    "generate_synthetic_aqi_data",
    "save_model",
    "load_model",
    "calculate_metrics",
    "AQIDataPreprocessor",
    "preprocess_pipeline",
]
