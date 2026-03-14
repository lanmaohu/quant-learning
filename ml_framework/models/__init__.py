"""
可插拔模型模块

支持的模型:
- sklearn_models: RidgeRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel
- pytorch_models: MLPModel, LSTMModel
"""

from .base_model import BaseModel
from .sklearn_models import RidgeRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel

try:
    from .pytorch_models import MLPModel, LSTMModel
    __all__ = [
        'BaseModel',
        'RidgeRegressionModel',
        'RandomForestModel',
        'XGBoostModel',
        'LightGBMModel',
        'MLPModel',
        'LSTMModel',
    ]
except ImportError:
    __all__ = [
        'BaseModel',
        'RidgeRegressionModel',
        'RandomForestModel',
        'XGBoostModel',
        'LightGBMModel',
    ]
