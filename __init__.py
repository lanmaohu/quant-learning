"""
ML量化框架 - 可插拔的机器学习量化交易系统

主要模块:
- data_loader: 数据加载器
- feature_engineering: 特征工程
- models: 可插拔模型（sklearn + pytorch）
- backtester: 回测系统
- main: 主流程

使用示例:
    from main import run_ml_pipeline
    from models.sklearn_models import XGBoostModel
    
    result = run_ml_pipeline(XGBoostModel)
"""

__version__ = "0.1.0"

# 导出主要类
from data_loader import StockDataLoader, time_series_split
from feature_engineering import FeatureEngineer
from backtester import Backtester

__all__ = [
    'StockDataLoader',
    'time_series_split', 
    'FeatureEngineer',
    'Backtester',
]
