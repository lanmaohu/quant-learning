"""
特征工程模块包

提供 Qlib Alpha 表达式体系和传统技术因子的特征生成工具。

主要类：
    QlibFeatureEngineer    — 30 个基础 Alpha 特征（推荐）
    QlibFeatureEngineerV2  — 基础 + 高级 Alpha 特征
    FeatureEngineer        — 传统技术指标（DEPRECATED）
    TechnicalFactorCalculator — 底层技术因子计算器
    FactorAnalyzer         — 因子分析（IC/IR/换手率）
    FactorPreprocessor     — 因子预处理（标准化/去极值）
    MultiFactorModel       — 多因子合成模型
"""

from features.qlib_features import (
    QlibFeatureEngineer,
    QlibFeatureEngineerV2,
    create_alpha_features,
)
from features.feature_engineering import FeatureEngineer
from features.factor_calculator import TechnicalFactorCalculator, FactorPipeline
from features.factor_analyzer import FactorAnalyzer
from features.factor_preprocessor import FactorPreprocessor
from features.multi_factor_model import FactorSynthesizer, MultiFactorStrategy, FactorResearchPipeline

__all__ = [
    "QlibFeatureEngineer",
    "QlibFeatureEngineerV2",
    "create_alpha_features",
    "FeatureEngineer",
    "TechnicalFactorCalculator",
    "FactorPipeline",
    "FactorAnalyzer",
    "FactorPreprocessor",
    "FactorSynthesizer",
    "MultiFactorStrategy",
    "FactorResearchPipeline",
]
