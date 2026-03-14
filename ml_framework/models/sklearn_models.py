"""
sklearn模型实现
"""

from ml_framework.model_base import SklearnModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class LogisticRegressionModel(SklearnModel):
    """逻辑回归（用于分类问题）"""
    
    def __init__(self, **kwargs):
        model = LogisticRegression(
            penalty='l2',
            C=kwargs.get('C', 1.0),
            max_iter=1000,
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        super().__init__('LogisticRegression', model, **kwargs)


class RidgeRegressionModel(SklearnModel):
    """岭回归（L2正则化线性回归）"""
    
    def __init__(self, **kwargs):
        model = Ridge(
            alpha=kwargs.get('alpha', 1.0),
            max_iter=1000,
            random_state=kwargs.get('random_state', 42)
        )
        super().__init__('RidgeRegression', model, **kwargs)


class RandomForestModel(SklearnModel):
    """随机森林"""
    
    def __init__(self, **kwargs):
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 50),
            min_samples_leaf=kwargs.get('min_samples_leaf', 20),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1,
            verbose=0
        )
        super().__init__('RandomForest', model, **kwargs)
    
    def feature_importances(self):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class XGBoostModel(SklearnModel):
    """XGBoost"""
    
    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
                verbosity=0
            )
            super().__init__('XGBoost', model, **kwargs)
        except ImportError:
            raise ImportError("请安装 xgboost: pip install xgboost")


class LightGBMModel(SklearnModel):
    """LightGBM"""
    
    def __init__(self, **kwargs):
        try:
            import lightgbm as lgb
            import warnings
            # 抑制特征名警告
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            
            model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
                verbosity=-1
            )
            super().__init__('LightGBM', model, **kwargs)
        except ImportError:
            raise ImportError("请安装 lightgbm: pip install lightgbm")
