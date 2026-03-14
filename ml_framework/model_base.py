"""
模型基类
定义可插拔模型接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    """
    模型基类
    所有模型必须实现这些接口
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.config = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        训练模型
        
        Returns:
        --------
        history: 训练历史信息
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        y_pred = self.predict(X)
        
        return {
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred),
        }
    
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'name': self.name,
            'config': self.config,
            'is_fitted': self.is_fitted
        }


class SklearnModel(BaseModel):
    """
    sklearn模型基类
    """
    
    def __init__(self, name: str, sklearn_model, **kwargs):
        super().__init__(name, **kwargs)
        self.model = sklearn_model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return {'message': 'Training completed'}
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True
