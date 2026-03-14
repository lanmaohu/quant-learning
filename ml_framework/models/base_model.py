"""
模型基类 - 定义所有模型的统一接口
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseModel(ABC):
    """
    机器学习模型基类
    
    所有模型需要实现以下方法：
    - fit: 训练模型
    - predict: 预测
    - evaluate: 评估模型性能
    """
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        训练模型
        
        Parameters:
        -----------
        X_train : np.ndarray
            训练特征
        y_train : np.ndarray
            训练标签
        X_val : np.ndarray, optional
            验证特征
        y_val : np.ndarray, optional
            验证标签
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Parameters:
        -----------
        X : np.ndarray
            输入特征
        
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        评估模型性能
        
        Parameters:
        -----------
        X : np.ndarray
            特征
        y : np.ndarray
            真实标签
        
        Returns:
        --------
        metrics : dict
            包含RMSE, MAE, R²等指标
        """
        y_pred = self.predict(X)
        
        return {
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred),
            'direction_accuracy': np.mean((y_pred > 0) == (y > 0))
        }
