# DEPRECATED: 请使用 qlib_features.py 中的 QlibFeatureEngineer / QlibFeatureEngineerV2
# 本模块保留仅供参考，三个 notebook 已全面迁移到 Qlib Alpha 表达式体系

"""
特征工程模块
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    特征工程器
    构建技术指标特征，准备训练数据
    """
    
    def __init__(self):
        self.feature_cols = None
        self.target_col = None
        
    def create_features(self, df: pd.DataFrame, pred_horizon: int = 5) -> pd.DataFrame:
        """
        构建特征数据集
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始价格数据
        pred_horizon : int
            预测未来N天收益率
        
        Returns:
        --------
        df_features : pd.DataFrame
            包含特征和目标的数据
        """
        print("\n🔧 构建特征...")
        
        # 导入技术指标计算器
        from features.factor_calculator import TechnicalFactorCalculator
        
        calc = TechnicalFactorCalculator()
        all_features = []
        
        for code in df['code'].unique():
            stock_df = df[df['code'] == code].copy()
            stock_df = stock_df.sort_values('date')
            
            if len(stock_df) < 30:
                continue
            
            # 计算技术指标
            stock_df = calc.moving_average(stock_df, windows=[5, 10, 20, 60])
            stock_df = calc.rsi(stock_df, windows=[6, 14])
            stock_df = calc.macd(stock_df)
            stock_df = calc.bollinger_bands(stock_df)
            stock_df = calc.volatility_factors(stock_df)
            stock_df = calc.volume_factors(stock_df)
            stock_df = calc.momentum_factors(stock_df)
            
            # 目标变量
            stock_df[f'target_return_{pred_horizon}d'] = (
                stock_df['close'].shift(-pred_horizon) / stock_df['close'] - 1
            )
            
            # 日期特征
            stock_df['dayofweek'] = stock_df['date'].dt.dayofweek
            stock_df['month'] = stock_df['date'].dt.month
            
            all_features.append(stock_df)
        
        df_features = pd.concat(all_features, ignore_index=True)
        
        # 确定特征列
        self.feature_cols = [c for c in df_features.columns 
                            if any(x in c for x in ['ma_', 'rsi_', 'macd_', 'bb_', 
                                                    'volatility_', 'momentum_', 'volume_', 
                                                    'dayofweek', 'month'])]
        self.target_col = f'target_return_{pred_horizon}d'
        
        # 删除NaN
        df_features = df_features.dropna(subset=self.feature_cols + [self.target_col])
        
        print(f"   特征数: {len(self.feature_cols)}")
        print(f"   有效样本: {len(df_features):,}")
        
        return df_features
    
    def prepare_xy(self, df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        准备X, y数据，可选标准化
        
        Returns:
        --------
        X, y, scaler
        """
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        
        if scaler is not None:
            if fit_scaler:
                X = scaler.fit_transform(X)
            else:
                X = scaler.transform(X)
        
        return X, y, scaler
        

    

class SequenceFeatureEngineer(FeatureEngineer):
    """
    序列特征工程器（用于LSTM等时序模型）
    """
    
    def __init__(self, seq_length: int = 20):
        super().__init__()
        self.seq_length = seq_length
    
    def create_sequences(self, df: pd.DataFrame, pred_horizon: int = 5
                        ) -> Tuple[np.ndarray, np.ndarray, List, List]:
        """
        构建时序序列数据
        
        Returns:
        --------
        X: (samples, seq_length, features)
        y: (samples,)
        dates: 每个样本的日期
        codes: 每个样本的股票代码
        """
        print(f"\n🔧 构建时序序列 (seq_length={self.seq_length})...")
        
        # 先创建基础特征
        df_features = self.create_features(df, pred_horizon)
        
        X_list, y_list, dates_list, codes_list = [], [], [], []
        
        for code in df_features['code'].unique():
            stock = df_features[df_features['code'] == code].sort_values('date')
            
            values = stock[self.feature_cols].values
            targets = stock[self.target_col].values
            dates = stock['date'].values
            
            for i in range(len(stock) - self.seq_length):
                X_list.append(values[i:i+self.seq_length])
                y_list.append(targets[i+self.seq_length-1])
                dates_list.append(dates[i+self.seq_length-1])
                codes_list.append(code)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"   序列数: {len(X)}, 形状: X{X.shape}, y{y.shape}")
        
        return X, y, dates_list, codes_list
