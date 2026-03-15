"""
机器学习量化框架主流程
可插拔模型示例
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, TOP_K_STOCKS
from data_loader import StockDataLoader, time_series_split
from feature_engineering import FeatureEngineer
from backtester import Backtester

# 导入模型（可插拔）
from models.sklearn_models import (
    RidgeRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel
)
from models.pytorch_models import MLPModel, LSTMModel


def run_ml_pipeline(ModelClass, model_params=None, sample_n=50):
    """
    运行完整的机器学习量化流程
    
    Parameters:
    -----------
    ModelClass : class
        模型类（可插拔）
    model_params : dict
        模型参数
    sample_n : int
        样本股票数量
    
    Returns:
    --------
    result : dict
        包含训练好的模型、评估结果、回测结果
    """
    print("=" * 70)
    print(f"🚀 ML量化流程 - {ModelClass.__name__}")
    print("=" * 70)
    
    # 1. 数据加载
    print("\n📂 Step 1: 数据加载")
    loader = StockDataLoader(DATA_PATH)
    df = loader.load(years_back=5)
    
    # 选择样本股票
    selected_codes = loader.select_sample_codes(n=sample_n)
    df = df[df['code'].isin(selected_codes)]
    
    # 2. 数据集划分
    print("\n📊 Step 2: 数据集划分")
    train_df, val_df, test_df = time_series_split(df)
    
    # 3. 特征工程
    print("\n🔧 Step 3: 特征工程")
    fe = FeatureEngineer()
    train_features = fe.create_features(train_df)
    val_features = fe.create_features(val_df)
    test_features = fe.create_features(test_df)
    
    # 4. 准备训练数据
    print("\n📋 Step 4: 准备训练数据")
    scaler = StandardScaler()
    
    X_train, y_train, _ = fe.prepare_xy(train_features, scaler, fit_scaler=True)
    X_val, y_val, _ = fe.prepare_xy(val_features, scaler, fit_scaler=False)
    X_test, y_test, _ = fe.prepare_xy(test_features, scaler, fit_scaler=False)
    
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 5. 模型训练（可插拔部分）
    print(f"\n🧠 Step 5: 模型训练 ({ModelClass.__name__})")
    
    if model_params is None:
        model_params = {}
    
    # 根据模型类型添加必要参数
    if 'MLP' in ModelClass.__name__ or 'LSTM' in ModelClass.__name__:
        model_params['input_dim'] = X_train.shape[1]
    
    model = ModelClass(**model_params)
    history = model.fit(X_train, y_train, X_val, y_val)
    
    # 6. 模型评估
    print("\n📈 Step 6: 模型评估")
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)
    
    print(f"   训练集 - RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   验证集 - RMSE: {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   测试集 - RMSE: {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    
    # 7. 预测并回测
    print("\n💰 Step 7: 策略回测")
    y_pred = model.predict(X_test)
    
    # 构建回测数据
    test_features['pred_return'] = y_pred
    
    backtester = Backtester(top_k=TOP_K_STOCKS)
    backtest_result = backtester.run(test_features)
    
    # 绘制回测结果
    if backtest_result:
        backtester.plot_results(backtest_result)
    
    print("\n" + "=" * 70)
    print("✅ 流程完成！")
    print("=" * 70)
    
    return {
        'model': model,
        'history': history,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'backtest': backtest_result
    }


if __name__ == '__main__':
    # 示例：运行不同模型
    
    # 1. 线性模型
    # result = run_ml_pipeline(RidgeRegressionModel, {'alpha': 1.0})
    
    # 2. 树模型
    result = run_ml_pipeline(RandomForestModel, {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    })
    
    # 3. 深度学习模型（需要PyTorch）
    # result = run_ml_pipeline(MLPModel, {
    #     'hidden_dims': [128, 64, 32],
    #     'dropout_rate': 0.3,
    #     'epochs': 50,
    #     'batch_size': 256
    # })
