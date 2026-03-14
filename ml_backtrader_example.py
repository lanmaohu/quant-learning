#!/usr/bin/env python
"""
ML + Backtrader 完整示例
展示如何使用 ml_framework 训练模型，然后用 Backtrader 回测
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ML Framework
from ml_framework.data_loader import StockDataLoader, time_series_split
from ml_framework.feature_engineering import FeatureEngineer
from ml_framework.config import DATA_PATH
from ml_framework.models import XGBoostModel, LightGBMModel, RandomForestModel

# Backtrader Integration
from ml_framework.backtrader_integration import run_backtrader_with_predictions


def run_ml_backtrader_workflow(ModelClass, model_params=None, top_k=5, sample_n=30):
    """
    完整流程：数据加载 → 特征工程 → ML训练 → Backtrader回测
    
    Parameters:
    -----------
    ModelClass : class
        模型类 (XGBoostModel, LightGBMModel, etc.)
    model_params : dict
        模型参数
    top_k : int
        每日选股数量
    sample_n : int
        样本股票数量
    """
    print("=" * 80)
    print(f"🚀 ML + Backtrader 完整流程 | 模型: {ModelClass.__name__}")
    print("=" * 80)
    
    # ========== 1. 数据加载 ==========
    print("\n📂 Step 1: 数据加载")
    loader = StockDataLoader(DATA_PATH)
    df = loader.load(years_back=3)  # 加载最近3年数据
    
    # 选择样本股票
    selected_codes = loader.select_sample_codes(n=sample_n)
    df = df[df['code'].isin(selected_codes)]
    print(f"   选中 {len(selected_codes)} 只股票")
    
    # ========== 2. 数据集划分 ==========
    print("\n📊 Step 2: 时序数据集划分")
    train_df, val_df, test_df = time_series_split(df, train_ratio=0.6, val_ratio=0.2)
    
    # ========== 3. 特征工程 ==========
    print("\n🔧 Step 3: 特征工程")
    fe = FeatureEngineer()
    
    train_features = fe.create_features(train_df, pred_horizon=5)
    val_features = fe.create_features(val_df, pred_horizon=5)
    test_features = fe.create_features(test_df, pred_horizon=5)
    
    # ========== 4. 准备训练数据 ==========
    print("\n📋 Step 4: 准备训练数据")
    scaler = StandardScaler()
    
    X_train, y_train, _ = fe.prepare_xy(train_features, scaler, fit_scaler=True)
    X_val, y_val, _ = fe.prepare_xy(val_features, scaler, fit_scaler=False)
    X_test, y_test, _ = fe.prepare_xy(test_features, scaler, fit_scaler=False)
    
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # ========== 5. 模型训练 ==========
    print(f"\n🧠 Step 5: 训练 {ModelClass.__name__}")
    if model_params is None:
        model_params = {}
    
    model = ModelClass(**model_params)
    history = model.fit(X_train, y_train, X_val, y_val)
    
    # 评估
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print(f"   训练集 R²: {train_metrics['R2']:.4f}")
    print(f"   测试集 R²: {test_metrics['R2']:.4f}")
    
    # ========== 6. 生成预测 ==========
    print("\n🔮 Step 6: 生成预测")
    y_pred = model.predict(X_test)
    
    # 将预测结果添加到 test_features
    test_features['pred_return'] = y_pred
    
    # ========== 7. Backtrader 回测 ==========
    print("\n📈 Step 7: Backtrader 回测")
    backtest_result = run_backtrader_with_predictions(
        test_df=test_features,
        pred_col='pred_return',
        price_col='close',
        code_col='code',
        date_col='date',
        top_k=top_k,
        initial_cash=100000.0,
        commission=0.001,
        print_log=False  # 关闭详细日志，只看结果
    )
    
    # ========== 8. 汇总结果 ==========
    print("\n" + "=" * 80)
    print("📊 完整流程结果汇总")
    print("=" * 80)
    print(f"模型: {ModelClass.__name__}")
    print(f"测试集 R²: {test_metrics['R2']:.4f}")
    print(f"Backtrader 回测收益: {backtest_result['total_return']*100:.2f}%")
    print(f"夏普比率: {backtest_result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {backtest_result['max_drawdown']*100:.2f}%")
    
    return {
        'model': model,
        'metrics': test_metrics,
        'backtest': backtest_result
    }


def compare_models_with_backtrader():
    """
    对比多个模型在 Backtrader 上的表现
    """
    print("=" * 80)
    print("🔍 多模型对比 (ML + Backtrader)")
    print("=" * 80)
    
    models_to_test = [
        (RandomForestModel, {'n_estimators': 100, 'max_depth': 10}),
        (XGBoostModel, {'n_estimators': 100, 'max_depth': 5}),
        (LightGBMModel, {'n_estimators': 100, 'num_leaves': 31}),
    ]
    
    results = []
    
    for ModelClass, params in models_to_test:
        try:
            result = run_ml_backtrader_workflow(
                ModelClass=ModelClass,
                model_params=params,
                top_k=5,
                sample_n=20  # 减少股票数量加快测试
            )
            
            results.append({
                'model': ModelClass.__name__,
                'r2': result['metrics']['R2'],
                'total_return': result['backtest']['total_return'],
                'sharpe': result['backtest']['sharpe_ratio'],
                'max_dd': result['backtest']['max_drawdown']
            })
            
        except Exception as e:
            print(f"❌ {ModelClass.__name__} 失败: {e}")
    
    # 对比表格
    if results:
        print("\n" + "=" * 80)
        print("📊 模型对比结果")
        print("=" * 80)
        print(f"{'模型':<20} {'R²':>8} {'总收益':>10} {'夏普':>8} {'最大回撤':>10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['model']:<20} {r['r2']:>8.4f} {r['total_return']*100:>9.2f}% {r['sharpe']:>8.3f} {r['max_dd']*100:>9.2f}%")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML + Backtrader 回测')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'lightgbm', 'random_forest'],
                       help='选择模型')
    parser.add_argument('--top-k', type=int, default=5, help='每日选股数量')
    parser.add_argument('--compare', action='store_true', help='对比多个模型')
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        compare_models_with_backtrader()
    else:
        # 单模型模式
        model_map = {
            'xgboost': (XGBoostModel, {'n_estimators': 100, 'max_depth': 5}),
            'lightgbm': (LightGBMModel, {'n_estimators': 100, 'num_leaves': 31}),
            'random_forest': (RandomForestModel, {'n_estimators': 100, 'max_depth': 10}),
        }
        
        ModelClass, params = model_map[args.model]
        run_ml_backtrader_workflow(ModelClass, params, top_k=args.top_k)
