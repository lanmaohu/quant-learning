"""
ML + Qlib 风格特征工程 + 回测 完整流程
独立 Python 脚本版本
"""

import sys
import os
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 核心模块
from data_loader import StockDataLoader, time_series_split
from qlib_features import QlibFeatureEngineer
from config import DATA_PATH
from models.sklearn_models import (
    RidgeRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print('✅ 所有模块导入成功！')
print(f'数据路径: {DATA_PATH}')

# ========== 2. 加载数据 ==========
print('\n' + '='*60)
print('📊 2. 加载数据')
print('='*60)

loader = StockDataLoader(DATA_PATH)
df = loader.load(years_back=None)  # 加载全部数据

print(f'\n📊 数据集信息:')
print(f'   总记录数: {len(df):,}')
print(f'   股票数量: {df["code"].nunique()}')
print(f'   日期范围: {df["date"].min().date()} ~ {df["date"].max().date()}')

# ========== 3. 时序数据集划分 ==========
print('\n' + '='*60)
print('⏰ 3. 时序数据集划分')
print('='*60)

train_df, val_df, test_df = time_series_split(
    df, 
    train_ratio=0.6,
    val_ratio=0.2
)

print('\n🔒 时序划分验证:')
print(f'   训练集: {train_df["date"].min().date()} ~ {train_df["date"].max().date()}')
print(f'   验证集: {val_df["date"].min().date()} ~ {val_df["date"].max().date()}')
print(f'   测试集: {test_df["date"].min().date()} ~ {test_df["date"].max().date()}')

# 确认无重叠
assert train_df['date'].max() < val_df['date'].min(), '训练集和验证集重叠'
assert val_df['date'].max() < test_df['date'].min(), '验证集和测试集重叠'
print('   ✅ 无时间穿越风险')

# ========== 4. Qlib 风格特征工程 ==========
print('\n' + '='*60)
print('🔧 4. Qlib 风格特征工程')
print('='*60)

engineer = QlibFeatureEngineer()
engineer.build_alpha_features()

print('\n步骤1: 训练集特征工程')
train_features = engineer.transform(train_df)

print('\n步骤2: 验证集特征工程')
val_features = engineer.transform(val_df)

print('\n步骤3: 测试集特征工程')
test_features = engineer.transform(test_df)

alpha_cols = [c for c in train_features.columns if c.startswith('alpha_')]
print(f'\n📋 特征列表 ({len(alpha_cols)} 个):')
for i, col in enumerate(alpha_cols[:10], 1):
    print(f'   {i:2d}. {col}')
print(f'   ... 共 {len(alpha_cols)} 个')

# ========== 5. 添加目标变量 ==========
print('\n' + '='*60)
print('📊 5. 添加目标变量')
print('='*60)

def add_target_variable(df: pd.DataFrame, pred_horizon: int = 5) -> pd.DataFrame:
    df = df.copy()
    df[f'target_return_{pred_horizon}d'] = df.groupby('code')['close'].shift(-pred_horizon) / df['close'] - 1
    df = df.dropna(subset=[f'target_return_{pred_horizon}d'])
    return df

PRED_HORIZON = 5
print(f'预测周期: {PRED_HORIZON} 日')

train_features = add_target_variable(train_features, PRED_HORIZON)
val_features = add_target_variable(val_features, PRED_HORIZON)
test_features = add_target_variable(test_features, PRED_HORIZON)

target_col = f'target_return_{PRED_HORIZON}d'
print(f'\n目标变量统计:')
print(f'   训练集 mean: {train_features[target_col].mean():.6f}')
print(f'   验证集 mean: {val_features[target_col].mean():.6f}')
print(f'   测试集 mean: {test_features[target_col].mean():.6f}')

# ========== 6. 数据清洗和标准化 ==========
print('\n' + '='*60)
print('🔧 6. 数据清洗和标准化')
print('='*60)

print('数据清洗...')
train_nan = train_features[alpha_cols + [target_col]].isna().sum().sum()
val_nan = val_features[alpha_cols + [target_col]].isna().sum().sum()
test_nan = test_features[alpha_cols + [target_col]].isna().sum().sum()

print(f'   训练集 NaN: {train_nan}')
print(f'   验证集 NaN: {val_nan}')
print(f'   测试集 NaN: {test_nan}')

train_clean = train_features.dropna(subset=alpha_cols + [target_col])
val_clean = val_features.dropna(subset=alpha_cols + [target_col])
test_clean = test_features.dropna(subset=alpha_cols + [target_col])

print(f'\n清洗后:')
print(f'   训练集: {len(train_clean):,}')
print(f'   验证集: {len(val_clean):,}')
print(f'   测试集: {len(test_clean):,}')

scaler = StandardScaler()

X_train = train_clean[alpha_cols].values
y_train = train_clean[target_col].values
X_val = val_clean[alpha_cols].values
y_val = val_clean[target_col].values
X_test = test_clean[alpha_cols].values
y_test = test_clean[target_col].values

print('\n标准化...')
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f'\n✅ 数据准备完成:')
print(f'   训练集: X{X_train_scaled.shape}')
print(f'   验证集: X{X_val_scaled.shape}')
print(f'   测试集: X{X_test_scaled.shape}')

# ========== 7. 训练 ML 模型 ==========
print('\n' + '='*60)
print('🤖 7. 训练 ML 模型')
print('='*60)

models_results = {}

# Ridge
print('\n模型 1: Ridge 回归')
model_ridge = RidgeRegressionModel(alpha=1.0)
history_ridge = model_ridge.fit(X_train_scaled, y_train, X_val_scaled, y_val)
metrics_ridge = model_ridge.evaluate(X_test_scaled, y_test)
models_results['Ridge'] = {'model': model_ridge, 'metrics': metrics_ridge}
print(f'   R2: {metrics_ridge["R2"]:.4f}')

# LightGBM
print('\n模型 2: LightGBM')
model_lgb = LightGBMModel(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    random_state=42
)
history_lgb = model_lgb.fit(X_train_scaled, y_train, X_val_scaled, y_val)
metrics_lgb = model_lgb.evaluate(X_test_scaled, y_test)
models_results['LightGBM'] = {'model': model_lgb, 'metrics': metrics_lgb}
print(f'   R2: {metrics_lgb["R2"]:.4f}')

# XGBoost
print('\n模型 3: XGBoost')
model_xgb = XGBoostModel(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
history_xgb = model_xgb.fit(X_train_scaled, y_train, X_val_scaled, y_val)
metrics_xgb = model_xgb.evaluate(X_test_scaled, y_test)
models_results['XGBoost'] = {'model': model_xgb, 'metrics': metrics_xgb}
print(f'   R2: {metrics_xgb["R2"]:.4f}')

# ========== 8. 模型对比 ==========
print('\n' + '='*60)
print('📊 8. 模型对比')
print('='*60)

comparison_df = pd.DataFrame(
    {name: result['metrics'] for name, result in models_results.items()}
).T
print(comparison_df.round(4))

best_model_name = comparison_df['R2'].idxmax()
print(f'\n🏆 最佳模型: {best_model_name}')
best_model = models_results[best_model_name]['model']

# ========== 9. 生成预测结果 ==========
print('\n' + '='*60)
print('🔮 9. 生成预测结果')
print('='*60)

y_pred_test = best_model.predict(X_test_scaled)
test_clean['pred_return'] = y_pred_test

y_pred_val = best_model.predict(X_val_scaled)
val_clean['pred_return'] = y_pred_val

print(f'预测分布:')
print(f'   验证集: mean={y_pred_val.mean():.4f}, std={y_pred_val.std():.4f}')
print(f'   测试集: mean={y_pred_test.mean():.4f}, std={y_pred_test.std():.4f}')

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_val, y_pred_val, alpha=0.3, s=1)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Return')
axes[0].set_ylabel('Predicted Return')
axes[0].set_title(f'{best_model_name} - Validation Set')

axes[1].scatter(y_test, y_pred_test, alpha=0.3, s=1)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Return')
axes[1].set_ylabel('Predicted Return')
axes[1].set_title(f'{best_model_name} - Test Set')

plt.tight_layout()
plt.savefig('./prediction_scatter.png', dpi=150)
plt.show()

print('\n✅ 预测散点图已保存: ./prediction_scatter.png')

# ========== 10. 特征重要性 ==========
print('\n' + '='*60)
print('📊 10. 特征重要性')
print('='*60)

if hasattr(best_model.model, 'feature_importances_'):
    importances = best_model.model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': alpha_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print('Top 10 重要特征:')
    print(importance_df.head(10).to_string(index=False))
else:
    print('当前模型不支持特征重要性分析')

print('\n🎉 全部流程完成！')
