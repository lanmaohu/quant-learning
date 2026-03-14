# ML量化框架

可插拔的机器学习量化交易系统，支持多种模型（随机森林、XGBoost、LightGBM、MLP、LSTM）的统一接口。

## 特点

- **可插拔模型**: 所有模型实现相同的`BaseModel`接口，一键切换
- **时序数据划分**: 避免未来信息泄漏的日期切分方式
- **特征工程**: 自动计算技术指标并标准化
- **回测系统**: Top-K选股策略回测
- **完整流程**: 数据加载 → 特征工程 → 模型训练 → 回测

## 项目结构

```
ml_framework/
├── __init__.py              # 包初始化
├── config.py                # 配置
├── data_loader.py           # 数据加载
├── feature_engineering.py   # 特征工程
├── backtester.py           # 回测系统
├── main.py                 # 主流程
├── example_usage.ipynb     # 使用示例
├── models/
│   ├── __init__.py
│   ├── base_model.py       # 基类
│   ├── sklearn_models.py   # sklearn模型
│   └── pytorch_models.py   # PyTorch模型
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib
```

### 2. 运行示例

```bash
# 运行完整流程
python -m ml_framework.main

# 或者使用Jupyter Notebook打开 example_usage.ipynb
```

### 3. 代码示例

```python
from ml_framework.main import run_ml_pipeline
from ml_framework.models.sklearn_models import XGBoostModel, RandomForestModel
from ml_framework.models.pytorch_models import MLPModel

# 运行XGBoost模型
result = run_ml_pipeline(XGBoostModel, {
    'n_estimators': 100,
    'max_depth': 5
})

# 切换为随机森林 - 代码完全相同！
result = run_ml_pipeline(RandomForestModel, {
    'n_estimators': 100,
    'max_depth': 10
})

# 切换为神经网络 - 代码完全相同！
result = run_ml_pipeline(MLPModel, {
    'hidden_dims': [128, 64, 32],
    'dropout_rate': 0.3
})
```

## 支持的模型

| 模型 | 文件 | 类型 |
|------|------|------|
| Ridge回归 | `sklearn_models.py` | 线性 |
| 随机森林 | `sklearn_models.py` | 树模型 |
| XGBoost | `sklearn_models.py` | 树模型 |
| LightGBM | `sklearn_models.py` | 树模型 |
| MLP | `pytorch_models.py` | 神经网络 |
| LSTM | `pytorch_models.py` | 时序网络 |

## 扩展模型

添加新模型只需实现4个方法：

```python
from ml_framework.models.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, **params):
        self.model = ...  # 初始化模型
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # 训练逻辑
        pass
    
    def predict(self, X):
        # 预测逻辑
        pass
    
    def evaluate(self, X, y):
        # 评估逻辑
        pass
```

## 配置

在 `config.py` 中设置：
- `DATA_PATH`: 股票数据路径
- `TARGET_HORIZON`: 预测目标周期（默认5天）
- `TOP_K_STOCKS`: 每日选股数量（默认10只）

## 数据格式

数据需要包含以下列：
- `code`: 股票代码
- `date`: 日期
- `open`, `high`, `low`, `close`: 价格数据
- `volume`: 成交量

## 许可证

MIT
