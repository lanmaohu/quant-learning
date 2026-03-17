"""
机器学习量化框架配置
"""

import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent
# 支持环境变量覆盖，默认使用项目目录下的数据文件
DATA_PATH = Path(os.getenv('STOCK_DATA_PATH', PROJECT_ROOT / 'data' ))

# 数据参数
HISTORY_YEARS = 10
PRED_HORIZON = 5  # 预测未来5天收益率
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
TOP_K_STOCKS = 10

# 特征参数
SEQ_LENGTH = 20  # 时序长度（用于LSTM等模型）

# 训练参数
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
RANDOM_SEED = 42

# 模型保存路径
MODEL_SAVE_PATH = PROJECT_ROOT / 'models'
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 回测公共参数
# ──────────────────────────────────────────────────────────────
BACKTEST_INITIAL_CASH: float = 1_000_000.0   # 初始资金（元）
BACKTEST_COMMISSION: float = 0.00025          # 佣金率（万分之2.5，单边）
BACKTEST_REBALANCE_FREQ: int = 5              # 调仓频率（交易日数）
BACKTEST_MIN_DATA_DAYS: int = 5              # 单只股票最少数据天数

# ──────────────────────────────────────────────────────────────
# Threshold 策略参数
# ──────────────────────────────────────────────────────────────
THRESHOLD_BUY: float = 0.02       # 买入阈值（预测收益率 > 2% 则买入）
THRESHOLD_SELL: float = -0.01     # 卖出阈值（预测收益率 < -1% 则卖出）
THRESHOLD_MAX_POSITIONS: int = 10  # 最大同时持仓数

# ──────────────────────────────────────────────────────────────
# TopK 策略参数
# ──────────────────────────────────────────────────────────────
TOPK_EQUAL_WEIGHT: bool = True    # True=等权重, False=按预测分数加权
TOPK_POSITION_PCT: float = 0.95   # 资金使用比例（保留 5% 现金）
TOPK_MIN_SCORE: float = 0.0       # 最低预测分数阈值
TOPK_SELL_AT_EXIT: bool = True    # 掉出 TopK 时立即卖出

# ──────────────────────────────────────────────────────────────
# 分类策略（logistic regression notebook）参数
# ──────────────────────────────────────────────────────────────
RETURN_THRESHOLD: float = 0.02   # 正样本标签阈值（5日收益 > 2% → 正样本）
OOS_HOLDOUT: float = 0.20        # 股票级别 OOS holdout 比例
