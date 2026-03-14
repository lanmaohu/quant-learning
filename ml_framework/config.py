"""
机器学习量化框架配置
"""

import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = os.getenv('STOCK_DATA_PATH', str(PROJECT_ROOT / 'data' / 'a_stock_history_price.csv'))

# 数据参数
HISTORY_YEARS = 5
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
MODEL_SAVE_PATH = PROJECT_ROOT / 'data' / 'models'
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
