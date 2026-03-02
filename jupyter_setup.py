#!/usr/bin/env python
"""
Jupyter Notebook 启动配置
在每个 notebook 开头运行: %run jupyter_setup.py
"""

import sys
import os

# 1. 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. 加载环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# 3. 验证配置
print("=" * 60)
print("🚀 Jupyter 环境初始化")
print("=" * 60)

# 检查 Tushare Token
token = os.getenv('TUSHARE_TOKEN', '')
if token and len(token) > 20:
    print(f"✅ Tushare Token: {token[:10]}...")
else:
    print("⚠️ Tushare Token 未配置")

# 4. 导入常用模块
print("\n📦 导入模块:")
import pandas as pd
import numpy as np
print("✅ pandas, numpy")

from utils.data_fetcher import DataFetcher
from utils.data_processor import DataProcessor
from utils.factor_calculator import FactorPipeline
from utils.factor_preprocessor import FactorPreprocessor
from utils.factor_analyzer import FactorAnalyzer
print("✅ data_fetcher, data_processor, factor_calculator")
print("✅ factor_preprocessor, factor_analyzer")

# 5. 创建全局实例
print("\n🔧 创建实例:")
fetcher = DataFetcher(tushare_token=token if len(token) > 20 else None)
print("✅ fetcher = DataFetcher()")

processor = DataProcessor()
print("✅ processor = DataProcessor()")

factor_pipeline = FactorPipeline()
print("✅ factor_pipeline = FactorPipeline()")

print("\n" + "=" * 60)
print("💡 现在可以直接使用:")
print("   df = fetcher.get_daily_data_ak('000001', '20240101', '20241231')")
print("=" * 60)
