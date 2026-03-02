#!/usr/bin/env python
"""
修复 Jupyter 的 Python 路径
确保 Jupyter 能正确找到项目模块
"""

import sys
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 添加项目路径到 sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"✅ 已添加项目路径: {PROJECT_ROOT}")
else:
    print(f"✅ 项目路径已在 sys.path 中")

# 验证导入
print("\n验证导入:")
try:
    from utils.data_fetcher import DataFetcher
    print(f"✅ DataFetcher 导入成功")
    print(f"   位置: {DataFetcher.__module__}")
    
    # 检查数据源
    fetcher = DataFetcher()
    print(f"✅ DataFetcher 初始化成功")
    print(f"   数据源: Tushare={'✓' if fetcher._tushare else '✗'}, Baostock={'✓' if hasattr(fetcher, '_smart_fetcher') else '✗'}")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")

print("\n" + "="*60)
print("💡 在 Jupyter 中使用:")
print("   在每个 notebook 开头运行:")
print("   %run fix_jupyter_path.py")
print("="*60)
