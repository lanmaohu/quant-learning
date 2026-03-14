"""
配置文件
用于存储 API Keys 等配置
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# Tushare Pro Token
# 优先级: 1. 环境变量 2. .env 文件 3. 手动设置
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# GLM 模型配置
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4")

# 其他配置
DATA_DIR = "./data"
CACHE_DIR = "./data/cache"


def check_tushare_token():
    """检查 Tushare Token 是否配置"""
    if not TUSHARE_TOKEN or TUSHARE_TOKEN == "你的token请替换这里":
        print("""
        ⚠️ Tushare Token 未配置！
        
        请按以下步骤配置:
        
        方式1 - 修改 .env 文件（推荐）:
            1. 打开 .env 文件
            2. 将 TUSHARE_TOKEN=你的token请替换这里
               改为 TUSHARE_TOKEN=你的真实token
        
        方式2 - 设置环境变量:
            export TUSHARE_TOKEN='你的token'
        
        方式3 - 在代码中传入:
            from utils.data_fetcher import DataFetcher
            fetcher = DataFetcher(tushare_token='你的token')
        """)
        return False

    # 验证 token 前几位格式
    if len(TUSHARE_TOKEN) < 20:
        print("⚠️ Token 格式似乎不正确，请检查")
        return False

    print(f"✅ Tushare Token 已配置 (前10位: {TUSHARE_TOKEN[:10]}...)")
    return True


if __name__ == "__main__":
    check_tushare_token()
