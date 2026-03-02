"""量化交易常数配置"""

# 交易日相关
TRADING_DAYS_PER_YEAR = 252  # 年均交易日
TRADING_DAYS_PER_MONTH = 21  # 月均交易日
TRADING_DAYS_PER_WEEK = 5    # 周均交易日

# 交易费用（默认值）
DEFAULT_COMMISSION_RATE = 0.001   # 佣金率 0.1%
DEFAULT_SLIPPAGE = 0.001          # 滑点 0.1%
DEFAULT_TAX_RATE = 0.001          # 印花税 0.1%（卖出时）

# 数据相关
DEFAULT_CACHE_DAYS = 7  # 默认缓存天数
