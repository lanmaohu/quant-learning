"""
Tushare 数据接口使用示例
Tushare Pro 是专业的金融数据接口，数据质量高

使用步骤:
1. 访问 https://tushare.pro/register 注册账号
2. 在"个人中心" -> "接口TOKEN" 获取 token
3. 将 token 填入下方代码或使用环境变量
"""

import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import os


class TushareDataFetcher:
    """
    Tushare 数据获取类
    """
    
    def __init__(self, token=None):
        """
        初始化 Tushare
        
        Parameters:
        -----------
        token : str
            Tushare Pro 的 token
            可以从 https://tushare.pro/user/token 获取
        """
        if token is None:
            # 尝试从环境变量获取
            token = os.environ.get('TUSHARE_TOKEN')
        
        if token is None:
            raise ValueError("""
            ❌ 缺少 Tushare Token!
            
            请按以下步骤获取:
            1. 访问 https://tushare.pro/register 注册账号
            2. 登录后进入"个人中心" -> "接口TOKEN"
            3. 复制你的 token
            4. 使用方式:
               
               # 方式一：直接传入
               fetcher = TushareDataFetcher(token='你的token')
               
               # 方式二：设置环境变量
               export TUSHARE_TOKEN='你的token'
            """)
        
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        print("✅ Tushare Pro 初始化成功！")
    
    def get_daily_data(self, ts_code, start_date=None, end_date=None):
        """
        获取股票日线数据
        
        Parameters:
        -----------
        ts_code : str
            股票代码，如 '000001.SZ', '600519.SH'
        start_date : str
            开始日期 'YYYYMMDD'
        end_date : str
            结束日期 'YYYYMMDD'
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"[Tushare] 获取 {ts_code} 从 {start_date} 到 {end_date} 的数据...")
        
        # 调用 Tushare Pro 接口
        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise Exception(f"未找到 {ts_code} 的数据")
        
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        
        # 标准化列名
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'pct_chg': 'pct_change',
            'change': 'change'
        })
        
        # 计算振幅
        df['amplitude'] = (df['high'] - df['low']) / df['pre_close'] * 100
        
        print(f"   ✅ 成功获取 {len(df)} 条数据")
        return df
    
    def get_daily_data_by_symbol(self, symbol, start_date=None, end_date=None):
        """
        根据股票代码获取数据（自动添加后缀）
        
        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001', '600519'
        """
        # 自动添加交易所后缀
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        return self.get_daily_data(ts_code, start_date, end_date)
    
    def get_stock_basic(self):
        """
        获取股票基础信息
        """
        print("[Tushare] 获取股票基础信息...")
        df = self.pro.stock_basic(
            exchange='',
            list_status='L',  # 上市
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        print(f"   ✅ 共 {len(df)} 只股票")
        return df
    
    def get_index_daily(self, ts_code='000300.SH', start_date=None, end_date=None):
        """
        获取指数日线数据
        
        Parameters:
        -----------
        ts_code : str
            指数代码，如 '000300.SH' (沪深300), '000001.SH' (上证指数)
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"[Tushare] 获取指数 {ts_code} 数据...")
        
        df = self.pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise Exception(f"未找到 {ts_code} 的数据")
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"   ✅ 成功获取 {len(df)} 条数据")
        return df
    
    def get_financial_data(self, ts_code, start_date=None, end_date=None):
        """
        获取财务数据（需要较高积分）
        
        Parameters:
        -----------
        ts_code : str
            股票代码，如 '000001.SZ'
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"[Tushare] 获取 {ts_code} 财务指标...")
        
        # 获取每日财务指标
        df = self.pro.fina_indicator(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            print("   ⚠️ 未获取到财务数据（可能需要更多积分）")
            return None
        
        df['end_date'] = pd.to_datetime(df['end_date'])
        df.set_index('end_date', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"   ✅ 成功获取 {len(df)} 条财务数据")
        return df
    
    def get_limit_list(self, trade_date=None):
        """
        获取每日涨跌停股票列表
        
        Parameters:
        -----------
        trade_date : str
            交易日期 'YYYYMMDD'，默认今天
        """
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
        
        print(f"[Tushare] 获取 {trade_date} 涨跌停数据...")
        
        df = self.pro.limit_list(
            trade_date=trade_date,
            limit_type='U'  # U: 涨停，D: 跌停
        )
        
        print(f"   ✅ 共 {len(df)} 只涨停股票")
        return df


def demo_tushare():
    """
    Tushare 使用演示
    """
    print("=" * 70)
    print("📊 Tushare Pro 数据接口演示")
    print("=" * 70)
    
    # 请替换为你的 token
    # 获取方式：https://tushare.pro/user/token
    TOKEN = "你的Tushare Token"
    
    # 尝试从环境变量读取
    import os
    TOKEN = os.environ.get('TUSHARE_TOKEN', TOKEN)
    
    if TOKEN == "你的Tushare Token":
        print("""
        ⚠️ 请先设置 Tushare Token！
        
        获取方式:
        1. 访问 https://tushare.pro/register 注册
        2. 进入"个人中心" -> "接口TOKEN" 复制 token
        3. 修改本文件的 TOKEN 变量，或设置环境变量:
           export TUSHARE_TOKEN='你的token'
        """)
        return
    
    try:
        # 初始化
        fetcher = TushareDataFetcher(token=TOKEN)
        
        # 1. 获取股票基础信息
        print("\n📋 获取股票基础信息:")
        basic = fetcher.get_stock_basic()
        print(basic.head(10))
        
        # 2. 获取个股日线数据
        print("\n📈 获取平安银行(000001.SZ)日线数据:")
        df = fetcher.get_daily_data_by_symbol('000001', '20240101', '20241231')
        print(df[['open', 'high', 'low', 'close', 'volume', 'pct_change']].head(10))
        print(f"\n数据统计:")
        print(df[['close', 'volume', 'pct_change']].describe())
        
        # 3. 获取指数数据
        print("\n📊 获取沪深300指数数据:")
        index_df = fetcher.get_index_daily('000300.SH', '20240101', '20241231')
        print(index_df[['close', 'pct_chg']].head())
        
        # 4. 获取贵州茅台
        print("\n🍷 获取贵州茅台(600519.SH)数据:")
        moutai = fetcher.get_daily_data('600519.SH', '20240101', '20241231')
        print(f"区间涨幅: {(moutai['close'].iloc[-1]/moutai['close'].iloc[0]-1)*100:.2f}%")
        
        # 5. 保存数据
        print("\n💾 保存数据...")
        df.to_csv('./data/000001_tushare.csv')
        print("✅ 数据已保存到 ./data/000001_tushare.csv")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == '__main__':
    demo_tushare()
