"""
股票数据加载器
支持本地 CSV 或在线 API
添加样本股票选择功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class StockDataLoader:
    """
    股票数据加载器
    支持本地 CSV 或在线 API
    """
    
    # A股指数成分股示例（用于快速测试）
    SAMPLE_STOCKS = {
        'hs300': [  # 沪深300样本
            '000001', '000002', '000063', '000100', '000333',
            '000538', '000568', '000651', '000725', '000768',
            '000858', '002024', '002142', '002304', '002415',
            '002594', '300003', '300014', '300015', '300033',
            '600000', '600009', '600016', '600028', '600030',
            '600031', '600036', '600048', '600050', '600276',
            '600309', '600519', '600585', '600588', '600690',
            '600745', '600809', '600837', '601012', '601066',
            '601088', '601100', '601138', '601166', '601288',
            '601318', '601319', '601398', '601601', '601628',
            '601668', '601688', '601766', '601888', '601899',
            '603288', '603501', '603659', '603986', '688981'
        ],
        'zz500': [  # 中证500样本
            '000009', '000012', '000021', '000025', '000039',
            '000060', '000061', '000066', '000078', '000090',
            '000156', '000166', '000400', '000425', '000538',
            '000581', '000625', '000630', '000651', '000709'
        ],
        'sz50': [   # 上证50样本
            '600000', '600009', '600016', '600028', '600030',
            '600031', '600036', '600048', '600050', '600276',
            '600309', '600406', '600436', '600438', '600519',
            '600585', '600588', '600690', '600745', '600809',
            '600837', '600893', '601012', '601066', '601088',
            '601100', '601138', '601166', '601211', '601288',
            '601318', '601319', '601390', '601398', '601601',
            '601628', '601668', '601688', '601766', '601857',
            '601888', '601899', '601939', '601985', '601988',
            '603288', '603259', '603501', '603986', '688981'
        ],
        'tech': [   # 科技板块样本
            '002594', '300750', '002230', '002415', '002236',
            '300014', '300033', '300059', '600584', '603501',
            '603986', '688981', '688012', '688008', '600171'
        ],
        'finance': [  # 金融板块样本
            '000001', '000166', '002142', '600000', '600016',
            '600030', '600036', '600048', '601166', '601169',
            '601288', '601318', '601328', '601398', '601601',
            '601628', '601658', '601688', '601818', '601939'
        ],
        'consumption': [  # 消费板块样本
            '000568', '000596', '000858', '002304', '002507',
            '600519', '600600', '600690', '600702', '600809',
            '600887', '601888', '603288', '603369', '603589'
        ]
    }
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.raw_df = None
        self.selected_codes = None  # 当前选中的股票代码
        
    def select_sample_codes(self, sample_type='hs300', n=None, random_select=False, seed=42):
        """
        选择样本股票代码
        
        Parameters:
        -----------
        sample_type : str
            样本类型: 'hs300'(沪深300), 'zz500'(中证500), 'sz50'(上证50),
                    'tech'(科技), 'finance'(金融), 'consumption'(消费),
                    'all'(全部)
        n : int, optional
            选择前n只股票，None表示全部
        random_select : bool
            是否随机选择
        seed : int
            随机种子
        
        Returns:
        --------
        list : 股票代码列表
        """
        print(f"📋 选择样本股票: {sample_type}")
        
        # 获取基础股票池
        if sample_type == 'all':
            # 合并所有样本
            codes = list(set(sum(self.SAMPLE_STOCKS.values(), [])))
        elif sample_type in self.SAMPLE_STOCKS:
            codes = self.SAMPLE_STOCKS[sample_type].copy()
        else:
            raise ValueError(f"未知的样本类型: {sample_type}. "
                           f"可选: {list(self.SAMPLE_STOCKS.keys())}")
        
        # 随机选择
        if random_select and n and n < len(codes):
            random.seed(seed)
            codes = random.sample(codes, n)
        elif n and n < len(codes):
            codes = codes[:n]
        
        self.selected_codes = codes
        print(f"   ✅ 选中 {len(codes)} 只股票")
        print(f"   示例: {', '.join(codes[:5])}{'...' if len(codes) > 5 else ''}")
        
        return codes
    
    def get_stock_list(self, df=None):
        """
        获取数据中的所有股票列表
        """
        if df is None:
            df = self.raw_df
        
        if df is None:
            raise ValueError("请先加载数据")
        
        stocks = df[['code', 'name']].drop_duplicates().sort_values('code')
        return stocks
    
    def filter_by_codes(self, df=None, codes=None):
        """
        根据股票代码过滤数据
        
        Parameters:
        -----------
        df : pd.DataFrame, optional
            要过滤的数据，默认使用 self.raw_df
        codes : list, optional
            股票代码列表，默认使用 self.selected_codes
        """
        if df is None:
            df = self.raw_df
        
        if df is None:
            raise ValueError("请先加载数据")
        
        if codes is None:
            codes = self.selected_codes
        
        if codes is None:
            print("⚠️ 未指定股票代码，返回全部数据")
            return df
        
        # 过滤
        filtered = df[df['code'].isin(codes)].copy()
        print(f"\n🔍 过滤数据:")
        print(f"   原始: {len(df):,} 条")
        print(f"   过滤后: {len(filtered):,} 条")
        print(f"   股票数: {filtered['code'].nunique()}")
        
        return filtered
    
    def load_from_csv(self, filepath, nrows=None, select_codes=None):
        """
        从 CSV 加载数据
        
        Parameters:
        -----------
        filepath : str
            CSV文件路径
        nrows : int, optional
            只加载前n行
        select_codes : list, optional
            只加载指定股票代码的数据（需CSV有code列）
        """
        print(f"📊 加载数据: {filepath}")
        
        # 如果指定了股票代码，使用chunksize分块读取
        if select_codes:
            chunks = []
            chunk_iter = pd.read_csv(filepath, chunksize=100000)
            for i, chunk in enumerate(chunk_iter):
                # 标准字段映射
                chunk = self._standardize_columns(chunk)
                # 过滤
                chunk = chunk[chunk['code'].isin(select_codes)]
                chunks.append(chunk)
                if i % 10 == 0:
                    print(f"   处理块 {i}...")
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(filepath, nrows=nrows)
            df = self._standardize_columns(df)
        
        self.raw_df = df
        print(f"✅ 加载完成: {len(df):,} 条记录, {df['code'].nunique()} 只股票")
        return df
    
    def _standardize_columns(self, df):
        """标准化列名"""
        # 标准字段映射
        column_mapping = {
            'stock_code': 'code',
            'ts_code': 'code',
            'trade_date': 'date',
            'close_price': 'close',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'vol': 'volume',
            'amount': 'amount',
            'market_cap': 'market_cap',
            'total_mv': 'market_cap',
            'circ_mv': 'circ_market_cap'
        }
        
        # 重命名列
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # 转换日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 标准化股票代码（6位数字）
        if 'code' in df.columns:
            df['code'] = df['code'].astype(str).str.zfill(6)
            # 提取交易所信息
            df['exchange'] = df['code'].apply(self._get_exchange)
        
        return df
    
    def _get_exchange(self, code):
        """根据代码判断交易所"""
        if code.startswith('6'):
            return 'SH'  # 上海主板
        elif code.startswith('0') or code.startswith('001'):
            return 'SZ'  # 深圳主板
        elif code.startswith('3'):
            return 'CY'  # 创业板
        elif code.startswith('68'):
            return 'KC'  # 科创板
        elif code.startswith('8') or code.startswith('4'):
            return 'BJ'  # 北交所
        return 'OTHER'
    
    def load_from_api(self, stock_codes=None, start_date=None, end_date=None):
        """
        从 AKShare 获取数据
        
        Parameters:
        -----------
        stock_codes : list, optional
            股票代码列表，默认使用 self.selected_codes
        start_date : str
            开始日期 'YYYYMMDD'
        end_date : str
            结束日期 'YYYYMMDD'
        """
        if stock_codes is None:
            stock_codes = self.selected_codes
        
        if stock_codes is None:
            raise ValueError("请提供股票代码或先调用 select_sample_codes()")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        try:
            import akshare as ak
            
            all_data = []
            for i, code in enumerate(stock_codes):
                print(f"📥 获取 {code} ({i+1}/{len(stock_codes)})...")
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=code, 
                        period="daily", 
                        start_date=start_date, 
                        end_date=end_date, 
                        adjust="qfq"
                    )
                    if not df.empty:
                        df['code'] = code
                        all_data.append(df)
                except Exception as e:
                    print(f"   ⚠️ {code} 获取失败: {e}")
                    continue
            
            if not all_data:
                raise Exception("没有成功获取任何数据")
            
            df = pd.concat(all_data, ignore_index=True)
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            df['date'] = pd.to_datetime(df['date'])
            df['exchange'] = df['code'].apply(self._get_exchange)
            
            self.raw_df = df
            print(f"✅ 获取完成: {len(df):,} 条记录, {df['code'].nunique()} 只股票")
            return df
            
        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return None
    
    def get_single_stock(self, code):
        """获取单只股票数据"""
        if self.raw_df is None:
            raise ValueError("请先加载数据")
        
        df = self.raw_df[self.raw_df['code'] == code].copy()
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df


# ==================== 使用示例 ====================

def demo_stock_loader():
    """StockDataLoader 使用演示"""
    print("=" * 70)
    print("🚀 StockDataLoader 使用演示")
    print("=" * 70)
    
    # 1. 初始化加载器
    loader = StockDataLoader()
    print("\n✅ 数据加载器初始化完成")
    
    # 2. 选择样本股票
    print("\n📋 选择沪深300样本股票 (前20只):")
    codes = loader.select_sample_codes('hs300', n=20)
    
    # 3. 从A股大数据文件中加载这20只股票
    print("\n📊 从A股历史数据文件加载...")
    try:
        df = loader.load_from_csv(
            '/Users/harry/Documents/a_stock_history_price_20260223.csv',
            select_codes=codes
        )
        
        # 4. 查看数据
        print(f"\n📈 数据预览:")
        print(df[['date', 'code', 'name', 'close', 'volume']].head(10))
        
        # 5. 获取股票列表
        print(f"\n📋 股票列表:")
        stocks = loader.get_stock_list()
        print(stocks.head(10))
        
        # 6. 获取单只股票
        print(f"\n🔍 平安银行(000001)最新数据:")
        pingan = loader.get_single_stock('000001')
        print(pingan[['close', 'volume', 'pct_change']].tail())
        
    except Exception as e:
        print(f"⚠️ 演示失败: {e}")
        print("   请确保A股数据文件存在")
    
    print("\n" + "=" * 70)
    print("✅ 演示完成!")
    print("=" * 70)
    
    return loader


if __name__ == '__main__':
    demo_stock_loader()
