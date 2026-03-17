"""
数据加载模块
统一的数据加载接口
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


class StockDataLoader:
    """
    股票数据加载器
    支持CSV文件加载、字段标准化、时间过滤
    """
    
    # 列名映射标准化
    COLUMN_MAPPING = {
        'stock_code': 'code',
        'code': 'code',
        'date': 'date',
        'name': 'name',
        'price_change_rate': 'price_change_rate',
        'vol': 'volume',
        'total_mv': 'market_cap',
        'close': 'close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
    }
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_df = None
        
    def load(self, years_back: int = 5, select_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载并标准化数据
        
        Parameters:
        -----------
        years_back : int
            使用最近N年数据
        select_codes : List[str], optional
            只加载指定股票代码
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        print(f"📂 加载数据: {self.data_path}")
        
        # 如果指定了代码，使用分块加载
        if select_codes:
            df = self._load_with_filter(select_codes)
        else:
            df = pd.read_csv(self.data_path, low_memory=False)
        
        print(f"   原始: {df.shape[0]:,} 行")
        
        # 打印列名和各列取值样例
        print(f"\n   原始列名: {list(df.columns)}")
        print(f"\n   各列取值样例（前3行）:")
        print(df.head(3).to_string())
        print(f"\n   各列数据类型:")
        print(df.dtypes.to_string())
        
        # 标准化列名
        df = self._normalize_columns(df)
        
        # 数据类型转换
        df = self._convert_types(df)
        
        # 时间过滤
        if years_back:
            cutoff = df['date'].max() - pd.DateOffset(years=years_back)
            df = df[df['date'] >= cutoff]
        
        # 排序
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        
        self.raw_df = df
        print(f"   加载后: {df.shape[0]:,} 行 | {df['code'].nunique()} 只股票")
        return df
    
    def _load_with_filter(self, select_codes: List[str]) -> pd.DataFrame:
        """分块加载并过滤"""
        chunks = []
        for chunk in pd.read_csv(self.data_path, chunksize=100000):
            chunk = self._normalize_columns(chunk)
            if 'code' in chunk.columns:
                chunk = chunk[chunk['code'].isin(select_codes)]
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        if 'code' in df.columns:
            df['code'] = df['code'].astype(str).str.zfill(6)
        
        return df
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        # 日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # 数值列
        num_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 如果没有amount列，通过 volume * close 计算
        if 'amount' not in df.columns and 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        
        return df
    
    def select_sample_codes(self, n: int = 50, random_seed: int = 42) -> List[str]:
        """
        随机选择样本股票
        """
        if self.raw_df is None:
            raise ValueError("请先调用load()加载数据")
        
        all_codes = self.raw_df['code'].unique()
        np.random.seed(random_seed)
        selected = np.random.choice(all_codes, size=min(n, len(all_codes)), replace=False)
        return list(selected)


def time_series_split(df: pd.DataFrame, 
                      train_ratio: float = 0.6,
                      val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    时间序列数据集划分（严格无穿越）
    
    Returns:
    --------
    train_df, val_df, test_df
    """
    print("\n⏰ 时间序列数据集划分...")
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()
    
    # 验证无穿越
    assert train_df['date'].max() < val_df['date'].min(), "训练集和验证集有重叠！"
    assert val_df['date'].max() < test_df['date'].min(), "验证集和测试集有重叠！"
    
    print(f"   训练集: {len(train_df):,} ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    print(f"   验证集: {len(val_df):,} ({val_df['date'].min().date()} ~ {val_df['date'].max().date()})")
    print(f"   测试集: {len(test_df):,} ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")
    
    return train_df, val_df, test_df
