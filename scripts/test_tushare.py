#!/usr/bin/env python
"""
Tushare 数据获取测试
使用你的 token 获取真实 A 股数据
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TUSHARE_TOKEN, check_tushare_token
from utils.data_fetcher import DataFetcher


def main():
    """主函数"""
    print("=" * 70)
    print("📊 Tushare Pro 数据获取测试")
    print("=" * 70)
    
    # 检查 token
    if not check_tushare_token():
        print("\n❌ Token 未配置，请修改 .env 文件")
        return
    
    # 创建数据获取器（使用配置的 token）
    print("\n🔌 连接 Tushare Pro...")
    fetcher = DataFetcher(tushare_token=TUSHARE_TOKEN)
    
    # 获取数据
    print("\n" + "-" * 70)
    print("📈 测试1: 获取平安银行(000001)数据")
    print("-" * 70)
    
    try:
        df = fetcher.get_daily_data_ak('000001', '20240101', '20241231')
        print(f"\n✅ 成功获取 {len(df)} 条数据")
        print(f"\n最新5天数据:")
        print(df[['open', 'high', 'low', 'close', 'volume', 'pct_change']].tail())
        
        # 保存
        fetcher.save_to_csv(df, '000001_tushare.csv')
        
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 获取多只股票
    print("\n" + "-" * 70)
    print("📈 测试2: 批量获取多只股票")
    print("-" * 70)
    
    stocks = [
        ('000001', '平安银行'),
        ('000002', '万科A'),
        ('600519', '贵州茅台'),
        ('002594', '比亚迪'),
        ('300750', '宁德时代'),
    ]
    
    for code, name in stocks:
        try:
            df = fetcher.get_daily_data_ak(code, '20240101', '20241231')
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            ret = (end_price / start_price - 1) * 100
            print(f"   ✅ {code} {name}: {len(df)}条, 收益 {ret:+.2f}%")
        except Exception as e:
            print(f"   ❌ {code} {name}: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！数据已保存到 ./data/ 目录")
    print("=" * 70)


if __name__ == '__main__':
    main()
