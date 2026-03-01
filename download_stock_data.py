"""
股票数据下载工具
提供多种方式获取A股数据
"""

import os
import sys


def print_manual_download_guide():
    """打印手动下载数据指南"""
    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║                📥 A股数据手动下载指南                                  ║
╚══════════════════════════════════════════════════════════════════════╝

【方案一】东方财富网（推荐，免费）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 打开 https://quote.eastmoney.com/concept/sh600519.html
2. 搜索你想要的股票，如 "贵州茅台" 或代码 "600519"
3. 点击页面顶部的 "数据" 或 "导出数据"
4. 选择时间范围，下载CSV文件
5. 将文件重命名为 {股票代码}.csv，如 600519.csv
6. 放到 ./data/offline/ 目录下

【方案二】同花顺/通达信导出
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 打开同花顺软件
2. 进入个股K线图
3. 右键 -> 数据导出 -> 选择CSV格式
4. 保存到 ./data/offline/ 目录

【方案三】Tushare Pro（推荐程序化获取）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 访问 https://tushare.pro/register 注册账号
2. 获取 Token（个人中心 -> 接口TOKEN）
3. 修改代码设置 Token:

    from utils.data_fetcher import DataFetcher
    fetcher = DataFetcher(tushare_token='你的token')
    df = fetcher.get_daily_data_ak('000001', '20240101', '20241231')

【方案四】使用 Yahoo Finance（无需注册）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
适合获取港股通标的和部分美股

    pip install yfinance
    
    from utils.data_solutions import YahooFinanceSource
    yf = YahooFinanceSource()
    df = yf.get_daily_data('000001', '20240101', '20241231')

【方案五】Baostock（A股免费数据）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install baostock
    
    from utils.data_solutions import BaostockSource
    bs = BaostockSource()
    df = bs.get_daily_data('000001', '20240101', '20241231')

【方案六】聚宽/米筐（在线平台）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 聚宽 https://www.joinquant.com/
- 米筐 https://www.ricequant.com/
- 申请账号后，可以在平台上直接研究，或导出数据

═══════════════════════════════════════════════════════════════════════
"""
    print(guide)


def setup_offline_directory():
    """设置离线数据目录"""
    offline_dir = './data/offline'
    os.makedirs(offline_dir, exist_ok=True)
    
    # 创建示例CSV
    sample_content = """date,open,high,low,close,volume
2024-01-02,10.50,10.80,10.40,10.75,1500000
2024-01-03,10.75,10.90,10.60,10.65,1800000
2024-01-04,10.65,10.70,10.45,10.50,1200000
2024-01-05,10.50,10.60,10.30,10.35,1600000
"""
    
    sample_file = os.path.join(offline_dir, 'README.txt')
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("""离线数据目录

请在此目录放置CSV格式的股票数据文件

文件命名格式: {股票代码}.csv
例如: 000001.csv, 600519.csv

CSV文件格式要求:
- 第一行必须是列名: date, open, high, low, close, volume
- date 列格式: YYYY-MM-DD
- 数据按日期升序排列

示例数据格式:
""")
        f.write(sample_content)
    
    print(f"✅ 离线数据目录已创建: {offline_dir}")
    print(f"   示例文件: {sample_file}")
    return offline_dir


def download_with_yahoo():
    """使用 Yahoo Finance 下载数据"""
    print("\n📊 使用 Yahoo Finance 下载数据...")
    
    try:
        from utils.data_solutions import YahooFinanceSource
        
        yf = YahooFinanceSource()
        
        if not yf.available:
            print("⚠️ yfinance 未安装，正在安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
            print("✅ 安装完成，请重新运行")
            return
        
        # 下载几只示例股票
        stocks = ['000001', '600519', '000858']
        
        for symbol in stocks:
            try:
                print(f"\n   下载 {symbol}...")
                df = yf.get_daily_data(symbol, '20230101', '20241231')
                
                # 保存
                filepath = f'./data/offline/{symbol}_yahoo.csv'
                df.to_csv(filepath)
                print(f"   ✅ 已保存: {filepath} ({len(df)} 条)")
                
            except Exception as e:
                print(f"   ❌ 失败: {e}")
    
    except Exception as e:
        print(f"⚠️ 错误: {e}")


def download_with_baostock():
    """使用 Baostock 下载数据"""
    print("\n📊 使用 Baostock 下载数据...")
    
    try:
        from utils.data_solutions import BaostockSource
        
        bs = BaostockSource()
        
        if not bs.available:
            print("⚠️ baostock 未安装，正在安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "baostock"])
            print("✅ 安装完成，请重新运行")
            return
        
        # 下载几只示例股票
        stocks = ['000001', '600519', '000858']
        
        for symbol in stocks:
            try:
                print(f"\n   下载 {symbol}...")
                df = bs.get_daily_data(symbol, '20230101', '20241231')
                
                # 保存
                filepath = f'./data/offline/{symbol}_baostock.csv'
                df.to_csv(filepath)
                print(f"   ✅ 已保存: {filepath} ({len(df)} 条)")
                
            except Exception as e:
                print(f"   ❌ 失败: {e}")
        
        # 登出
        import baostock
        baostock.logout()
    
    except Exception as e:
        print(f"⚠️ 错误: {e}")


def test_all_sources():
    """测试所有数据源"""
    print("\n🧪 测试所有数据源...")
    
    from utils.data_solutions import SmartDataFetcher
    
    fetcher = SmartDataFetcher()
    
    test_stocks = [
        ('000001', '平安银行'),
        ('600519', '贵州茅台'),
        ('000858', '五粮液'),
    ]
    
    print("\n测试结果:")
    print("-" * 50)
    
    for code, name in test_stocks:
        print(f"\n   {code} {name}:")
        try:
            df = fetcher.get_daily_data(code, '20240101', '20241231')
            print(f"      ✅ 成功 ({len(df)} 条)")
            print(f"         日期范围: {df.index[0].date()} ~ {df.index[-1].date()}")
            print(f"         最新价格: {df['close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"      ❌ 失败: {str(e)[:40]}")
    
    print("\n" + "=" * 50)


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 70)
        print("📊 股票数据下载工具")
        print("=" * 70)
        print("\n请选择操作:")
        print("   1. 查看手动下载指南")
        print("   2. 设置离线数据目录")
        print("   3. 使用 Yahoo Finance 下载")
        print("   4. 使用 Baostock 下载")
        print("   5. 测试所有数据源")
        print("   6. 安装所需依赖包")
        print("   0. 退出")
        print("-" * 70)
        
        choice = input("请输入选项 (0-6): ").strip()
        
        if choice == '1':
            print_manual_download_guide()
        elif choice == '2':
            setup_offline_directory()
        elif choice == '3':
            download_with_yahoo()
        elif choice == '4':
            download_with_baostock()
        elif choice == '5':
            test_all_sources()
        elif choice == '6':
            from utils.data_solutions import install_data_packages
            install_data_packages()
        elif choice == '0':
            print("👋 再见！")
            break
        else:
            print("⚠️ 无效选项")


if __name__ == '__main__':
    main()
