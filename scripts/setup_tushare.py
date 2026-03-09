#!/usr/bin/env python
"""
交互式配置 Tushare Token
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import getpass


def setup_token():
    """设置 Tushare Token"""
    print("=" * 70)
    print("🔐 Tushare Token 配置")
    print("=" * 70)
    
    print("""
Tushare Pro 是专业的金融数据接口，需要 token 才能使用。

获取方式:
1. 访问 https://tushare.pro/register 注册账号
2. 登录后进入"个人中心" -> "接口TOKEN"
3. 复制你的 token（约40位字母数字组合）
""")
    
    # 提示输入 token
    print("请输入你的 Tushare Token（输入时不会显示）:")
    token = getpass.getpass("Token: ").strip()
    
    if not token:
        print("❌ 未输入 token")
        return False
    
    if len(token) < 20:
        print(f"⚠️ Token 长度似乎不正确（当前{len(token)}位），请确认")
        confirm = input("是否继续? (y/n): ").lower()
        if confirm != 'y':
            return False
    
    # 保存到 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    with open(env_path, 'w') as f:
        f.write(f"# Tushare Pro Token\n")
        f.write(f"# 获取方式: https://tushare.pro/user/token\n")
        f.write(f"TUSHARE_TOKEN={token}\n")
    
    # 设置文件权限（仅限当前用户可读）
    os.chmod(env_path, 0o600)
    
    print(f"\n✅ Token 已保存到: {env_path}")
    print(f"   Token 前10位: {token[:10]}...")
    print(f"   文件权限已设置为仅限当前用户读取")
    
    # 测试 token
    print("\n🧪 测试 token 是否有效...")
    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 尝试获取一条数据
        df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240105')
        
        if not df.empty:
            print(f"✅ Token 验证成功！获取到 {len(df)} 条测试数据")
            return True
        else:
            print("⚠️ Token 验证返回空数据，可能需要更多积分")
            return True
            
    except Exception as e:
        print(f"❌ Token 验证失败: {e}")
        print("   请检查 token 是否正确")
        return False


def test_data_fetching():
    """测试数据获取"""
    print("\n" + "=" * 70)
    print("📊 测试数据获取")
    print("=" * 70)
    
    from utils.data_fetcher import DataFetcher
    
    # 从 .env 加载 token
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv('TUSHARE_TOKEN', '')
    
    if not token:
        print("❌ 未找到 token，请先运行配置")
        return
    
    fetcher = DataFetcher(tushare_token=token)
    
    # 获取平安银行数据
    print("\n获取平安银行(000001)数据...")
    try:
        df = fetcher.get_daily_data_ak('000001', '20240101', '20241231')
        print(f"✅ 成功获取 {len(df)} 条数据")
        print(f"\n最新5天数据:")
        print(df[['open', 'high', 'low', 'close', 'volume', 'pct_change']].tail())
    except Exception as e:
        print(f"❌ 失败: {e}")


def main():
    """主菜单"""
    while True:
        print("\n" + "=" * 70)
        print("🔧 Tushare 配置工具")
        print("=" * 70)
        print("\n选项:")
        print("   1. 配置/更新 Tushare Token")
        print("   2. 测试数据获取")
        print("   3. 查看当前配置")
        print("   0. 退出")
        print("-" * 70)
        
        choice = input("请选择 (0-3): ").strip()
        
        if choice == '1':
            setup_token()
        elif choice == '2':
            test_data_fetching()
        elif choice == '3':
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.startswith('TUSHARE_TOKEN='):
                            token = line.split('=', 1)[1]
                            print(f"\n✅ Token 已配置")
                            print(f"   前10位: {token[:10]}...")
                            print(f"   长度: {len(token)} 位")
                            break
            else:
                print("\n❌ 未找到配置文件")
        elif choice == '0':
            print("👋 再见！")
            break
        else:
            print("⚠️ 无效选项")


if __name__ == '__main__':
    main()
