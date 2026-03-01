#!/usr/bin/env python
"""
直接配置 Tushare Token - 简单版本
"""

import os
import sys

def main():
    print("=" * 60)
    print("🔐 配置 Tushare Token")
    print("=" * 60)
    print("")
    print("请输入你的 Tushare Token（从 https://tushare.pro/user/token 获取）:")
    print("提示: Token 通常是 40 位左右的字母数字组合")
    print("")
    
    # 直接读取输入
    try:
        token = input("Token: ").strip()
    except EOFError:
        # 如果是非交互式环境，使用参数
        if len(sys.argv) > 1:
            token = sys.argv[1]
        else:
            print("❌ 无法读取输入，请使用: python configure_token.py YOUR_TOKEN")
            sys.exit(1)
    
    if not token or len(token) < 10:
        print("❌ Token 无效或太短")
        sys.exit(1)
    
    # 保存到 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    with open(env_path, 'w') as f:
        f.write(f"# Tushare Pro Token\n")
        f.write(f"# 获取方式: https://tushare.pro/user/token\n")
        f.write(f"TUSHARE_TOKEN={token}\n")
    
    os.chmod(env_path, 0o600)
    
    print("")
    print(f"✅ Token 已保存！")
    print(f"   前10位: {token[:10]}...")
    print(f"   长度: {len(token)} 位")
    print("")
    print("现在运行测试:")
    print("   python test_tushare.py")

if __name__ == '__main__':
    main()
