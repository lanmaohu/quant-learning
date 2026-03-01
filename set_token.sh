#!/bin/bash
# 设置 Tushare Token 脚本

echo "=================================="
echo "🔐 Tushare Token 配置"
echo "=================================="
echo ""
echo "请访问 https://tushare.pro/user/token 获取你的 token"
echo ""

# 提示输入 token
read -p "请输入你的 Tushare Token: " TOKEN

if [ -z "$TOKEN" ]; then
    echo "❌ 未输入 token"
    exit 1
fi

# 保存到 .env 文件
ENV_FILE="$(dirname "$0")/.env"

echo "# Tushare Pro Token" > "$ENV_FILE"
echo "# 获取方式: https://tushare.pro/user/token" >> "$ENV_FILE"
echo "TUSHARE_TOKEN=$TOKEN" >> "$ENV_FILE"

# 设置权限
chmod 600 "$ENV_FILE"

echo ""
echo "✅ Token 已保存！"
echo "   文件: $ENV_FILE"
echo "   前10位: ${TOKEN:0:10}..."
echo ""
echo "现在可以运行测试:"
echo "   python test_tushare.py"
