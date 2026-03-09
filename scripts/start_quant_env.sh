#!/bin/bash
# 启动量化开发环境的便捷脚本

echo "🚀 启动量化开发环境..."
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quant
echo "✅ 已进入 quant 环境，Python版本:"
python --version

echo ""
echo "📦 已安装的核心包:"
pip list | grep -E "(pandas|numpy|backtrader|vectorbt|akshare|lightgbm|jupyter)"

echo ""
echo "💡 提示:"
echo "   - 运行回测: python examples/backtrader_demo.py"
echo "   - 启动Jupyter: jupyter lab"
echo ""

# 保持shell在quant环境中
exec zsh
