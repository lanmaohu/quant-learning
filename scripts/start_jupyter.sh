#!/bin/bash
# 一键启动 Jupyter Lab

cd "$(dirname "$(dirname "$0")")"

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quant

echo "🚀 启动 Jupyter Lab..."
echo "📂 工作目录: $(pwd)"
echo "🐍 Python: $(which python)"
echo ""

# 启动 Jupyter Lab
jupyter lab \
  --ip=127.0.0.1 \
  --port=8888 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --NotebookApp.notebook_dir="$(pwd)"
