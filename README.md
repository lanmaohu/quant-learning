# A股量化投资学习项目

## 📁 项目结构

```
quant-learning/
├── data/                          # 数据存储目录
├── strategies/                    # 策略文件目录
├── utils/                         # 工具函数目录
│   └── data_fetcher.py           # 数据获取工具
├── backtrader_demo.py             # Backtrader回测示例
├── vectorbt_demo.py               # VectorBT回测示例
├── start_quant_env.sh             # 一键启动环境脚本
├── requirements.txt               # pip依赖
├── environment.yml                # Conda环境配置
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 方式一：一键启动（推荐）

```bash
cd ~/quant-learning
./start_quant_env.sh
```

### 方式二：手动激活Conda环境

```bash
# 激活环境
conda activate quant

# 退出环境
conda deactivate
```

### 方式三：直接使用（如果不使用Conda）

```bash
pip install -r requirements.txt
```

## 📊 运行回测

```bash
# 确保在quant环境中
conda activate quant

# 运行Backtrader回测
python backtrader_demo.py

# 运行VectorBT回测
python vectorbt_demo.py

# 测试数据接口
python utils/data_fetcher.py
```

## 🐍 Conda环境管理

### 环境信息
- **环境名称**: `quant`
- **Python版本**: 3.11
- **安装位置**: `~/miniconda3/envs/quant`

### 常用命令

```bash
# 查看所有环境
conda env list

# 查看当前环境的包
conda list

# 安装新包
conda install 包名        # 优先用conda
pip install 包名          # conda没有时再用pip

# 导出环境（备份）
conda env export --no-builds > environment.yml

# 从文件创建环境
conda env create -f environment.yml

# 删除环境
conda remove -n quant --all
```

## 📦 核心依赖

| 包名 | 用途 | 安装方式 |
|------|------|---------|
| pandas | 数据处理 | conda |
| numpy | 数值计算 | conda |
| backtrader | 事件驱动回测 | pip |
| vectorbt | 向量化回测 | pip |
| akshare | A股数据获取 | pip |
| lightgbm | 机器学习 | conda/pip |
| jupyter | 交互式开发 | conda |

## 💡 使用Jupyter Lab进行开发

```bash
# 启动Jupyter Lab
jupyter lab

# 在浏览器中打开 http://localhost:8888
```

## 🔧 故障排除

### 1. Conda命令找不到
```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

### 2. 网络问题导致数据获取失败
代码已内置模拟数据模式，断网也能运行学习。

### 3. 包冲突
```bash
# 重新创建干净环境
conda remove -n quant --all
conda create -n quant python=3.11
conda activate quant
pip install -r requirements.txt
```

## 📚 学习资源

1. [AKShare文档](https://www.akshare.xyz/)
2. [Backtrader文档](https://www.backtrader.com/docu/)
3. [VectorBT文档](https://vectorbt.dev/)
4. [聚宽](https://www.joinquant.com/)

## 📝 量化学习计划

见 `30天学习计划.md`
