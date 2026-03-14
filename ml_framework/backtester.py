"""
回测模块
"""

import pandas as pd
import numpy as np
from typing import Dict


class Backtester:
    """
    回测器
    Top-K 选股策略回测
    """
    
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        
    def run(self, test_df: pd.DataFrame, pred_col: str = 'pred_return', 
            actual_col: str = 'target_return_5d') -> Dict:
        """
        运行回测
        
        Parameters:
        -----------
        test_df : pd.DataFrame
            测试集数据，包含预测和实际收益
        pred_col : str
            预测收益列名
        actual_col : str
            实际收益列名
        
        Returns:
        --------
        result : dict
            回测结果
        """
        print(f"\n🚀 回测策略 (每日选Top {self.top_k})...")
        
        daily_returns = []
        
        for date, group in test_df.groupby('date'):
            if len(group) < self.top_k:
                continue
            
            # 选预测收益最高的top_k
            top_stocks = group.nlargest(self.top_k, pred_col)
            
            # 计算平均实际收益
            avg_return = top_stocks[actual_col].mean()
            daily_returns.append({
                'date': date,
                'return': avg_return,
                'n_stocks': len(top_stocks)
            })
        
        returns_df = pd.DataFrame(daily_returns)
        if len(returns_df) == 0:
            print("⚠️ 没有可回测的数据")
            return None
        
        returns_df.set_index('date', inplace=True)
        returns_df['cum_return'] = (1 + returns_df['return']).cumprod() - 1
        
        # 计算指标
        total_return = returns_df['cum_return'].iloc[-1]
        sharpe = (returns_df['return'].mean() / returns_df['return'].std() * np.sqrt(252) 
                 if returns_df['return'].std() > 0 else 0)
        max_dd = (returns_df['cum_return'] - returns_df['cum_return'].cummax()).min()
        win_rate = (returns_df['return'] > 0).mean()
        
        print(f"\n📊 回测结果:")
        print(f"   总收益率: {total_return*100:.2f}%")
        print(f"   夏普比率: {sharpe:.3f}")
        print(f"   最大回撤: {max_dd*100:.2f}%")
        print(f"   胜率: {win_rate*100:.1f}%")
        
        return {
            'returns_df': returns_df,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }
    
    def plot_results(self, result: Dict, save_path: str = None):
        """绘制回测结果"""
        import matplotlib.pyplot as plt
        
        returns_df = result['returns_df']
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 累计收益
        axes[0].plot(returns_df.index, returns_df['cum_return']*100, linewidth=2)
        axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].set_title(f'Top-{self.top_k} Strategy Backtest')
        axes[0].grid(True, alpha=0.3)
        
        # 每日收益分布
        axes[1].bar(returns_df.index, returns_df['return']*100, alpha=0.6)
        axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Daily Return (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ 回测图保存: {save_path}")
        
        plt.show()
