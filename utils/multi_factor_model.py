"""
多因子合成模型
Day 11-12: 多因子加权、合成、完整因子研究框架
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


class FactorSynthesizer:
    """
    多因子合成器
    将多个因子合成为一个复合因子
    """
    
    @staticmethod
    def equal_weight(factor_df, factor_cols):
        """
        等权合成
        """
        return factor_df[factor_cols].mean(axis=1)
    
    @staticmethod
    def ic_weighted(factor_df, factor_cols, ic_values):
        """
        IC加权合成
        
        Parameters:
        -----------
        ic_values : dict
            每个因子对应的IC值 {factor_name: ic_value}
        """
        weights = pd.Series({col: abs(ic_values.get(col, 0)) for col in factor_cols})
        weights = weights / weights.sum()  # 归一化
        
        print("📊 IC权重分配:")
        for col, w in weights.items():
            print(f"   {col}: {w:.4f}")
        
        composite = pd.Series(0, index=factor_df.index)
        for col in factor_cols:
            # IC为负的因子反向
            sign = 1 if ic_values.get(col, 0) >= 0 else -1
            composite += weights[col] * sign * factor_df[col]
        
        return composite
    
    @staticmethod
    def ir_weighted(factor_df, factor_cols, ic_series_dict):
        """
        IR加权（IC均值/IC标准差）
        更稳健，考虑了因子稳定性
        """
        ir_values = {}
        for col in factor_cols:
            if col in ic_series_dict:
                ic_mean = ic_series_dict[col].mean()
                ic_std = ic_series_dict[col].std()
                ir_values[col] = abs(ic_mean / ic_std) if ic_std != 0 else 0
            else:
                ir_values[col] = 0
        
        weights = pd.Series(ir_values)
        weights = weights / weights.sum()
        
        print("📊 IR权重分配:")
        for col, w in weights.items():
            print(f"   {col}: {w:.4f} (IR={ir_values[col]:.4f})")
        
        composite = pd.Series(0, index=factor_df.index)
        for col in factor_cols:
            sign = 1 if ic_series_dict[col].mean() >= 0 else -1
            composite += weights[col] * sign * factor_df[col]
        
        return composite
    
    @staticmethod
    def ml_weighted(factor_df, factor_cols, forward_returns, model_type='ridge'):
        """
        机器学习加权（预测模型权重）
        
        Parameters:
        -----------
        model_type : str
            'ridge', 'lasso', 'rf', 'lightgbm'
        """
        X = factor_df[factor_cols].fillna(0)
        y = forward_returns.fillna(0)
        
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=5)
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(n_estimators=100, verbosity=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        
        # 获取特征重要性（权重）
        if hasattr(model, 'coef_'):
            weights = pd.Series(model.coef_, index=factor_cols)
        else:
            weights = pd.Series(model.feature_importances_, index=factor_cols)
        
        weights = weights / weights.abs().sum()
        
        print(f"📊 ML({model_type})权重分配:")
        for col, w in weights.items():
            print(f"   {col}: {w:.4f}")
        
        # 预测值作为复合因子
        composite = pd.Series(model.predict(X), index=factor_df.index)
        
        return composite
    
    @staticmethod
    def pca_weighted(factor_df, factor_cols, n_components=3):
        """
        PCA降维合成（去除因子间相关性）
        """
        from sklearn.decomposition import PCA
        
        X = factor_df[factor_cols].fillna(0)
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)
        
        print(f"📊 PCA解释方差比: {pca.explained_variance_ratio_}")
        print(f"   累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")
        
        # 使用第一主成分
        return pd.Series(components[:, 0], index=factor_df.index)


class MultiFactorStrategy:
    """
    多因子选股策略框架
    """
    
    def __init__(self):
        self.selected_factors = []
        self.factor_weights = {}
        self.composite_factor = None
    
    def select_factors(self, factor_test_results, min_ic=0.03, min_ir=0.3):
        """
        基于IC/IR筛选有效因子
        
        Parameters:
        -----------
        factor_test_results : dict
            因子测试结果 {factor_name: {'ic_mean': x, 'ic_ir': y}}
        """
        selected = []
        for factor_name, results in factor_test_results.items():
            ic_mean = abs(results.get('ic_mean', 0))
            ic_ir = abs(results.get('ic_ir', 0))
            
            if ic_mean >= min_ic and ic_ir >= min_ir:
                selected.append(factor_name)
                print(f"✅ 选中因子: {factor_name} (IC={ic_mean:.4f}, IR={ic_ir:.4f})")
            else:
                print(f"❌ 过滤因子: {factor_name} (IC={ic_mean:.4f}, IR={ic_ir:.4f})")
        
        self.selected_factors = selected
        return selected
    
    def build_composite_factor(self, df, factor_cols, method='ic', **kwargs):
        """
        构建复合因子
        """
        synthesizer = FactorSynthesizer()
        
        if method == 'equal':
            composite = synthesizer.equal_weight(df, factor_cols)
        elif method == 'ic':
            composite = synthesizer.ic_weighted(df, factor_cols, kwargs.get('ic_values', {}))
        elif method == 'ir':
            composite = synthesizer.ir_weighted(df, factor_cols, kwargs.get('ic_series_dict', {}))
        elif method == 'ml':
            composite = synthesizer.ml_weighted(
                df, factor_cols, 
                kwargs.get('forward_returns'),
                model_type=kwargs.get('model_type', 'ridge')
            )
        elif method == 'pca':
            composite = synthesizer.pca_weighted(df, factor_cols, kwargs.get('n_components', 3))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.composite_factor = composite
        return composite
    
    def select_stocks(self, df, composite_col='composite_factor', n_top=20, n_bottom=0):
        """
        基于复合因子选股
        
        Returns:
        --------
        long_stocks : list
            多头组合股票列表
        short_stocks : list
            空头组合股票列表（如n_bottom>0）
        """
        latest_data = df.sort_values('date').groupby('code').last()
        
        # 多头：因子值最高的N只
        long_stocks = latest_data.nlargest(n_top, composite_col).index.tolist()
        
        # 空头：因子值最低的N只（如需要）
        short_stocks = []
        if n_bottom > 0:
            short_stocks = latest_data.nsmallest(n_bottom, composite_col).index.tolist()
        
        return long_stocks, short_stocks
    
    def backtest_simple(self, df, composite_col='composite_factor', n_quantiles=5):
        """
        简单分层回测
        """
        returns = []
        
        for date, group in df.groupby('date'):
            group['quantile'] = pd.qcut(group[composite_col], n_quantiles, labels=False)
            
            # 每层平均收益
            daily_return = group.groupby('quantile')['forward_return'].mean()
            daily_return['date'] = date
            returns.append(daily_return)
        
        returns_df = pd.DataFrame(returns).set_index('date')
        cumulative = (1 + returns_df).cumprod()
        
        return returns_df, cumulative


class FactorResearchPipeline:
    """
    完整因子研究流水线
    Day 12: 从原始数据到选股策略的完整框架
    """
    
    def __init__(self):
        from utils.data_processor import DataProcessor
        from utils.factor_calculator import FactorPipeline
        from utils.factor_preprocessor import FactorPreprocessor
        from utils.factor_analyzer import FactorAnalyzer
        
        self.data_processor = DataProcessor()
        self.factor_calc = FactorPipeline()
        self.preprocessor = FactorPreprocessor()
        self.analyzer = FactorAnalyzer()
        self.strategy = MultiFactorStrategy()
    
    def run_full_pipeline(self, raw_df, factor_cols=None, **kwargs):
        """
        运行完整研究流程
        
        Parameters:
        -----------
        raw_df : DataFrame
            原始股票数据（含OHLCV）
        **kwargs : 配置参数
        """
        print("=" * 70)
        print("🚀 启动完整因子研究流水线")
        print("=" * 70)
        
        # Step 1: 数据预处理
        print("\n📌 Step 1: 数据预处理")
        df = self.data_processor.clean_price_data(raw_df)
        df = self.data_processor.calculate_returns(df, periods=[5, 10, 20])
        
        # Step 2: 计算因子
        print("\n📌 Step 2: 计算原始因子")
        df = self.factor_calc.calculate_all_factors(df)
        
        if factor_cols is None:
            factor_cols = self.factor_calc.get_factor_list(df)
        
        print(f"   共计算 {len(factor_cols)} 个因子")
        
        # Step 3: 因子预处理
        print("\n📌 Step 3: 因子预处理")
        df = self.preprocessor.preprocess_pipeline(
            df, factor_cols,
            winsorize_method='mad',
            standardize_method='zscore'
        )
        
        # Step 4: 因子有效性检验
        print("\n📌 Step 4: 因子有效性检验")
        factor_results = {}
        
        for factor in factor_cols[:5]:  # 只测试前5个作为示例
            print(f"\n   测试因子: {factor}")
            result = self.analyzer.generate_factor_report(
                df, factor, 
                return_col='return_5d',
                date_col='date' if 'date' in df.columns else None
            )
            factor_results[factor] = result['ic_stats']
        
        # Step 5: 多因子合成
        print("\n📌 Step 5: 多因子合成")
        selected_factors = self.strategy.select_factors(factor_results, min_ic=0.01, min_ir=0.1)
        
        if len(selected_factors) >= 2:
            composite = self.strategy.build_composite_factor(
                df, selected_factors, method='equal'
            )
            df['composite_factor'] = composite
            
            # 测试复合因子
            print("\n📌 Step 6: 复合因子效果测试")
            composite_result = self.analyzer.generate_factor_report(
                df, 'composite_factor', return_col='return_5d'
            )
        
        print("\n" + "=" * 70)
        print("✅ 因子研究流水线完成！")
        print("=" * 70)
        
        return df, factor_results


# ============== 测试代码 ==============

def test_multi_factor():
    """测试多因子合成"""
    print("=" * 70)
    print("🧪 测试多因子合成与策略")
    print("=" * 70)
    
    # 创建模拟数据
    np.random.seed(42)
    n_dates = 20
    n_stocks = 50
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    
    data = []
    for date in dates:
        for i in range(n_stocks):
            # 多个相关因子
            base = np.random.randn()
            
            f1 = base + 0.3 * np.random.randn()  # 强预测因子
            f2 = base * 0.5 + 0.5 * np.random.randn()  # 中等因子
            f3 = np.random.randn()  # 噪声因子
            
            # 未来收益与f1正相关，与f2负相关
            ret = 0.15 * f1 - 0.1 * f2 + 0.05 * np.random.randn()
            
            data.append({
                'date': date,
                'code': f'STOCK_{i:03d}',
                'factor_momentum': f1,
                'factor_value': f2,
                'factor_volatility': f3,
                'forward_return': ret
            })
    
    df = pd.DataFrame(data)
    
    # 测试不同合成方法
    factor_cols = ['factor_momentum', 'factor_value', 'factor_volatility']
    synthesizer = FactorSynthesizer()
    
    print("\n📊 测试等权合成:")
    df['composite_equal'] = synthesizer.equal_weight(df, factor_cols)
    
    print("\n📊 测试IC加权合成:")
    ic_values = {
        'factor_momentum': 0.12,
        'factor_value': -0.08,
        'factor_volatility': 0.02
    }
    df['composite_ic'] = synthesizer.ic_weighted(df, factor_cols, ic_values)
    
    # 评估合成效果
    from utils.factor_analyzer import FactorAnalyzer
    analyzer = FactorAnalyzer()
    
    print("\n" + "-" * 40)
    print("📈 合成因子效果对比:")
    
    for col in ['composite_equal', 'composite_ic']:
        ic, _ = analyzer.calculate_ic(df[col], df['forward_return'])
        print(f"   {col}: IC = {ic:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 多因子合成测试完成！")
    print("=" * 70)
    
    return df


if __name__ == '__main__':
    test_multi_factor()
