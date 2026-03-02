"""
常量配置模块测试
测试 utils/constants.py 中的常量定义
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import constants


class TestTradingDaysConstants:
    """测试交易日相关常量"""
    
    def test_trading_days_per_year(self):
        """测试年均交易日常量"""
        assert hasattr(constants, 'TRADING_DAYS_PER_YEAR')
        assert constants.TRADING_DAYS_PER_YEAR == 252
        assert isinstance(constants.TRADING_DAYS_PER_YEAR, int)
    
    def test_trading_days_per_month(self):
        """测试月均交易日常量"""
        assert hasattr(constants, 'TRADING_DAYS_PER_MONTH')
        assert constants.TRADING_DAYS_PER_MONTH == 21
        assert isinstance(constants.TRADING_DAYS_PER_MONTH, int)
    
    def test_trading_days_per_week(self):
        """测试周均交易日常量"""
        assert hasattr(constants, 'TRADING_DAYS_PER_WEEK')
        assert constants.TRADING_DAYS_PER_WEEK == 5
        assert isinstance(constants.TRADING_DAYS_PER_WEEK, int)
    
    def test_trading_days_consistency(self):
        """测试交易日常量之间的一致性"""
        # 年均交易日 ≈ 月均交易日 * 12
        assert constants.TRADING_DAYS_PER_YEAR == constants.TRADING_DAYS_PER_MONTH * 12
        # 月均交易日 ≈ 周均交易日 * 4.2
        assert abs(constants.TRADING_DAYS_PER_MONTH - constants.TRADING_DAYS_PER_WEEK * 4.2) < 2


class TestTradingCostConstants:
    """测试交易费用相关常量"""
    
    def test_default_commission_rate(self):
        """测试默认佣金率"""
        assert hasattr(constants, 'DEFAULT_COMMISSION_RATE')
        assert constants.DEFAULT_COMMISSION_RATE == 0.001
        assert isinstance(constants.DEFAULT_COMMISSION_RATE, float)
        # 佣金率应在合理范围（0-1%）
        assert 0 <= constants.DEFAULT_COMMISSION_RATE <= 0.01
    
    def test_default_slippage(self):
        """测试默认滑点"""
        assert hasattr(constants, 'DEFAULT_SLIPPAGE')
        assert constants.DEFAULT_SLIPPAGE == 0.001
        assert isinstance(constants.DEFAULT_SLIPPAGE, float)
        # 滑点应在合理范围（0-1%）
        assert 0 <= constants.DEFAULT_SLIPPAGE <= 0.01
    
    def test_default_tax_rate(self):
        """测试默认印花税率"""
        assert hasattr(constants, 'DEFAULT_TAX_RATE')
        assert constants.DEFAULT_TAX_RATE == 0.001
        assert isinstance(constants.DEFAULT_TAX_RATE, float)
        # 印花税率应在合理范围（0-1%）
        assert 0 <= constants.DEFAULT_TAX_RATE <= 0.01
    
    def test_trading_costs_reasonable(self):
        """测试交易成本合理性"""
        total_cost = (constants.DEFAULT_COMMISSION_RATE + 
                     constants.DEFAULT_SLIPPAGE + 
                     constants.DEFAULT_TAX_RATE)
        # 单边总成本应小于 1%
        assert total_cost < 0.01, "单边总交易成本应小于 1%"


class TestDataConstants:
    """测试数据相关常量"""
    
    def test_default_cache_days(self):
        """测试默认缓存天数"""
        assert hasattr(constants, 'DEFAULT_CACHE_DAYS')
        assert constants.DEFAULT_CACHE_DAYS == 7
        assert isinstance(constants.DEFAULT_CACHE_DAYS, int)
        # 缓存天数应为正数
        assert constants.DEFAULT_CACHE_DAYS > 0


class TestConstantsImport:
    """测试常量模块导入"""
    
    def test_import_constants_module(self):
        """测试导入常量模块"""
        try:
            from utils import constants as const
            assert const is not None
        except ImportError as e:
            pytest.fail(f"导入常量模块失败: {e}")
    
    def test_import_specific_constants(self):
        """测试导入特定常量"""
        from utils.constants import (
            TRADING_DAYS_PER_YEAR,
            TRADING_DAYS_PER_MONTH,
            TRADING_DAYS_PER_WEEK,
            DEFAULT_COMMISSION_RATE,
            DEFAULT_SLIPPAGE,
            DEFAULT_TAX_RATE,
            DEFAULT_CACHE_DAYS
        )
        
        assert TRADING_DAYS_PER_YEAR == 252
        assert TRADING_DAYS_PER_MONTH == 21
        assert TRADING_DAYS_PER_WEEK == 5
        assert DEFAULT_COMMISSION_RATE == 0.001
        assert DEFAULT_SLIPPAGE == 0.001
        assert DEFAULT_TAX_RATE == 0.001
        assert DEFAULT_CACHE_DAYS == 7
    
    def test_all_constants_defined(self):
        """测试所有预期常量都已定义"""
        expected_constants = [
            'TRADING_DAYS_PER_YEAR',
            'TRADING_DAYS_PER_MONTH',
            'TRADING_DAYS_PER_WEEK',
            'DEFAULT_COMMISSION_RATE',
            'DEFAULT_SLIPPAGE',
            'DEFAULT_TAX_RATE',
            'DEFAULT_CACHE_DAYS'
        ]
        
        for const_name in expected_constants:
            assert hasattr(constants, const_name), f"缺少常量: {const_name}"


class TestConstantsImmutability:
    """测试常量不可变性（约定俗成）"""
    
    def test_constant_naming_convention(self):
        """测试常量命名规范（全大写）"""
        import re
        
        for name in dir(constants):
            if not name.startswith('_'):  # 忽略私有属性
                value = getattr(constants, name)
                # 数值型常量应使用全大写下划线命名
                if isinstance(value, (int, float)):
                    assert re.match(r'^[A-Z][A-Z_0-9]*$', name), \
                        f"常量 {name} 应使用全大写下划线命名"
