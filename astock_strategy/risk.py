"""
A股风控引擎
"""
from typing import Optional
from loguru import logger
from datetime import datetime

from .config import AStockStrategyConfig


class AStockRiskEngine:
    """A股风控引擎"""
    
    def __init__(self):
        self.config = AStockStrategyConfig()
    
    def check_stock_safety(self, stock_info: dict) -> bool:
        """
        检查股票是否安全
        
        Args:
            stock_info: 股票信息字典
                - code: 股票代码
                - name: 股票名称
                - market_cap: 市值
                - turnover: 换手率
                - pct_chg: 涨跌幅
                - list_date: 上市日期
        
        Returns:
            是否安全
        """
        code = stock_info.get('code', '')
        name = stock_info.get('name', '')
        
        # 排除 ST 股票
        if self.config.AVOID_ST and 'ST' in name:
            logger.debug(f"风控: {name} 是 ST 股票，跳过")
            return False
        
        # 市值检查
        market_cap = stock_info.get('market_cap', 0)
        if market_cap < self.config.MIN_MARKET_CAP:
            logger.debug(f"风控: {name} 市值过小 ({market_cap/1e8:.1f}亿)")
            return False
        
        # 换手率检查
        turnover = stock_info.get('turnover', 0)
        if turnover < self.config.MIN_TURNOVER:
            logger.debug(f"风控: {name} 换手率过低 ({turnover*100:.2f}%)")
            return False
        
        # 涨跌停检查 (涨停不买)
        pct_chg = stock_info.get('pct_chg', 0)
        if pct_chg >= 9.5:  # 接近涨停
            logger.debug(f"风控: {name} 接近涨停 ({pct_chg:.2f}%)")
            return False
        
        # 新股检查
        list_date = stock_info.get('list_date')
        if list_date and self.config.AVOID_NEW_DAYS > 0:
            try:
                if isinstance(list_date, str):
                    list_dt = datetime.strptime(list_date, '%Y-%m-%d')
                else:
                    list_dt = list_date
                
                days = (datetime.now() - list_dt).days
                if days < self.config.AVOID_NEW_DAYS:
                    logger.debug(f"风控: {name} 是次新股 (上市 {days} 天)")
                    return False
            except:
                pass
        
        return True
    
    def calculate_position_size(self, balance: float, current_positions: int,
                                total_position_value: float) -> float:
        """
        计算仓位大小
        
        Args:
            balance: 可用资金
            current_positions: 当前持仓数量
            total_position_value: 当前持仓市值
        
        Returns:
            建议的入场金额
        """
        # 检查持仓数量上限
        if current_positions >= self.config.MAX_OPEN_POSITIONS:
            logger.debug("风控: 已达到最大持仓数量")
            return 0.0
        
        # 检查总仓位上限
        total_asset = balance + total_position_value
        max_position_amount = total_asset * self.config.MAX_TOTAL_RATIO
        
        if total_position_value >= max_position_amount:
            logger.debug("风控: 已达到总仓位上限")
            return 0.0
        
        # 单只股票最大仓位
        max_single = total_asset * self.config.MAX_SINGLE_RATIO
        
        # 默认入场金额
        entry_amount = self.config.ENTRY_AMOUNT
        
        # 取最小值
        entry_amount = min(entry_amount, max_single, balance)
        
        # 预留一些资金
        if balance - entry_amount < 5000:
            entry_amount = max(0, balance - 5000)
        
        return entry_amount
    
    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
        
        Returns:
            是否触发止损
        """
        pnl_pct = (current_price - entry_price) / entry_price
        return pnl_pct <= self.config.STOP_LOSS_PCT
    
    def check_take_profit(self, entry_price: float, current_price: float, 
                          tp_level: int) -> tuple:
        """
        检查是否触发止盈
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            tp_level: 当前止盈档位
        
        Returns:
            (是否触发, 卖出比例, 新档位)
        """
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 第一档止盈
        if tp_level < 1 and pnl_pct >= self.config.TP_LEVEL_1:
            return True, self.config.TP_LEVEL_1_RATIO, 1
        
        # 第二档止盈
        if tp_level < 2 and pnl_pct >= self.config.TP_LEVEL_2:
            return True, self.config.TP_LEVEL_2_RATIO, 2
        
        return False, 0.0, tp_level
    
    def check_trailing_stop(self, entry_price: float, current_price: float,
                           highest_price: float) -> bool:
        """
        检查是否触发移动止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            highest_price: 历史最高价
        
        Returns:
            是否触发移动止损
        """
        # 计算最大涨幅
        max_gain = (highest_price - entry_price) / entry_price
        
        # 是否达到启动条件
        if max_gain < self.config.TRAILING_ACTIVATION:
            return False
        
        # 计算从最高点的回撤
        drawdown = (highest_price - current_price) / highest_price
        
        return drawdown >= self.config.TRAILING_DROP

