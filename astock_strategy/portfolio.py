"""
A股持仓管理
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from loguru import logger


@dataclass
class Position:
    """持仓数据结构"""
    stock_code: str          # 股票代码
    stock_name: str          # 股票名称
    entry_price: float       # 入场价格
    entry_time: float        # 入场时间戳
    entry_date: str          # 入场日期 (YYYY-MM-DD)
    volume: int              # 持仓数量
    can_sell_volume: int     # 可卖数量 (T+1)
    cost_amount: float       # 投入成本
    highest_price: float     # 历史最高价
    current_price: float     # 当前价格
    tp_level: int = 0        # 已触发的止盈档位
    is_moonbag: bool = False # 是否为留底仓位


class AStockPortfolioManager:
    """A股持仓管理器"""
    
    def __init__(self, state_file: str = "astock_portfolio.json"):
        self.state_file = state_file
        self.positions: Dict[str, Position] = {}
        self.trade_history = []  # 交易历史
        self.load_state()
    
    def add_position(self, code: str, name: str, price: float, 
                     volume: int, amount: float):
        """
        添加持仓
        
        Args:
            code: 股票代码
            name: 股票名称
            price: 入场价格
            volume: 持仓数量
            amount: 投入成本
        """
        from datetime import datetime
        
        self.positions[code] = Position(
            stock_code=code,
            stock_name=name,
            entry_price=price,
            entry_time=time.time(),
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            volume=volume,
            can_sell_volume=0,  # T+1，当日买入不可卖
            cost_amount=amount,
            highest_price=price,
            current_price=price,
        )
        
        self.save_state()
        logger.info(f"[+] 新增持仓: {name}({code}) | 价格: {price} | 数量: {volume}")
    
    def update_price(self, code: str, current_price: float):
        """
        更新价格
        
        Args:
            code: 股票代码
            current_price: 当前价格
        """
        if code in self.positions:
            pos = self.positions[code]
            pos.current_price = current_price
            
            if current_price > pos.highest_price:
                pos.highest_price = current_price
            
            self.save_state()
    
    def update_can_sell(self, code: str, can_sell_volume: int):
        """
        更新可卖数量
        
        Args:
            code: 股票代码
            can_sell_volume: 可卖数量
        """
        if code in self.positions:
            self.positions[code].can_sell_volume = can_sell_volume
            self.save_state()
    
    def reduce_position(self, code: str, sell_volume: int, sell_price: float, reason: str):
        """
        减仓
        
        Args:
            code: 股票代码
            sell_volume: 卖出数量
            sell_price: 卖出价格
            reason: 卖出原因
        """
        if code not in self.positions:
            return
        
        pos = self.positions[code]
        
        # 记录交易历史
        self.trade_history.append({
            'time': time.time(),
            'code': code,
            'name': pos.stock_name,
            'action': 'SELL',
            'volume': sell_volume,
            'price': sell_price,
            'reason': reason,
            'pnl': (sell_price - pos.entry_price) / pos.entry_price,
        })
        
        # 更新持仓
        pos.volume -= sell_volume
        pos.can_sell_volume = max(0, pos.can_sell_volume - sell_volume)
        
        if pos.volume <= 0:
            self.close_position(code, reason)
        else:
            self.save_state()
            logger.info(f"[-] 减仓: {pos.stock_name} | 卖出: {sell_volume} | 剩余: {pos.volume}")
    
    def close_position(self, code: str, reason: str = ""):
        """
        完全平仓
        
        Args:
            code: 股票代码
            reason: 平仓原因
        """
        if code in self.positions:
            pos = self.positions[code]
            logger.info(f"[X] 平仓: {pos.stock_name}({code}) | 原因: {reason}")
            del self.positions[code]
            self.save_state()
    
    def get_position(self, code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(code)
    
    def get_open_count(self) -> int:
        """获取持仓数量"""
        return len(self.positions)
    
    def get_all_codes(self) -> list:
        """获取所有持仓代码"""
        return list(self.positions.keys())
    
    def get_total_market_value(self) -> float:
        """获取总市值"""
        return sum(p.volume * p.current_price for p in self.positions.values())
    
    def get_total_cost(self) -> float:
        """获取总成本"""
        return sum(p.cost_amount for p in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return self.get_total_market_value() - self.get_total_cost()
    
    def get_total_pnl_pct(self) -> float:
        """获取总盈亏比例"""
        cost = self.get_total_cost()
        if cost <= 0:
            return 0.0
        return self.get_total_pnl() / cost
    
    def save_state(self):
        """保存状态"""
        data = {
            'positions': {k: asdict(v) for k, v in self.positions.items()},
            'trade_history': self.trade_history[-100:],  # 只保留最近 100 条
        }
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_state(self):
        """加载状态"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for k, v in data.get('positions', {}).items():
                    self.positions[k] = Position(**v)
                
                self.trade_history = data.get('trade_history', [])
                
            logger.info(f"已加载 {len(self.positions)} 个持仓")
        except FileNotFoundError:
            self.positions = {}
            self.trade_history = []

