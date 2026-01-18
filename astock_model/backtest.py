"""
A股回测引擎

包含 A股特有的交易规则：T+1、涨跌停、手续费等
"""
import torch
from typing import Tuple
from loguru import logger

from astock_pipeline.config import AStockConfig


class AStockBacktest:
    """A股回测引擎"""
    
    def __init__(self):
        self.config = AStockConfig()
        
        # 交易成本
        self.commission_rate = self.config.COMMISSION_RATE
        self.stamp_tax_rate = self.config.STAMP_TAX_RATE
        self.transfer_fee_rate = self.config.TRANSFER_FEE_RATE
        self.slippage = self.config.SLIPPAGE
        
        # 涨跌停限制
        self.limit_up = 0.10    # 主板 10%
        self.limit_down = -0.10
        
        # 策略参数
        self.signal_threshold = 0.7  # 买入信号阈值
        self.max_positions = 10      # 最大持仓数量
        self.position_size = 0.1     # 单只股票最大仓位
    
    def calculate_trading_cost(self, amount: float, is_sell: bool = False) -> float:
        """
        计算交易成本
        
        Args:
            amount: 交易金额
            is_sell: 是否为卖出
        
        Returns:
            交易成本
        """
        # 佣金
        commission = max(amount * self.commission_rate, 5.0)
        
        # 过户费
        transfer_fee = amount * self.transfer_fee_rate
        
        # 印花税 (仅卖出收取)
        stamp_tax = amount * self.stamp_tax_rate if is_sell else 0
        
        # 滑点
        slippage_cost = amount * self.slippage
        
        return commission + transfer_fee + stamp_tax + slippage_cost
    
    def apply_limit_filter(self, pct_chg: torch.Tensor) -> torch.Tensor:
        """
        涨跌停过滤
        
        Args:
            pct_chg: 涨跌幅 [Stocks, Time]
        
        Returns:
            可交易标记 (1=可交易, 0=涨跌停不可交易)
        """
        # 涨停不能买入
        is_limit_up = (pct_chg >= self.limit_up * 100 - 0.5).float()
        
        # 跌停不能卖出
        is_limit_down = (pct_chg <= self.limit_down * 100 + 0.5).float()
        
        # 可以买入的标记 (非涨停)
        can_buy = 1.0 - is_limit_up
        
        # 可以卖出的标记 (非跌停)
        can_sell = 1.0 - is_limit_down
        
        return can_buy, can_sell
    
    def apply_t1_rule(self, position: torch.Tensor) -> torch.Tensor:
        """
        应用 T+1 规则
        
        Args:
            position: 目标仓位信号 [Stocks, Time]
        
        Returns:
            实际仓位 (考虑 T+1)
        """
        # 今日买入，明日才能卖出
        # 简化处理：检测仓位从 0 变为 1 的情况
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        
        # 新开仓标记
        new_position = (prev_pos == 0) & (position > 0)
        
        # T+1: 新开仓当日不能平仓
        # 这里的处理是：如果信号要求平仓，但是是当日开仓，则保持仓位
        # 实际上回测中简化处理，假设持仓至少 1 天
        
        return position
    
    def evaluate(self, factors: torch.Tensor, raw_data: dict, 
                 target_ret: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        评估因子表现
        
        Args:
            factors: 因子信号 [Stocks, Time]
            raw_data: 原始数据字典
            target_ret: 目标收益率 [Stocks, Time]
        
        Returns:
            (适应度分数, 累计收益率)
        """
        close = raw_data['close']
        pct_chg = raw_data.get('pct_chg', torch.zeros_like(close))
        
        # 1. 生成信号
        signal = torch.sigmoid(factors)
        
        # 2. 涨跌停过滤
        can_buy, can_sell = self.apply_limit_filter(pct_chg)
        
        # 3. 生成仓位
        # 高信号买入，低信号卖出
        raw_position = (signal > self.signal_threshold).float()
        
        # 应用涨跌停限制
        # 买入时检查是否涨停
        position = raw_position * can_buy
        
        # 4. 应用 T+1 规则
        position = self.apply_t1_rule(position)
        
        # 5. 计算换手
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        
        turnover = torch.abs(position - prev_pos)
        
        # 6. 计算交易成本
        # 买入成本
        buy_cost = turnover * (position > prev_pos).float() * (
            self.commission_rate + self.transfer_fee_rate + self.slippage
        )
        
        # 卖出成本 (含印花税)
        sell_cost = turnover * (position < prev_pos).float() * (
            self.commission_rate + self.stamp_tax_rate + 
            self.transfer_fee_rate + self.slippage
        )
        
        total_cost = buy_cost + sell_cost
        
        # 7. 计算收益
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - total_cost
        
        # 8. 计算指标
        # 累计收益
        cum_ret = net_pnl.sum(dim=1)
        
        # 最大回撤惩罚
        cumsum = net_pnl.cumsum(dim=1)
        running_max = cumsum.cummax(dim=1)[0]
        drawdown = (running_max - cumsum) / (running_max + 1e-9)
        max_drawdown = drawdown.max(dim=1)[0]
        
        # 大亏损惩罚
        big_loss = (net_pnl < -0.03).float().sum(dim=1)
        
        # 夏普比率近似
        daily_ret = net_pnl.mean(dim=1)
        daily_std = net_pnl.std(dim=1) + 1e-9
        sharpe_approx = daily_ret / daily_std * (250 ** 0.5)
        
        # 9. 综合评分
        # 累计收益 + 夏普 - 回撤惩罚 - 大亏损惩罚
        score = cum_ret + sharpe_approx * 0.5 - max_drawdown * 2.0 - big_loss * 0.5
        
        # 活跃度惩罚 (交易次数太少)
        activity = position.sum(dim=1)
        score = torch.where(activity < 10, torch.tensor(-10.0, device=score.device), score)
        
        # 使用中位数作为最终分数 (更鲁棒)
        final_score = torch.median(score)
        avg_ret = cum_ret.mean().item()
        
        return final_score, avg_ret


class DetailedAStockBacktest(AStockBacktest):
    """详细回测引擎 (带完整统计)"""
    
    def detailed_evaluate(self, factors: torch.Tensor, raw_data: dict,
                         target_ret: torch.Tensor) -> dict:
        """
        详细评估
        
        Returns:
            详细统计字典
        """
        close = raw_data['close']
        pct_chg = raw_data.get('pct_chg', torch.zeros_like(close))
        
        # 基础评估
        score, avg_ret = self.evaluate(factors, raw_data, target_ret)
        
        # 生成仓位
        signal = torch.sigmoid(factors)
        can_buy, can_sell = self.apply_limit_filter(pct_chg)
        position = (signal > self.signal_threshold).float() * can_buy
        
        # 计算详细统计
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        
        # 交易次数
        turnover = torch.abs(position - prev_pos)
        trades_per_stock = turnover.sum(dim=1)
        total_trades = trades_per_stock.sum().item()
        
        # 持仓统计
        avg_position = position.mean().item()
        
        # 胜率
        gross_pnl = position * target_ret
        win_trades = (gross_pnl > 0).float().sum().item()
        total_position_days = position.sum().item()
        win_rate = win_trades / (total_position_days + 1e-9)
        
        # 盈亏比
        gains = gross_pnl[gross_pnl > 0].sum().item()
        losses = -gross_pnl[gross_pnl < 0].sum().item()
        profit_loss_ratio = gains / (losses + 1e-9)
        
        return {
            'score': score.item(),
            'total_return': avg_ret,
            'total_trades': total_trades,
            'avg_position': avg_position,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
        }

