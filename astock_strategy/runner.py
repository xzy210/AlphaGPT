"""
A股策略执行器
"""
import asyncio
import json
import os
import time
import torch
from datetime import datetime
from typing import List, Optional
from loguru import logger

from astock_pipeline.data_manager import AStockDataManager
from astock_pipeline.xtquant_provider import XtQuantProvider
from astock_model.vm import AStockStackVM
from astock_model.data_loader import AStockDataLoader
from astock_execution.trader import AStockTrader
from .config import AStockStrategyConfig
from .portfolio import AStockPortfolioManager
from .risk import AStockRiskEngine

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AStockStrategyRunner:
    """A股策略执行器"""
    
    def __init__(self):
        self.config = AStockStrategyConfig()
        
        # 数据
        self.data_mgr = AStockDataManager()
        self.data_provider = XtQuantProvider()
        self.data_loader = AStockDataLoader()
        
        # 交易
        self.trader = AStockTrader()
        self.portfolio = AStockPortfolioManager()
        self.risk = AStockRiskEngine()
        
        # 策略
        self.vm = AStockStackVM()
        self.formula = None
        
        # 状态
        self.last_sync_time = 0
        self.stock_map = {}  # {code: index}
        
        # 加载策略
        self._load_strategy()
    
    def _load_strategy(self):
        """加载策略"""
        try:
            strategy_path = os.path.join(PROJECT_ROOT, "best_astock_strategy.json")
            with open(strategy_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                self.formula = data.get('formula') or data
                
                formula_str = self.vm.decode_formula(self.formula) if isinstance(self.formula, list) else str(self.formula)
                logger.success(f"策略已加载: {formula_str}")
        except FileNotFoundError:
            logger.critical("策略文件未找到! 请先训练模型。")
            exit(1)
    
    async def initialize(self):
        """初始化"""
        logger.info("正在初始化...")
        
        # 初始化数据
        await self.data_mgr.initialize()
        self.data_provider.connect()
        
        # 初始化交易
        self.trader.connect()
        
        # 显示账户信息
        balance = self.trader.get_balance()
        logger.info(f"账户资金: 总资产 {balance['total_asset']:.2f} | "
                   f"可用 {balance['cash']:.2f}")
        
        # 保存初始状态供 Dashboard 读取
        self._save_state()
        
        logger.success("初始化完成")
    
    async def run_loop(self):
        """主循环"""
        logger.info("=" * 50)
        logger.info("🚀 A股策略执行器启动")
        logger.info("=" * 50)
        
        while True:
            try:
                loop_start = time.time()
                
                # 保存状态供 Dashboard 读取
                self._save_state()
                
                # 检查交易时间
                if not self.trader.is_trading_time():
                    logger.info("当前非交易时间，等待...")
                    await asyncio.sleep(60)
                    continue
                
                # 定期同步数据
                if time.time() - self.last_sync_time > self.config.DATA_SYNC_INTERVAL:
                    logger.info("正在同步数据...")
                    await self.data_mgr.sync_realtime_data()
                    self._reload_data()
                    self.last_sync_time = time.time()
                
                # 更新持仓可卖数量 (T+1)
                self._update_can_sell()
                
                # 监控持仓
                await self._monitor_positions()
                
                # 扫描买入机会
                if self.portfolio.get_open_count() < self.config.MAX_OPEN_POSITIONS:
                    await self._scan_for_entries()
                
                # 休眠
                elapsed = time.time() - loop_start
                sleep_time = max(10, self.config.SCAN_INTERVAL - elapsed)
                logger.debug(f"循环完成，休眠 {sleep_time:.0f} 秒")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"循环异常: {e}")
                await asyncio.sleep(30)
    
    def _reload_data(self):
        """重新加载数据"""
        self.data_loader.load_data(limit_stocks=500, lookback_days=60)
        
        # 构建映射
        codes = self.data_loader.get_all_codes()
        self.stock_map = {code: idx for idx, code in enumerate(codes)}
        
        logger.info(f"数据已加载，共 {len(codes)} 只股票")
    
    def _update_can_sell(self):
        """更新可卖数量"""
        positions = self.trader.get_positions()
        
        for pos in positions:
            code = pos['code']
            if code in self.portfolio.positions:
                self.portfolio.update_can_sell(code, pos['can_use_volume'])
    
    async def _monitor_positions(self):
        """监控持仓"""
        if not self.portfolio.positions:
            return
        
        logger.info(f"监控 {len(self.portfolio.positions)} 个持仓...")
        
        for code, pos in list(self.portfolio.positions.items()):
            # 获取实时价格
            current_price = self._get_realtime_price(code)
            if current_price <= 0:
                continue
            
            # 更新价格
            self.portfolio.update_price(code, current_price)
            
            # 计算盈亏
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            
            # === 止损检查 ===
            if self.risk.check_stop_loss(pos.entry_price, current_price):
                logger.warning(f"🔴 止损: {pos.stock_name} | 亏损: {pnl_pct:.2%}")
                await self._execute_sell(code, 1.0, "止损")
                continue
            
            # === 止盈检查 ===
            triggered, sell_ratio, new_level = self.risk.check_take_profit(
                pos.entry_price, current_price, pos.tp_level
            )
            if triggered:
                logger.success(f"🟢 止盈: {pos.stock_name} | 盈利: {pnl_pct:.2%}")
                await self._execute_sell(code, sell_ratio, f"止盈L{new_level}")
                self.portfolio.positions[code].tp_level = new_level
                continue
            
            # === 移动止损检查 ===
            if self.risk.check_trailing_stop(pos.entry_price, current_price, pos.highest_price):
                logger.warning(f"🟡 移动止损: {pos.stock_name} | 回撤触发")
                await self._execute_sell(code, 1.0, "移动止损")
                continue
            
            # === AI 信号检查 ===
            if not pos.is_moonbag:
                ai_score = self._run_inference(code)
                if ai_score != -1 and ai_score < self.config.SELL_THRESHOLD:
                    logger.info(f"🤖 AI 卖出: {pos.stock_name} | 信号: {ai_score:.2f}")
                    await self._execute_sell(code, 1.0, "AI信号")
    
    async def _scan_for_entries(self):
        """扫描买入机会"""
        if self.data_loader.feat_tensor is None:
            return
        
        # 执行公式
        signals = self.vm.execute(self.formula, self.data_loader.feat_tensor)
        if signals is None:
            return
        
        # 获取最新信号
        latest_signals = signals[:, -1]
        scores = torch.sigmoid(latest_signals).cpu().numpy()
        
        # 排序
        sorted_indices = scores.argsort()[::-1]
        
        # 反向映射
        idx_to_code = {v: k for k, v in self.stock_map.items()}
        
        for idx in sorted_indices:
            score = float(scores[idx])
            
            if score < self.config.BUY_THRESHOLD:
                break
            
            code = idx_to_code.get(idx)
            if not code:
                continue
            
            # 跳过已持有
            if code in self.portfolio.positions:
                continue
            
            # 获取股票信息
            stock_info = self._get_stock_info(code)
            if not stock_info:
                continue
            
            logger.info(f"🔍 候选: {stock_info.get('name', code)} | "
                       f"信号: {score:.2f}")
            
            # 风控检查
            if not self.risk.check_stock_safety(stock_info):
                continue
            
            # 执行买入
            await self._execute_buy(code, stock_info, score)
            
            # 检查是否已满仓
            if self.portfolio.get_open_count() >= self.config.MAX_OPEN_POSITIONS:
                break
    
    async def _execute_buy(self, code: str, stock_info: dict, score: float):
        """执行买入"""
        # 计算仓位
        balance = self.trader.get_balance()
        position_value = self.portfolio.get_total_market_value()
        
        amount = self.risk.calculate_position_size(
            balance['cash'],
            self.portfolio.get_open_count(),
            position_value
        )
        
        if amount < 5000:
            logger.warning("可用资金不足")
            return
        
        current_price = stock_info.get('price', 0)
        if current_price <= 0:
            return
        
        # 计算数量
        volume = int(amount / current_price)
        volume = (volume // 100) * 100
        
        if volume < 100:
            return
        
        # 执行买入
        logger.info(f"🎯 买入: {stock_info.get('name', code)} | "
                   f"金额: {amount:.0f} | 数量: {volume}")
        
        order_id = self.trader.buy_by_amount(code, amount, current_price)
        
        if order_id:
            # 记录持仓
            actual_amount = volume * current_price
            self.portfolio.add_position(
                code=code,
                name=stock_info.get('name', code),
                price=current_price,
                volume=volume,
                amount=actual_amount
            )
            logger.success(f"✅ 买入成功: {stock_info.get('name', code)}")
    
    async def _execute_sell(self, code: str, ratio: float, reason: str):
        """执行卖出"""
        pos = self.portfolio.get_position(code)
        if not pos:
            return
        
        # 检查可卖数量
        if pos.can_sell_volume <= 0:
            logger.warning(f"T+1 限制，今日不可卖出: {pos.stock_name}")
            return
        
        # 计算卖出数量
        sell_volume = int(pos.can_sell_volume * ratio)
        sell_volume = (sell_volume // 100) * 100
        
        if sell_volume < 100:
            sell_volume = pos.can_sell_volume  # 不足 100 股，全部卖出
        
        logger.info(f"📤 卖出: {pos.stock_name} | 数量: {sell_volume} | 原因: {reason}")
        
        order_id = self.trader.sell_all(code, pos.current_price)
        
        if order_id:
            self.portfolio.reduce_position(code, sell_volume, pos.current_price, reason)
            logger.success(f"✅ 卖出成功: {pos.stock_name}")
    
    def _run_inference(self, code: str) -> float:
        """运行推理"""
        idx = self.stock_map.get(code)
        if idx is None:
            return -1
        
        if self.data_loader.feat_tensor is None:
            return -1
        
        features = self.data_loader.feat_tensor[idx:idx+1]
        result = self.vm.execute(self.formula, features)
        
        if result is None:
            return -1
        
        latest_score = torch.sigmoid(result[0, -1]).item()
        return latest_score
    
    def _get_realtime_price(self, code: str) -> float:
        """获取实时价格"""
        try:
            df = self.data_provider.get_market_data([code])
            if not df.empty:
                return float(df.iloc[0]['price'])
        except:
            pass
        return 0.0
    
    def _get_stock_info(self, code: str) -> Optional[dict]:
        """获取股票信息"""
        try:
            df = self.data_provider.get_market_data([code])
            if not df.empty:
                row = df.iloc[0]
                detail = self.data_provider.get_instrument_detail(code)
                
                return {
                    'code': code,
                    'name': detail.get('name', '') if detail else '',
                    'price': float(row.get('price', 0)),
                    'market_cap': float(row.get('market_cap', 0)),
                    'turnover': float(row.get('turnover', 0)) if 'turnover' in row else 0.01,
                    'pct_chg': float(row.get('pct_chg', 0)) if 'pct_chg' in row else 0,
                    'list_date': detail.get('list_date') if detail else None,
                }
        except Exception as e:
            logger.debug(f"获取 {code} 信息失败: {e}")
        return None
    
    def _save_state(self):
        """保存状态到文件，供 Dashboard 读取"""
        try:
            # 保存账户状态
            balance = self.trader.get_balance()
            account_state = {
                "total_asset": balance.get('total_asset', 0),
                "cash": balance.get('cash', 0),
                "market_value": balance.get('market_value', 0),
                "frozen": balance.get('frozen_cash', 0),
                "profit": 0,  # 需要计算
                "profit_pct": 0,
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            account_path = os.path.join(PROJECT_ROOT, "astock_account_state.json")
            with open(account_path, "w", encoding="utf-8") as f:
                json.dump(account_state, f, ensure_ascii=False, indent=2)
            
            # 保存持仓状态
            portfolio_state = {}
            for code, pos in self.portfolio.positions.items():
                portfolio_state[code] = {
                    "code": code,
                    "name": pos.stock_name,
                    "volume": pos.volume,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "amount": pos.amount,
                    "can_sell_volume": pos.can_sell_volume,
                    "entry_time": pos.entry_time.strftime("%Y-%m-%d %H:%M:%S") if pos.entry_time else "",
                }
            
            portfolio_path = os.path.join(PROJECT_ROOT, "astock_portfolio_state.json")
            with open(portfolio_path, "w", encoding="utf-8") as f:
                json.dump(portfolio_state, f, ensure_ascii=False, indent=2)
                
            logger.debug("状态已保存")
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")

    async def shutdown(self):
        """关闭"""
        logger.info("正在关闭...")
        self._save_state()  # 关闭前保存状态
        await self.data_mgr.close()
        self.data_provider.disconnect()
        self.trader.disconnect()
        logger.info("已关闭")


async def run_astock_strategy():
    """运行A股策略"""
    runner = AStockStrategyRunner()
    
    try:
        await runner.initialize()
        await runner.run_loop()
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(run_astock_strategy())

