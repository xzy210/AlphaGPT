"""
A股交易执行器

使用 xtquant/miniqmt 进行交易
"""
import time
from datetime import datetime
from typing import Optional, Dict, List
from loguru import logger

try:
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount
    from xtquant import xtconstant
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False
    logger.warning("xtquant 未安装")

from .config import AStockExecutionConfig


class OrderCallback:
    """订单回调"""
    
    def __init__(self, trader):
        self.trader = trader
    
    def on_disconnected(self):
        logger.warning("交易连接断开")
    
    def on_stock_order(self, order):
        """订单回报"""
        logger.info(f"订单回报: {order.stock_code} | "
                   f"状态: {order.order_status} | "
                   f"成交: {order.traded_volume}/{order.order_volume}")
    
    def on_stock_trade(self, trade):
        """成交回报"""
        logger.success(f"成交回报: {trade.stock_code} | "
                      f"价格: {trade.traded_price} | "
                      f"数量: {trade.traded_volume}")
    
    def on_order_error(self, order_id, error_id, error_msg):
        """订单错误"""
        logger.error(f"订单错误: {order_id} | {error_id} | {error_msg}")
    
    def on_order_stock_async_response(self, response):
        """异步下单响应"""
        logger.info(f"异步响应: {response}")


class AStockTrader:
    """A股交易执行器"""
    
    def __init__(self):
        self.config = AStockExecutionConfig()
        self.trader = None
        self.account = None
        self._connected = False
        self._orders = {}  # 订单缓存
    
    def connect(self):
        """连接交易服务"""
        if not XTQUANT_AVAILABLE:
            raise ImportError("xtquant 未安装")
        
        try:
            # 创建交易对象
            session_id = int(time.time())
            self.trader = XtQuantTrader(self.config.QMT_PATH, session_id)
            
            # 注册回调
            callback = OrderCallback(self)
            self.trader.register_callback(callback)
            
            # 启动交易线程
            self.trader.start()
            
            # 连接
            result = self.trader.connect()
            if result != 0:
                raise ConnectionError(f"连接失败: {result}")
            
            # 创建账户对象
            self.account = StockAccount(self.config.ACCOUNT_ID)
            
            # 订阅账户
            subscribe_result = self.trader.subscribe(self.account)
            if subscribe_result != 0:
                raise ConnectionError(f"订阅账户失败: {subscribe_result}")
            
            self._connected = True
            logger.success(f"已连接交易服务: {self.config.ACCOUNT_ID}")
            
        except Exception as e:
            logger.error(f"连接交易服务失败: {e}")
            raise
    
    def disconnect(self):
        """断开连接"""
        if self.trader and self._connected:
            try:
                self.trader.stop()
                self._connected = False
                logger.info("已断开交易服务")
            except:
                pass
    
    def is_trading_time(self) -> bool:
        """检查是否在交易时间"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        for start, end in self.config.TRADING_HOURS:
            if start <= current_time <= end:
                return True
        
        return False
    
    def get_balance(self) -> Dict:
        """
        获取账户资金
        
        Returns:
            资金信息字典
        """
        if not self._connected:
            self.connect()
        
        try:
            asset = self.trader.query_stock_asset(self.account)
            if asset:
                return {
                    'total_asset': asset.total_asset,       # 总资产
                    'cash': asset.cash,                     # 可用资金
                    'frozen_cash': asset.frozen_cash,       # 冻结资金
                    'market_value': asset.market_value,     # 持仓市值
                }
        except Exception as e:
            logger.error(f"查询资金失败: {e}")
        
        return {'total_asset': 0, 'cash': 0, 'frozen_cash': 0, 'market_value': 0}
    
    def get_positions(self) -> List[Dict]:
        """
        获取持仓
        
        Returns:
            持仓列表
        """
        if not self._connected:
            self.connect()
        
        positions = []
        
        try:
            pos_list = self.trader.query_stock_positions(self.account)
            if pos_list:
                for pos in pos_list:
                    positions.append({
                        'code': pos.stock_code,
                        'volume': pos.volume,              # 持仓数量
                        'can_use_volume': pos.can_use_volume,  # 可卖数量
                        'avg_price': pos.avg_price,        # 成本价
                        'market_value': pos.market_value,  # 市值
                    })
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
        
        return positions
    
    def buy(self, stock_code: str, price: float, volume: int, 
            price_type: int = None) -> Optional[int]:
        """
        买入股票
        
        Args:
            stock_code: 股票代码 (如 '000001.SZ')
            price: 价格 (限价单使用)
            volume: 数量 (必须是 100 的整数倍)
            price_type: 价格类型
        
        Returns:
            订单ID，失败返回 None
        """
        if not self._connected:
            self.connect()
        
        # 检查交易时间
        if not self.is_trading_time():
            logger.warning("当前非交易时间")
            return None
        
        # 数量检查 (必须是 100 的整数倍)
        volume = (volume // 100) * 100
        if volume < 100:
            logger.warning(f"买入数量不足 100 股: {volume}")
            return None
        
        # 价格类型
        if price_type is None:
            price_type = self.config.DEFAULT_PRICE_TYPE
        
        try:
            logger.info(f"买入 {stock_code} | 价格: {price} | 数量: {volume}")
            
            order_id = self.trader.order_stock(
                self.account,
                stock_code,
                xtconstant.STOCK_BUY,
                volume,
                price_type,
                price
            )
            
            if order_id > 0:
                self._orders[order_id] = {
                    'code': stock_code,
                    'direction': 'BUY',
                    'price': price,
                    'volume': volume,
                    'time': datetime.now(),
                }
                logger.success(f"买入订单已提交: {order_id}")
                return order_id
            else:
                logger.error(f"买入订单提交失败: {order_id}")
                return None
                
        except Exception as e:
            logger.error(f"买入失败: {e}")
            return None
    
    def sell(self, stock_code: str, price: float, volume: int,
             price_type: int = None) -> Optional[int]:
        """
        卖出股票
        
        Args:
            stock_code: 股票代码
            price: 价格
            volume: 数量
            price_type: 价格类型
        
        Returns:
            订单ID
        """
        if not self._connected:
            self.connect()
        
        # 检查交易时间
        if not self.is_trading_time():
            logger.warning("当前非交易时间")
            return None
        
        # 检查可卖数量
        positions = self.get_positions()
        pos = next((p for p in positions if p['code'] == stock_code), None)
        
        if not pos:
            logger.warning(f"未持有 {stock_code}")
            return None
        
        if pos['can_use_volume'] < volume:
            logger.warning(f"可卖数量不足: {pos['can_use_volume']} < {volume}")
            volume = pos['can_use_volume']
        
        if volume < 100:
            logger.warning(f"卖出数量不足 100 股")
            return None
        
        # 价格类型
        if price_type is None:
            price_type = self.config.DEFAULT_PRICE_TYPE
        
        try:
            logger.info(f"卖出 {stock_code} | 价格: {price} | 数量: {volume}")
            
            order_id = self.trader.order_stock(
                self.account,
                stock_code,
                xtconstant.STOCK_SELL,
                volume,
                price_type,
                price
            )
            
            if order_id > 0:
                self._orders[order_id] = {
                    'code': stock_code,
                    'direction': 'SELL',
                    'price': price,
                    'volume': volume,
                    'time': datetime.now(),
                }
                logger.success(f"卖出订单已提交: {order_id}")
                return order_id
            else:
                logger.error(f"卖出订单提交失败: {order_id}")
                return None
                
        except Exception as e:
            logger.error(f"卖出失败: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        撤销订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            是否成功
        """
        if not self._connected:
            return False
        
        try:
            result = self.trader.cancel_order_stock(self.account, order_id)
            if result == 0:
                logger.info(f"订单已撤销: {order_id}")
                return True
            else:
                logger.warning(f"撤单失败: {result}")
                return False
        except Exception as e:
            logger.error(f"撤单异常: {e}")
            return False
    
    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """
        查询订单状态
        
        Args:
            order_id: 订单ID
        
        Returns:
            订单状态字典
        """
        if not self._connected:
            return None
        
        try:
            orders = self.trader.query_stock_orders(self.account)
            if orders:
                for order in orders:
                    if order.order_id == order_id:
                        return {
                            'order_id': order.order_id,
                            'code': order.stock_code,
                            'status': order.order_status,
                            'order_volume': order.order_volume,
                            'traded_volume': order.traded_volume,
                            'traded_price': order.traded_price,
                        }
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
        
        return None
    
    def buy_by_amount(self, stock_code: str, amount: float, 
                      current_price: float) -> Optional[int]:
        """
        按金额买入
        
        Args:
            stock_code: 股票代码
            amount: 金额
            current_price: 当前价格
        
        Returns:
            订单ID
        """
        # 计算数量
        volume = int(amount / current_price)
        volume = (volume // 100) * 100
        
        if volume < 100:
            logger.warning(f"金额不足买入 100 股")
            return None
        
        # 买入价格加一点溢价
        buy_price = round(current_price * 1.003, 2)
        
        return self.buy(stock_code, buy_price, volume)
    
    def sell_all(self, stock_code: str, current_price: float) -> Optional[int]:
        """
        全部卖出
        
        Args:
            stock_code: 股票代码
            current_price: 当前价格
        
        Returns:
            订单ID
        """
        positions = self.get_positions()
        pos = next((p for p in positions if p['code'] == stock_code), None)
        
        if not pos or pos['can_use_volume'] <= 0:
            logger.warning(f"没有可卖出的 {stock_code}")
            return None
        
        # 卖出价格减一点折价
        sell_price = round(current_price * 0.997, 2)
        
        return self.sell(stock_code, sell_price, pos['can_use_volume'])

