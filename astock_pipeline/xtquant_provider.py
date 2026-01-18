"""
XtQuant 数据提供者

使用 xtquant (MiniQMT) 获取A股行情数据

注意：
- xtdata 模块不需要显式调用 connect()，会自动连接本地 MiniQMT
- 使用前需要确保 MiniQMT 客户端已启动
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger

try:
    from xtquant import xtdata
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False
    logger.warning("xtquant 未安装，请先安装: pip install xtquant")

from .config import AStockConfig


class XtQuantProvider:
    """XtQuant 数据提供者"""
    
    def __init__(self):
        self.config = AStockConfig()
        self._connected = False
        
    def connect(self):
        """
        初始化 XtQuant 数据服务
        
        注意：xtdata 不需要显式 connect，但我们在这里验证服务是否可用
        """
        if not XTQUANT_AVAILABLE:
            raise ImportError("xtquant 未安装")
        
        try:
            # 测试连接：尝试获取股票列表来验证服务是否可用
            test_stocks = xtdata.get_stock_list_in_sector("上证A股")
            if test_stocks is not None:
                self._connected = True
                logger.success(f"XtQuant 数据服务可用，获取到 {len(test_stocks)} 只上证股票")
            else:
                raise ConnectionError("无法获取股票列表，请确保 MiniQMT 已启动")
        except Exception as e:
            logger.error(f"XtQuant 初始化失败: {e}")
            logger.error("请确保：1) MiniQMT 客户端已启动  2) 已登录账户")
            raise
    
    def disconnect(self):
        """断开连接 (xtdata 无需显式断开)"""
        self._connected = False
        logger.info("已标记 XtQuant 连接为关闭")
    
    def get_stock_list(self) -> List[Dict]:
        """
        获取股票列表
        
        Returns:
            股票列表，包含 code, name, market 等信息
        """
        if not self._connected:
            self.connect()
        
        stocks = []
        
        # 获取上证股票
        if "SH" in self.config.MARKET:
            sh_stocks = xtdata.get_stock_list_in_sector("上证A股")
            for code in sh_stocks:
                stocks.append({
                    'code': code,
                    'market': 'SH'
                })
        
        # 获取深证股票
        if "SZ" in self.config.MARKET:
            sz_stocks = xtdata.get_stock_list_in_sector("深证A股")
            for code in sz_stocks:
                stocks.append({
                    'code': code,
                    'market': 'SZ'
                })
        
        logger.info(f"获取到 {len(stocks)} 只股票")
        return stocks
    
    def get_instrument_detail(self, stock_code: str) -> Optional[Dict]:
        """
        获取股票详细信息
        
        Args:
            stock_code: 股票代码 (如 '000001.SZ')
        
        Returns:
            股票信息字典
        """
        if not XTQUANT_AVAILABLE:
            return None
        
        try:
            detail = xtdata.get_instrument_detail(stock_code)
            if detail:
                return {
                    'code': stock_code,
                    'name': detail.get('InstrumentName', ''),
                    'market': 'SH' if stock_code.endswith('.SH') else 'SZ',
                    'list_date': detail.get('OpenDate', ''),
                    'industry': detail.get('IndustryName', ''),
                }
        except Exception as e:
            logger.warning(f"获取 {stock_code} 详情失败: {e}")
        
        return None
    
    def get_market_data(self, stock_codes: List[str], fields: List[str] = None) -> pd.DataFrame:
        """
        获取实时行情数据
        
        Args:
            stock_codes: 股票代码列表
            fields: 字段列表
        
        Returns:
            行情数据 DataFrame
        """
        if not XTQUANT_AVAILABLE:
            return pd.DataFrame()
        
        if fields is None:
            fields = ['lastPrice', 'open', 'high', 'low', 'lastClose', 
                      'volume', 'amount', 'peTTM', 'pbMRQ', 'totalValue', 'floatValue']
        
        try:
            data = xtdata.get_full_tick(stock_codes)
            
            if not data:
                return pd.DataFrame()
            
            records = []
            for code, tick in data.items():
                if tick:
                    records.append({
                        'code': code,
                        'price': tick.get('lastPrice', 0),
                        'open': tick.get('open', 0),
                        'high': tick.get('high', 0),
                        'low': tick.get('low', 0),
                        'pre_close': tick.get('lastClose', 0),
                        'volume': tick.get('volume', 0),
                        'amount': tick.get('amount', 0),
                        'pe': tick.get('peTTM', 0),
                        'pb': tick.get('pbMRQ', 0),
                        'market_cap': tick.get('totalValue', 0),
                        'float_cap': tick.get('floatValue', 0),
                    })
            
            return pd.DataFrame(records)
        
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()
    
    def get_history_data(self, stock_code: str, period: str = '1d', 
                         start_time: str = '', end_time: str = '',
                         count: int = -1) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            stock_code: 股票代码
            period: K线周期 (1m, 5m, 15m, 30m, 60m, 1d, 1w, 1mon)
            start_time: 开始时间 (格式: 20240101 或 20240101093000)
            end_time: 结束时间
            count: 数据条数 (-1 表示全部)
        
        Returns:
            K线数据 DataFrame
        """
        if not XTQUANT_AVAILABLE:
            return pd.DataFrame()
        
        try:
            # 直接获取本地数据 (download_all_history 已经下载过了)
            data = xtdata.get_market_data(
                field_list=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                stock_list=[stock_code],
                period=period,
                start_time=start_time,
                end_time=end_time,
                count=count,
                dividend_type='front',  # 前复权
                fill_data=True
            )
            
            if data is None:
                return pd.DataFrame()
            
            # xtquant 返回 DataFrame 格式: data[field] 是 DataFrame, 行索引是股票代码, 列是日期
            if isinstance(data, dict) and 'close' in data:
                close_df = data['close']
                
                # 检查是否是 DataFrame 格式
                if isinstance(close_df, pd.DataFrame):
                    # 检查股票代码是否在数据中
                    if stock_code not in close_df.index:
                        return pd.DataFrame()
                    
                    # 提取该股票的数据 (转置: 列变成行)
                    dates = close_df.columns.tolist()  # 日期列表
                    
                    df = pd.DataFrame({
                        'time': dates,
                        'open': data['open'].loc[stock_code].values,
                        'high': data['high'].loc[stock_code].values,
                        'low': data['low'].loc[stock_code].values,
                        'close': data['close'].loc[stock_code].values,
                        'volume': data['volume'].loc[stock_code].values,
                        'amount': data['amount'].loc[stock_code].values,
                    })
                    
                    if len(df) > 0:
                        # 日期列是字符串格式 "20260105"，转换为 datetime
                        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')
                        df['code'] = stock_code
                        return df
                
                # 兼容字典格式 (以防万一)
                elif isinstance(close_df, dict) and stock_code in close_df:
                    df = pd.DataFrame({
                        'time': data['time'][stock_code],
                        'open': data['open'][stock_code],
                        'high': data['high'][stock_code],
                        'low': data['low'][stock_code],
                        'close': data['close'][stock_code],
                        'volume': data['volume'][stock_code],
                        'amount': data['amount'][stock_code],
                    })
                    
                    if len(df) > 0:
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                        df['code'] = stock_code
                        return df
            
        except Exception as e:
            logger.warning(f"获取 {stock_code} 历史数据失败: {e}")
        
        return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str) -> Optional[Dict]:
        """
        获取财务数据
        
        Args:
            stock_code: 股票代码
        
        Returns:
            财务数据字典
        """
        if not XTQUANT_AVAILABLE:
            return None
        
        try:
            # 获取财务数据
            fin_data = xtdata.get_financial_data(
                [stock_code],
                table_list=['Balance', 'Income', 'Indicator'],
                start_time='',
                end_time='',
                report_type='report_time'
            )
            
            if fin_data and stock_code in fin_data:
                return fin_data[stock_code]
                
        except Exception as e:
            logger.warning(f"获取 {stock_code} 财务数据失败: {e}")
        
        return None
    
    def download_all_history(self, stock_codes: List[str], period: str = '1d', 
                             start_time: str = '', end_time: str = ''):
        """
        批量下载历史数据
        
        Args:
            stock_codes: 股票代码列表
            period: K线周期
            start_time: 开始时间
            end_time: 结束时间
        """
        if not XTQUANT_AVAILABLE:
            logger.error("xtquant 不可用")
            return
        
        logger.info(f"开始下载 {len(stock_codes)} 只股票的历史数据...")
        
        # 分批下载
        batch_size = self.config.BATCH_SIZE
        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i:i+batch_size]
            
            for code in batch:
                try:
                    xtdata.download_history_data(code, period, start_time, end_time)
                except Exception as e:
                    logger.warning(f"下载 {code} 数据失败: {e}")
            
            logger.info(f"已下载 {min(i+batch_size, len(stock_codes))}/{len(stock_codes)}")
        
        logger.success("历史数据下载完成")
    
    def filter_stocks(self, stocks: List[Dict]) -> List[Dict]:
        """
        筛选股票
        
        Args:
            stocks: 原始股票列表
        
        Returns:
            筛选后的股票列表
        """
        filtered = []
        
        for stock in stocks:
            code = stock['code']
            
            # 排除 ST 股票
            if self.config.EXCLUDE_ST:
                detail = self.get_instrument_detail(code)
                if detail and 'ST' in detail.get('name', ''):
                    continue
            
            # 排除新股
            if self.config.EXCLUDE_NEW_DAYS > 0:
                detail = detail or self.get_instrument_detail(code)
                if detail:
                    list_date = detail.get('list_date', '')
                    if list_date:
                        try:
                            list_dt = datetime.strptime(str(list_date), '%Y%m%d')
                            if (datetime.now() - list_dt).days < self.config.EXCLUDE_NEW_DAYS:
                                continue
                        except:
                            pass
            
            filtered.append(stock)
        
        logger.info(f"筛选后剩余 {len(filtered)} 只股票")
        return filtered


# 单例实例
_provider = None

def get_provider() -> XtQuantProvider:
    """获取数据提供者单例"""
    global _provider
    if _provider is None:
        _provider = XtQuantProvider()
    return _provider

