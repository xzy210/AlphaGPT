"""
A股数据加载器

从数据库加载数据并转换为 Tensor
"""
import pandas as pd
import torch
import sqlalchemy
from typing import Optional, List
from loguru import logger

from astock_pipeline.config import AStockConfig
from .factors import AStockFeatureEngineer


class AStockDataLoader:
    """A股数据加载器"""
    
    def __init__(self, device: str = 'cpu'):
        self.config = AStockConfig()
        self.engine = sqlalchemy.create_engine(self.config.DB_DSN)
        self.device = torch.device(device)
        
        # 数据缓存
        self.raw_data_cache = None
        self.feat_tensor = None
        self.target_ret = None
        self.stock_codes = None
    
    def load_data(self, limit_stocks: int = 500, lookback_days: int = 250, min_days: int = 20):
        """
        加载数据
        
        Args:
            limit_stocks: 最大股票数量
            lookback_days: 历史数据天数
            min_days: 最小数据天数要求 (默认20天)
        """
        logger.info(f"正在加载数据 (最多 {limit_stocks} 只股票, {lookback_days} 天)...")
        
        # 先检查数据库中有多少数据
        try:
            check_query = """
            SELECT COUNT(DISTINCT code) as stock_count, 
                   COUNT(*) as total_rows,
                   MAX(trade_date) as latest_date,
                   MIN(trade_date) as earliest_date
            FROM daily_kline
            """
            check_df = pd.read_sql(check_query, self.engine)
            if not check_df.empty:
                row = check_df.iloc[0]
                logger.info(f"数据库状态: {row['stock_count']} 只股票, "
                           f"{row['total_rows']} 条记录, "
                           f"日期范围: {row['earliest_date']} ~ {row['latest_date']}")
        except Exception as e:
            logger.warning(f"检查数据库状态失败: {e}")
        
        # 1. 获取活跃股票列表 (降低要求，只要有 min_days 天数据就行)
        stock_query = f"""
        SELECT code FROM (
            SELECT code, COUNT(*) as cnt 
            FROM daily_kline 
            GROUP BY code 
            HAVING COUNT(*) >= {min_days}
            ORDER BY cnt DESC
        ) t
        LIMIT {limit_stocks}
        """
        
        try:
            stock_df = pd.read_sql(stock_query, self.engine)
            self.stock_codes = stock_df['code'].tolist()
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return
        
        if not self.stock_codes:
            logger.warning("没有找到符合条件的股票")
            logger.warning(f"请检查: 1) 数据管线是否成功同步K线数据  2) 数据库 daily_kline 表是否有数据")
            return
        
        logger.info(f"找到 {len(self.stock_codes)} 只股票")
        
        # 2. 加载 K线数据
        codes_str = "'" + "','".join(self.stock_codes) + "'"
        data_query = f"""
        SELECT trade_date, code, open, high, low, close, 
               volume, amount, turnover, pct_chg, pe, pb, 
               market_cap, float_cap
        FROM daily_kline
        WHERE code IN ({codes_str})
        ORDER BY trade_date DESC
        LIMIT {limit_stocks * lookback_days}
        """
        
        df = pd.read_sql(data_query, self.engine)
        
        if df.empty:
            logger.warning("没有加载到数据")
            return
        
        logger.info(f"加载了 {len(df)} 条 K线数据")
        
        # 3. 转换为张量
        def to_tensor(col):
            pivot = df.pivot(index='trade_date', columns='code', values=col)
            pivot = pivot.sort_index()  # 按日期排序
            pivot = pivot.ffill().fillna(0.0)
            
            # [Time, Stocks] -> [Stocks, Time]
            arr = pivot.values.T
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'amount': to_tensor('amount'),
            'pct_chg': to_tensor('pct_chg'),
            'pe': to_tensor('pe'),
            'pb': to_tensor('pb'),
            'market_cap': to_tensor('market_cap'),
            'float_cap': to_tensor('float_cap'),
            # 注意：turnover 因子已移除，因子计算不再使用换手率
        }
        
        # 4. 计算特征
        logger.info("正在计算特征...")
        self.feat_tensor = AStockFeatureEngineer.compute_features(self.raw_data_cache)
        
        # 5. 计算目标收益率 (T+1 开盘买入, T+2 开盘卖出)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)  # T+1 开盘价
        t2 = torch.roll(op, -2, dims=1)  # T+2 开盘价
        self.target_ret = (t2 - t1) / (t1 + 1e-9)
        
        # 最后两天的收益设为 0
        self.target_ret[:, -2:] = 0.0
        
        logger.success(f"数据加载完成. 特征维度: {self.feat_tensor.shape}")
    
    def get_latest_features(self) -> Optional[torch.Tensor]:
        """
        获取最新的特征数据 (用于实盘)
        
        Returns:
            最新特征 [Stocks, Features]
        """
        if self.feat_tensor is None:
            return None
        
        return self.feat_tensor[:, :, -1]
    
    def get_stock_index(self, stock_code: str) -> int:
        """
        获取股票在张量中的索引
        
        Args:
            stock_code: 股票代码
        
        Returns:
            索引 (-1 表示未找到)
        """
        if self.stock_codes is None:
            return -1
        
        try:
            return self.stock_codes.index(stock_code)
        except ValueError:
            return -1
    
    def get_all_codes(self) -> List[str]:
        """获取所有股票代码"""
        return self.stock_codes or []


class RealtimeDataLoader:
    """实时数据加载器"""
    
    def __init__(self, base_loader: AStockDataLoader):
        self.base_loader = base_loader
        self.device = base_loader.device
    
    def update_with_realtime(self, realtime_data: pd.DataFrame):
        """
        用实时数据更新特征
        
        Args:
            realtime_data: 实时数据 DataFrame
        """
        if self.base_loader.raw_data_cache is None:
            logger.warning("基础数据未加载")
            return
        
        # 将实时数据追加到缓存
        for _, row in realtime_data.iterrows():
            code = row['code']
            idx = self.base_loader.get_stock_index(code)
            if idx < 0:
                continue
            
            # 追加到每个字段的最后
            for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if field in row:
                    # 滚动并更新最后一个值
                    self.base_loader.raw_data_cache[field][idx, -1] = row[field]
        
        # 重新计算特征
        self.base_loader.feat_tensor = AStockFeatureEngineer.compute_features(
            self.base_loader.raw_data_cache
        )

