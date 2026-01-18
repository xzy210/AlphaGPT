"""
A股数据库管理

管理 PostgreSQL 数据存储
"""
import asyncpg
from loguru import logger
from .config import AStockConfig


class AStockDBManager:
    """A股数据库管理器"""
    
    def __init__(self):
        self.pool = None
        self.config = AStockConfig()
    
    async def connect(self):
        """建立数据库连接池"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=self.config.DB_DSN)
            logger.info("A股数据库连接已建立")
    
    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("数据库连接已关闭")
    
    async def init_schema(self):
        """初始化数据库表结构"""
        async with self.pool.acquire() as conn:
            # 股票基础信息表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    industry TEXT,
                    list_date DATE,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # 日线行情表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_kline (
                    trade_date DATE NOT NULL,
                    code TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    amount DOUBLE PRECISION,
                    turnover DOUBLE PRECISION,
                    pct_chg DOUBLE PRECISION,
                    pe DOUBLE PRECISION,
                    pb DOUBLE PRECISION,
                    market_cap DOUBLE PRECISION,
                    float_cap DOUBLE PRECISION,
                    PRIMARY KEY (trade_date, code)
                );
            """)
            
            # 尝试转换为 TimescaleDB 超表
            try:
                await conn.execute("""
                    SELECT create_hypertable('daily_kline', 'trade_date', 
                                             if_not_exists => TRUE);
                """)
                logger.info("已启用 TimescaleDB 超表")
            except Exception:
                logger.warning("TimescaleDB 未启用，使用普通表")
            
            # 创建索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_kline_code 
                ON daily_kline (code);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_kline_date 
                ON daily_kline (trade_date DESC);
            """)
            
            # 因子缓存表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_cache (
                    trade_date DATE NOT NULL,
                    code TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    factor_value DOUBLE PRECISION,
                    PRIMARY KEY (trade_date, code, factor_name)
                );
            """)
            
            logger.success("数据库表结构初始化完成")
    
    async def upsert_stocks(self, stocks: list):
        """
        更新或插入股票信息
        
        Args:
            stocks: 股票信息列表 [(code, name, market, industry, list_date), ...]
        """
        if not stocks:
            return
        
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO stocks (code, name, market, industry, list_date, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (code) DO UPDATE 
                SET name = EXCLUDED.name,
                    industry = EXCLUDED.industry,
                    last_updated = NOW();
            """, stocks)
        
        logger.info(f"已更新 {len(stocks)} 只股票信息")
    
    async def batch_insert_kline(self, records: list):
        """
        批量插入K线数据
        
        Args:
            records: K线数据列表
                [(trade_date, code, open, high, low, close, volume, amount, 
                  turnover, pct_chg, pe, pb, market_cap, float_cap), ...]
        """
        if not records:
            return
        
        async with self.pool.acquire() as conn:
            try:
                await conn.copy_records_to_table(
                    'daily_kline',
                    records=records,
                    columns=['trade_date', 'code', 'open', 'high', 'low', 'close',
                             'volume', 'amount', 'turnover', 'pct_chg', 
                             'pe', 'pb', 'market_cap', 'float_cap'],
                    timeout=60
                )
                logger.info(f"已插入 {len(records)} 条K线数据")
            except asyncpg.UniqueViolationError:
                # 忽略重复数据
                pass
            except Exception as e:
                logger.error(f"批量插入K线数据失败: {e}")
    
    async def get_stock_list(self) -> list:
        """获取股票列表"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT code, name, market FROM stocks")
            return [dict(row) for row in rows]
    
    async def get_kline_data(self, codes: list, start_date: str, end_date: str) -> list:
        """
        获取K线数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期
        
        Returns:
            K线数据列表
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT trade_date, code, open, high, low, close, 
                       volume, amount, turnover, pct_chg, pe, pb, 
                       market_cap, float_cap
                FROM daily_kline
                WHERE code = ANY($1)
                  AND trade_date >= $2
                  AND trade_date <= $3
                ORDER BY code, trade_date
            """, codes, start_date, end_date)
            return [dict(row) for row in rows]
    
    async def get_latest_date(self) -> str:
        """获取最新数据日期"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(trade_date) as max_date FROM daily_kline"
            )
            if row and row['max_date']:
                return row['max_date'].strftime('%Y-%m-%d')
            return ''

