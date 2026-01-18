"""
A股数据管理器

协调数据获取、处理和存储
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger

from .config import AStockConfig
from .db_manager import AStockDBManager
from .xtquant_provider import XtQuantProvider


class AStockDataManager:
    """A股数据管理器"""
    
    def __init__(self):
        self.config = AStockConfig()
        self.db = AStockDBManager()
        self.provider = XtQuantProvider()
    
    async def initialize(self):
        """初始化"""
        await self.db.connect()
        await self.db.init_schema()
        self.provider.connect()
        logger.success("A股数据管理器初始化完成")
    
    async def close(self):
        """关闭"""
        await self.db.close()
        self.provider.disconnect()
    
    async def sync_stock_list(self):
        """同步股票列表"""
        logger.info("正在同步股票列表...")
        
        # 获取股票列表
        stocks = self.provider.get_stock_list()
        
        # 筛选股票
        filtered = self.provider.filter_stocks(stocks)
        
        # 获取详细信息并存储
        db_records = []
        for stock in filtered:
            detail = self.provider.get_instrument_detail(stock['code'])
            if detail:
                # 转换日期格式
                list_date = detail.get('list_date', None)
                if list_date:
                    try:
                        # 将 '19970410' 格式转换为 date 对象
                        list_date = datetime.strptime(str(list_date), '%Y%m%d').date()
                    except:
                        list_date = None
                
                db_records.append((
                    detail['code'],
                    detail['name'],
                    detail['market'],
                    detail.get('industry', ''),
                    list_date
                ))
        
        await self.db.upsert_stocks(db_records)
        logger.success(f"股票列表同步完成，共 {len(db_records)} 只")
        
        return [r[0] for r in db_records]  # 返回股票代码列表
    
    async def sync_history_data(self, stock_codes: List[str] = None, 
                                 start_date: str = '', end_date: str = ''):
        """
        同步历史数据
        
        Args:
            stock_codes: 股票代码列表，为空则从数据库获取
            start_date: 开始日期
            end_date: 结束日期
        """
        # 获取股票列表
        if not stock_codes:
            stocks = await self.db.get_stock_list()
            stock_codes = [s['code'] for s in stocks]
        
        if not stock_codes:
            logger.warning("没有股票需要同步")
            return
        
        # 设置日期范围
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.config.HISTORY_DAYS)).strftime('%Y%m%d')
        
        logger.info(f"开始同步 {len(stock_codes)} 只股票的历史数据 ({start_date} ~ {end_date})")
        
        # 批量下载
        self.provider.download_all_history(
            stock_codes, 
            period=self.config.PERIOD,
            start_time=start_date,
            end_time=end_date
        )
        
        # 处理并存储数据
        total_records = 0
        batch_size = self.config.BATCH_SIZE
        empty_count = 0
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            batch_records = []
            
            for code in batch_codes:
                df = self.provider.get_history_data(
                    code, 
                    period=self.config.PERIOD,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if df.empty:
                    empty_count += 1
                    # 前几个股票打印调试信息
                    if empty_count <= 3:
                        logger.debug(f"股票 {code} 返回空数据")
                    continue
                
                # 计算衍生字段
                df['pct_chg'] = df['close'].pct_change() * 100
                df['turnover'] = 0.0  # 需要从其他来源获取
                df['pe'] = 0.0
                df['pb'] = 0.0
                df['market_cap'] = 0.0
                df['float_cap'] = 0.0
                
                # 转换为记录
                for _, row in df.iterrows():
                    batch_records.append((
                        row['time'].date(),
                        code,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume']),
                        float(row['amount']),
                        float(row['turnover']),
                        float(row['pct_chg']) if pd.notna(row['pct_chg']) else 0.0,
                        float(row['pe']),
                        float(row['pb']),
                        float(row['market_cap']),
                        float(row['float_cap']),
                    ))
            
            # 批量插入
            if batch_records:
                await self.db.batch_insert_kline(batch_records)
                total_records += len(batch_records)
            
            logger.info(f"进度: {min(i+batch_size, len(stock_codes))}/{len(stock_codes)}, "
                       f"已插入 {total_records} 条数据")
        
        logger.success(f"历史数据同步完成，共 {total_records} 条")
    
    async def sync_realtime_data(self, stock_codes: List[str] = None):
        """
        同步实时数据
        
        Args:
            stock_codes: 股票代码列表
        """
        if not stock_codes:
            stocks = await self.db.get_stock_list()
            stock_codes = [s['code'] for s in stocks]
        
        if not stock_codes:
            return
        
        logger.info(f"获取 {len(stock_codes)} 只股票的实时数据...")
        
        # 批量获取
        batch_size = 100
        all_records = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i:i+batch_size]
            df = self.provider.get_market_data(batch)
            
            if df.empty:
                continue
            
            today = datetime.now().date()
            
            for _, row in df.iterrows():
                # 计算涨跌幅
                pct_chg = 0.0
                if row['pre_close'] > 0:
                    pct_chg = (row['price'] - row['pre_close']) / row['pre_close'] * 100
                
                all_records.append((
                    today,
                    row['code'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['price']),
                    int(row['volume']),
                    float(row['amount']),
                    0.0,  # turnover
                    pct_chg,
                    float(row.get('pe', 0)),
                    float(row.get('pb', 0)),
                    float(row.get('market_cap', 0)),
                    float(row.get('float_cap', 0)),
                ))
        
        if all_records:
            await self.db.batch_insert_kline(all_records)
            logger.info(f"已更新 {len(all_records)} 条实时数据")
    
    async def pipeline_daily(self):
        """每日数据同步管线"""
        logger.info("========== 开始每日数据同步 ==========")
        
        # 1. 同步股票列表
        stock_codes = await self.sync_stock_list()
        
        # 2. 同步历史数据 (增量)
        latest_date = await self.db.get_latest_date()
        if latest_date:
            start_date = (datetime.strptime(latest_date, '%Y-%m-%d') + 
                         timedelta(days=1)).strftime('%Y%m%d')
        else:
            start_date = ''
        
        await self.sync_history_data(stock_codes, start_date=start_date)
        
        logger.success("========== 每日数据同步完成 ==========")


async def run_pipeline():
    """运行数据管线"""
    manager = AStockDataManager()
    
    try:
        await manager.initialize()
        await manager.pipeline_daily()
    except Exception as e:
        logger.exception(f"数据管线执行失败: {e}")
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(run_pipeline())

