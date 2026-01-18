#!/usr/bin/env python3
"""
A股数据管线启动脚本
"""
import asyncio
from loguru import logger

from astock_pipeline.data_manager import AStockDataManager


async def main():
    """运行数据管线"""
    logger.info("=" * 50)
    logger.info("A股数据管线启动")
    logger.info("=" * 50)
    
    manager = AStockDataManager()
    
    try:
        await manager.initialize()
        await manager.pipeline_daily()
    except Exception as e:
        logger.exception(f"数据管线执行失败: {e}")
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())

