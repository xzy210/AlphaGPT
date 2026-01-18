#!/usr/bin/env python3
"""
A股策略执行启动脚本
"""
import asyncio
from loguru import logger

from astock_strategy.runner import AStockStrategyRunner


async def main():
    """运行策略"""
    logger.info("=" * 50)
    logger.info("A股策略执行器启动")
    logger.info("=" * 50)
    
    runner = AStockStrategyRunner()
    
    try:
        await runner.initialize()
        await runner.run_loop()
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.exception(f"策略执行异常: {e}")
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

