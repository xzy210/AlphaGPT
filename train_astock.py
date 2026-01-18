#!/usr/bin/env python3
"""
A股模型训练启动脚本
"""
from loguru import logger

from astock_model.engine import AStockAlphaEngine


def main():
    """训练A股策略"""
    logger.info("=" * 50)
    logger.info("A股 AlphaGPT 模型训练")
    logger.info("=" * 50)
    
    # 创建训练引擎
    engine = AStockAlphaEngine(
        use_lord=True,        # 使用 LoRD 正则化
        lord_decay_rate=1e-3  # 衰减率
    )
    
    # 开始训练
    engine.train()


if __name__ == "__main__":
    main()

