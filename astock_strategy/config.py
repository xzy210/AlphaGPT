"""
A股策略配置
"""

class AStockStrategyConfig:
    """A股策略配置"""
    
    # ========== 仓位控制 ==========
    MAX_OPEN_POSITIONS = 10       # 最大持仓数量
    ENTRY_AMOUNT = 50000          # 单笔入场金额 (元)
    MAX_SINGLE_RATIO = 0.15       # 单只股票最大仓位比例
    MAX_TOTAL_RATIO = 0.8         # 总仓位上限
    
    # ========== 止损止盈 ==========
    STOP_LOSS_PCT = -0.05         # 止损线 (-5%)
    TAKE_PROFIT_PCT = 0.15        # 止盈线 (+15%)
    
    # 分批止盈
    TP_LEVEL_1 = 0.08             # 第一档止盈 (+8%)
    TP_LEVEL_1_RATIO = 0.3        # 第一档卖出比例 (30%)
    TP_LEVEL_2 = 0.15             # 第二档止盈 (+15%)
    TP_LEVEL_2_RATIO = 0.5        # 第二档卖出比例 (50%)
    
    # ========== 移动止损 ==========
    TRAILING_ACTIVATION = 0.08    # 启动移动止损的最小涨幅 (+8%)
    TRAILING_DROP = 0.05          # 从最高点回撤触发阈值 (5%)
    
    # ========== 信号阈值 ==========
    BUY_THRESHOLD = 0.70          # 买入信号阈值
    SELL_THRESHOLD = 0.40         # AI 卖出信号阈值
    
    # ========== 风控参数 ==========
    MIN_MARKET_CAP = 5e9          # 最小市值 (50亿)
    MIN_TURNOVER = 0.005          # 最小换手率 (0.5%)
    AVOID_ST = True               # 避开 ST 股票
    AVOID_NEW_DAYS = 30           # 避开上市不满 N 天的新股
    
    # ========== 运行参数 ==========
    SCAN_INTERVAL = 60            # 扫描间隔 (秒)
    DATA_SYNC_INTERVAL = 900      # 数据同步间隔 (秒)

