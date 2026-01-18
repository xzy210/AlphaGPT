"""
A股执行层配置
"""
import os
from dotenv import load_dotenv

load_dotenv()


class AStockExecutionConfig:
    """A股执行配置"""
    
    # ========== QMT 配置 ==========
    # MiniQMT 路径
    QMT_PATH = os.getenv("QMT_PATH", r"D:\国金证券QMT交易端\userdata_mini")
    
    # 账户信息
    ACCOUNT_ID = os.getenv("QMT_ACCOUNT_ID", "")
    ACCOUNT_TYPE = "STOCK"  # 账户类型: STOCK, CREDIT
    
    # ========== 交易参数 ==========
    # 默认滑点 (基点)
    DEFAULT_SLIPPAGE_BPS = 30  # 0.3%
    
    # 单笔最小交易金额
    MIN_TRADE_AMOUNT = 5000.0
    
    # 单笔最大交易金额
    MAX_TRADE_AMOUNT = 100000.0
    
    # 价格类型
    PRICE_TYPE_LIMIT = 0       # 限价单
    PRICE_TYPE_MARKET = 1      # 市价单 (五档即成剩撤)
    PRICE_TYPE_BEST = 2        # 最优五档即时成交剩余撤销
    
    # 默认价格类型
    DEFAULT_PRICE_TYPE = PRICE_TYPE_LIMIT
    
    # ========== 风控参数 ==========
    # 单只股票最大仓位比例
    MAX_SINGLE_POSITION_RATIO = 0.2
    
    # 最大持仓股票数
    MAX_POSITIONS = 10
    
    # 总仓位上限
    MAX_TOTAL_POSITION_RATIO = 0.8
    
    # 涨跌停价差保护 (距离涨跌停价的最小距离)
    LIMIT_PRICE_BUFFER = 0.005  # 0.5%
    
    # ========== 时间配置 ==========
    # 交易时间段
    TRADING_HOURS = [
        ("09:30", "11:30"),
        ("13:00", "15:00"),
    ]
    
    # 集合竞价时间
    AUCTION_TIME = [
        ("09:15", "09:25"),  # 开盘集合竞价
        ("14:57", "15:00"),  # 收盘集合竞价
    ]

