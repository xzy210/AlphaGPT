"""
A股数据管道配置

使用 xtquant/miniqmt 作为数据源
"""
import os
from dotenv import load_dotenv

load_dotenv()

class AStockConfig:
    """A股配置"""
    
    # ========== 数据库配置 ==========
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "astock_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # ========== XtQuant 配置 ==========
    # MiniQMT 路径 (安装目录下的 userdata_mini 文件夹)
    QMT_PATH = os.getenv("QMT_PATH", r"D:\国金证券QMT交易端\userdata_mini")
    
    # 账户ID (用于交易)
    ACCOUNT_ID = os.getenv("QMT_ACCOUNT_ID", "")
    
    # ========== 数据参数 ==========
    # 股票池筛选
    MARKET = "SH,SZ"                   # 市场: SH=上证, SZ=深证
    EXCLUDE_ST = True                  # 排除 ST 股票
    EXCLUDE_NEW_DAYS = 60              # 排除上市不满 N 天的新股
    MIN_MARKET_CAP = 3e9               # 最小市值 (30亿)
    MAX_MARKET_CAP = float('inf')      # 最大市值
    MIN_TURNOVER = 0.005               # 最小换手率
    
    # K线周期
    PERIOD = "1d"                      # 日线 (支持: 1m, 5m, 15m, 30m, 60m, 1d)
    HISTORY_DAYS = 250                 # 历史数据天数 (约一年)
    
    # 并发控制
    BATCH_SIZE = 50                    # 批量请求大小
    
    # ========== 交易参数 ==========
    COMMISSION_RATE = 0.00025          # 佣金费率 (万2.5)
    MIN_COMMISSION = 5.0               # 最低佣金 (5元)
    STAMP_TAX_RATE = 0.001             # 印花税 (千1，卖出收取)
    TRANSFER_FEE_RATE = 0.00002        # 过户费 (万0.2)
    SLIPPAGE = 0.001                   # 滑点估计 (0.1%)

