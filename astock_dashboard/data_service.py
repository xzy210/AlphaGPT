"""
A股 Dashboard 数据服务层

提供数据库查询、账户信息、持仓管理等功能
"""
import json
import os
import pandas as pd
import sqlalchemy
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

# 项目根目录 (astock_dashboard 的上级目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

try:
    import psycopg2
    from psycopg2 import extensions as psy_ext
    HAS_PSYCOPG2 = True
except Exception:
    HAS_PSYCOPG2 = False


def _robust_text_decoder(value, cursor):
    """
    健壮的文本解码器 - 跳过无法解码的字节
    
    这个解码器会在全局级别注册，处理所有文本类型
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    
    # 直接用 GBK 解码，忽略无法解码的字节
    try:
        return value.decode("gbk", errors="ignore")
    except Exception:
        pass
    
    # 备选：UTF-8 解码，忽略错误
    return value.decode("utf-8", errors="ignore")


def _register_global_decoders():
    """在全局级别注册文本解码器"""
    if not HAS_PSYCOPG2:
        return
    
    try:
        # PostgreSQL 文本类型 OID
        # 25: TEXT, 1042: BPCHAR (char(n)), 1043: VARCHAR, 19: NAME, 18: CHAR
        text_oids = (25, 1042, 1043, 19, 18)
        
        # 创建并注册类型适配器
        text_type = psy_ext.new_type(text_oids, "TEXT_ROBUST", _robust_text_decoder)
        psy_ext.register_type(text_type)  # 全局注册，不绑定特定连接
        
        logger.info("已注册全局文本解码器")
    except Exception as e:
        logger.warning(f"注册全局文本解码器失败: {e}")


# 模块加载时立即注册全局解码器
_register_global_decoders()

# 可选：QMT 交易接口
try:
    from xtquant import xtdata
    from xtquant.xttrader import XtQuantTrader
    HAS_XTQUANT = True
except ImportError:
    HAS_XTQUANT = False
    logger.warning("xtquant 未安装，部分功能不可用")


class AStockDashboardService:
    """A股 Dashboard 数据服务"""
    
    def __init__(self):
        # 数据库连接
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "password")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "astock_quant")
        db_port = os.getenv("DB_PORT", "5432")
        
        # 客户端编码（Windows 上常见 GBK）
        self.db_client_encoding = os.getenv("DB_CLIENT_ENCODING", "GBK")
        self.db_dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        self.engine = self._create_engine(self.db_client_encoding)
        
        # QMT 交易接口（可选）
        self.trader = None
        self.account_id = os.getenv("QMT_ACCOUNT_ID", "")

    def _create_engine(self, encoding: str):
        """创建数据库引擎并设置编码"""
        # 简化连接配置 - 全局解码器已注册，只需设置客户端编码
        engine = sqlalchemy.create_engine(
            self.db_dsn,
            connect_args={"options": f"-c client_encoding={encoding}"},
            pool_pre_ping=True
        )
        return engine

    def _read_sql(self, query: str) -> pd.DataFrame:
        """读取 SQL - 使用原生 psycopg2 连接避免编码问题"""
        try:
            # 直接用 psycopg2 连接，绕过 SQLAlchemy 的编码处理
            import psycopg2
            conn = psycopg2.connect(self.db_dsn)
            # 设置客户端编码为 GBK（与数据实际编码匹配）
            conn.set_client_encoding('GBK')
            
            try:
                df = pd.read_sql(query, conn)
                return df
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"SQL 查询失败: {e}")
            raise
    
    def connect_trader(self, path: str = None, session_id: int = None):
        """
        连接 QMT 交易接口（可选）
        
        Args:
            path: MiniQMT 路径
            session_id: 会话ID
        """
        if not HAS_XTQUANT:
            return False
        
        try:
            if path and session_id and self.account_id:
                self.trader = XtQuantTrader(path, session_id)
                self.trader.start()
                self.trader.connect()
                logger.info(f"QMT 交易接口已连接: {self.account_id}")
                return True
        except Exception as e:
            logger.error(f"QMT 连接失败: {e}")
        return False
    
    def get_account_info(self) -> dict:
        """
        获取账户信息
        
        Returns:
            账户信息字典
        """
        default_info = {
            "total_asset": 0.0,
            "cash": 0.0,
            "market_value": 0.0,
            "frozen": 0.0,
            "profit": 0.0,
            "profit_pct": 0.0
        }
        
        if not self.trader or not self.account_id:
            # 尝试从本地文件读取
            try:
                filepath = os.path.join(PROJECT_ROOT, "astock_account_state.json")
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return default_info
        
        try:
            account = self.trader.query_stock_asset(self.account_id)
            if account:
                return {
                    "total_asset": account.total_asset,
                    "cash": account.cash,
                    "market_value": account.market_value,
                    "frozen": account.frozen_cash,
                    "profit": account.total_asset - account.total_asset,  # 需要历史数据计算
                    "profit_pct": 0.0
                }
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
        
        return default_info
    
    def load_portfolio(self) -> pd.DataFrame:
        """
        加载当前持仓
        
        Returns:
            持仓 DataFrame
        """
        try:
            filepath = os.path.join(PROJECT_ROOT, "astock_portfolio_state.json")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data.values())
                
                # 计算盈亏
                if 'current_price' in df.columns and 'entry_price' in df.columns:
                    df['pnl_pct'] = (df['current_price'] - df['entry_price']) / df['entry_price']
                    df['pnl_amount'] = (df['current_price'] - df['entry_price']) * df.get('amount', 0)
                
                return df
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"加载持仓失败: {e}")
            return pd.DataFrame()
    
    def load_strategy_info(self) -> dict:
        """
        加载策略信息
        
        Returns:
            策略信息字典
        """
        try:
            filepath = os.path.join(PROJECT_ROOT, "best_astock_strategy.json")
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"formula_str": "未训练", "score": 0.0}
    
    def load_training_history(self) -> pd.DataFrame:
        """
        加载训练历史
        
        Returns:
            训练历史 DataFrame
        """
        try:
            filepath = os.path.join(PROJECT_ROOT, "astock_training_history.json")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 找出最长的数组长度
                max_len = max(len(v) for v in data.values() if isinstance(v, list))
                
                # 只保留长度一致的字段，或者填充短的数组
                filtered_data = {}
                for key, val in data.items():
                    if isinstance(val, list):
                        if len(val) == max_len:
                            filtered_data[key] = val
                        elif len(val) > 0:
                            # 对于较短的数组（如 stable_rank），用 None 填充
                            # 或者跳过不一致长度的字段
                            pass  # 跳过长度不一致的字段
                    else:
                        filtered_data[key] = val
                
                return pd.DataFrame(filtered_data)
        except Exception as e:
            logger.warning(f"加载训练历史失败: {e}")
            return pd.DataFrame()
    
    def get_market_overview(self, limit: int = 50) -> pd.DataFrame:
        """
        获取市场概览（按涨跌幅排序）
        
        Args:
            limit: 返回数量
            
        Returns:
            市场数据 DataFrame
        """
        query = f"""
        SELECT 
            d.code,
            s.name,
            d.close,
            d.pct_chg,
            d.volume,
            d.amount,
            d.turnover,
            d.market_cap,
            d.pe,
            d.pb
        FROM daily_kline d
        JOIN stocks s ON d.code = s.code
        WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_kline)
        ORDER BY d.pct_chg DESC
        LIMIT {limit}
        """
        try:
            return self._read_sql(query)
        except Exception as e:
            logger.error(f"获取市场概览失败: {e}")
            return pd.DataFrame()
    
    def get_market_stats(self) -> dict:
        """
        获取市场统计
        
        Returns:
            市场统计字典
        """
        query = """
        SELECT 
            COUNT(*) as total_stocks,
            SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) as up_count,
            SUM(CASE WHEN pct_chg < 0 THEN 1 ELSE 0 END) as down_count,
            SUM(CASE WHEN pct_chg = 0 THEN 1 ELSE 0 END) as flat_count,
            SUM(CASE WHEN pct_chg >= 9.9 THEN 1 ELSE 0 END) as limit_up,
            SUM(CASE WHEN pct_chg <= -9.9 THEN 1 ELSE 0 END) as limit_down,
            AVG(pct_chg) as avg_pct_chg,
            SUM(amount) / 100000000 as total_amount_yi,
            MAX(trade_date) as latest_date
        FROM daily_kline
        WHERE trade_date = (SELECT MAX(trade_date) FROM daily_kline)
        """
        try:
            df = self._read_sql(query)
            if not df.empty:
                row = df.iloc[0]
                return {
                    "total_stocks": int(row['total_stocks'] or 0),
                    "up_count": int(row['up_count'] or 0),
                    "down_count": int(row['down_count'] or 0),
                    "flat_count": int(row['flat_count'] or 0),
                    "limit_up": int(row['limit_up'] or 0),
                    "limit_down": int(row['limit_down'] or 0),
                    "avg_pct_chg": float(row['avg_pct_chg'] or 0),
                    "total_amount_yi": float(row['total_amount_yi'] or 0),
                    "latest_date": str(row['latest_date'] or '')
                }
        except Exception as e:
            logger.error(f"获取市场统计失败: {e}")
        
        return {
            "total_stocks": 0, "up_count": 0, "down_count": 0, 
            "flat_count": 0, "limit_up": 0, "limit_down": 0,
            "avg_pct_chg": 0, "total_amount_yi": 0, "latest_date": ""
        }
    
    def get_kline_data(self, code: str, days: int = 60) -> pd.DataFrame:
        """
        获取个股K线数据
        
        Args:
            code: 股票代码
            days: 天数
            
        Returns:
            K线 DataFrame
        """
        query = f"""
        SELECT trade_date, open, high, low, close, volume, amount
        FROM daily_kline
        WHERE code = '{code}'
        ORDER BY trade_date DESC
        LIMIT {days}
        """
        try:
            df = self._read_sql(query)
            df = df.sort_values('trade_date')
            return df
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()
    
    def search_stock(self, keyword: str) -> pd.DataFrame:
        """
        搜索股票
        
        Args:
            keyword: 关键词（代码或名称）
            
        Returns:
            搜索结果 DataFrame
        """
        query = f"""
        SELECT code, name, market, industry
        FROM stocks
        WHERE code LIKE '%{keyword}%' 
           OR name LIKE '%{keyword}%'
        LIMIT 20
        """
        try:
            return self._read_sql(query)
        except Exception:
            return pd.DataFrame()
    
    def get_db_stats(self) -> dict:
        """
        获取数据库统计
        
        Returns:
            数据库统计字典
        """
        try:
            stock_count = self._read_sql(
                "SELECT COUNT(*) as cnt FROM stocks"
            ).iloc[0]['cnt']
            
            kline_stats = self._read_sql("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT code) as code_count
                FROM daily_kline
            """).iloc[0]
            
            return {
                "stock_count": int(stock_count),
                "kline_records": int(kline_stats['total_records']),
                "min_date": str(kline_stats['min_date']),
                "max_date": str(kline_stats['max_date']),
                "kline_codes": int(kline_stats['code_count'])
            }
        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            return {
                "stock_count": 0, "kline_records": 0,
                "min_date": "", "max_date": "", "kline_codes": 0
            }
    
    def get_recent_logs(self, n: int = 50) -> list:
        """
        获取最近的日志
        
        Args:
            n: 日志行数
            
        Returns:
            日志行列表
        """
        log_file = os.path.join(PROJECT_ROOT, "astock_strategy.log")
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return lines[-n:]
        except:
            return []
    
    def get_sector_stats(self) -> pd.DataFrame:
        """
        获取板块统计
        
        Returns:
            板块统计 DataFrame
        """
        query = """
        SELECT 
            s.industry,
            COUNT(*) as stock_count,
            AVG(d.pct_chg) as avg_pct_chg,
            SUM(d.amount) / 100000000 as total_amount_yi
        FROM daily_kline d
        JOIN stocks s ON d.code = s.code
        WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily_kline)
          AND s.industry IS NOT NULL 
          AND s.industry != ''
        GROUP BY s.industry
        ORDER BY avg_pct_chg DESC
        """
        try:
            return self._read_sql(query)
        except:
            return pd.DataFrame()

    def run_backtest(self, lookback_days: int = 60, limit_stocks: int = 200) -> dict:
        """
        运行策略回测
        
        Args:
            lookback_days: 回测天数
            limit_stocks: 股票数量
            
        Returns:
            回测结果字典
        """
        try:
            import torch
            from astock_model.vm import AStockStackVM
            from astock_model.backtest import DetailedAStockBacktest
            from astock_model.data_loader import AStockDataLoader
            
            # 加载策略
            strategy = self.load_strategy_info()
            formula = strategy.get('formula')
            
            if not formula:
                return {"error": "未找到训练好的策略，请先运行训练"}
            
            logger.info(f"开始回测，股票数: {limit_stocks}，天数: {lookback_days}")
            
            # 加载数据
            loader = AStockDataLoader(device='cpu')
            loader.load_data(limit_stocks=limit_stocks, lookback_days=lookback_days, min_days=20)
            
            if loader.feat_tensor is None:
                return {"error": "数据加载失败"}
            
            # 执行公式
            vm = AStockStackVM()
            factors = vm.execute(formula, loader.feat_tensor)
            
            if factors is None:
                return {"error": "策略执行失败"}
            
            # 运行详细回测
            bt = DetailedAStockBacktest()
            stats = bt.detailed_evaluate(factors, loader.raw_data_cache, loader.target_ret)
            
            # 计算每日收益曲线
            signal = torch.sigmoid(factors)
            position = (signal > 0.7).float()
            
            # 每只股票的累计收益
            gross_pnl = position * loader.target_ret
            daily_returns = gross_pnl.mean(dim=0).cpu().numpy()  # 每日平均收益
            
            # 累计收益曲线
            cum_returns = daily_returns.cumsum()
            
            # 获取日期
            dates = []
            try:
                query = f"""
                SELECT DISTINCT trade_date FROM daily_kline 
                ORDER BY trade_date DESC LIMIT {lookback_days}
                """
                date_df = self._read_sql(query)
                dates = date_df['trade_date'].sort_values().tolist()
            except:
                dates = list(range(len(cum_returns)))
            
            # 确保长度一致
            min_len = min(len(dates), len(cum_returns))
            dates = dates[-min_len:]
            cum_returns = cum_returns[-min_len:]
            daily_returns = daily_returns[-min_len:]
            
            return {
                "success": True,
                "score": stats['score'],
                "total_return": stats['total_return'],
                "total_trades": stats['total_trades'],
                "win_rate": stats['win_rate'],
                "profit_loss_ratio": stats['profit_loss_ratio'],
                "avg_position": stats['avg_position'],
                "formula_str": strategy.get('formula_str', ''),
                "dates": [str(d) for d in dates],
                "cum_returns": cum_returns.tolist(),
                "daily_returns": daily_returns.tolist(),
                "stock_count": limit_stocks,
                "lookback_days": lookback_days
            }
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

