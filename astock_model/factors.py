"""
A股因子工程

针对 A股市场设计的因子体系
"""
import torch
import torch.nn as nn
import numpy as np


class AStockIndicators:
    """A股技术指标"""
    
    @staticmethod
    def momentum(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        动量因子
        
        Args:
            close: 收盘价 [Stocks, Time]
            window: 回溯窗口
        
        Returns:
            动量值
        """
        past_close = torch.roll(close, window, dims=1)
        mom = (close - past_close) / (past_close + 1e-9)
        mom[:, :window] = 0
        return mom
    
    @staticmethod
    def volatility(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        波动率因子 (已实现波动率)
        
        Args:
            close: 收盘价
            window: 计算窗口
        
        Returns:
            波动率
        """
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret[:, 0] = 0
        
        # 滚动标准差
        ret_sq = ret ** 2
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol = torch.sqrt(ret_sq_pad.unfold(1, window, 1).mean(dim=-1) + 1e-9)
        
        return vol
    
    @staticmethod
    def volume_price_divergence(close: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """
        量价背离
        
        Args:
            close: 收盘价
            volume: 成交量
        
        Returns:
            量价背离信号 (-1, 0, 1)
        """
        price_chg = torch.sign(close - torch.roll(close, 1, dims=1))
        vol_chg = torch.sign(volume - torch.roll(volume, 1, dims=1))
        
        # 背离 = 价格和成交量方向相反
        divergence = -price_chg * vol_chg
        divergence[:, 0] = 0
        
        return divergence
    
    @staticmethod
    def rsi(close: torch.Tensor, window: int = 14) -> torch.Tensor:
        """
        相对强弱指数 (RSI)
        
        Args:
            close: 收盘价
            window: 计算窗口
        
        Returns:
            RSI 值 (0-100)
        """
        delta = close - torch.roll(close, 1, dims=1)
        delta[:, 0] = 0
        
        gains = torch.relu(delta)
        losses = torch.relu(-delta)
        
        # 滚动均值
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(close: torch.Tensor, fast: int = 12, slow: int = 26, signal: int = 9) -> torch.Tensor:
        """
        MACD 指标
        
        Args:
            close: 收盘价
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
        
        Returns:
            MACD 柱状图
        """
        # 简化版 EMA (使用 SMA 近似)
        def sma(x, window):
            pad = torch.zeros((x.shape[0], window-1), device=x.device)
            x_pad = torch.cat([pad, x], dim=1)
            return x_pad.unfold(1, window, 1).mean(dim=-1)
        
        ema_fast = sma(close, fast)
        ema_slow = sma(close, slow)
        
        dif = ema_fast - ema_slow
        dea = sma(dif, signal)
        
        macd = 2 * (dif - dea)
        
        return macd
    
    @staticmethod
    def price_position(close: torch.Tensor, high: torch.Tensor, low: torch.Tensor, 
                       window: int = 20) -> torch.Tensor:
        """
        价格位置因子 (当前价格在 N 日区间的位置)
        
        Args:
            close: 收盘价
            high: 最高价
            low: 最低价
            window: 计算窗口
        
        Returns:
            位置值 (0-1)
        """
        pad = torch.zeros((high.shape[0], window-1), device=close.device)
        
        high_pad = torch.cat([pad, high], dim=1)
        low_pad = torch.cat([pad, low], dim=1)
        
        highest = high_pad.unfold(1, window, 1).max(dim=-1)[0]
        lowest = low_pad.unfold(1, window, 1).min(dim=-1)[0]
        
        position = (close - lowest) / (highest - lowest + 1e-9)
        
        return position
    
    @staticmethod
    def amplitude(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> torch.Tensor:
        """
        振幅因子
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
        
        Returns:
            振幅
        """
        amp = (high - low) / (close + 1e-9)
        return amp
    
    @staticmethod
    def limit_up_distance(close: torch.Tensor, pct_chg: torch.Tensor) -> torch.Tensor:
        """
        距离涨停距离
        
        Args:
            close: 收盘价
            pct_chg: 涨跌幅 (%)
        
        Returns:
            距离涨停的空间 (%)
        """
        # 涨停幅度 (主板10%, 科创板/创业板20%)
        limit_rate = 10.0
        distance = limit_rate - pct_chg
        return torch.clamp(distance, 0.0, 20.0)
    
    @staticmethod
    def pe_percentile(pe: torch.Tensor, window: int = 250) -> torch.Tensor:
        """
        PE 百分位 (估值水平)
        
        Args:
            pe: 市盈率
            window: 历史窗口
        
        Returns:
            PE 百分位 (0-1)
        """
        # 简化计算：当前 PE 在历史区间的位置
        pad = torch.zeros((pe.shape[0], window-1), device=pe.device)
        pe_pad = torch.cat([pad, pe], dim=1)
        
        pe_max = pe_pad.unfold(1, window, 1).max(dim=-1)[0]
        pe_min = pe_pad.unfold(1, window, 1).min(dim=-1)[0]
        
        percentile = (pe - pe_min) / (pe_max - pe_min + 1e-9)
        
        return percentile


class AStockFeatureEngineer:
    """A股特征工程"""
    
    INPUT_DIM = 10  # 特征维度 (移除了 turnover 和 pe_rank)
    
    FEATURE_NAMES = [
        'ret',           # 收益率
        'momentum',      # 动量
        'volatility',    # 波动率
        'vol_price_div', # 量价背离
        'rsi',           # RSI
        'macd',          # MACD
        'price_pos',     # 价格位置
        'amplitude',     # 振幅
        'limit_dist',    # 涨停距离
        'vol_ratio',     # 量比
    ]
    
    @staticmethod
    def robust_norm(t: torch.Tensor) -> torch.Tensor:
        """
        鲁棒标准化 (使用 MAD)
        
        Args:
            t: 输入张量 [Stocks, Time]
        
        Returns:
            标准化后的张量
        """
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)
    
    @staticmethod
    def cross_sectional_rank(t: torch.Tensor) -> torch.Tensor:
        """
        截面排名 (每个时间点对所有股票排名)
        
        Args:
            t: 输入张量 [Stocks, Time]
        
        Returns:
            排名百分位 (0-1)
        """
        # 转置后排名
        n_stocks = t.shape[0]
        ranks = torch.zeros_like(t)
        
        for i in range(t.shape[1]):
            col = t[:, i]
            sorted_idx = col.argsort()
            rank = torch.zeros_like(col)
            rank[sorted_idx] = torch.arange(n_stocks, device=t.device, dtype=t.dtype)
            ranks[:, i] = rank / n_stocks
        
        return ranks
    
    @classmethod
    def compute_features(cls, raw_dict: dict) -> torch.Tensor:
        """
        计算特征张量
        
        Args:
            raw_dict: 原始数据字典
                - 'open': [Stocks, Time]
                - 'high': [Stocks, Time]
                - 'low': [Stocks, Time]
                - 'close': [Stocks, Time]
                - 'volume': [Stocks, Time]
                - 'amount': [Stocks, Time]
                - 'pct_chg': [Stocks, Time]
                - 'pe': [Stocks, Time]
                - 'pb': [Stocks, Time]
        
        Returns:
            特征张量 [Stocks, Features, Time]
        """
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        pct_chg = raw_dict.get('pct_chg', torch.zeros_like(c))
        
        # 1. 收益率
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        ret[:, 0] = 0
        
        # 2. 动量 (20日)
        momentum = AStockIndicators.momentum(c, window=20)
        
        # 3. 波动率 (20日)
        volatility = AStockIndicators.volatility(c, window=20)
        
        # 4. 量价背离
        vol_price_div = AStockIndicators.volume_price_divergence(c, v)
        
        # 5. RSI (14日)
        rsi = AStockIndicators.rsi(c, window=14)
        rsi_norm = (rsi - 50) / 50  # 归一化到 (-1, 1)
        
        # 6. MACD
        macd = AStockIndicators.macd(c)
        macd_norm = cls.robust_norm(macd)
        
        # 7. 价格位置 (20日)
        price_pos = AStockIndicators.price_position(c, h, l, window=20)
        
        # 8. 振幅
        amplitude = AStockIndicators.amplitude(h, l, c)
        amplitude_norm = cls.robust_norm(amplitude)
        
        # 9. 距离涨停
        limit_dist = AStockIndicators.limit_up_distance(c, pct_chg)
        limit_dist_norm = limit_dist / 10.0  # 归一化
        
        # 10. 量比 (当日成交量 / 5日均量)
        v_ma5 = torch.zeros_like(v)
        for i in range(5, v.shape[1]):
            v_ma5[:, i] = v[:, i-5:i].mean(dim=1)
        vol_ratio = v / (v_ma5 + 1e-9)
        vol_ratio_norm = cls.robust_norm(vol_ratio)
        
        # 堆叠特征 (10个)
        features = torch.stack([
            cls.robust_norm(ret),
            cls.robust_norm(momentum),
            cls.robust_norm(volatility),
            vol_price_div,
            rsi_norm,
            macd_norm,
            price_pos,
            amplitude_norm,
            limit_dist_norm,
            vol_ratio_norm,
        ], dim=1)
        
        # 处理 NaN
        features = torch.nan_to_num(features, nan=0.0)
        
        return features


class AdvancedAStockFeatureEngineer(AStockFeatureEngineer):
    """高级A股特征工程 (扩展版)"""
    
    INPUT_DIM = 14  # 10 基础 + 4 扩展
    
    FEATURE_NAMES = AStockFeatureEngineer.FEATURE_NAMES + [
        'skewness',      # 偏度
        'kurtosis',      # 峰度
        'up_shadow',     # 上影线
        'down_shadow',   # 下影线
    ]
    
    @classmethod
    def compute_features(cls, raw_dict: dict) -> torch.Tensor:
        """计算扩展特征"""
        # 获取基础特征
        base_features = super().compute_features(raw_dict)
        
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        
        # 计算收益率
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        ret[:, 0] = 0
        
        # 13. 偏度 (20日)
        skew = cls._rolling_skewness(ret, window=20)
        
        # 14. 峰度 (20日)
        kurt = cls._rolling_kurtosis(ret, window=20)
        
        # 15. 上影线
        body_top = torch.max(o, c)
        up_shadow = (h - body_top) / (h - l + 1e-9)
        
        # 16. 下影线
        body_bottom = torch.min(o, c)
        down_shadow = (body_bottom - l) / (h - l + 1e-9)
        
        # 合并特征
        extra_features = torch.stack([
            cls.robust_norm(skew),
            cls.robust_norm(kurt),
            up_shadow,
            down_shadow,
        ], dim=1)
        
        features = torch.cat([base_features, extra_features], dim=1)
        
        return features
    
    @staticmethod
    def _rolling_skewness(x: torch.Tensor, window: int) -> torch.Tensor:
        """计算滚动偏度"""
        pad = torch.zeros((x.shape[0], window-1), device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        
        result = torch.zeros_like(x)
        for i in range(window-1, x_pad.shape[1]):
            window_data = x_pad[:, i-window+1:i+1]
            mean = window_data.mean(dim=1, keepdim=True)
            std = window_data.std(dim=1, keepdim=True) + 1e-9
            skew = ((window_data - mean) ** 3).mean(dim=1) / (std.squeeze() ** 3)
            result[:, i-window+1] = skew
        
        return result
    
    @staticmethod
    def _rolling_kurtosis(x: torch.Tensor, window: int) -> torch.Tensor:
        """计算滚动峰度"""
        pad = torch.zeros((x.shape[0], window-1), device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        
        result = torch.zeros_like(x)
        for i in range(window-1, x_pad.shape[1]):
            window_data = x_pad[:, i-window+1:i+1]
            mean = window_data.mean(dim=1, keepdim=True)
            std = window_data.std(dim=1, keepdim=True) + 1e-9
            kurt = ((window_data - mean) ** 4).mean(dim=1) / (std.squeeze() ** 4) - 3
            result[:, i-window+1] = kurt
        
        return result

