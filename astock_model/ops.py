"""
A股算子定义

扩展原有算子，增加 A股特有算子
"""
import torch

# ========== 时序延迟函数 ==========

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序延迟"""
    if d == 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)


@torch.jit.script
def _ts_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动求和"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    result = torch.zeros_like(x)
    for i in range(window-1, x_pad.shape[1]):
        result[:, i-window+1] = x_pad[:, i-window+1:i+1].sum(dim=1)
    return result


@torch.jit.script
def _ts_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动均值"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    result = torch.zeros_like(x)
    for i in range(window-1, x_pad.shape[1]):
        result[:, i-window+1] = x_pad[:, i-window+1:i+1].mean(dim=1)
    return result


@torch.jit.script
def _ts_std(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动标准差"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    result = torch.zeros_like(x)
    for i in range(window-1, x_pad.shape[1]):
        result[:, i-window+1] = x_pad[:, i-window+1:i+1].std(dim=1)
    return result


@torch.jit.script
def _ts_max(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动最大值"""
    pad = torch.full((x.shape[0], window-1), float('-inf'), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    result = torch.zeros_like(x)
    for i in range(window-1, x_pad.shape[1]):
        result[:, i-window+1] = x_pad[:, i-window+1:i+1].max(dim=1)[0]
    return result


@torch.jit.script
def _ts_min(x: torch.Tensor, window: int) -> torch.Tensor:
    """滚动最小值"""
    pad = torch.full((x.shape[0], window-1), float('inf'), device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    result = torch.zeros_like(x)
    for i in range(window-1, x_pad.shape[1]):
        result[:, i-window+1] = x_pad[:, i-window+1:i+1].min(dim=1)[0]
    return result


# ========== 基础算子 ==========

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """条件选择: condition > 0 ? x : y"""
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """极端值检测 (z-score > 3)"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)


@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    """指数衰减"""
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2) + 0.4 * _ts_delay(x, 3)


# ========== A股特有算子 ==========

@torch.jit.script
def _op_rank(x: torch.Tensor) -> torch.Tensor:
    """截面排名 (每个时间点排名)"""
    n_stocks = x.shape[0]
    ranks = torch.zeros_like(x)
    
    for i in range(x.shape[1]):
        col = x[:, i]
        sorted_idx = col.argsort()
        rank = torch.zeros_like(col)
        rank[sorted_idx] = torch.arange(n_stocks, device=x.device, dtype=x.dtype)
        ranks[:, i] = rank / n_stocks
    
    return ranks


@torch.jit.script
def _op_zscore(x: torch.Tensor) -> torch.Tensor:
    """截面 Z-score 标准化"""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std


@torch.jit.script
def _op_neutralize(x: torch.Tensor) -> torch.Tensor:
    """截面中性化 (减去均值)"""
    mean = x.mean(dim=0, keepdim=True)
    return x - mean


@torch.jit.script
def _op_winsorize(x: torch.Tensor) -> torch.Tensor:
    """缩尾处理 (3 sigma)"""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    upper = mean + 3 * std
    lower = mean - 3 * std
    return torch.clamp(x, lower, upper)


@torch.jit.script
def _op_ts_corr(x: torch.Tensor, y: torch.Tensor, window: int = 20) -> torch.Tensor:
    """时序相关性"""
    result = torch.zeros_like(x)
    
    for i in range(window-1, x.shape[1]):
        x_win = x[:, i-window+1:i+1]
        y_win = y[:, i-window+1:i+1]
        
        x_mean = x_win.mean(dim=1, keepdim=True)
        y_mean = y_win.mean(dim=1, keepdim=True)
        
        x_std = x_win.std(dim=1, keepdim=True) + 1e-6
        y_std = y_win.std(dim=1, keepdim=True) + 1e-6
        
        cov = ((x_win - x_mean) * (y_win - y_mean)).mean(dim=1)
        corr = cov / (x_std.squeeze() * y_std.squeeze())
        
        result[:, i] = corr
    
    return result


@torch.jit.script
def _op_delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """差分"""
    return x - _ts_delay(x, d)


@torch.jit.script
def _op_pct_change(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """百分比变化"""
    prev = _ts_delay(x, d)
    return (x - prev) / (prev + 1e-9)


# ========== 算子配置 ==========

OPS_CONFIG = [
    # 基础算术 (索引 12-17)
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    
    # 数学函数 (索引 18-21)
    ('SIGN', torch.sign, 1),
    ('LOG', lambda x: torch.log(torch.abs(x) + 1e-9), 1),
    ('SQRT', lambda x: torch.sqrt(torch.abs(x)), 1),
    ('SQUARE', lambda x: x ** 2, 1),
    
    # 时序算子 (索引 22-29)
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('DELAY5', lambda x: _ts_delay(x, 5), 1),
    ('MA5', lambda x: _ts_mean(x, 5), 1),
    ('MA10', lambda x: _ts_mean(x, 10), 1),
    ('MA20', lambda x: _ts_mean(x, 20), 1),
    ('STD20', lambda x: _ts_std(x, 20), 1),
    ('MAX20', lambda x: _ts_max(x, 20), 1),
    ('MIN20', lambda x: _ts_min(x, 20), 1),
    
    # 特殊算子 (索引 30-35)
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELTA', lambda x: _op_delta(x, 1), 1),
    ('PCTCHG', lambda x: _op_pct_change(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
    
    # A股特有算子 (索引 36-40)
    ('RANK', _op_rank, 1),
    ('ZSCORE', _op_zscore, 1),
    ('NEUTRAL', _op_neutralize, 1),
    ('WINSOR', _op_winsorize, 1),
    ('CORR20', lambda x, y: _op_ts_corr(x, y, 20), 2),
]

# 算子名称列表
OPS_NAMES = [cfg[0] for cfg in OPS_CONFIG]

# 算子数量
NUM_OPS = len(OPS_CONFIG)

