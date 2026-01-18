"""
A股版 AlphaGPT 模型

基于 Looped Transformer 的因子公式生成器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .factors import AStockFeatureEngineer
from .ops import OPS_CONFIG


class RMSNorm(nn.Module):
    """Root Mean Square 归一化"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    """Query-Key 归一化"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    """Swish GLU 激活"""
    
    def __init__(self, d_in: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.fc(x)


class LoopedTransformerLayer(nn.Module):
    """循环 Transformer 层"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 num_loops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead
        
        # QK-Norm
        self.qk_norm = QKNorm(d_model // nhead)
        
        # 注意力
        self.attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        
        # RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU FFN
        self.ffn = SwiGLU(d_model, dim_feedforward)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask=None, is_causal: bool = False):
        for _ in range(self.num_loops):
            # Self-attention
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(
                x_norm, x_norm, x_norm, 
                attn_mask=mask, is_causal=is_causal
            )
            x = x + self.dropout(attn_out)
            
            # FFN
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)
        
        return x


class LoopedTransformer(nn.Module):
    """循环 Transformer 编码器"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int, num_loops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask=None, is_causal: bool = False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class MTPHead(nn.Module):
    """多任务输出头"""
    
    def __init__(self, d_model: int, vocab_size: int, num_tasks: int = 3):
        super().__init__()
        self.num_tasks = num_tasks
        
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
        ])
        
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks)
        )
    
    def forward(self, x: torch.Tensor):
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)
        
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
        return weighted, task_probs


class AStockAlphaGPT(nn.Module):
    """A股版 AlphaGPT"""
    
    def __init__(self, d_model: int = 128, max_formula_len: int = 16):
        super().__init__()
        
        self.d_model = d_model
        self.max_formula_len = max_formula_len
        
        # 词表
        self.features_list = AStockFeatureEngineer.FEATURE_NAMES
        self.ops_list = [cfg[0] for cfg in OPS_CONFIG]
        self.vocab = self.features_list + self.ops_list
        self.vocab_size = len(self.vocab)
        
        # Embedding
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_formula_len + 1, d_model))
        
        # Transformer
        self.blocks = LoopedTransformer(
            d_model=d_model,
            nhead=8,
            num_layers=3,
            dim_feedforward=d_model * 4,
            num_loops=3,
            dropout=0.1
        )
        
        # 最终归一化
        self.ln_f = RMSNorm(d_model)
        
        # 输出头
        self.mtp_head = MTPHead(d_model, self.vocab_size, num_tasks=3)
        self.head_critic = nn.Linear(d_model, 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, idx: torch.Tensor):
        """
        前向传播
        
        Args:
            idx: Token 索引 [Batch, SeqLen]
        
        Returns:
            logits: 下一个 Token 的概率 [Batch, VocabSize]
            value: 价值估计 [Batch, 1]
            task_probs: 任务概率 [Batch, NumTasks]
        """
        B, T = idx.size()
        
        # Embedding
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        # Transformer
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        # 取最后一个位置
        last_emb = x[:, -1, :]
        
        # 输出
        logits, task_probs = self.mtp_head(last_emb)
        value = self.head_critic(last_emb)
        
        return logits, value, task_probs
    
    def generate(self, start_token: int = 0, max_len: int = None, 
                 temperature: float = 1.0) -> torch.Tensor:
        """
        生成公式
        
        Args:
            start_token: 起始 Token
            max_len: 最大长度
            temperature: 采样温度
        
        Returns:
            生成的 Token 序列
        """
        if max_len is None:
            max_len = self.max_formula_len
        
        device = next(self.parameters()).device
        
        # 初始化序列
        tokens = torch.tensor([[start_token]], dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits, _, _ = self.forward(tokens)
            
            # 温度调整
            logits = logits / temperature
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens[0].tolist()


class NewtonSchulzLowRankDecay:
    """Low-Rank Decay (LoRD) 正则化"""
    
    def __init__(self, named_parameters, decay_rate: float = 1e-3, 
                 num_iterations: int = 5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self):
        """应用低秩衰减"""
        for name, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            norm = X.norm() + 1e-8
            X = X / norm
            
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    """稳定秩监控器"""
    
    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj"]
        self.history = []
    
    @torch.no_grad()
    def compute(self) -> float:
        """计算平均稳定秩"""
        ranks = []
        
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank

