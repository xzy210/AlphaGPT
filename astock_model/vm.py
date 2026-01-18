"""
A股栈式虚拟机

执行因子公式
"""
import torch
from typing import List, Optional
from .ops import OPS_CONFIG
from .factors import AStockFeatureEngineer


class AStockStackVM:
    """A股栈式虚拟机"""
    
    def __init__(self):
        # 特征数量作为偏移量
        self.feat_offset = AStockFeatureEngineer.INPUT_DIM
        
        # 构建算子映射
        self.op_map = {
            i + self.feat_offset: cfg[1] 
            for i, cfg in enumerate(OPS_CONFIG)
        }
        
        self.arity_map = {
            i + self.feat_offset: cfg[2] 
            for i, cfg in enumerate(OPS_CONFIG)
        }
    
    def execute(self, formula_tokens: List[int], 
                feat_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        执行公式
        
        Args:
            formula_tokens: Token 序列 (整数列表)
            feat_tensor: 特征张量 [Stocks, Features, Time]
        
        Returns:
            因子信号 [Stocks, Time]，失败返回 None
        """
        stack = []
        
        try:
            for token in formula_tokens:
                token = int(token)
                
                if token < self.feat_offset:
                    # 特征 Token: 将对应特征压栈
                    # feat_tensor: [Stocks, Features, Time]
                    stack.append(feat_tensor[:, token, :])
                    
                elif token in self.op_map:
                    # 算子 Token: 弹出操作数，计算，压栈
                    arity = self.arity_map[token]
                    
                    if len(stack) < arity:
                        return None  # 参数不足
                    
                    # 弹出参数
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()  # 恢复顺序
                    
                    # 执行算子
                    func = self.op_map[token]
                    result = func(*args)
                    
                    # 处理异常值
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    stack.append(result)
                    
                else:
                    # 无效 Token
                    return None
            
            # 最终栈中应该只有一个元素
            if len(stack) == 1:
                return stack[0]
            else:
                return None
                
        except Exception as e:
            return None
    
    def validate_formula(self, formula_tokens: List[int]) -> bool:
        """
        验证公式是否有效
        
        Args:
            formula_tokens: Token 序列
        
        Returns:
            是否有效
        """
        stack_depth = 0
        
        for token in formula_tokens:
            token = int(token)
            
            if token < self.feat_offset:
                # 特征: 压栈
                stack_depth += 1
                
            elif token in self.arity_map:
                # 算子: 弹出参数，压入结果
                arity = self.arity_map[token]
                
                if stack_depth < arity:
                    return False
                
                stack_depth -= arity  # 弹出参数
                stack_depth += 1      # 压入结果
                
            else:
                return False
        
        return stack_depth == 1
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return self.feat_offset + len(OPS_CONFIG)
    
    def decode_formula(self, formula_tokens: List[int]) -> str:
        """
        将公式 Token 解码为可读字符串
        
        Args:
            formula_tokens: Token 序列
        
        Returns:
            可读字符串
        """
        feature_names = AStockFeatureEngineer.FEATURE_NAMES
        op_names = [cfg[0] for cfg in OPS_CONFIG]
        
        parts = []
        for token in formula_tokens:
            token = int(token)
            
            if token < self.feat_offset:
                if token < len(feature_names):
                    parts.append(feature_names[token])
                else:
                    parts.append(f"F{token}")
            else:
                op_idx = token - self.feat_offset
                if op_idx < len(op_names):
                    parts.append(op_names[op_idx])
                else:
                    parts.append(f"OP{op_idx}")
        
        return " ".join(parts)

