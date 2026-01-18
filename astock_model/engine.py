"""
A股训练引擎

使用强化学习训练 AlphaGPT
"""
import torch
import json
from torch.distributions import Categorical
from tqdm import tqdm
from loguru import logger

from .alphagpt import AStockAlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import AStockStackVM
from .backtest import AStockBacktest
from .data_loader import AStockDataLoader


class ModelConfig:
    """模型配置"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据 GPU 显存调整 batch size
    # 10GB 显存建议 256-512，8GB 显存建议 128-256
    BATCH_SIZE = 256
    
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 16
    D_MODEL = 128
    
    # 保存路径
    BEST_STRATEGY_PATH = "best_astock_strategy.json"
    TRAINING_HISTORY_PATH = "astock_training_history.json"


class AStockAlphaEngine:
    """A股训练引擎"""
    
    def __init__(self, use_lord: bool = True, lord_decay_rate: float = 1e-3):
        """
        初始化训练引擎
        
        Args:
            use_lord: 是否使用 LoRD 正则化
            lord_decay_rate: LoRD 衰减率
        """
        # 加载数据
        logger.info("正在加载数据...")
        self.loader = AStockDataLoader(device=str(ModelConfig.DEVICE))
        # min_days=20: 只要有20天数据就可以训练 (适合刚同步的数据)
        self.loader.load_data(limit_stocks=500, lookback_days=250, min_days=20)
        
        if self.loader.feat_tensor is None:
            raise ValueError("数据加载失败！请检查:\n"
                           "  1) 是否运行了数据管线: python run_astock_pipeline.py\n"
                           "  2) MiniQMT 是否启动并登录\n"
                           "  3) 数据库 daily_kline 表是否有数据")
        
        # 初始化模型
        self.model = AStockAlphaGPT(
            d_model=ModelConfig.D_MODEL,
            max_formula_len=ModelConfig.MAX_FORMULA_LEN
        ).to(ModelConfig.DEVICE)
        
        # 优化器
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # LoRD 正则化
        self.use_lord = use_lord
        if use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=5,
                target_keywords=["attention", "q_proj", "k_proj"]
            )
            self.rank_monitor = StableRankMonitor(self.model)
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        # 虚拟机和回测
        self.vm = AStockStackVM()
        self.bt = AStockBacktest()
        
        # 记录
        self.best_score = float('-inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }
    
    def train(self):
        """训练模型"""
        logger.info("=" * 50)
        logger.info("🚀 A股 AlphaGPT 训练开始")
        if self.use_lord:
            logger.info("   LoRD 正则化已启用")
        logger.info("=" * 50)
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS), desc="Training")
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            
            # 初始输入 (起始 Token)
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            # 自回归生成公式
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            # 组合公式
            seqs = torch.stack(tokens_list, dim=1)  # [Batch, SeqLen]
            
            # 评估奖励
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                # 执行公式
                result = self.vm.execute(formula, self.loader.feat_tensor)
                
                if result is None:
                    rewards[i] = -5.0
                    continue
                
                # 检查变异性
                if result.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                # 回测评估
                score, ret_val = self.bt.evaluate(
                    result, 
                    self.loader.raw_data_cache, 
                    self.loader.target_ret
                )
                rewards[i] = score
                
                # 记录最佳
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    
                    formula_str = self.vm.decode_formula(formula)
                    tqdm.write(f"[!] 新最优: Score={score:.3f} | Ret={ret_val:.2%}")
                    tqdm.write(f"    公式: {formula_str}")
            
            # 计算优势
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            # 策略梯度损失
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            loss = loss.mean()
            
            # 反向传播
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # 应用 LoRD
            if self.use_lord:
                self.lord_opt.step()
            
            # 记录
            avg_reward = rewards.mean().item()
            postfix = {'AvgRew': f"{avg_reward:.3f}", 'Best': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                rank = self.rank_monitor.compute()
                postfix['Rank'] = f"{rank:.2f}"
                self.training_history['stable_rank'].append(rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix)
        
        # 保存结果
        self._save_results()
        
        logger.info("=" * 50)
        logger.info("✅ 训练完成!")
        logger.info(f"   最优得分: {self.best_score:.4f}")
        logger.info(f"   最优公式: {self.vm.decode_formula(self.best_formula)}")
        logger.info("=" * 50)
    
    def _save_results(self):
        """保存结果"""
        # 保存策略
        strategy_data = {
            'formula': self.best_formula,
            'formula_str': self.vm.decode_formula(self.best_formula),
            'score': self.best_score,
        }
        
        with open(ModelConfig.BEST_STRATEGY_PATH, 'w', encoding='utf-8') as f:
            json.dump(strategy_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"策略已保存到: {ModelConfig.BEST_STRATEGY_PATH}")
        
        # 保存训练历史
        with open(ModelConfig.TRAINING_HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练历史已保存到: {ModelConfig.TRAINING_HISTORY_PATH}")


def train_astock():
    """训练A股策略"""
    engine = AStockAlphaEngine(use_lord=True, lord_decay_rate=1e-3)
    engine.train()


if __name__ == "__main__":
    train_astock()

