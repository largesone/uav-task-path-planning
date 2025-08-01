# -*- coding: utf-8 -*-
# 简单的NoisyLinear测试

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformer_gnn import TransformerGNN
from noisy_linear import NoisyLinear

# 创建最简单的测试
def simple_test():
    print("=== 简单NoisyLinear测试 ===")
    
    # 使用扁平观测空间（更简单）
    class SimpleBox:
        def __init__(self, shape):
            self.shape = shape
    
    obs_space = SimpleBox((64,))
    action_space = SimpleBox((10,))
    
    # 启用NoisyLinear的配置
    model_config = {
        "use_noisy_linear": True,
        "noisy_std_init": 0.3,
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1
    }
    
    print("创建模型...")
    model = TransformerGNN(obs_space, action_space, 10, model_config, "test")
    
    print("检查模型中的层类型...")
    
    # 详细检查每个模块
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, NoisyLinear)):
            print(f"{name}: {type(module).__name__}")
    
    # 统计
    noisy_count = sum(1 for m in model.modules() if isinstance(m, NoisyLinear))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    
    print(f"\n统计结果:")
    print(f"NoisyLinear层: {noisy_count}")
    print(f"Linear层: {linear_count}")
    
    # 测试功能
    print("\n测试功能...")
    x = torch.randn(2, 64)
    input_dict = {"obs": x}
    state = []
    seq_lens = torch.tensor([1, 1])
    
    model.train()
    logits1, _ = model(input_dict, state, seq_lens)
    logits2, _ = model(input_dict, state, seq_lens)
    
    print(f"训练模式下结果不同: {not torch.allclose(logits1, logits2, atol=1e-6)}")
    
    model.eval()
    logits3, _ = model(input_dict, state, seq_lens)
    logits4, _ = model(input_dict, state, seq_lens)
    
    print(f"推理模式下结果相同: {torch.allclose(logits3, logits4)}")

if __name__ == "__main__":
    simple_test()