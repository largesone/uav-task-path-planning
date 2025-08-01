# -*- coding: utf-8 -*-
# 调试NoisyLinear替换问题

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformer_gnn import TransformerGNN
from noisy_linear import NoisyLinear

# 创建简单的观测空间
class SimpleSpace:
    def __init__(self, shape):
        self.shape = shape

class DictSpace:
    def __init__(self, spaces_dict):
        self.spaces = spaces_dict
        for key, value in spaces_dict.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return self.spaces[key]

def test_noisy_replacement():
    print("=== 调试NoisyLinear替换 ===")
    
    # 创建观测空间
    obs_space = DictSpace({
        'uav_features': SimpleSpace((5, 8)),
        'target_features': SimpleSpace((3, 6)),
        'relative_positions': SimpleSpace((15, 2)),
        'distances': SimpleSpace((5, 3)),
        'masks': DictSpace({
            'uav_mask': SimpleSpace((5,)),
            'target_mask': SimpleSpace((3,))
        })
    })
    
    action_space = SimpleSpace((10,))
    
    # 测试配置
    model_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "use_noisy_linear": True,
        "noisy_std_init": 0.3
    }
    
    print("创建模型...")
    model = TransformerGNN(obs_space, action_space, 10, model_config, "test")
    
    print("检查模型结构...")
    
    # 检查各个组件
    print(f"uav_encoder类型: {type(model.uav_encoder)}")
    print(f"target_encoder类型: {type(model.target_encoder)}")
    
    # 检查输出层
    print("输出层结构:")
    for i, layer in enumerate(model.output_layer):
        print(f"  {i}: {type(layer)} - {layer}")
    
    # 检查值函数头
    print("值函数头结构:")
    for i, layer in enumerate(model.value_head):
        print(f"  {i}: {type(layer)} - {layer}")
    
    # 统计层类型
    linear_count = 0
    noisy_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_count += 1
            print(f"Linear层: {name}")
        elif isinstance(module, NoisyLinear):
            noisy_count += 1
            print(f"NoisyLinear层: {name}")
    
    print(f"\n总计 - Linear: {linear_count}, NoisyLinear: {noisy_count}")
    
    # 测试_replace_with_noisy_linear方法
    print("\n手动调用替换方法...")
    if hasattr(model, '_replace_with_noisy_linear'):
        model._replace_with_noisy_linear()
        
        # 重新统计
        linear_count_after = 0
        noisy_count_after = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_count_after += 1
            elif isinstance(module, NoisyLinear):
                noisy_count_after += 1
        
        print(f"替换后 - Linear: {linear_count_after}, NoisyLinear: {noisy_count_after}")
    else:
        print("模型没有_replace_with_noisy_linear方法")

if __name__ == "__main__":
    test_noisy_replacement()