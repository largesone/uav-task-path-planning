# -*- coding: utf-8 -*-
# 调试局部注意力初始化

import torch
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from transformer_gnn import TransformerGNN

class MockSpace:
    def __init__(self, shape):
        self.shape = shape

def test_local_attention_init():
    print("=== 调试局部注意力初始化 ===")
    
    # 创建观测空间
    obs_space = {
        'uav_features': MockSpace((5, 8)),
        'target_features': MockSpace((8, 6)),
        'relative_positions': MockSpace((5, 8, 2)),
        'distances': MockSpace((5, 8)),
        'masks': {
            'uav_mask': MockSpace((5,)),
            'target_mask': MockSpace((8,)),
        }
    }
    
    action_space = MockSpace((10,))
    
    # 测试配置
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "use_local_attention": True,
        "k_adaptive": True,
        "k_min": 2,
        "k_max": 8,
        "use_flash_attention": False,
        "use_noisy_linear": False
    }
    
    print("创建模型...")
    print(f"配置: {model_config}")
    
    # 创建模型
    try:
        model = TransformerGNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=10,
            model_config=model_config,
            name="debug_model"
        )
        print("模型创建成功")
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n模型属性检查:")
    print(f"- use_local_attention: {getattr(model, 'use_local_attention', 'NOT_FOUND')}")
    print(f"- local_attention: {getattr(model, 'local_attention', 'NOT_FOUND')}")
    print(f"- k_adaptive: {getattr(model, 'k_adaptive', 'NOT_FOUND')}")
    print(f"- k_min: {getattr(model, 'k_min', 'NOT_FOUND')}")
    print(f"- k_max: {getattr(model, 'k_max', 'NOT_FOUND')}")
    
    # 检查模型的所有属性
    print("\n所有模型属性:")
    for attr_name in dir(model):
        if not attr_name.startswith('_') and not callable(getattr(model, attr_name)):
            attr_value = getattr(model, attr_name)
            print(f"- {attr_name}: {type(attr_value)} = {attr_value}")

if __name__ == "__main__":
    test_local_attention_init()