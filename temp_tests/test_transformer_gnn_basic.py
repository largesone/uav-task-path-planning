# -*- coding: utf-8 -*-
# 文件名: temp_tests/test_transformer_gnn_basic.py
# 描述: TransformerGNN基础功能测试

import torch
import numpy as np
from gymnasium import spaces
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_gnn import TransformerGNN, create_transformer_gnn_model

def test_transformer_gnn_dict_obs():
    """测试TransformerGNN在图模式观测下的基本功能"""
    print("=== 测试TransformerGNN图模式观测 ===")
    
    # 定义图模式观测空间
    obs_space = spaces.Dict({
        'uav_features': spaces.Box(low=0.0, high=1.0, shape=(3, 9), dtype=np.float32),  # 3个UAV，每个9个特征
        'target_features': spaces.Box(low=0.0, high=1.0, shape=(2, 8), dtype=np.float32),  # 2个目标，每个8个特征
        'relative_positions': spaces.Box(low=-1.0, high=1.0, shape=(3, 2, 2), dtype=np.float32),  # 3x2的相对位置
        'distances': spaces.Box(low=0.0, high=1.0, shape=(3, 2), dtype=np.float32),  # 3x2的距离矩阵
        'masks': spaces.Dict({
            'uav_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32),
            'target_mask': spaces.Box(low=0, high=1, shape=(2,), dtype=np.int32)
        })
    })
    
    action_space = spaces.Discrete(36)  # 3 UAV * 2 目标 * 6 方向
    num_outputs = 36
    
    # 模型配置
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": False  # 暂时关闭噪声线性层以简化测试
    }
    
    # 创建模型
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test_model")
    
    # 创建测试输入
    batch_size = 2
    test_obs = {
        'uav_features': torch.randn(batch_size, 3, 9),
        'target_features': torch.randn(batch_size, 2, 8),
        'relative_positions': torch.randn(batch_size, 3, 2, 2),
        'distances': torch.rand(batch_size, 3, 2),
        'masks': {
            'uav_mask': torch.ones(batch_size, 3, dtype=torch.int32),
            'target_mask': torch.ones(batch_size, 2, dtype=torch.int32)
        }
    }
    
    # 前向传播测试
    input_dict = {"obs": test_obs}
    state = []
    seq_lens = torch.tensor([1] * batch_size)
    
    print(f"输入观测形状:")
    for key, value in test_obs.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    # 执行前向传播
    logits, new_state = model.forward(input_dict, state, seq_lens)
    value = model.value_function()
    
    print(f"输出logits形状: {logits.shape}")
    print(f"输出值函数形状: {value.shape}")
    print(f"logits范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"值函数范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    # 验证输出形状
    assert logits.shape == (batch_size, num_outputs), f"期望logits形状: ({batch_size}, {num_outputs}), 实际: {logits.shape}"
    assert value.shape == (batch_size,), f"期望值函数形状: ({batch_size},), 实际: {value.shape}"
    
    print("✓ 图模式观测测试通过")
    return True

def test_transformer_gnn_flat_obs():
    """测试TransformerGNN在扁平模式观测下的基本功能"""
    print("\n=== 测试TransformerGNN扁平模式观测 ===")
    
    # 定义扁平模式观测空间
    input_dim = 100
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(input_dim,), dtype=np.float32)
    action_space = spaces.Discrete(36)
    num_outputs = 36
    
    # 模型配置
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": False
    }
    
    # 创建模型
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test_model_flat")
    
    # 创建测试输入
    batch_size = 2
    test_obs = torch.randn(batch_size, input_dim)
    
    # 前向传播测试
    input_dict = {"obs": test_obs}
    state = []
    seq_lens = torch.tensor([1] * batch_size)
    
    print(f"输入观测形状: {test_obs.shape}")
    
    # 执行前向传播
    logits, new_state = model.forward(input_dict, state, seq_lens)
    value = model.value_function()
    
    print(f"输出logits形状: {logits.shape}")
    print(f"输出值函数形状: {value.shape}")
    print(f"logits范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"值函数范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    # 验证输出形状
    assert logits.shape == (batch_size, num_outputs), f"期望logits形状: ({batch_size}, {num_outputs}), 实际: {logits.shape}"
    assert value.shape == (batch_size,), f"期望值函数形状: ({batch_size},), 实际: {value.shape}"
    
    print("✓ 扁平模式观测测试通过")
    return True

def test_model_parameters():
    """测试模型参数数量和结构"""
    print("\n=== 测试模型参数 ===")
    
    # 创建一个简单的模型
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
    action_space = spaces.Discrete(10)
    num_outputs = 10
    
    model_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": False
    }
    
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test_params")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 检查主要组件
    print(f"UAV编码器参数: {sum(p.numel() for p in model.uav_encoder.parameters()):,}")
    print(f"目标编码器参数: {sum(p.numel() for p in model.target_encoder.parameters()):,}")
    print(f"Transformer编码器参数: {sum(p.numel() for p in model.transformer_encoder.parameters()):,}")
    print(f"输出层参数: {sum(p.numel() for p in model.output_layer.parameters()):,}")
    print(f"值函数头参数: {sum(p.numel() for p in model.value_head.parameters()):,}")
    
    if model.use_position_encoding:
        print(f"位置编码器参数: {sum(p.numel() for p in model.position_encoder.parameters()):,}")
    
    print("✓ 参数统计完成")
    return True

def test_factory_function():
    """测试工厂函数"""
    print("\n=== 测试工厂函数 ===")
    
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    action_space = spaces.Discrete(5)
    num_outputs = 5
    model_config = {"embed_dim": 16}
    
    model = create_transformer_gnn_model(obs_space, action_space, num_outputs, model_config)
    
    assert isinstance(model, TransformerGNN), "工厂函数应该返回TransformerGNN实例"
    print("✓ 工厂函数测试通过")
    return True

def main():
    """运行所有测试"""
    print("开始TransformerGNN基础功能测试...")
    
    try:
        # 运行所有测试
        test_transformer_gnn_dict_obs()
        test_transformer_gnn_flat_obs()
        test_model_parameters()
        test_factory_function()
        
        print("\n🎉 所有测试通过！TransformerGNN基础架构实现正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()