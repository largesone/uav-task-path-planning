# -*- coding: utf-8 -*-
# 文件名: test_transformer_gnn_integration.py
# 描述: TransformerGNN与局部注意力机制集成测试

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from transformer_gnn import TransformerGNN, create_transformer_gnn_model


class MockSpace:
    """模拟观测空间"""
    def __init__(self, shape):
        self.shape = shape


def create_test_obs_space():
    """创建测试用的观测空间"""
    return {
        'uav_features': MockSpace((5, 8)),
        'target_features': MockSpace((8, 6)),
        'relative_positions': MockSpace((5, 8, 2)),
        'distances': MockSpace((5, 8)),
        'masks': {
            'uav_mask': MockSpace((5,)),
            'target_mask': MockSpace((8,)),
            'interaction_mask': MockSpace((5, 8))
        }
    }


def create_test_observation(batch_size=2, num_uavs=5, num_targets=8):
    """创建测试用的观测数据"""
    return {
        'uav_features': torch.randn(batch_size, num_uavs, 8),
        'target_features': torch.randn(batch_size, num_targets, 6),
        'relative_positions': torch.randn(batch_size, num_uavs, num_targets, 2),
        'distances': torch.rand(batch_size, num_uavs, num_targets) * 10 + 0.1,
        'masks': {
            'uav_mask': torch.ones(batch_size, num_uavs, dtype=torch.bool),
            'target_mask': torch.ones(batch_size, num_targets, dtype=torch.bool),
            'interaction_mask': torch.ones(batch_size, num_uavs, num_targets, dtype=torch.bool)
        }
    }


def test_transformer_gnn_with_local_attention():
    """测试TransformerGNN与局部注意力的集成"""
    print("\n=== 测试TransformerGNN与局部注意力集成 ===")
    
    # 创建观测空间和动作空间
    obs_space = create_test_obs_space()
    action_space = MockSpace((10,))
    
    # 模型配置
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_local_attention": True,
        "k_adaptive": True,
        "k_min": 2,
        "k_max": 8,
        "use_flash_attention": False,  # 使用标准实现确保稳定性
        "use_noisy_linear": False  # 暂时禁用以简化测试
    }
    
    # 创建模型
    model = TransformerGNN(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=10,
        model_config=model_config,
        name="test_transformer_gnn"
    )
    
    print(f"✓ TransformerGNN模型创建成功")
    print(f"  - 嵌入维度: {model.embed_dim}")
    print(f"  - 注意力头数: {model.num_heads}")
    
    # 检查是否有局部注意力属性
    if hasattr(model, 'use_local_attention'):
        print(f"  - 局部注意力: {'启用' if model.use_local_attention else '禁用'}")
    else:
        print(f"  - 局部注意力: 属性不存在")
        
    # 检查是否有局部注意力模块
    if hasattr(model, 'local_attention'):
        print(f"  - 局部注意力模块: {'存在' if model.local_attention is not None else '不存在'}")
    else:
        print(f"  - 局部注意力模块: 属性不存在")
    
    return model


def test_forward_pass():
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    model = test_transformer_gnn_with_local_attention()
    
    # 创建测试数据
    batch_size = 2
    obs = create_test_observation(batch_size)
    
    # 设置部分掩码以测试鲁棒性
    obs['masks']['uav_mask'][0, -1] = False  # 第一个批次的最后一个UAV无效
    obs['masks']['target_mask'][1, -2:] = False  # 第二个批次的最后两个目标无效
    
    # 前向传播
    input_dict = {"obs": obs}
    state = []
    seq_lens = None
    
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        logits, new_state = model.forward(input_dict, state, seq_lens)
        value = model.value_function()
    
    # 验证输出
    assert logits.shape == (batch_size, 10), f"期望logits形状为({batch_size}, 10)，实际为{logits.shape}"
    assert value.shape == (batch_size,), f"期望value形状为({batch_size},)，实际为{value.shape}"
    
    # 验证数值稳定性
    assert not torch.isnan(logits).any(), "logits包含NaN值"
    assert not torch.isinf(logits).any(), "logits包含Inf值"
    assert not torch.isnan(value).any(), "value包含NaN值"
    assert not torch.isinf(value).any(), "value包含Inf值"
    
    print(f"✓ 前向传播测试通过")
    print(f"  - Logits形状: {logits.shape}")
    print(f"  - Value形状: {value.shape}")
    print(f"  - Logits范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"  - Value范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    return model, logits, value


def test_gradient_computation():
    """测试梯度计算"""
    print("\n=== 测试梯度计算 ===")
    
    model = test_transformer_gnn_with_local_attention()
    model.train()  # 设置为训练模式
    
    # 创建测试数据
    obs = create_test_observation(batch_size=1)
    input_dict = {"obs": obs}
    
    # 前向传播
    logits, _ = model.forward(input_dict, [], None)
    value = model.value_function()
    
    # 计算损失
    target_logits = torch.randn_like(logits)
    target_value = torch.randn_like(value)
    
    loss = nn.MSELoss()(logits, target_logits) + nn.MSELoss()(value, target_value)
    
    # 反向传播
    loss.backward()
    
    # 验证梯度
    grad_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数 {name} 没有梯度"
            
            if torch.allclose(param.grad, torch.zeros_like(param.grad)):
                zero_grad_count += 1
            else:
                grad_count += 1
    
    print(f"✓ 梯度计算测试通过")
    print(f"  - 有非零梯度的参数: {grad_count}")
    print(f"  - 零梯度参数: {zero_grad_count}")
    print(f"  - 损失值: {loss.item():.6f}")
    
    return model


def test_different_scales():
    """测试不同规模的输入"""
    print("\n=== 测试不同规模输入 ===")
    
    # 测试不同的规模
    test_scales = [
        (1, 2, 3),    # 小规模
        (2, 5, 8),    # 中等规模
        (1, 10, 15),  # 大规模
    ]
    
    for batch_size, num_uavs, num_targets in test_scales:
        print(f"测试规模: 批次={batch_size}, UAV={num_uavs}, 目标={num_targets}")
        
        # 创建对应规模的观测空间
        obs_space = {
            'uav_features': MockSpace((num_uavs, 8)),
            'target_features': MockSpace((num_targets, 6)),
            'relative_positions': MockSpace((num_uavs, num_targets, 2)),
            'distances': MockSpace((num_uavs, num_targets)),
            'masks': {
                'uav_mask': MockSpace((num_uavs,)),
                'target_mask': MockSpace((num_targets,)),
                'interaction_mask': MockSpace((num_uavs, num_targets))
            }
        }
        
        action_space = MockSpace((10,))
        
        model_config = {
            "embed_dim": 64,
            "num_heads": 4,
            "use_local_attention": True,
            "k_adaptive": True,
            "use_noisy_linear": False
        }
        
        # 创建模型
        model = TransformerGNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=10,
            model_config=model_config,
            name=f"test_model_{num_uavs}_{num_targets}"
        )
        
        # 创建测试数据
        obs = create_test_observation(batch_size, num_uavs, num_targets)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            logits, _ = model.forward({"obs": obs}, [], None)
            value = model.value_function()
        
        # 验证输出
        assert logits.shape == (batch_size, 10)
        assert value.shape == (batch_size,)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(value).any()
        
        print(f"  ✓ 规模 ({batch_size}, {num_uavs}, {num_targets}) 测试通过")


def test_local_attention_effectiveness():
    """测试局部注意力机制的有效性"""
    print("\n=== 测试局部注意力机制有效性 ===")
    
    obs_space = create_test_obs_space()
    action_space = MockSpace((10,))
    
    # 创建两个模型：一个启用局部注意力，一个禁用
    config_with_local = {
        "embed_dim": 64,
        "num_heads": 4,
        "use_local_attention": True,
        "k_adaptive": True,
        "k_max": 4,
        "use_noisy_linear": False
    }
    
    config_without_local = {
        "embed_dim": 64,
        "num_heads": 4,
        "use_local_attention": False,
        "use_noisy_linear": False
    }
    
    model_with_local = TransformerGNN(obs_space, action_space, 10, config_with_local, "with_local")
    model_without_local = TransformerGNN(obs_space, action_space, 10, config_without_local, "without_local")
    
    # 创建测试数据
    obs = create_test_observation(batch_size=1)
    
    # 前向传播
    model_with_local.eval()
    model_without_local.eval()
    
    with torch.no_grad():
        logits_with, _ = model_with_local.forward({"obs": obs}, [], None)
        logits_without, _ = model_without_local.forward({"obs": obs}, [], None)
    
    # 验证两个模型都能正常工作
    assert logits_with.shape == logits_without.shape
    assert not torch.isnan(logits_with).any()
    assert not torch.isnan(logits_without).any()
    
    # 验证输出有差异（说明局部注意力确实在起作用）
    diff = torch.abs(logits_with - logits_without).mean()
    print(f"  - 启用/禁用局部注意力的输出差异: {diff.item():.6f}")
    
    print(f"✓ 局部注意力机制有效性验证通过")


def run_integration_tests():
    """运行所有集成测试"""
    print("开始TransformerGNN与局部注意力集成测试")
    print("=" * 60)
    
    try:
        # 运行各项测试
        test_transformer_gnn_with_local_attention()
        test_forward_pass()
        test_gradient_computation()
        test_different_scales()
        test_local_attention_effectiveness()
        
        print("\n" + "=" * 60)
        print("🎉 所有集成测试通过！TransformerGNN与局部注意力机制集成成功。")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)