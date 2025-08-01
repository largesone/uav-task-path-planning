# -*- coding: utf-8 -*-
# 文件名: test_transformer_gnn_noisy.py
# 描述: 测试TransformerGNN与NoisyLinear的集成

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 尝试导入gymnasium，如果失败则创建简单的spaces模拟
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # 创建简单的spaces模拟
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
        
        class Dict:
            def __init__(self, space_dict):
                self.spaces = space_dict
                for key, value in space_dict.items():
                    setattr(self, key, value)
        
        class Discrete:
            def __init__(self, n):
                self.n = n

# 添加路径以导入TransformerGNN
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transformer_gnn import TransformerGNN, create_transformer_gnn_model
from temp_tests.noisy_linear import NoisyLinear


def create_test_obs_space():
    """创建测试用的观测空间"""
    spaces_dict = {
        'uav_features': spaces.Box(low=-1, high=1, shape=(5, 8), dtype=np.float32),
        'target_features': spaces.Box(low=-1, high=1, shape=(3, 6), dtype=np.float32),
        'relative_positions': spaces.Box(low=-10, high=10, shape=(15, 2), dtype=np.float32),
        'distances': spaces.Box(low=0, high=20, shape=(5, 3), dtype=np.float32),
        'masks': spaces.Dict({
            'uav_mask': spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32),
            'target_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32)
        })
    }
    
    # 创建支持字典访问的观测空间
    class DictObsSpace:
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict
            for key, value in spaces_dict.items():
                setattr(self, key, value)
        
        def __getitem__(self, key):
            return self.spaces[key]
    
    return DictObsSpace(spaces_dict)


def create_test_observation():
    """创建测试用的观测数据"""
    return {
        'uav_features': torch.randn(2, 5, 8),
        'target_features': torch.randn(2, 3, 6),
        'relative_positions': torch.randn(2, 15, 2),
        'distances': torch.abs(torch.randn(2, 5, 3)),
        'masks': {
            'uav_mask': torch.ones(2, 5, dtype=torch.int32),
            'target_mask': torch.ones(2, 3, dtype=torch.int32)
        }
    }


def test_transformer_gnn_with_noisy_linear():
    """测试TransformerGNN与NoisyLinear的集成"""
    print("=== 测试TransformerGNN与NoisyLinear集成 ===")
    
    # 创建观测空间和动作空间
    obs_space = create_test_obs_space()
    action_space = spaces.Discrete(10)
    num_outputs = 10
    
    # 模型配置
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": True,
        "noisy_std_init": 0.3
    }
    
    # 创建模型
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test")
    
    # 检查是否成功替换了Linear层
    noisy_count = sum(1 for m in model.modules() if isinstance(m, NoisyLinear))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    
    print(f"模型中NoisyLinear层数量: {noisy_count}")
    print(f"模型中剩余Linear层数量: {linear_count}")
    print(f"NoisyLinear替换成功: {noisy_count > 0}")
    
    # 创建测试输入
    obs = create_test_observation()
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1])
    
    # 测试训练模式下的随机性
    model.train()
    
    logits1, _ = model(input_dict, state, seq_lens)
    logits2, _ = model(input_dict, state, seq_lens)
    
    print(f"训练模式下两次前向传播结果是否不同: {not torch.allclose(logits1, logits2, atol=1e-6)}")
    
    # 测试推理模式下的确定性
    model.eval()
    
    logits3, _ = model(input_dict, state, seq_lens)
    logits4, _ = model(input_dict, state, seq_lens)
    
    print(f"推理模式下两次前向传播结果是否相同: {torch.allclose(logits3, logits4)}")
    
    # 测试值函数
    value1 = model.value_function()
    value2 = model.value_function()
    
    print(f"值函数输出形状: {value1.shape}")
    print(f"值函数输出是否一致: {torch.allclose(value1, value2)}")
    
    print()


def test_transformer_gnn_without_noisy_linear():
    """测试不使用NoisyLinear的TransformerGNN"""
    print("=== 测试不使用NoisyLinear的TransformerGNN ===")
    
    # 创建观测空间和动作空间
    obs_space = create_test_obs_space()
    action_space = spaces.Discrete(10)
    num_outputs = 10
    
    # 模型配置（禁用NoisyLinear）
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": False,
        "noisy_std_init": 0.3
    }
    
    # 创建模型
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test")
    
    # 检查Linear层数量
    noisy_count = sum(1 for m in model.modules() if isinstance(m, NoisyLinear))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    
    print(f"模型中NoisyLinear层数量: {noisy_count}")
    print(f"模型中Linear层数量: {linear_count}")
    print(f"未使用NoisyLinear: {noisy_count == 0 and linear_count > 0}")
    
    # 创建测试输入
    obs = create_test_observation()
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1])
    
    # 测试确定性（应该在训练和推理模式下都是确定的）
    model.train()
    logits1, _ = model(input_dict, state, seq_lens)
    logits2, _ = model(input_dict, state, seq_lens)
    
    print(f"训练模式下两次前向传播结果是否相同: {torch.allclose(logits1, logits2)}")
    
    model.eval()
    logits3, _ = model(input_dict, state, seq_lens)
    logits4, _ = model(input_dict, state, seq_lens)
    
    print(f"推理模式下两次前向传播结果是否相同: {torch.allclose(logits3, logits4)}")
    
    print()


def test_noise_reset_functionality():
    """测试噪声重置功能"""
    print("=== 测试噪声重置功能 ===")
    
    # 创建使用NoisyLinear的模型
    obs_space = create_test_obs_space()
    action_space = spaces.Discrete(10)
    num_outputs = 10
    
    model_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "use_noisy_linear": True,
        "noisy_std_init": 0.5
    }
    
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test")
    model.train()
    
    # 创建测试输入
    obs = create_test_observation()
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1])
    
    # 第一次前向传播
    logits1, _ = model(input_dict, state, seq_lens)
    
    # 手动重置噪声
    model.reset_noise()
    
    # 第二次前向传播
    logits2, _ = model(input_dict, state, seq_lens)
    
    print(f"手动重置噪声后结果是否不同: {not torch.allclose(logits1, logits2, atol=1e-6)}")
    
    # 测试自动重置（每次forward调用都会重置）
    logits3, _ = model(input_dict, state, seq_lens)
    logits4, _ = model(input_dict, state, seq_lens)
    
    print(f"自动重置噪声后结果是否不同: {not torch.allclose(logits3, logits4, atol=1e-6)}")
    
    print()


def test_gradient_flow():
    """测试梯度流动"""
    print("=== 测试梯度流动 ===")
    
    # 创建模型
    obs_space = create_test_obs_space()
    action_space = spaces.Discrete(5)
    num_outputs = 5
    
    model_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "use_noisy_linear": True
    }
    
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test")
    model.train()
    
    # 创建测试输入和目标
    obs = create_test_observation()
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1])
    
    target_logits = torch.randn(2, 5)
    target_values = torch.randn(2)
    
    # 前向传播
    logits, _ = model(input_dict, state, seq_lens)
    values = model.value_function()
    
    # 计算损失
    logits_loss = nn.MSELoss()(logits, target_logits)
    value_loss = nn.MSELoss()(values, target_values)
    total_loss = logits_loss + value_loss
    
    # 反向传播
    total_loss.backward()
    
    # 检查梯度
    has_gradients = []
    gradient_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients.append(True)
            gradient_norms.append(param.grad.norm().item())
        else:
            has_gradients.append(False)
            gradient_norms.append(0.0)
    
    print(f"有梯度的参数比例: {sum(has_gradients) / len(has_gradients):.2%}")
    print(f"平均梯度范数: {np.mean(gradient_norms):.6f}")
    print(f"最大梯度范数: {max(gradient_norms):.6f}")
    print(f"梯度流动正常: {sum(has_gradients) > 0 and max(gradient_norms) > 0}")
    
    print()


def test_flat_observation_compatibility():
    """测试扁平观测的兼容性"""
    print("=== 测试扁平观测兼容性 ===")
    
    # 创建扁平观测空间
    obs_space = spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)
    action_space = spaces.Discrete(8)
    num_outputs = 8
    
    model_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "use_noisy_linear": True
    }
    
    model = TransformerGNN(obs_space, action_space, num_outputs, model_config, "test")
    
    # 创建扁平观测
    obs = torch.randn(3, 64)
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1, 1])
    
    # 测试前向传播
    try:
        logits, _ = model(input_dict, state, seq_lens)
        values = model.value_function()
        
        print(f"扁平观测输入形状: {obs.shape}")
        print(f"输出logits形状: {logits.shape}")
        print(f"输出values形状: {values.shape}")
        print(f"扁平观测兼容性测试: 成功")
        
    except Exception as e:
        print(f"扁平观测兼容性测试: 失败 - {e}")
    
    print()


if __name__ == "__main__":
    print("开始测试TransformerGNN与NoisyLinear集成...")
    print()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有测试
    test_transformer_gnn_with_noisy_linear()
    test_transformer_gnn_without_noisy_linear()
    test_noise_reset_functionality()
    test_gradient_flow()
    test_flat_observation_compatibility()
    
    print("所有集成测试完成！")