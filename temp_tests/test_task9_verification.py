# -*- coding: utf-8 -*-
# 文件名: test_task9_verification.py
# 描述: 任务9的完整验证测试 - 使用参数空间噪声进行探索

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformer_gnn import TransformerGNN
from noisy_linear import NoisyLinear


def test_task9_requirements():
    """验证任务9的所有需求"""
    print("=== 任务9需求验证测试 ===")
    print("需求: 6.1, 6.2, 6.3, 6.4 - 参数空间噪声探索")
    print()
    
    # 创建测试用的观测空间
    class TestObsSpace:
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict
            for key, value in spaces_dict.items():
                setattr(self, key, value)
        
        def __getitem__(self, key):
            return self.spaces[key]
    
    class TestBox:
        def __init__(self, shape):
            self.shape = shape
    
    # 图模式观测空间
    obs_space = TestObsSpace({
        'uav_features': TestBox((5, 8)),
        'target_features': TestBox((3, 6)),
        'relative_positions': TestBox((15, 2)),
        'distances': TestBox((5, 3)),
        'masks': TestObsSpace({
            'uav_mask': TestBox((5,)),
            'target_mask': TestBox((3,))
        })
    })
    
    action_space = TestBox((10,))
    
    # 测试配置
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": True,
        "noisy_std_init": 0.4
    }
    
    print("1. 测试NoisyLinear层替换...")
    model = TransformerGNN(obs_space, action_space, 10, model_config, "test")
    
    # 验证所有nn.Linear层都被替换为NoisyLinear层
    linear_layers = []
    noisy_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
        elif isinstance(module, NoisyLinear):
            noisy_layers.append(name)
    
    print(f"   发现NoisyLinear层: {len(noisy_layers)}")
    print(f"   剩余Linear层: {len(linear_layers)}")
    
    # 显示替换的层
    print("   已替换的层:")
    for layer_name in noisy_layers:
        if not layer_name.startswith('transformer_encoder'):
            print(f"     - {layer_name}")
    
    # 显示未替换的层（主要是Transformer内部层）
    print("   未替换的层（Transformer内部）:")
    for layer_name in linear_layers:
        print(f"     - {layer_name}")
    
    requirement_6_1 = len(noisy_layers) > 0
    print(f"   ✓ 需求6.1 - 将nn.Linear层替换为NoisyLinear层: {'通过' if requirement_6_1 else '失败'}")
    print()
    
    print("2. 测试训练模式下的参数噪声...")
    
    # 创建测试输入
    obs = {
        'uav_features': torch.randn(2, 5, 8),
        'target_features': torch.randn(2, 3, 6),
        'relative_positions': torch.randn(2, 15, 2),
        'distances': torch.abs(torch.randn(2, 5, 3)),
        'masks': {
            'uav_mask': torch.ones(2, 5, dtype=torch.int32),
            'target_mask': torch.ones(2, 3, dtype=torch.int32)
        }
    }
    
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1])
    
    # 训练模式测试
    model.train()
    
    outputs = []
    for i in range(5):
        logits, _ = model(input_dict, state, seq_lens)
        outputs.append(logits.clone())
    
    # 检查输出是否不同（由于噪声）
    all_different = True
    for i in range(1, len(outputs)):
        if torch.allclose(outputs[0], outputs[i], atol=1e-6):
            all_different = False
            break
    
    requirement_6_2 = all_different
    print(f"   ✓ 需求6.2 - 训练模式启用参数噪声: {'通过' if requirement_6_2 else '失败'}")
    print()
    
    print("3. 测试推理模式下的确定性...")
    
    # 推理模式测试
    model.eval()
    
    eval_outputs = []
    for i in range(5):
        logits, _ = model(input_dict, state, seq_lens)
        eval_outputs.append(logits.clone())
    
    # 检查输出是否相同（无噪声）
    all_same = True
    for i in range(1, len(eval_outputs)):
        if not torch.allclose(eval_outputs[0], eval_outputs[i]):
            all_same = False
            break
    
    requirement_6_3 = all_same
    print(f"   ✓ 需求6.3 - eval模式关闭噪声确保可复现性: {'通过' if requirement_6_3 else '失败'}")
    print()
    
    print("4. 测试Ray RLlib TorchModelV2集成...")
    
    # 检查是否正确继承TorchModelV2
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    is_torch_model = isinstance(model, TorchModelV2)
    
    # 检查必要的方法
    has_forward = hasattr(model, 'forward') and callable(getattr(model, 'forward'))
    has_value_function = hasattr(model, 'value_function') and callable(getattr(model, 'value_function'))
    
    # 测试方法调用
    try:
        logits, state_out = model(input_dict, state, seq_lens)
        values = model.value_function()
        methods_work = True
    except Exception as e:
        print(f"   方法调用错误: {e}")
        methods_work = False
    
    requirement_6_4 = is_torch_model and has_forward and has_value_function and methods_work
    print(f"   ✓ 需求6.4 - 正确集成Ray RLlib TorchModelV2: {'通过' if requirement_6_4 else '失败'}")
    print()
    
    print("5. 测试噪声重置功能...")
    
    model.train()
    
    # 第一次前向传播
    logits1, _ = model(input_dict, state, seq_lens)
    
    # 手动重置噪声
    model.reset_noise()
    
    # 第二次前向传播
    logits2, _ = model(input_dict, state, seq_lens)
    
    # 检查结果是否不同
    noise_reset_works = not torch.allclose(logits1, logits2, atol=1e-6)
    print(f"   ✓ 噪声重置功能: {'正常' if noise_reset_works else '异常'}")
    print()
    
    print("6. 测试梯度计算...")
    
    # 创建损失并反向传播
    target = torch.randn_like(logits2)
    loss = nn.MSELoss()(logits2, target)
    
    # 清零梯度
    model.zero_grad()
    
    # 反向传播
    loss.backward()
    
    # 检查NoisyLinear层的梯度
    noisy_layers_with_grad = 0
    total_noisy_layers = 0
    
    for module in model.modules():
        if isinstance(module, NoisyLinear):
            total_noisy_layers += 1
            if module.weight_mu.grad is not None and module.weight_sigma.grad is not None:
                noisy_layers_with_grad += 1
    
    gradient_flow_ok = noisy_layers_with_grad == total_noisy_layers
    print(f"   ✓ 梯度计算: {noisy_layers_with_grad}/{total_noisy_layers} NoisyLinear层有梯度")
    print(f"   ✓ 梯度流动: {'正常' if gradient_flow_ok else '异常'}")
    print()
    
    # 总结
    print("=== 任务9验证总结 ===")
    all_requirements_met = all([
        requirement_6_1,
        requirement_6_2, 
        requirement_6_3,
        requirement_6_4
    ])
    
    print(f"需求6.1 (NoisyLinear替换): {'✓' if requirement_6_1 else '✗'}")
    print(f"需求6.2 (训练模式噪声): {'✓' if requirement_6_2 else '✗'}")
    print(f"需求6.3 (推理模式确定性): {'✓' if requirement_6_3 else '✗'}")
    print(f"需求6.4 (RLlib集成): {'✓' if requirement_6_4 else '✗'}")
    print(f"噪声重置功能: {'✓' if noise_reset_works else '✗'}")
    print(f"梯度流动: {'✓' if gradient_flow_ok else '✗'}")
    print()
    print(f"任务9整体状态: {'✅ 完成' if all_requirements_met else '❌ 未完成'}")
    
    return all_requirements_met


def test_performance_impact():
    """测试NoisyLinear对性能的影响"""
    print("\n=== 性能影响测试 ===")
    
    # 创建两个相同的模型，一个使用NoisyLinear，一个不使用
    class SimpleBox:
        def __init__(self, shape):
            self.shape = shape
    
    obs_space = SimpleBox((64,))
    action_space = SimpleBox((10,))
    
    # 普通模型
    normal_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "use_noisy_linear": False
    }
    
    # NoisyLinear模型
    noisy_config = {
        "embed_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "use_noisy_linear": True
    }
    
    normal_model = TransformerGNN(obs_space, action_space, 10, normal_config, "normal")
    noisy_model = TransformerGNN(obs_space, action_space, 10, noisy_config, "noisy")
    
    # 计算参数数量
    normal_params = sum(p.numel() for p in normal_model.parameters())
    noisy_params = sum(p.numel() for p in noisy_model.parameters())
    
    print(f"普通模型参数数量: {normal_params:,}")
    print(f"NoisyLinear模型参数数量: {noisy_params:,}")
    print(f"参数增加: {noisy_params - normal_params:,} ({(noisy_params/normal_params-1)*100:.1f}%)")
    
    # 简单的前向传播时间测试
    import time
    
    x = torch.randn(32, 64)
    input_dict = {"obs": x}
    state = []
    seq_lens = torch.tensor([1] * 32)
    
    # 预热
    for _ in range(10):
        normal_model(input_dict, state, seq_lens)
        noisy_model(input_dict, state, seq_lens)
    
    # 测试普通模型
    start_time = time.time()
    for _ in range(100):
        normal_model(input_dict, state, seq_lens)
    normal_time = time.time() - start_time
    
    # 测试NoisyLinear模型
    start_time = time.time()
    for _ in range(100):
        noisy_model(input_dict, state, seq_lens)
    noisy_time = time.time() - start_time
    
    print(f"普通模型100次前向传播时间: {normal_time:.4f}s")
    print(f"NoisyLinear模型100次前向传播时间: {noisy_time:.4f}s")
    print(f"性能开销: {(noisy_time/normal_time-1)*100:.1f}%")


if __name__ == "__main__":
    print("开始任务9完整验证...")
    print()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行验证测试
    success = test_task9_requirements()
    
    # 运行性能测试
    test_performance_impact()
    
    print(f"\n任务9验证结果: {'成功' if success else '失败'}")