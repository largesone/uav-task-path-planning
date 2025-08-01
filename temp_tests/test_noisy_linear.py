# -*- coding: utf-8 -*-
# 文件名: test_noisy_linear.py
# 描述: NoisyLinear层的测试文件

import torch
import torch.nn as nn
import numpy as np
from noisy_linear import NoisyLinear, replace_linear_with_noisy, reset_noise_in_module


def test_noisy_linear_basic():
    """测试NoisyLinear层的基本功能"""
    print("=== 测试NoisyLinear基本功能 ===")
    
    # 创建NoisyLinear层
    layer = NoisyLinear(10, 5, bias=True, std_init=0.5)
    
    # 测试输入
    x = torch.randn(3, 10)
    
    # 训练模式下的输出
    layer.train()
    output1 = layer(x)
    output2 = layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    print(f"训练模式下两次前向传播结果是否相同: {torch.allclose(output1, output2)}")
    
    # 重置噪声后应该不同
    layer.reset_noise()
    output3 = layer(x)
    print(f"重置噪声后结果是否不同: {not torch.allclose(output1, output3)}")
    
    # 推理模式下的输出
    layer.eval()
    output4 = layer(x)
    output5 = layer(x)
    
    print(f"推理模式下两次前向传播结果是否相同: {torch.allclose(output4, output5)}")
    print()


def test_replace_linear_with_noisy():
    """测试Linear层替换功能"""
    print("=== 测试Linear层替换功能 ===")
    
    # 创建包含Linear层的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Sequential(
                nn.Linear(20, 15),
                nn.ReLU(),
                nn.Linear(15, 5)
            )
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    net = SimpleNet()
    
    # 检查原始网络中的Linear层数量
    linear_count = sum(1 for m in net.modules() if isinstance(m, nn.Linear))
    print(f"原始网络中Linear层数量: {linear_count}")
    
    # 替换为NoisyLinear层
    net = replace_linear_with_noisy(net, std_init=0.3)
    
    # 检查替换后的NoisyLinear层数量
    noisy_count = sum(1 for m in net.modules() if isinstance(m, NoisyLinear))
    remaining_linear = sum(1 for m in net.modules() if isinstance(m, nn.Linear))
    
    print(f"替换后NoisyLinear层数量: {noisy_count}")
    print(f"剩余Linear层数量: {remaining_linear}")
    print(f"替换是否成功: {noisy_count == linear_count and remaining_linear == 0}")
    
    # 测试网络功能
    x = torch.randn(2, 10)
    
    net.train()
    output1 = net(x)
    reset_noise_in_module(net)
    output2 = net(x)
    
    print(f"训练模式下重置噪声前后结果是否不同: {not torch.allclose(output1, output2)}")
    
    net.eval()
    output3 = net(x)
    output4 = net(x)
    
    print(f"推理模式下结果是否一致: {torch.allclose(output3, output4)}")
    print()


def test_noisy_linear_gradients():
    """测试NoisyLinear层的梯度计算"""
    print("=== 测试NoisyLinear梯度计算 ===")
    
    layer = NoisyLinear(5, 3, bias=True)
    x = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 3)
    
    # 前向传播
    output = layer(x)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_weight_mu_grad = layer.weight_mu.grad is not None
    has_weight_sigma_grad = layer.weight_sigma.grad is not None
    has_bias_mu_grad = layer.bias_mu.grad is not None
    has_bias_sigma_grad = layer.bias_sigma.grad is not None
    has_input_grad = x.grad is not None
    
    print(f"weight_mu有梯度: {has_weight_mu_grad}")
    print(f"weight_sigma有梯度: {has_weight_sigma_grad}")
    print(f"bias_mu有梯度: {has_bias_mu_grad}")
    print(f"bias_sigma有梯度: {has_bias_sigma_grad}")
    print(f"输入有梯度: {has_input_grad}")
    
    all_grads_present = all([
        has_weight_mu_grad, has_weight_sigma_grad, 
        has_bias_mu_grad, has_bias_sigma_grad, has_input_grad
    ])
    print(f"所有必要梯度都存在: {all_grads_present}")
    print()


def test_noisy_linear_deterministic_inference():
    """测试NoisyLinear层在推理模式下的确定性"""
    print("=== 测试推理模式确定性 ===")
    
    layer = NoisyLinear(8, 4, bias=True)
    x = torch.randn(5, 8)
    
    # 设置为推理模式
    layer.eval()
    
    # 多次前向传播
    outputs = []
    for i in range(10):
        output = layer(x)
        outputs.append(output.clone())
    
    # 检查所有输出是否相同
    all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
    print(f"推理模式下10次前向传播结果是否完全相同: {all_same}")
    
    # 切换到训练模式
    layer.train()
    
    # 多次前向传播（每次重置噪声）
    train_outputs = []
    for i in range(5):
        layer.reset_noise()
        output = layer(x)
        train_outputs.append(output.clone())
    
    # 检查训练模式下的输出是否不同
    train_different = not all(torch.allclose(train_outputs[0], out) for out in train_outputs[1:])
    print(f"训练模式下重置噪声后结果是否不同: {train_different}")
    print()


def test_noise_scaling():
    """测试噪声缩放函数"""
    print("=== 测试噪声缩放 ===")
    
    layer = NoisyLinear(10, 5)
    
    # 测试不同大小的噪声
    for size in [1, 5, 10, 100]:
        noise = layer._scale_noise(size)
        print(f"大小{size}的噪声 - 形状: {noise.shape}, 均值: {noise.mean():.4f}, 标准差: {noise.std():.4f}")
    
    print()


if __name__ == "__main__":
    print("开始测试NoisyLinear实现...")
    print()
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有测试
    test_noisy_linear_basic()
    test_replace_linear_with_noisy()
    test_noisy_linear_gradients()
    test_noisy_linear_deterministic_inference()
    test_noise_scaling()
    
    print("所有测试完成！")