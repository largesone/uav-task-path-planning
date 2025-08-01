# -*- coding: utf-8 -*-
# 文件名: transformer_gnn_usage_example.py
# 描述: TransformerGNN与NoisyLinear的使用示例

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformer_gnn import TransformerGNN, create_transformer_gnn_model


def create_example_observation():
    """创建示例观测数据"""
    batch_size = 4
    
    return {
        'uav_features': torch.randn(batch_size, 5, 8),  # 5个UAV，每个8维特征
        'target_features': torch.randn(batch_size, 3, 6),  # 3个目标，每个6维特征
        'relative_positions': torch.randn(batch_size, 15, 2),  # 5*3=15个UAV-目标对的相对位置
        'distances': torch.abs(torch.randn(batch_size, 5, 3)),  # 5个UAV到3个目标的距离
        'masks': {
            'uav_mask': torch.ones(batch_size, 5, dtype=torch.int32),  # 所有UAV都有效
            'target_mask': torch.ones(batch_size, 3, dtype=torch.int32)  # 所有目标都有效
        }
    }


def example_training_loop():
    """示例训练循环"""
    print("=== TransformerGNN训练示例 ===")
    
    # 创建观测空间（简化版本）
    class ObsSpace:
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict
            for key, value in spaces_dict.items():
                setattr(self, key, value)
        
        def __getitem__(self, key):
            return self.spaces[key]
    
    class Box:
        def __init__(self, shape):
            self.shape = shape
    
    obs_space = ObsSpace({
        'uav_features': Box((5, 8)),
        'target_features': Box((3, 6)),
        'relative_positions': Box((15, 2)),
        'distances': Box((5, 3)),
        'masks': ObsSpace({
            'uav_mask': Box((5,)),
            'target_mask': Box((3,))
        })
    })
    
    action_space = Box((10,))
    
    # 模型配置
    model_config = {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "use_position_encoding": True,
        "use_noisy_linear": True,  # 启用参数空间噪声探索
        "noisy_std_init": 0.5
    }
    
    # 创建模型
    print("创建TransformerGNN模型...")
    model = TransformerGNN(obs_space, action_space, 10, model_config, "example")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 模拟训练数据
    print("开始训练循环...")
    
    for epoch in range(5):
        model.train()  # 训练模式，启用噪声
        
        epoch_loss = 0.0
        num_batches = 10
        
        for batch in range(num_batches):
            # 创建批次数据
            obs = create_example_observation()
            input_dict = {"obs": obs}
            state = []
            seq_lens = torch.tensor([1, 1, 1, 1])
            
            # 前向传播
            logits, _ = model(input_dict, state, seq_lens)
            values = model.value_function()
            
            # 创建虚拟目标
            target_logits = torch.randn_like(logits)
            target_values = torch.randn_like(values)
            
            # 计算损失
            policy_loss = nn.MSELoss()(logits, target_logits)
            value_loss = nn.MSELoss()(values, target_values)
            total_loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/5, 平均损失: {avg_loss:.4f}")
    
    print("训练完成！")
    return model


def example_inference():
    """示例推理过程"""
    print("\n=== TransformerGNN推理示例 ===")
    
    # 使用训练好的模型
    model = example_training_loop()
    
    # 切换到推理模式
    model.eval()  # 推理模式，关闭噪声
    
    print("进行推理...")
    
    # 创建测试数据
    test_obs = create_example_observation()
    input_dict = {"obs": test_obs}
    state = []
    seq_lens = torch.tensor([1, 1, 1, 1])
    
    # 多次推理验证确定性
    with torch.no_grad():
        results = []
        for i in range(3):
            logits, _ = model(input_dict, state, seq_lens)
            values = model.value_function()
            results.append((logits.clone(), values.clone()))
        
        # 验证结果一致性
        all_same = True
        for i in range(1, len(results)):
            if not torch.allclose(results[0][0], results[i][0]) or \
               not torch.allclose(results[0][1], results[i][1]):
                all_same = False
                break
        
        print(f"推理结果一致性: {'✓ 一致' if all_same else '✗ 不一致'}")
        print(f"动作logits形状: {results[0][0].shape}")
        print(f"值函数输出形状: {results[0][1].shape}")
        
        # 显示示例输出
        print(f"示例动作logits (第一个样本): {results[0][0][0]}")
        print(f"示例值函数输出 (第一个样本): {results[0][1][0]}")


def example_noise_exploration():
    """示例噪声探索功能"""
    print("\n=== 噪声探索功能示例 ===")
    
    # 创建简单模型用于演示
    class SimpleBox:
        def __init__(self, shape):
            self.shape = shape
    
    obs_space = SimpleBox((32,))
    action_space = SimpleBox((5,))
    
    model_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "use_noisy_linear": True,
        "noisy_std_init": 0.3
    }
    
    model = TransformerGNN(obs_space, action_space, 5, model_config, "noise_demo")
    
    # 创建测试输入
    obs = torch.randn(3, 32)
    input_dict = {"obs": obs}
    state = []
    seq_lens = torch.tensor([1, 1, 1])
    
    print("训练模式下的探索行为:")
    model.train()
    
    for i in range(5):
        logits, _ = model(input_dict, state, seq_lens)
        action_probs = torch.softmax(logits, dim=-1)
        print(f"  步骤 {i+1}: 动作概率 = {action_probs[0].detach().numpy()}")
    
    print("\n推理模式下的确定性行为:")
    model.eval()
    
    for i in range(3):
        logits, _ = model(input_dict, state, seq_lens)
        action_probs = torch.softmax(logits, dim=-1)
        print(f"  步骤 {i+1}: 动作概率 = {action_probs[0].detach().numpy()}")
    
    print("\n手动重置噪声:")
    model.train()
    
    # 第一次前向传播
    logits1, _ = model(input_dict, state, seq_lens)
    probs1 = torch.softmax(logits1, dim=-1)
    
    # 重置噪声
    model.reset_noise()
    
    # 第二次前向传播
    logits2, _ = model(input_dict, state, seq_lens)
    probs2 = torch.softmax(logits2, dim=-1)
    
    print(f"  重置前: {probs1[0].detach().numpy()}")
    print(f"  重置后: {probs2[0].detach().numpy()}")
    print(f"  结果不同: {not torch.allclose(probs1, probs2, atol=1e-6)}")


if __name__ == "__main__":
    print("TransformerGNN与NoisyLinear使用示例")
    print("=" * 50)
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行示例
    example_inference()
    example_noise_exploration()
    
    print("\n示例运行完成！")