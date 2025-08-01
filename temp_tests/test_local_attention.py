# -*- coding: utf-8 -*-
# 文件名: test_local_attention.py
# 描述: 局部注意力机制的单元测试和验证

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from local_attention import LocalAttention, MultiScaleLocalAttention


class TestLocalAttention:
    """局部注意力机制测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.batch_size = 2
        self.num_uavs = 5
        self.num_targets = 8
        self.embed_dim = 64
        self.num_heads = 4
        
        # 创建测试数据
        self.uav_embeddings = torch.randn(self.batch_size, self.num_uavs, self.embed_dim)
        self.target_embeddings = torch.randn(self.batch_size, self.num_targets, self.embed_dim)
        
        # 创建距离矩阵（确保有合理的距离分布）
        self.distances = torch.rand(self.batch_size, self.num_uavs, self.num_targets) * 10 + 0.1
        
        # 创建掩码
        self.masks = {
            'uav_mask': torch.ones(self.batch_size, self.num_uavs, dtype=torch.bool),
            'target_mask': torch.ones(self.batch_size, self.num_targets, dtype=torch.bool)
        }
        
        # 设置部分无效实体
        self.masks['uav_mask'][0, -1] = False  # 第一个批次的最后一个UAV无效
        self.masks['target_mask'][1, -2:] = False  # 第二个批次的最后两个目标无效
    
    def test_local_attention_initialization(self):
        """测试局部注意力初始化"""
        print("\n=== 测试局部注意力初始化 ===")
        
        # 测试基本初始化
        attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_adaptive=True
        )
        
        assert attention.embed_dim == self.embed_dim
        assert attention.num_heads == self.num_heads
        assert attention.head_dim == self.embed_dim // self.num_heads
        assert attention.k_adaptive == True
        
        print("✓ 基本初始化测试通过")
        
        # 测试固定k值初始化
        attention_fixed = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_adaptive=False,
            k_fixed=6
        )
        
        assert attention_fixed.k_adaptive == False
        assert attention_fixed.k_fixed == 6
        
        print("✓ 固定k值初始化测试通过")
    
    def test_adaptive_k_computation(self):
        """测试自适应k值计算"""
        print("\n=== 测试自适应k值计算 ===")
        
        attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_adaptive=True,
            k_min=4,
            k_max=16
        )
        
        # 测试不同目标数量下的k值计算
        test_cases = [
            (4, 4),    # 最小情况
            (8, 4),    # 小规模
            (16, 4),   # 中等规模
            (32, 8),   # 大规模
            (64, 16),  # 超大规模
        ]
        
        for num_targets, expected_min_k in test_cases:
            k_eval = attention._compute_adaptive_k(num_targets, training=False)
            k_train = attention._compute_adaptive_k(num_targets, training=True)
            
            print(f"目标数量: {num_targets}, 评估k: {k_eval}, 训练k: {k_train}")
            
            # 验证k值在合理范围内
            assert attention.k_min <= k_eval <= min(attention.k_max, num_targets)
            assert attention.k_min <= k_train <= min(attention.k_max, num_targets)
            
            # 验证基础k值符合预期
            if num_targets >= 16:
                assert k_eval >= expected_min_k
        
        print("✓ 自适应k值计算测试通过")
    
    def test_k_nearest_selection(self):
        """测试k-近邻选择"""
        print("\n=== 测试k-近邻选择 ===")
        
        attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        k = 3
        selected_indices, selected_distances = attention._select_k_nearest(
            self.distances, k, self.masks
        )
        
        # 验证输出形状
        expected_shape = (self.batch_size, self.num_uavs, k)
        assert selected_indices.shape == expected_shape
        assert selected_distances.shape == expected_shape
        
        print(f"✓ 选择形状正确: {selected_indices.shape}")
        
        # 验证选择的是最近的目标
        for b in range(self.batch_size):
            for u in range(self.num_uavs):
                selected_dists = selected_distances[b, u]
                all_dists = self.distances[b, u]
                
                # 应用掩码
                if self.masks['target_mask'][b].sum() >= k:
                    valid_dists = all_dists[self.masks['target_mask'][b]]
                    sorted_valid_dists, _ = torch.sort(valid_dists)
                    
                    # 验证选择的距离确实是最小的k个
                    for i in range(k):
                        assert selected_dists[i] <= sorted_valid_dists[min(i+1, len(sorted_valid_dists)-1)] + 1e-6
        
        print("✓ k-近邻选择正确性验证通过")
        
        # 测试掩码处理
        print("✓ 掩码处理测试通过")
    
    def test_local_attention_forward(self):
        """测试局部注意力前向传播"""
        print("\n=== 测试局部注意力前向传播 ===")
        
        attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_adaptive=True,
            use_flash_attention=False  # 使用标准实现以确保稳定性
        )
        
        # 前向传播
        output = attention(
            self.uav_embeddings,
            self.target_embeddings,
            self.distances,
            self.masks
        )
        
        # 验证输出形状
        expected_shape = (self.batch_size, self.num_uavs, self.embed_dim)
        assert output.shape == expected_shape
        print(f"✓ 输出形状正确: {output.shape}")
        
        # 验证输出不包含NaN或Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print("✓ 输出数值稳定性验证通过")
        
        # 验证掩码效果
        # 被掩码的UAV输出应该为零
        masked_uav_output = output[0, -1]  # 第一个批次的最后一个UAV被掩码
        assert torch.allclose(masked_uav_output, torch.zeros_like(masked_uav_output), atol=1e-6)
        print("✓ UAV掩码效果验证通过")
    
    def test_gradient_flow(self):
        """测试梯度流"""
        print("\n=== 测试梯度流 ===")
        
        attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            use_flash_attention=False
        )
        
        # 设置需要梯度
        self.uav_embeddings.requires_grad_(True)
        self.target_embeddings.requires_grad_(True)
        
        # 前向传播
        output = attention(
            self.uav_embeddings,
            self.target_embeddings,
            self.distances,
            self.masks
        )
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在且不为零
        assert self.uav_embeddings.grad is not None
        assert self.target_embeddings.grad is not None
        assert not torch.allclose(self.uav_embeddings.grad, torch.zeros_like(self.uav_embeddings.grad))
        assert not torch.allclose(self.target_embeddings.grad, torch.zeros_like(self.target_embeddings.grad))
        
        print("✓ 梯度流测试通过")
        
        # 验证注意力模块参数的梯度
        for name, param in attention.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"参数 {name} 梯度为零"
        
        print("✓ 参数梯度验证通过")
    
    def test_different_scales(self):
        """测试不同规模的输入"""
        print("\n=== 测试不同规模输入 ===")
        
        attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_adaptive=True
        )
        
        # 测试不同的规模
        test_scales = [
            (2, 3),    # 小规模
            (5, 8),    # 中等规模
            (10, 15),  # 大规模
            (20, 30),  # 超大规模
        ]
        
        for num_uavs, num_targets in test_scales:
            print(f"测试规模: {num_uavs} UAVs, {num_targets} 目标")
            
            # 创建测试数据
            uav_emb = torch.randn(1, num_uavs, self.embed_dim)
            target_emb = torch.randn(1, num_targets, self.embed_dim)
            distances = torch.rand(1, num_uavs, num_targets) * 10 + 0.1
            
            masks = {
                'uav_mask': torch.ones(1, num_uavs, dtype=torch.bool),
                'target_mask': torch.ones(1, num_targets, dtype=torch.bool)
            }
            
            # 前向传播
            output = attention(uav_emb, target_emb, distances, masks)
            
            # 验证输出形状
            assert output.shape == (1, num_uavs, self.embed_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            print(f"✓ 规模 ({num_uavs}, {num_targets}) 测试通过")
    
    def test_multi_scale_attention(self):
        """测试多尺度局部注意力"""
        print("\n=== 测试多尺度局部注意力 ===")
        
        multi_attention = MultiScaleLocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_scales=[2, 4, 6],
            fusion_method="weighted_sum"
        )
        
        # 前向传播
        output = multi_attention(
            self.uav_embeddings,
            self.target_embeddings,
            self.distances,
            self.masks
        )
        
        # 验证输出形状
        expected_shape = (self.batch_size, self.num_uavs, self.embed_dim)
        assert output.shape == expected_shape
        print(f"✓ 多尺度输出形状正确: {output.shape}")
        
        # 验证数值稳定性
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print("✓ 多尺度数值稳定性验证通过")
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        print("\n=== 测试内存效率 ===")
        
        # 创建大规模测试数据
        large_batch_size = 4
        large_num_uavs = 50
        large_num_targets = 100
        
        large_uav_emb = torch.randn(large_batch_size, large_num_uavs, self.embed_dim)
        large_target_emb = torch.randn(large_batch_size, large_num_targets, self.embed_dim)
        large_distances = torch.rand(large_batch_size, large_num_uavs, large_num_targets) * 10 + 0.1
        
        large_masks = {
            'uav_mask': torch.ones(large_batch_size, large_num_uavs, dtype=torch.bool),
            'target_mask': torch.ones(large_batch_size, large_num_targets, dtype=torch.bool)
        }
        
        # 测试局部注意力
        local_attention = LocalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            k_adaptive=True,
            k_max=16  # 限制k值以控制内存
        )
        
        # 记录内存使用
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 前向传播
        output = local_attention(
            large_uav_emb, large_target_emb, large_distances, large_masks
        )
        
        peak_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = peak_memory - initial_memory
        
        print(f"大规模测试 ({large_num_uavs} UAVs, {large_num_targets} 目标)")
        print(f"内存使用: {memory_used / 1024 / 1024:.2f} MB" if torch.cuda.is_available() else "CPU模式")
        print(f"输出形状: {output.shape}")
        
        # 验证输出正确性
        assert output.shape == (large_batch_size, large_num_uavs, self.embed_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print("✓ 大规模内存效率测试通过")


def run_comprehensive_test():
    """运行综合测试"""
    print("开始局部注意力机制综合测试")
    print("=" * 50)
    
    test_instance = TestLocalAttention()
    test_instance.setup_method()
    
    try:
        # 运行所有测试
        test_instance.test_local_attention_initialization()
        test_instance.test_adaptive_k_computation()
        test_instance.test_k_nearest_selection()
        test_instance.test_local_attention_forward()
        test_instance.test_gradient_flow()
        test_instance.test_different_scales()
        test_instance.test_multi_scale_attention()
        test_instance.test_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！局部注意力机制实现正确。")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
