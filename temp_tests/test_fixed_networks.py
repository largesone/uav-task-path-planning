# -*- coding: utf-8 -*-
# 文件名: test_fixed_networks.py
# 描述: 修复版网络的测试脚本，验证三个风险点的修复效果

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks_fixed import (
    RobustFeatureExtractor,
    TrueGraphAttentionNetwork,
    RelativePositionEncoder,
    create_fixed_network
)
from transformer_gnn_fixed import FixedTransformerGNN, create_fixed_transformer_gnn_model
from gymnasium import spaces
import unittest


class TestFixedNetworks(unittest.TestCase):
    """修复版网络的测试类"""
    
    def setUp(self):
        """测试设置"""
        self.batch_size = 4
        self.input_dim = 128
        self.output_dim = 10
        self.hidden_dims = [256, 128, 64]
        
        # 测试配置
        self.test_config = {
            'extraction_strategy': 'semantic',
            'total_input_dim': self.input_dim,
            'embedding_dim': 128,
            'num_heads': 8,
            'dropout': 0.1,
            'num_attention_layers': 2,
            'n_uavs': 2,
            'n_targets': 3,
            'uav_features_per_entity': 8,
            'target_features_per_entity': 7,
            'max_distance': 1000.0,
            'target_feature_ratio': 0.6
        }
    
    def test_robust_feature_extractor_semantic(self):
        """测试鲁棒特征提取器 - 语义策略"""
        print("\n=== 测试风险点1修复：鲁棒特征提取器（语义策略） ===")
        
        # 创建特征提取器
        extractor = RobustFeatureExtractor(self.test_config)
        
        # 创建测试输入
        test_input = torch.randn(self.batch_size, self.input_dim)
        
        # 提取特征
        uav_features, target_features, additional_features = extractor.extract_features(test_input)
        
        # 验证输出形状
        expected_uav_dim = self.test_config['uav_features_per_entity'] * self.test_config['n_uavs']
        expected_target_dim = self.test_config['target_features_per_entity'] * self.test_config['n_targets']
        
        self.assertEqual(uav_features.shape, (self.batch_size, expected_uav_dim))
        self.assertEqual(target_features.shape, (self.batch_size, expected_target_dim))
        
        # 验证额外特征
        self.assertIn('collaboration', additional_features)
        self.assertIn('global', additional_features)
        
        print(f"✓ 语义特征提取成功 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        print(f"✓ 额外特征: {list(additional_features.keys())}")
    
    def test_robust_feature_extractor_ratio(self):
        """测试鲁棒特征提取器 - 比例策略"""
        print("\n=== 测试风险点1修复：鲁棒特征提取器（比例策略） ===")
        
        # 修改配置为比例策略
        ratio_config = self.test_config.copy()
        ratio_config['extraction_strategy'] = 'ratio'
        
        # 创建特征提取器
        extractor = RobustFeatureExtractor(ratio_config)
        
        # 创建测试输入
        test_input = torch.randn(self.batch_size, self.input_dim)
        
        # 提取特征
        uav_features, target_features, additional_features = extractor.extract_features(test_input)
        
        # 验证输出形状
        expected_target_dim = int(self.input_dim * self.test_config['target_feature_ratio'])
        expected_uav_dim = self.input_dim - expected_target_dim
        
        self.assertEqual(uav_features.shape, (self.batch_size, expected_uav_dim))
        self.assertEqual(target_features.shape, (self.batch_size, expected_target_dim))
        
        print(f"✓ 比例特征提取成功 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        print(f"✓ 比例分配: 目标{self.test_config['target_feature_ratio']:.1f}, UAV{1-self.test_config['target_feature_ratio']:.1f}")
    
    def test_robust_feature_extractor_fixed(self):
        """测试鲁棒特征提取器 - 固定维度策略"""
        print("\n=== 测试风险点1修复：鲁棒特征提取器（固定维度策略） ===")
        
        # 修改配置为固定维度策略
        fixed_config = self.test_config.copy()
        fixed_config['extraction_strategy'] = 'fixed'
        fixed_config['uav_feature_dim'] = 32
        fixed_config['target_feature_dim'] = 48
        
        # 创建特征提取器
        extractor = RobustFeatureExtractor(fixed_config)
        
        # 创建测试输入
        test_input = torch.randn(self.batch_size, self.input_dim)
        
        # 提取特征
        uav_features, target_features, additional_features = extractor.extract_features(test_input)
        
        # 验证输出形状
        self.assertEqual(uav_features.shape, (self.batch_size, 32))
        self.assertEqual(target_features.shape, (self.batch_size, 48))
        
        # 验证剩余特征
        remaining_dim = self.input_dim - 32 - 48
        if remaining_dim > 0:
            self.assertIn('remaining', additional_features)
            self.assertEqual(additional_features['remaining'].shape, (self.batch_size, remaining_dim))
        
        print(f"✓ 固定维度特征提取成功 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        if remaining_dim > 0:
            print(f"✓ 剩余特征维度: {remaining_dim}")
    
    def test_relative_position_encoder(self):
        """测试相对位置编码器"""
        print("\n=== 测试风险点3修复：相对位置编码器 ===")
        
        # 创建位置编码器
        position_encoder = RelativePositionEncoder(
            position_dim=2,
            embed_dim=64,
            max_distance=1000.0,
            num_distance_bins=32,
            num_angle_bins=16
        )
        
        # 创建测试相对位置
        num_pairs = 6  # 2 UAVs * 3 targets
        relative_positions = torch.randn(self.batch_size, num_pairs, 2) * 500  # 随机相对位置
        
        # 生成位置编码
        position_embeddings = position_encoder(relative_positions)
        
        # 验证输出形状
        self.assertEqual(position_embeddings.shape, (self.batch_size, num_pairs, 64))
        
        # 验证编码的有效性（不应该全为零）
        self.assertGreater(torch.abs(position_embeddings).sum().item(), 0)
        
        print(f"✓ 位置编码生成成功 - 输入: {relative_positions.shape}, 输出: {position_embeddings.shape}")
        print(f"✓ 编码范围: [{position_embeddings.min().item():.3f}, {position_embeddings.max().item():.3f}]")
    
    def test_graph_attention_layer(self):
        """测试图注意力层"""
        print("\n=== 测试风险点2修复：真正的图注意力层 ===")
        
        from networks_fixed import GraphAttentionLayer
        
        # 创建图注意力层
        attention_layer = GraphAttentionLayer(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            use_position_encoding=True
        )
        
        # 创建测试节点嵌入
        num_nodes = 5  # 2 UAVs + 3 targets
        node_embeddings = torch.randn(self.batch_size, num_nodes, 128)
        
        # 创建位置编码
        position_embeddings = torch.randn(self.batch_size, 1, 128)
        
        # 应用注意力层
        output_embeddings = attention_layer(node_embeddings, position_embeddings)
        
        # 验证输出形状
        self.assertEqual(output_embeddings.shape, node_embeddings.shape)
        
        # 验证输出不等于输入（确实进行了变换）
        self.assertFalse(torch.allclose(output_embeddings, node_embeddings, atol=1e-6))
        
        print(f"✓ 图注意力计算成功 - 输入: {node_embeddings.shape}, 输出: {output_embeddings.shape}")
        print(f"✓ 注意力变换有效性验证通过")
    
    def test_true_graph_attention_network(self):
        """测试真正的图注意力网络"""
        print("\n=== 测试风险点2修复：真正的图注意力网络 ===")
        
        # 创建网络
        network = TrueGraphAttentionNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # 创建测试输入
        test_input = torch.randn(self.batch_size, self.input_dim)
        
        # 前向传播
        output = network(test_input)
        
        # 验证输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # 验证网络参数可训练
        total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        self.assertGreater(total_params, 0)
        
        print(f"✓ 图注意力网络前向传播成功 - 输入: {test_input.shape}, 输出: {output.shape}")
        print(f"✓ 可训练参数数量: {total_params:,}")
    
    def test_fixed_transformer_gnn_flat_mode(self):
        """测试修复版TransformerGNN - 扁平模式"""
        print("\n=== 测试修复版TransformerGNN：扁平模式 ===")
        
        # 创建观测空间
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dim,), dtype=np.float32)
        action_space = spaces.Discrete(self.output_dim)
        
        # 模型配置
        model_config = {
            "embed_dim": 128,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_position_encoding": True,
            "use_noisy_linear": False,  # 简化测试
            "extraction_strategy": "semantic",
            "n_uavs": 2,
            "n_targets": 3,
            "uav_features_per_entity": 8,
            "target_features_per_entity": 7,
            "max_distance": 1000.0
        }
        
        # 创建模型
        model = FixedTransformerGNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=self.output_dim,
            model_config=model_config,
            name="TestFixedTransformerGNN"
        )
        
        # 创建测试输入
        input_dict = {
            "obs": torch.randn(self.batch_size, self.input_dim)
        }
        
        # 前向传播
        logits, state = model.forward(input_dict, [], None)
        
        # 验证输出
        self.assertEqual(logits.shape, (self.batch_size, self.output_dim))
        
        # 验证值函数
        values = model.value_function()
        self.assertEqual(values.shape, (self.batch_size,))
        
        print(f"✓ 扁平模式前向传播成功 - logits: {logits.shape}, values: {values.shape}")
    
    def test_fixed_transformer_gnn_dict_mode(self):
        """测试修复版TransformerGNN - 字典模式"""
        print("\n=== 测试修复版TransformerGNN：字典模式 ===")
        
        # 创建字典观测空间
        obs_space = spaces.Dict({
            'uav_features': spaces.Box(low=-np.inf, high=np.inf, shape=(2, 9), dtype=np.float32),
            'target_features': spaces.Box(low=-np.inf, high=np.inf, shape=(3, 8), dtype=np.float32),
            'relative_positions': spaces.Box(low=-1.0, high=1.0, shape=(2, 3, 2), dtype=np.float32),
            'distances': spaces.Box(low=0.0, high=1.0, shape=(2, 3), dtype=np.float32),
            'masks': spaces.Dict({
                'uav_mask': spaces.Box(low=0, high=1, shape=(2,), dtype=np.int32),
                'target_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32)
            })
        })
        action_space = spaces.Discrete(self.output_dim)
        
        # 模型配置
        model_config = {
            "embed_dim": 128,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "use_position_encoding": True,
            "use_local_attention": True,
            "use_noisy_linear": False,  # 简化测试
            "k_adaptive": True,
            "k_min": 2,
            "k_max": 3
        }
        
        # 创建模型
        model = FixedTransformerGNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=self.output_dim,
            model_config=model_config,
            name="TestFixedTransformerGNN_Dict"
        )
        
        # 创建测试输入
        input_dict = {
            "obs": {
                'uav_features': torch.randn(self.batch_size, 2, 9),
                'target_features': torch.randn(self.batch_size, 3, 8),
                'relative_positions': torch.randn(self.batch_size, 2, 3, 2),
                'distances': torch.rand(self.batch_size, 2, 3),
                'masks': {
                    'uav_mask': torch.ones(self.batch_size, 2, dtype=torch.int32),
                    'target_mask': torch.ones(self.batch_size, 3, dtype=torch.int32)
                }
            }
        }
        
        # 前向传播
        logits, state = model.forward(input_dict, [], None)
        
        # 验证输出
        self.assertEqual(logits.shape, (self.batch_size, self.output_dim))
        
        # 验证值函数
        values = model.value_function()
        self.assertEqual(values.shape, (self.batch_size,))
        
        print(f"✓ 字典模式前向传播成功 - logits: {logits.shape}, values: {values.shape}")
    
    def test_feature_extraction_robustness(self):
        """测试特征提取的鲁棒性"""
        print("\n=== 测试特征提取鲁棒性 ===")
        
        # 测试不同输入维度
        test_dims = [64, 128, 256, 512]
        
        for dim in test_dims:
            print(f"\n测试输入维度: {dim}")
            
            # 更新配置
            config = self.test_config.copy()
            config['total_input_dim'] = dim
            
            # 创建特征提取器
            extractor = RobustFeatureExtractor(config)
            
            # 创建测试输入
            test_input = torch.randn(self.batch_size, dim)
            
            # 提取特征
            try:
                uav_features, target_features, additional_features = extractor.extract_features(test_input)
                print(f"  ✓ 维度 {dim} 测试通过 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
            except Exception as e:
                print(f"  ✗ 维度 {dim} 测试失败: {e}")
                raise
    
    def test_attention_mechanism_effectiveness(self):
        """测试注意力机制的有效性"""
        print("\n=== 测试注意力机制有效性 ===")
        
        from networks_fixed import GraphAttentionLayer
        
        # 创建两个相同的注意力层
        attention_layer1 = GraphAttentionLayer(embed_dim=128, num_heads=8, dropout=0.0)
        attention_layer2 = GraphAttentionLayer(embed_dim=128, num_heads=8, dropout=0.0)
        
        # 复制权重以确保初始状态相同
        attention_layer2.load_state_dict(attention_layer1.state_dict())
        
        # 创建两个不同的输入
        input1 = torch.randn(self.batch_size, 5, 128)
        input2 = torch.randn(self.batch_size, 5, 128)
        
        # 前向传播
        output1 = attention_layer1(input1)
        output2 = attention_layer2(input2)
        
        # 验证不同输入产生不同输出
        self.assertFalse(torch.allclose(output1, output2, atol=1e-4))
        
        # 验证相同输入产生相同输出
        output1_repeat = attention_layer1(input1)
        self.assertTrue(torch.allclose(output1, output1_repeat, atol=1e-6))
        
        print("✓ 注意力机制对不同输入产生不同输出")
        print("✓ 注意力机制对相同输入产生相同输出")
    
    def test_position_encoding_effectiveness(self):
        """测试位置编码的有效性"""
        print("\n=== 测试位置编码有效性 ===")
        
        # 创建位置编码器
        position_encoder = RelativePositionEncoder(
            position_dim=2,
            embed_dim=64,
            max_distance=1000.0
        )
        
        # 创建两组不同的相对位置
        pos1 = torch.tensor([[[100.0, 0.0], [0.0, 100.0]]], dtype=torch.float32)  # 近距离
        pos2 = torch.tensor([[[500.0, 0.0], [0.0, 500.0]]], dtype=torch.float32)  # 远距离
        
        # 生成位置编码
        emb1 = position_encoder(pos1)
        emb2 = position_encoder(pos2)
        
        # 验证不同位置产生不同编码
        self.assertFalse(torch.allclose(emb1, emb2, atol=1e-4))
        
        # 验证编码的一致性
        emb1_repeat = position_encoder(pos1)
        self.assertTrue(torch.allclose(emb1, emb1_repeat, atol=1e-6))
        
        print("✓ 不同相对位置产生不同编码")
        print("✓ 相同相对位置产生相同编码")
        print(f"✓ 近距离编码范围: [{emb1.min().item():.3f}, {emb1.max().item():.3f}]")
        print(f"✓ 远距离编码范围: [{emb2.min().item():.3f}, {emb2.max().item():.3f}]")
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        print("\n=== 测试梯度流动 ===")
        
        # 创建网络
        network = TrueGraphAttentionNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            config=self.test_config
        )
        
        # 创建测试输入和目标
        test_input = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        target = torch.randn(self.batch_size, self.output_dim)
        
        # 前向传播
        output = network(test_input)
        
        # 计算损失
        loss = nn.MSELoss()(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        input_grad_norm = test_input.grad.norm().item()
        param_grads = [p.grad.norm().item() for p in network.parameters() if p.grad is not None]
        
        # 验证梯度存在且非零
        self.assertGreater(input_grad_norm, 0)
        self.assertGreater(len(param_grads), 0)
        self.assertTrue(all(grad > 0 for grad in param_grads))
        
        print(f"✓ 输入梯度范数: {input_grad_norm:.6f}")
        print(f"✓ 参数梯度数量: {len(param_grads)}")
        print(f"✓ 参数梯度范数范围: [{min(param_grads):.6f}, {max(param_grads):.6f}]")
    
    def test_comparison_with_original(self):
        """对比原始实现和修复版实现"""
        print("\n=== 对比原始实现和修复版实现 ===")
        
        # 导入原始网络（如果可用）
        try:
            from networks import GATNetwork
            
            # 创建原始网络
            original_network = GATNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                dropout=0.1
            )
            
            # 创建修复版网络
            fixed_network = TrueGraphAttentionNetwork(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                config=self.test_config
            )
            
            # 创建测试输入
            test_input = torch.randn(self.batch_size, self.input_dim)
            
            # 前向传播
            original_output = original_network(test_input)
            fixed_output = fixed_network(test_input)
            
            # 验证输出形状相同
            self.assertEqual(original_output.shape, fixed_output.shape)
            
            # 验证输出不完全相同（说明实现确实不同）
            self.assertFalse(torch.allclose(original_output, fixed_output, atol=1e-4))
            
            # 统计参数数量
            original_params = sum(p.numel() for p in original_network.parameters())
            fixed_params = sum(p.numel() for p in fixed_network.parameters())
            
            print(f"✓ 原始网络参数数量: {original_params:,}")
            print(f"✓ 修复版网络参数数量: {fixed_params:,}")
            print(f"✓ 参数数量比例: {fixed_params/original_params:.2f}")
            print("✓ 输出形状一致，但数值不同（确认实现差异）")
            
        except ImportError:
            print("⚠ 无法导入原始网络，跳过对比测试")


def run_comprehensive_tests():
    """运行全面的测试"""
    print("=" * 80)
    print("修复版网络全面测试")
    print("=" * 80)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFixedNetworks)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 所有测试通过！修复版网络实现正确。")
        print("\n修复效果总结：")
        print("✅ 风险点1：鲁棒的特征提取 - 支持语义、比例、固定维度三种策略")
        print("✅ 风险点2：真正的图注意力机制 - 实现完整的多头注意力计算")
        print("✅ 风险点3：完整的空间信息处理 - 相对位置编码和结构感知")
    else:
        print("❌ 部分测试失败，请检查实现。")
        print(f"失败数量: {len(result.failures)}")
        print(f"错误数量: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    success = run_comprehensive_tests()
    
    # 退出码
    exit(0 if success else 1)
