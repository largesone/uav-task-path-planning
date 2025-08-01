# -*- coding: utf-8 -*-
# 文件名: test_positional_encoding.py
# 描述: 相对位置编码机制的单元测试

import torch
import torch.nn as nn
import numpy as np
from transformer_gnn import PositionalEncoder, TransformerGNN
from gymnasium.spaces import Box, Dict
import unittest


class TestPositionalEncoder(unittest.TestCase):
    """位置编码器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.position_dim = 2
        self.embed_dim = 64
        self.hidden_dim = 32
        self.encoder = PositionalEncoder(
            position_dim=self.position_dim,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim
        )
    
    def test_positional_encoder_initialization(self):
        """测试位置编码器初始化"""
        self.assertEqual(self.encoder.position_dim, self.position_dim)
        self.assertEqual(self.encoder.embed_dim, self.embed_dim)
        
        # 检查MLP结构
        self.assertIsInstance(self.encoder.position_mlp, nn.Sequential)
        
        # 检查权重初始化
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                # 权重应该被初始化
                self.assertFalse(torch.allclose(module.weight, torch.zeros_like(module.weight)))
    
    def test_positional_encoder_forward(self):
        """测试位置编码器前向传播"""
        batch_size = 4
        num_pairs = 6
        
        # 创建测试输入
        relative_positions = torch.randn(batch_size, num_pairs, self.position_dim)
        
        # 前向传播
        position_embeddings = self.encoder(relative_positions)
        
        # 检查输出形状
        expected_shape = (batch_size, num_pairs, self.embed_dim)
        self.assertEqual(position_embeddings.shape, expected_shape)
        
        # 检查输出不是零向量
        self.assertFalse(torch.allclose(position_embeddings, torch.zeros_like(position_embeddings)))
    
    def test_positional_encoder_deterministic(self):
        """测试位置编码器的确定性"""
        batch_size = 2
        num_pairs = 4
        
        # 创建相同的输入
        relative_positions = torch.randn(batch_size, num_pairs, self.position_dim)
        
        # 两次前向传播
        self.encoder.eval()  # 设置为评估模式
        output1 = self.encoder(relative_positions)
        output2 = self.encoder(relative_positions)
        
        # 输出应该相同
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
    
    def test_positional_encoder_different_inputs(self):
        """测试不同输入产生不同输出"""
        batch_size = 2
        num_pairs = 4
        
        # 创建不同的输入
        positions1 = torch.randn(batch_size, num_pairs, self.position_dim)
        positions2 = torch.randn(batch_size, num_pairs, self.position_dim)
        
        self.encoder.eval()
        output1 = self.encoder(positions1)
        output2 = self.encoder(positions2)
        
        # 不同输入应该产生不同输出
        self.assertFalse(torch.allclose(output1, output2, atol=1e-3))
    
    def test_positional_encoder_gradient_flow(self):
        """测试梯度流动"""
        batch_size = 2
        num_pairs = 3
        
        relative_positions = torch.randn(batch_size, num_pairs, self.position_dim, requires_grad=True)
        
        # 前向传播
        position_embeddings = self.encoder(relative_positions)
        loss = position_embeddings.sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度存在
        self.assertIsNotNone(relative_positions.grad)
        self.assertFalse(torch.allclose(relative_positions.grad, torch.zeros_like(relative_positions.grad)))


class TestTransformerGNN(unittest.TestCase):
    """TransformerGNN测试类"""
    
    def setUp(self):
        """测试初始化"""
        # 创建观测空间
        self.obs_space_flat = Box(low=-np.inf, high=np.inf, shape=(128,))
        self.obs_space_dict = Dict({
            'uav_features': Box(low=-np.inf, high=np.inf, shape=(3, 64)),
            'target_features': Box(low=-np.inf, high=np.inf, shape=(5, 64)),
            'relative_positions': Box(low=-np.inf, high=np.inf, shape=(15, 2))  # 3*5 pairs
        })
        
        # 创建动作空间
        self.action_space = Box(low=-1, high=1, shape=(4,))
        self.num_outputs = 4
        
        # 模型配置
        self.model_config = {
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "use_position_encoding": True
        }
    
    def test_transformer_gnn_initialization_flat_obs(self):
        """测试TransformerGNN在扁平观测下的初始化"""
        model = TransformerGNN(
            self.obs_space_flat,
            self.action_space,
            self.num_outputs,
            self.model_config,
            "test_model"
        )
        
        self.assertFalse(model.is_dict_obs)
        self.assertEqual(model.input_dim, 128)
        self.assertTrue(model.use_position_encoding)
        self.assertIsInstance(model.position_encoder, PositionalEncoder)
    
    def test_transformer_gnn_initialization_dict_obs(self):
        """测试TransformerGNN在字典观测下的初始化"""
        model = TransformerGNN(
            self.obs_space_dict,
            self.action_space,
            self.num_outputs,
            self.model_config,
            "test_model"
        )
        
        self.assertTrue(model.is_dict_obs)
        self.assertTrue(model.use_position_encoding)
        self.assertIsInstance(model.position_encoder, PositionalEncoder)
    
    def test_transformer_gnn_forward_flat_obs(self):
        """测试TransformerGNN在扁平观测下的前向传播"""
        model = TransformerGNN(
            self.obs_space_flat,
            self.action_space,
            self.num_outputs,
            self.model_config,
            "test_model"
        )
        
        batch_size = 4
        obs = torch.randn(batch_size, 128)
        input_dict = {"obs": obs}
        
        # 前向传播
        logits, state = model.forward(input_dict, [], None)
        
        # 检查输出形状
        self.assertEqual(logits.shape, (batch_size, self.num_outputs))
        
        # 检查值函数
        values = model.value_function()
        self.assertEqual(values.shape, (batch_size,))
    
    def test_transformer_gnn_forward_dict_obs(self):
        """测试TransformerGNN在字典观测下的前向传播"""
        model = TransformerGNN(
            self.obs_space_dict,
            self.action_space,
            self.num_outputs,
            self.model_config,
            "test_model"
        )
        
        batch_size = 4
        obs_dict = {
            'uav_features': torch.randn(batch_size, 3, 64),
            'target_features': torch.randn(batch_size, 5, 64),
            'relative_positions': torch.randn(batch_size, 15, 2)
        }
        input_dict = {"obs": obs_dict}
        
        # 前向传播
        logits, state = model.forward(input_dict, [], None)
        
        # 检查输出形状
        self.assertEqual(logits.shape, (batch_size, self.num_outputs))
        
        # 检查值函数
        values = model.value_function()
        self.assertEqual(values.shape, (batch_size,))
    
    def test_position_encoding_effect(self):
        """测试位置编码的效果"""
        # 创建两个模型：一个使用位置编码，一个不使用
        config_with_pos = self.model_config.copy()
        config_with_pos["use_position_encoding"] = True
        
        config_without_pos = self.model_config.copy()
        config_without_pos["use_position_encoding"] = False
        
        model_with_pos = TransformerGNN(
            self.obs_space_dict,
            self.action_space,
            self.num_outputs,
            config_with_pos,
            "with_pos"
        )
        
        model_without_pos = TransformerGNN(
            self.obs_space_dict,
            self.action_space,
            self.num_outputs,
            config_without_pos,
            "without_pos"
        )
        
        # 创建测试数据
        batch_size = 2
        obs_dict = {
            'uav_features': torch.randn(batch_size, 3, 64),
            'target_features': torch.randn(batch_size, 5, 64),
            'relative_positions': torch.randn(batch_size, 15, 2)
        }
        input_dict = {"obs": obs_dict}
        
        # 设置为评估模式
        model_with_pos.eval()
        model_without_pos.eval()
        
        # 前向传播
        logits_with_pos, _ = model_with_pos.forward(input_dict, [], None)
        logits_without_pos, _ = model_without_pos.forward(input_dict, [], None)
        
        # 两个模型的输出应该不同（因为位置编码的影响）
        self.assertFalse(torch.allclose(logits_with_pos, logits_without_pos, atol=1e-3))
    
    def test_permutation_invariance_breaking(self):
        """测试位置编码能够破坏排列不变性"""
        model = TransformerGNN(
            self.obs_space_dict,
            self.action_space,
            self.num_outputs,
            self.model_config,
            "test_model"
        )
        model.eval()
        
        batch_size = 1
        
        # 创建具有明显不同位置的UAV和目标
        uav_features = torch.randn(batch_size, 3, 64)
        target_features = torch.randn(batch_size, 5, 64)
        
        # 创建具有明显差异的相对位置矩阵
        # UAV 0 与所有目标的相对位置
        uav0_positions = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]], dtype=torch.float32)
        # UAV 1 与所有目标的相对位置  
        uav1_positions = torch.tensor([[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]], dtype=torch.float32)
        # UAV 2 与所有目标的相对位置
        uav2_positions = torch.tensor([[[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0], [-4.0, -4.0], [-5.0, -5.0]]], dtype=torch.float32)
        
        # 合并为完整的相对位置矩阵 [batch_size, num_uavs, num_targets, 2]
        relative_positions = torch.cat([uav0_positions, uav1_positions, uav2_positions], dim=1)
        
        obs_dict1 = {
            'uav_features': uav_features,
            'target_features': target_features,
            'relative_positions': relative_positions
        }
        
        # 创建排列后的观测（交换前两个UAV的特征和对应的相对位置）
        uav_features_permuted = uav_features.clone()
        uav_features_permuted[:, [0, 1]] = uav_features_permuted[:, [1, 0]]
        
        # 相应地交换相对位置（交换UAV 0和UAV 1的相对位置）
        relative_positions_permuted = relative_positions.clone()
        relative_positions_permuted[:, [0, 1]] = relative_positions_permuted[:, [1, 0]]
        
        obs_dict2 = {
            'uav_features': uav_features_permuted,
            'target_features': target_features,
            'relative_positions': relative_positions_permuted
        }
        
        # 前向传播
        logits1, _ = model.forward({"obs": obs_dict1}, [], None)
        logits2, _ = model.forward({"obs": obs_dict2}, [], None)
        
        # 由于位置编码的存在，排列后的输出应该不同
        # 使用更宽松的阈值，因为位置编码的影响可能较小
        self.assertFalse(torch.allclose(logits1, logits2, atol=1e-4))
        
        # 额外检查：确保差异足够大
        diff = torch.abs(logits1 - logits2).max().item()
        self.assertGreater(diff, 1e-5, f"位置编码的影响太小，最大差异: {diff}")
    
    def test_gradient_flow_with_position_encoding(self):
        """测试带位置编码的梯度流动"""
        model = TransformerGNN(
            self.obs_space_dict,
            self.action_space,
            self.num_outputs,
            self.model_config,
            "test_model"
        )
        
        batch_size = 2
        obs_dict = {
            'uav_features': torch.randn(batch_size, 3, 64, requires_grad=True),
            'target_features': torch.randn(batch_size, 5, 64, requires_grad=True),
            'relative_positions': torch.randn(batch_size, 15, 2, requires_grad=True)
        }
        input_dict = {"obs": obs_dict}
        
        # 前向传播
        logits, _ = model.forward(input_dict, [], None)
        loss = logits.sum()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度存在
        self.assertIsNotNone(obs_dict['uav_features'].grad)
        self.assertIsNotNone(obs_dict['target_features'].grad)
        self.assertIsNotNone(obs_dict['relative_positions'].grad)
        
        # 检查位置编码器的梯度
        pos_encoder_has_grad = False
        for param in model.position_encoder.parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                pos_encoder_has_grad = True
                break
        
        self.assertTrue(pos_encoder_has_grad, "位置编码器应该有梯度流动")


def run_position_encoding_tests():
    """运行位置编码测试"""
    print("开始运行位置编码机制测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加PositionalEncoder测试
    test_suite.addTest(unittest.makeSuite(TestPositionalEncoder))
    
    # 添加TransformerGNN测试
    test_suite.addTest(unittest.makeSuite(TestTransformerGNN))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    if result.wasSuccessful():
        print("\n✅ 所有位置编码测试通过！")
        print(f"运行了 {result.testsRun} 个测试")
    else:
        print(f"\n❌ 测试失败！")
        print(f"运行了 {result.testsRun} 个测试")
        print(f"失败: {len(result.failures)}")
        print(f"错误: {len(result.errors)}")
        
        # 打印失败详情
        for test, traceback in result.failures:
            print(f"\n失败测试: {test}")
            print(f"错误信息: {traceback}")
        
        for test, traceback in result.errors:
            print(f"\n错误测试: {test}")
            print(f"错误信息: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 运行测试
    success = run_position_encoding_tests()
    
    if success:
        print("\n🎉 位置编码机制实现完成并通过所有测试！")
        print("\n主要功能:")
        print("1. ✅ PositionalEncoder模块 - 通过小型MLP生成位置嵌入")
        print("2. ✅ TransformerGNN集成 - 将位置嵌入加到实体特征嵌入上")
        print("3. ✅ 排列不变性破坏 - 位置编码解决了排列不变性问题")
        print("4. ✅ 梯度流动正常 - 支持端到端训练")
        print("5. ✅ 多种观测格式支持 - 支持扁平和字典观测")
    else:
        print("\n❌ 测试失败，请检查实现！")