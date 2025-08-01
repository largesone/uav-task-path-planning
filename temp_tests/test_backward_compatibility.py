# -*- coding: utf-8 -*-
# 文件名: temp_tests/test_backward_compatibility.py
# 描述: 向后兼容性测试，验证现有功能不受影响

import sys
import os
import unittest
import numpy as np
import torch
import tempfile
import shutil
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compatibility_manager import CompatibilityManager, CompatibilityConfig
from entities import UAV, Target
from environment import DirectedGraph, UAVTaskEnv
from config import Config
from networks import create_network
from main import GraphRLSolver


class TestBackwardCompatibility(unittest.TestCase):
    """向后兼容性测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试实体
        self.uavs = [
            UAV(0, [100, 100], 0, [10, 10], 500, [20, 50], 30),
            UAV(1, [200, 200], 0, [8, 12], 600, [25, 45], 35),
            UAV(2, [300, 300], 0, [12, 8], 550, [30, 40], 32)
        ]
        
        self.targets = [
            Target(0, [400, 400], [5, 5], 100),
            Target(1, [500, 500], [7, 3], 150)
        ]
        
        # 创建图
        self.graph = DirectedGraph(self.uavs, self.targets, 6, [], self.config)
        
        print(f"[TestBackwardCompatibility] 测试环境准备完成")
        print(f"  - UAV数量: {len(self.uavs)}")
        print(f"  - 目标数量: {len(self.targets)}")
        print(f"  - 临时目录: {self.temp_dir}")
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_traditional_network_creation(self):
        """测试传统网络创建"""
        print("\n[测试] 传统网络创建")
        
        # 测试所有传统网络类型
        traditional_types = ["SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual"]
        
        for network_type in traditional_types:
            with self.subTest(network_type=network_type):
                print(f"  测试网络类型: {network_type}")
                
                try:
                    network = create_network(
                        network_type=network_type,
                        input_dim=128,
                        hidden_dims=[256, 128],
                        output_dim=64
                    )
                    
                    # 验证网络结构
                    self.assertIsNotNone(network)
                    
                    # 测试前向传播
                    test_input = torch.randn(4, 128)
                    output = network(test_input)
                    
                    self.assertEqual(output.shape, (4, 64))
                    print(f"    ✓ {network_type} 创建和前向传播成功")
                    
                except Exception as e:
                    self.fail(f"传统网络 {network_type} 创建失败: {e}")
    
    def test_environment_flat_mode(self):
        """测试环境扁平模式（向后兼容）"""
        print("\n[测试] 环境扁平模式")
        
        try:
            # 创建扁平模式环境
            env = UAVTaskEnv(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                config=self.config,
                obs_mode="flat"
            )
            
            # 验证观测空间
            self.assertIsNotNone(env.observation_space)
            print(f"    观测空间形状: {env.observation_space.shape}")
            
            # 测试环境重置
            state = env.reset()
            self.assertIsInstance(state, np.ndarray)
            print(f"    初始状态形状: {state.shape}")
            
            # 测试环境步进
            action = 0  # 第一个有效动作
            next_state, reward, done, truncated, info = env.step(action)
            
            self.assertIsInstance(next_state, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            
            print(f"    ✓ 扁平模式环境测试成功")
            print(f"      - 状态形状: {next_state.shape}")
            print(f"      - 奖励: {reward:.2f}")
            print(f"      - 完成状态: {done}")
            
        except Exception as e:
            self.fail(f"扁平模式环境测试失败: {e}")
    
    def test_environment_graph_mode(self):
        """测试环境图模式"""
        print("\n[测试] 环境图模式")
        
        try:
            # 创建图模式环境
            env = UAVTaskEnv(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                config=self.config,
                obs_mode="graph"
            )
            
            # 验证观测空间
            self.assertIsNotNone(env.observation_space)
            self.assertTrue(hasattr(env.observation_space, 'spaces'))
            print(f"    观测空间键: {list(env.observation_space.spaces.keys())}")
            
            # 测试环境重置
            state = env.reset()
            self.assertIsInstance(state, dict)
            
            # 验证状态结构
            expected_keys = ["uav_features", "target_features", "relative_positions", "distances", "masks"]
            for key in expected_keys:
                self.assertIn(key, state, f"缺少状态键: {key}")
            
            print(f"    ✓ 图模式环境测试成功")
            print(f"      - UAV特征形状: {state['uav_features'].shape}")
            print(f"      - 目标特征形状: {state['target_features'].shape}")
            print(f"      - 相对位置形状: {state['relative_positions'].shape}")
            print(f"      - 距离矩阵形状: {state['distances'].shape}")
            
            # 测试环境步进
            action = 0
            next_state, reward, done, truncated, info = env.step(action)
            
            self.assertIsInstance(next_state, dict)
            print(f"      - 步进测试成功，奖励: {reward:.2f}")
            
        except Exception as e:
            self.fail(f"图模式环境测试失败: {e}")
    
    def test_compatibility_manager_traditional_mode(self):
        """测试兼容性管理器传统模式"""
        print("\n[测试] 兼容性管理器传统模式")
        
        try:
            # 创建传统模式配置
            config = CompatibilityConfig(
                network_mode="traditional",
                traditional_network_type="DeepFCNResidual",
                obs_mode="flat",
                debug_mode=True
            )
            
            manager = CompatibilityManager(config)
            
            # 测试网络创建
            network = manager.create_network(
                input_dim=128,
                hidden_dims=[256, 128],
                output_dim=64
            )
            
            self.assertIsNotNone(network)
            print(f"    ✓ 传统网络创建成功")
            
            # 测试环境创建
            env = manager.create_environment(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                config=self.config
            )
            
            self.assertIsNotNone(env)
            self.assertEqual(env.obs_mode, "flat")
            print(f"    ✓ 传统环境创建成功")
            
            # 测试求解器创建
            solver = manager.create_solver(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                i_dim=128,
                h_dim=[256, 128],
                o_dim=64,
                config=self.config
            )
            
            self.assertIsNotNone(solver)
            print(f"    ✓ 传统求解器创建成功")
            
        except Exception as e:
            self.fail(f"兼容性管理器传统模式测试失败: {e}")
    
    def test_compatibility_manager_transformer_mode(self):
        """测试兼容性管理器TransformerGNN模式"""
        print("\n[测试] 兼容性管理器TransformerGNN模式")
        
        try:
            # 创建TransformerGNN模式配置
            config = CompatibilityConfig(
                network_mode="transformer_gnn",
                obs_mode="graph",
                debug_mode=True
            )
            
            manager = CompatibilityManager(config)
            
            # 首先创建环境以获取观测空间
            env = manager.create_environment(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                config=self.config
            )
            
            self.assertIsNotNone(env)
            self.assertEqual(env.obs_mode, "graph")
            print(f"    ✓ TransformerGNN环境创建成功")
            
            # 测试网络创建
            network = manager.create_network(
                input_dim=None,  # TransformerGNN不使用
                hidden_dims=None,  # TransformerGNN不使用
                output_dim=env.n_actions,
                obs_space=env.observation_space,
                action_space=env.action_space
            )
            
            self.assertIsNotNone(network)
            print(f"    ✓ TransformerGNN网络创建成功")
            
            # 测试网络前向传播
            state = env.reset()
            state_tensor = self._convert_state_to_tensor(state)
            
            with torch.no_grad():
                output, _ = network.forward({"obs": state_tensor}, [], [])
                self.assertEqual(output.shape[1], env.n_actions)
                print(f"    ✓ TransformerGNN前向传播成功，输出形状: {output.shape}")
            
        except Exception as e:
            self.fail(f"兼容性管理器TransformerGNN模式测试失败: {e}")
    
    def test_config_save_load(self):
        """测试配置保存和加载"""
        print("\n[测试] 配置保存和加载")
        
        try:
            # 创建配置
            original_config = CompatibilityConfig(
                network_mode="transformer_gnn",
                traditional_network_type="GAT",
                obs_mode="graph",
                enable_compatibility_checks=False,
                debug_mode=True
            )
            
            original_manager = CompatibilityManager(original_config)
            
            # 保存配置
            config_path = os.path.join(self.temp_dir, "test_config.json")
            original_manager.save_config(config_path)
            
            self.assertTrue(os.path.exists(config_path))
            print(f"    ✓ 配置保存成功: {config_path}")
            
            # 加载配置
            loaded_manager = CompatibilityManager.load_config(config_path)
            
            # 验证配置一致性
            self.assertEqual(loaded_manager.config.network_mode, original_config.network_mode)
            self.assertEqual(loaded_manager.config.traditional_network_type, original_config.traditional_network_type)
            self.assertEqual(loaded_manager.config.obs_mode, original_config.obs_mode)
            self.assertEqual(loaded_manager.config.enable_compatibility_checks, original_config.enable_compatibility_checks)
            self.assertEqual(loaded_manager.config.debug_mode, original_config.debug_mode)
            
            print(f"    ✓ 配置加载和验证成功")
            
        except Exception as e:
            self.fail(f"配置保存和加载测试失败: {e}")
    
    def test_compatibility_checks(self):
        """测试兼容性检查"""
        print("\n[测试] 兼容性检查")
        
        try:
            # 创建启用兼容性检查的管理器
            config = CompatibilityConfig(
                enable_compatibility_checks=True,
                debug_mode=True
            )
            
            manager = CompatibilityManager(config)
            
            # 运行兼容性检查
            results = manager.run_compatibility_checks()
            
            self.assertIsInstance(results, dict)
            self.assertIn("overall_compatibility", results)
            
            # 验证关键检查项
            expected_checks = [
                "traditional_network_creation",
                "transformer_gnn_creation",
                "overall_compatibility"
            ]
            
            for check in expected_checks:
                self.assertIn(check, results, f"缺少检查项: {check}")
            
            print(f"    ✓ 兼容性检查完成")
            print(f"      - 总体状态: {results['overall_compatibility']}")
            print(f"      - 检查项数量: {len(results)}")
            
            # 如果有失败的检查，输出详细信息
            failed_checks = [k for k, v in results.items() if v is False]
            if failed_checks:
                print(f"      - 失败的检查: {failed_checks}")
            
        except Exception as e:
            self.fail(f"兼容性检查测试失败: {e}")
    
    def test_run_scenario_compatibility(self):
        """测试run_scenario流程兼容性"""
        print("\n[测试] run_scenario流程兼容性")
        
        try:
            # 测试传统模式的run_scenario兼容性
            config = CompatibilityConfig(
                network_mode="traditional",
                traditional_network_type="DeepFCNResidual",
                obs_mode="flat"
            )
            
            manager = CompatibilityManager(config)
            
            # 创建环境
            env = manager.create_environment(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                config=self.config
            )
            
            # 模拟run_scenario的核心流程
            state = env.reset()
            self.assertIsInstance(state, np.ndarray)
            
            # 执行几步
            for step in range(3):
                action = step % env.n_actions  # 简单的动作选择
                next_state, reward, done, truncated, info = env.step(action)
                
                self.assertIsInstance(next_state, np.ndarray)
                self.assertIsInstance(reward, (int, float))
                
                if done or truncated:
                    break
                
                state = next_state
            
            print(f"    ✓ 传统模式run_scenario兼容性测试成功")
            
            # 测试图模式的兼容性
            config.obs_mode = "graph"
            manager = CompatibilityManager(config)
            
            env_graph = manager.create_environment(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=[],
                config=self.config
            )
            
            state_graph = env_graph.reset()
            self.assertIsInstance(state_graph, dict)
            
            print(f"    ✓ 图模式run_scenario兼容性测试成功")
            
        except Exception as e:
            self.fail(f"run_scenario流程兼容性测试失败: {e}")
    
    def _convert_state_to_tensor(self, state):
        """将状态转换为张量（用于TransformerGNN测试）"""
        if isinstance(state, dict):
            # 图模式状态
            tensor_state = {}
            for key, value in state.items():
                if key == "masks":
                    tensor_state[key] = {
                        sub_key: torch.tensor(sub_value, dtype=torch.float32).unsqueeze(0)
                        for sub_key, sub_value in value.items()
                    }
                else:
                    tensor_state[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
            return tensor_state
        else:
            # 扁平模式状态
            return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def run_compatibility_tests():
    """运行所有兼容性测试"""
    print("=" * 60)
    print("开始向后兼容性测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBackwardCompatibility)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("向后兼容性测试结果摘要")
    print("=" * 60)
    print(f"运行测试数量: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n总体结果: {'✓ 全部通过' if success else '✗ 存在问题'}")
    
    return success


if __name__ == "__main__":
    success = run_compatibility_tests()
    sys.exit(0 if success else 1)