# -*- coding: utf-8 -*-
# 文件名: temp_tests/test_compatibility_integration.py
# 描述: 兼容性集成测试，验证新旧系统的集成和兼容性

import sys
import os
import unittest
import numpy as np
import torch
import tempfile
import shutil
import time
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compatibility_manager import CompatibilityManager, CompatibilityConfig
from main_compatible import CompatibleGraphRLSolver, run_scenario_compatible
from entities import UAV, Target
from environment import DirectedGraph
from config import Config
from scenarios import get_small_scenario, get_balanced_scenario


class TestCompatibilityIntegration(unittest.TestCase):
    """兼容性集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试场景
        self.uavs, self.targets, self.obstacles = get_small_scenario()
        self.graph = DirectedGraph(self.uavs, self.targets, 6, self.obstacles, self.config)
        
        print(f"[TestCompatibilityIntegration] 测试环境准备完成")
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_traditional_to_transformer_migration(self):
        """测试从传统网络到TransformerGNN的迁移"""
        print("\n[测试] 传统网络到TransformerGNN迁移")
        
        # 第一阶段：使用传统网络
        traditional_config = CompatibilityConfig(
            network_mode="traditional",
            traditional_network_type="DeepFCNResidual",
            obs_mode="flat",
            debug_mode=True
        )
        
        traditional_manager = CompatibilityManager(traditional_config)
        
        # 创建传统求解器
        traditional_solver = CompatibleGraphRLSolver(
            uavs=self.uavs,
            targets=self.targets,
            graph=self.graph,
            obstacles=self.obstacles,
            i_dim=128,
            h_dim=[256, 128],
            o_dim=36,
            config=self.config,
            compatibility_manager=traditional_manager
        )
        
        # 简短训练
        model_path = os.path.join(self.temp_dir, "traditional_model.pth")
        training_time = traditional_solver.train(
            episodes=10,
            patience=50,
            log_interval=5,
            model_save_path=model_path
        )
        
        self.assertGreater(training_time, 0)
        self.assertTrue(os.path.exists(model_path))
        print(f"    ✓ 传统网络训练完成，耗时: {training_time:.2f}秒")
        
        # 第二阶段：切换到TransformerGNN
        transformer_config = CompatibilityConfig(
            network_mode="transformer_gnn",
            obs_mode="graph",
            debug_mode=True
        )
        
        transformer_manager = CompatibilityManager(transformer_config)
        
        # 创建TransformerGNN求解器
        transformer_solver = CompatibleGraphRLSolver(
            uavs=self.uavs,
            targets=self.targets,
            graph=self.graph,
            obstacles=self.obstacles,
            i_dim=128,
            h_dim=[256, 128],
            o_dim=36,
            config=self.config,
            compatibility_manager=transformer_manager
        )
        
        # 简短训练
        transformer_model_path = os.path.join(self.temp_dir, "transformer_model.pth")
        transformer_training_time = transformer_solver.train(
            episodes=10,
            patience=50,
            log_interval=5,
            model_save_path=transformer_model_path
        )
        
        self.assertGreater(transformer_training_time, 0)
        self.assertTrue(os.path.exists(transformer_model_path))
        print(f"    ✓ TransformerGNN训练完成，耗时: {transformer_training_time:.2f}秒")
        
        # 验证两种方法都能正常工作
        self.assertIsNotNone(traditional_solver.episode_rewards)
        self.assertIsNotNone(transformer_solver.episode_rewards)
        self.assertGreater(len(traditional_solver.episode_rewards), 0)
        self.assertGreater(len(transformer_solver.episode_rewards), 0)
        
        print(f"    ✓ 迁移测试成功")
    
    def test_run_scenario_compatibility(self):
        """测试run_scenario函数的兼容性"""
        print("\n[测试] run_scenario函数兼容性")
        
        # 测试传统模式
        traditional_result = run_scenario_compatible(
            scenario_func=get_small_scenario,
            scenario_name="test_traditional",
            config_override={
                "network_mode": "traditional",
                "traditional_network_type": "SimpleNetwork",
                "obs_mode": "flat",
                "enable_compatibility_checks": False  # 跳过检查以加快测试
            }
        )
        
        self.assertIsInstance(traditional_result, dict)
        self.assertEqual(traditional_result["network_mode"], "traditional")
        self.assertEqual(traditional_result["obs_mode"], "flat")
        self.assertIn("training_time", traditional_result)
        self.assertIn("final_completion_rate", traditional_result)
        
        print(f"    ✓ 传统模式测试成功")
        print(f"      - 训练时间: {traditional_result['training_time']:.2f}秒")
        print(f"      - 完成率: {traditional_result['final_completion_rate']:.3f}")
        
        # 测试TransformerGNN模式
        transformer_result = run_scenario_compatible(
            scenario_func=get_small_scenario,
            scenario_name="test_transformer",
            config_override={
                "network_mode": "transformer_gnn",
                "obs_mode": "graph",
                "enable_compatibility_checks": False  # 跳过检查以加快测试
            }
        )
        
        self.assertIsInstance(transformer_result, dict)
        self.assertEqual(transformer_result["network_mode"], "transformer_gnn")
        self.assertEqual(transformer_result["obs_mode"], "graph")
        self.assertIn("training_time", transformer_result)
        self.assertIn("final_completion_rate", transformer_result)
        
        print(f"    ✓ TransformerGNN模式测试成功")
        print(f"      - 训练时间: {transformer_result['training_time']:.2f}秒")
        print(f"      - 完成率: {transformer_result['final_completion_rate']:.3f}")
    
    def test_output_format_compatibility(self):
        """测试输出格式兼容性"""
        print("\n[测试] 输出格式兼容性")
        
        # 创建两种模式的求解器
        configs = [
            ("traditional", "flat"),
            ("transformer_gnn", "graph")
        ]
        
        results = {}
        
        for network_mode, obs_mode in configs:
            config = CompatibilityConfig(
                network_mode=network_mode,
                obs_mode=obs_mode,
                debug_mode=False
            )
            
            manager = CompatibilityManager(config)
            
            solver = CompatibleGraphRLSolver(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=self.obstacles,
                i_dim=128,
                h_dim=[256, 128],
                o_dim=36,
                config=self.config,
                compatibility_manager=manager
            )
            
            # 运行几步训练
            solver.train(episodes=3, patience=10, log_interval=1, 
                        model_save_path=os.path.join(self.temp_dir, f"{network_mode}_test.pth"))
            
            # 检查输出格式
            self.assertIsInstance(solver.episode_rewards, list)
            self.assertIsInstance(solver.episode_losses, list)
            self.assertIsInstance(solver.epsilon_values, list)
            self.assertIsInstance(solver.completion_rates, list)
            
            # 验证输出数据的一致性
            self.assertEqual(len(solver.episode_rewards), len(solver.episode_losses))
            self.assertEqual(len(solver.episode_rewards), len(solver.epsilon_values))
            self.assertEqual(len(solver.episode_rewards), len(solver.completion_rates))
            
            results[network_mode] = {
                "rewards": solver.episode_rewards,
                "losses": solver.episode_losses,
                "completion_rates": solver.completion_rates
            }
            
            print(f"    ✓ {network_mode}模式输出格式验证成功")
        
        # 验证两种模式的输出格式一致
        traditional_keys = set(results["traditional"].keys())
        transformer_keys = set(results["transformer_gnn"].keys())
        self.assertEqual(traditional_keys, transformer_keys)
        
        print(f"    ✓ 输出格式兼容性验证成功")
    
    def test_model_save_load_compatibility(self):
        """测试模型保存和加载的兼容性"""
        print("\n[测试] 模型保存加载兼容性")
        
        # 测试传统网络
        traditional_config = CompatibilityConfig(
            network_mode="traditional",
            traditional_network_type="DeepFCN",
            obs_mode="flat"
        )
        
        manager = CompatibilityManager(traditional_config)
        
        solver = CompatibleGraphRLSolver(
            uavs=self.uavs,
            targets=self.targets,
            graph=self.graph,
            obstacles=self.obstacles,
            i_dim=128,
            h_dim=[256, 128],
            o_dim=36,
            config=self.config,
            compatibility_manager=manager
        )
        
        # 训练几轮
        solver.train(episodes=5, patience=10, log_interval=1,
                    model_save_path=os.path.join(self.temp_dir, "test_model.pth"))
        
        # 保存模型
        model_path = os.path.join(self.temp_dir, "manual_save.pth")
        solver.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # 创建新的求解器并加载模型
        new_solver = CompatibleGraphRLSolver(
            uavs=self.uavs,
            targets=self.targets,
            graph=self.graph,
            obstacles=self.obstacles,
            i_dim=128,
            h_dim=[256, 128],
            o_dim=36,
            config=self.config,
            compatibility_manager=manager
        )
        
        new_solver.load_model(model_path)
        
        # 验证模型加载成功（通过比较网络参数）
        original_params = list(solver.policy_net.parameters())
        loaded_params = list(new_solver.policy_net.parameters())
        
        self.assertEqual(len(original_params), len(loaded_params))
        
        for orig, loaded in zip(original_params, loaded_params):
            self.assertTrue(torch.allclose(orig, loaded, atol=1e-6))
        
        print(f"    ✓ 模型保存加载兼容性验证成功")
    
    def test_performance_comparison(self):
        """测试性能对比"""
        print("\n[测试] 性能对比")
        
        configs = [
            ("traditional", "SimpleNetwork", "flat"),
            ("traditional", "DeepFCN", "flat"),
            ("transformer_gnn", None, "graph")
        ]
        
        performance_results = {}
        
        for network_mode, network_type, obs_mode in configs:
            config_dict = {
                "network_mode": network_mode,
                "obs_mode": obs_mode,
                "enable_compatibility_checks": False
            }
            
            if network_type:
                config_dict["traditional_network_type"] = network_type
            
            config = CompatibilityConfig(**config_dict)
            manager = CompatibilityManager(config)
            
            solver = CompatibleGraphRLSolver(
                uavs=self.uavs,
                targets=self.targets,
                graph=self.graph,
                obstacles=self.obstacles,
                i_dim=128,
                h_dim=[256, 128],
                o_dim=36,
                config=self.config,
                compatibility_manager=manager
            )
            
            # 测量训练时间
            start_time = time.time()
            solver.train(episodes=5, patience=10, log_interval=1,
                        model_save_path=os.path.join(self.temp_dir, f"{network_mode}_{network_type}_test.pth"))
            training_time = time.time() - start_time
            
            # 记录性能指标
            final_reward = solver.episode_rewards[-1] if solver.episode_rewards else 0
            final_completion = solver.completion_rates[-1] if solver.completion_rates else 0
            
            key = f"{network_mode}_{network_type}" if network_type else network_mode
            performance_results[key] = {
                "training_time": training_time,
                "final_reward": final_reward,
                "final_completion": final_completion,
                "total_episodes": len(solver.episode_rewards)
            }
            
            print(f"    {key}:")
            print(f"      - 训练时间: {training_time:.2f}秒")
            print(f"      - 最终奖励: {final_reward:.2f}")
            print(f"      - 完成率: {final_completion:.3f}")
        
        # 验证所有配置都能正常运行
        for key, result in performance_results.items():
            self.assertGreater(result["training_time"], 0)
            self.assertGreaterEqual(result["total_episodes"], 5)
        
        print(f"    ✓ 性能对比测试完成")


def run_integration_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("开始兼容性集成测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCompatibilityIntegration)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("兼容性集成测试结果摘要")
    print("=" * 60)
    print(f"运行测试数量: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n总体结果: {'✓ 全部通过' if success else '✗ 存在问题'}")
    
    return success


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
