# -*- coding: utf-8 -*-
"""
文件名: test_curriculum_training.py
描述: 课程学习训练协调器测试代码
作者: AI Assistant
日期: 2024

测试内容:
1. CurriculumStage和CurriculumConfig配置正确性
2. CurriculumTrainer初始化和基本功能
3. 训练阶段配置管理
4. 回退机制测试
5. 检查点管理测试
6. 与现有系统的兼容性测试
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入待测试模块
try:
    from run_curriculum_training import (
        CurriculumStage, CurriculumConfig, CurriculumTrainer, 
        CheckpointManager, create_env_wrapper
    )
    from config import Config
    from environment import UAVTaskEnv
    from scenarios import get_small_scenario, get_balanced_scenario
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"导入错误: {e}")
    IMPORTS_SUCCESS = False

class TestCurriculumStage(unittest.TestCase):
    """测试CurriculumStage配置类"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
    
    def test_curriculum_stage_creation(self):
        """测试课程阶段创建"""
        stage = CurriculumStage(
            name="test_stage",
            description="测试阶段",
            scenario_func=get_small_scenario,
            scenario_params={"obstacle_tolerance": 30.0},
            max_episodes=100,
            success_threshold=0.8
        )
        
        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.description, "测试阶段")
        self.assertEqual(stage.max_episodes, 100)
        self.assertEqual(stage.success_threshold, 0.8)
        self.assertEqual(stage.fallback_threshold, 0.6)  # 默认值
        self.assertEqual(stage.fallback_patience, 3)     # 默认值
    
    def test_curriculum_stage_defaults(self):
        """测试课程阶段默认参数"""
        stage = CurriculumStage(
            name="test",
            description="test",
            scenario_func=get_small_scenario,
            scenario_params={"obstacle_tolerance": 50.0},
            max_episodes=100
        )
        
        # 检查默认值
        self.assertEqual(stage.success_threshold, 0.8)
        self.assertEqual(stage.min_episodes, 100)
        self.assertEqual(stage.performance_window, 50)
        self.assertEqual(stage.learning_rate, 3e-4)
        self.assertEqual(stage.batch_size, 64)
        self.assertEqual(stage.gamma, 0.99)

class TestCurriculumConfig(unittest.TestCase):
    """测试CurriculumConfig配置类"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
    
    def test_curriculum_config_creation(self):
        """测试课程配置创建"""
        config = CurriculumConfig(
            mixed_replay_ratio=0.3,
            evaluation_frequency=10,
            output_dir="test_output"
        )
        
        self.assertEqual(config.mixed_replay_ratio, 0.3)
        self.assertEqual(config.evaluation_frequency, 10)
        self.assertEqual(config.output_dir, "test_output")
        self.assertEqual(len(config.stages), 4)  # 默认4个阶段
    
    def test_default_curriculum_stages(self):
        """测试默认课程阶段"""
        config = CurriculumConfig()
        
        # 检查默认阶段数量
        self.assertEqual(len(config.stages), 4)
        
        # 检查阶段名称
        expected_names = [
            "stage_1_simple", "stage_2_balanced", 
            "stage_3_complex", "stage_4_experimental"
        ]
        actual_names = [stage.name for stage in config.stages]
        self.assertEqual(actual_names, expected_names)
        
        # 检查阶段难度递增
        max_episodes = [stage.max_episodes for stage in config.stages]
        self.assertEqual(max_episodes, [500, 800, 1200, 1500])
        
        # 检查成功阈值递减（更难的场景要求更低）
        thresholds = [stage.success_threshold for stage in config.stages]
        self.assertEqual(thresholds, [0.85, 0.80, 0.75, 0.70])

class TestCheckpointManager(unittest.TestCase):
    """测试检查点管理器"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_manager_init(self):
        """测试检查点管理器初始化"""
        self.assertTrue(os.path.exists(self.checkpoint_manager.checkpoints_dir))
        expected_path = os.path.join(self.temp_dir, "checkpoints")
        self.assertEqual(self.checkpoint_manager.checkpoints_dir, expected_path)
    
    def test_save_stage_checkpoint(self):
        """测试保存阶段检查点"""
        stage_result = {
            "stage_name": "test_stage",
            "episodes_trained": 100,
            "success": True,
            "final_performance": 0.85
        }
        
        self.checkpoint_manager.save_stage_checkpoint(0, stage_result)
        
        # 检查文件是否创建
        checkpoint_path = os.path.join(
            self.checkpoint_manager.checkpoints_dir,
            "stage_0_checkpoint.json"
        )
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # 检查文件内容
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["stage_name"], "test_stage")
        self.assertEqual(saved_data["episodes_trained"], 100)
        self.assertTrue(saved_data["success"])

class TestCurriculumTrainer(unittest.TestCase):
    """测试课程学习训练器"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.base_config = Config()
        self.curriculum_config = CurriculumConfig(
            output_dir=self.temp_dir,
            num_workers=1,  # 减少资源使用
            evaluation_frequency=5
        )
        
        # 简化课程阶段用于测试
        self.curriculum_config.stages = [
            CurriculumStage(
                name="test_stage_1",
                description="测试阶段1",
                scenario_func=get_small_scenario,
                scenario_params={"obstacle_tolerance": 20.0},
                max_episodes=10,  # 减少训练轮数
                success_threshold=0.5,  # 降低成功阈值
                min_episodes=5
            ),
            CurriculumStage(
                name="test_stage_2", 
                description="测试阶段2",
                scenario_func=get_balanced_scenario,
                scenario_params={"obstacle_tolerance": 30.0},
                max_episodes=10,
                success_threshold=0.4,
                min_episodes=5
            )
        ]
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('run_curriculum_training.ray')
    def test_curriculum_trainer_init(self, mock_ray):
        """测试课程训练器初始化"""
        mock_ray.is_initialized.return_value = False
        
        trainer = CurriculumTrainer(self.curriculum_config, self.base_config)
        
        self.assertEqual(trainer.current_stage_idx, 0)
        self.assertEqual(len(trainer.curriculum_config.stages), 2)
        self.assertTrue(os.path.exists(self.curriculum_config.output_dir))
    
    def test_create_env_config(self):
        """测试环境配置创建"""
        trainer = CurriculumTrainer(self.curriculum_config, self.base_config)
        stage = self.curriculum_config.stages[0]
        
        env_config = trainer._create_env_config(stage)
        
        self.assertIn("uavs", env_config)
        self.assertIn("targets", env_config)
        self.assertIn("obstacles", env_config)
        self.assertIn("config", env_config)
        self.assertEqual(env_config["obs_mode"], "graph")
        
        # 检查场景数据类型
        self.assertIsInstance(env_config["uavs"], list)
        self.assertIsInstance(env_config["targets"], list)
        self.assertIsInstance(env_config["obstacles"], list)
    
    @patch('run_curriculum_training.run_scenario')
    def test_train_stage_fallback(self, mock_run_scenario):
        """测试回退训练方法"""
        # 模拟run_scenario返回值
        mock_run_scenario.return_value = (
            {},  # final_plan
            10.0,  # training_time
            {"episode_rewards": [1, 2, 3, 4, 5]},  # training_history
            {"completion_rate": 0.6}  # evaluation_metrics
        )
        
        trainer = CurriculumTrainer(self.curriculum_config, self.base_config)
        stage = self.curriculum_config.stages[0]
        env_config = trainer._create_env_config(stage)
        
        result = trainer._train_stage_fallback(stage, 0, env_config)
        
        # 检查结果结构
        self.assertIn("stage_name", result)
        self.assertIn("episodes_trained", result)
        self.assertIn("success", result)
        self.assertIn("final_performance", result)
        
        # 检查具体值
        self.assertEqual(result["stage_name"], "test_stage_1")
        self.assertEqual(result["episodes_trained"], 5)
        self.assertTrue(result["success"])  # 0.6 >= 0.5
        self.assertEqual(result["final_performance"], 0.6)

class TestEnvironmentWrapper(unittest.TestCase):
    """测试环境包装器"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
    
    def test_create_env_wrapper(self):
        """测试环境包装器创建"""
        # 创建测试场景
        uavs, targets, obstacles = get_small_scenario(20.0)
        base_config = Config()
        
        env_config = {
            "uavs": uavs,
            "targets": targets,
            "obstacles": obstacles,
            "config": base_config,
            "obs_mode": "graph"
        }
        
        # 创建环境
        env = create_env_wrapper(env_config)
        
        # 检查环境类型
        self.assertIsInstance(env, UAVTaskEnv)
        
        # 检查环境配置
        self.assertEqual(env.obs_mode, "graph")
        self.assertEqual(len(env.uavs), len(uavs))
        self.assertEqual(len(env.targets), len(targets))

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('run_curriculum_training.RLLIB_AVAILABLE', False)
    @patch('run_curriculum_training.run_scenario')
    def test_full_curriculum_training_fallback(self, mock_run_scenario):
        """测试完整课程学习训练（回退模式）"""
        # 模拟不同阶段的训练结果
        def mock_scenario_side_effect(*args, **kwargs):
            scenario_name = kwargs.get('scenario_name', '')
            if 'stage_1' in scenario_name:
                completion_rate = 0.6  # 成功
            else:
                completion_rate = 0.3  # 失败
            
            return (
                {},  # final_plan
                5.0,  # training_time
                {"episode_rewards": [1, 2, 3]},  # training_history
                {"completion_rate": completion_rate}  # evaluation_metrics
            )
        
        mock_run_scenario.side_effect = mock_scenario_side_effect
        
        # 创建简化的课程配置
        curriculum_config = CurriculumConfig(
            output_dir=self.temp_dir,
            evaluation_frequency=2
        )
        curriculum_config.stages = [
            CurriculumStage(
                name="test_stage_1",
                description="测试阶段1",
                scenario_func=get_small_scenario,
                scenario_params={"obstacle_tolerance": 20.0},
                max_episodes=5,
                success_threshold=0.5,
                min_episodes=2
            ),
            CurriculumStage(
                name="test_stage_2",
                description="测试阶段2", 
                scenario_func=get_balanced_scenario,
                scenario_params={"obstacle_tolerance": 30.0},
                max_episodes=5,
                success_threshold=0.5,
                min_episodes=2
            )
        ]
        
        base_config = Config()
        trainer = CurriculumTrainer(curriculum_config, base_config)
        
        # 执行训练
        result = trainer.train()
        
        # 检查结果
        self.assertIn("stages_completed", result)
        self.assertIn("total_episodes", result)
        self.assertIn("stage_results", result)
        
        # 检查阶段结果
        self.assertEqual(len(result["stage_results"]), 2)
        self.assertTrue(result["stage_results"][0]["success"])  # 第一阶段成功
        self.assertFalse(result["stage_results"][1]["success"])  # 第二阶段失败
        
        # 检查训练摘要文件
        summary_path = os.path.join(self.temp_dir, "training_summary.json")
        self.assertTrue(os.path.exists(summary_path))

def run_basic_functionality_test():
    """运行基本功能测试"""
    print("=" * 60)
    print("课程学习训练协调器 - 基本功能测试")
    print("=" * 60)
    
    try:
        # 测试1: 导入测试
        print("测试1: 模块导入...")
        if IMPORTS_SUCCESS:
            print("✓ 所有模块导入成功")
        else:
            print("✗ 模块导入失败")
            return False
        
        # 测试2: 配置创建测试
        print("\n测试2: 配置创建...")
        try:
            config = CurriculumConfig()
            print(f"✓ 课程配置创建成功，包含{len(config.stages)}个阶段")
            
            base_config = Config()
            print("✓ 基础配置创建成功")
        except Exception as e:
            print(f"✗ 配置创建失败: {e}")
            return False
        
        # 测试3: 训练器初始化测试
        print("\n测试3: 训练器初始化...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config.output_dir = temp_dir
                trainer = CurriculumTrainer(config, base_config)
                print("✓ 训练器初始化成功")
        except Exception as e:
            print(f"✗ 训练器初始化失败: {e}")
            return False
        
        # 测试4: 环境配置测试
        print("\n测试4: 环境配置...")
        try:
            stage = config.stages[0]
            env_config = trainer._create_env_config(stage)
            
            # 检查环境配置
            required_keys = ["uavs", "targets", "obstacles", "config", "obs_mode"]
            for key in required_keys:
                if key not in env_config:
                    raise ValueError(f"缺少环境配置键: {key}")
            
            print("✓ 环境配置创建成功")
            print(f"  - UAV数量: {len(env_config['uavs'])}")
            print(f"  - 目标数量: {len(env_config['targets'])}")
            print(f"  - 障碍物数量: {len(env_config['obstacles'])}")
            print(f"  - 观测模式: {env_config['obs_mode']}")
        except Exception as e:
            print(f"✗ 环境配置失败: {e}")
            return False
        
        # 测试5: 检查点管理器测试
        print("\n测试5: 检查点管理...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_manager = CheckpointManager(temp_dir)
                
                # 测试保存检查点
                test_result = {
                    "stage_name": "test",
                    "episodes_trained": 10,
                    "success": True
                }
                checkpoint_manager.save_stage_checkpoint(0, test_result)
                
                # 检查文件是否存在
                checkpoint_path = os.path.join(
                    checkpoint_manager.checkpoints_dir,
                    "stage_0_checkpoint.json"
                )
                if os.path.exists(checkpoint_path):
                    print("✓ 检查点保存成功")
                else:
                    raise FileNotFoundError("检查点文件未创建")
        except Exception as e:
            print(f"✗ 检查点管理失败: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✓ 所有基本功能测试通过!")
        print("课程学习训练协调器已准备就绪")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生未预期错误: {e}")
        return False

def main():
    """主测试函数"""
    print("课程学习训练协调器测试套件")
    print("=" * 50)
    
    # 运行基本功能测试
    basic_test_passed = run_basic_functionality_test()
    
    if not basic_test_passed:
        print("\n基本功能测试失败，跳过单元测试")
        return
    
    # 运行单元测试
    print("\n开始运行单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCurriculumStage,
        TestCurriculumConfig, 
        TestCheckpointManager,
        TestCurriculumTrainer,
        TestEnvironmentWrapper,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"跳过数: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
        
        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
