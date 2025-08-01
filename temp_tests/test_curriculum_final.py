# -*- coding: utf-8 -*-
"""
文件名: test_curriculum_final.py
描述: 课程学习训练协调器最终测试版本（解决日志冲突）
作者: AI Assistant
日期: 2024

特点:
- 解决Ray和unittest的输出缓冲区冲突
- 简化日志输出
- 稳定的测试环境
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 禁用过多的日志输出
logging.getLogger('ray').setLevel(logging.ERROR)
logging.getLogger('run_curriculum_training').setLevel(logging.ERROR)

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
    from entities import UAV, Target
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"导入错误: {e}")
    IMPORTS_SUCCESS = False

class TestCurriculumStageBasic(unittest.TestCase):
    """测试CurriculumStage配置类 - 基础版"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
    
    def test_stage_creation(self):
        """测试阶段创建"""
        def dummy_scenario(obstacle_tolerance=50.0):
            return [], [], []
        
        stage = CurriculumStage(
            name="test_stage",
            description="测试阶段",
            scenario_func=dummy_scenario,
            scenario_params={"obstacle_tolerance": 30.0},
            max_episodes=100,
            success_threshold=0.8
        )
        
        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.max_episodes, 100)
        self.assertEqual(stage.success_threshold, 0.8)

class TestCurriculumConfigBasic(unittest.TestCase):
    """测试CurriculumConfig配置类 - 基础版"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
    
    def test_config_creation(self):
        """测试配置创建"""
        config = CurriculumConfig(
            mixed_replay_ratio=0.3,
            evaluation_frequency=10,
            output_dir="test_output"
        )
        
        self.assertEqual(config.mixed_replay_ratio, 0.3)
        self.assertEqual(config.evaluation_frequency, 10)
        self.assertEqual(len(config.stages), 4)  # 默认4个阶段

class TestCheckpointManagerBasic(unittest.TestCase):
    """测试检查点管理器 - 基础版"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
        
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_save(self):
        """测试检查点保存"""
        stage_result = {
            "stage_name": "test_stage",
            "episodes_trained": 100,
            "success": True,
            "final_performance": 0.85
        }
        
        # 临时禁用日志
        with patch('run_curriculum_training.logger'):
            self.checkpoint_manager.save_stage_checkpoint(0, stage_result)
        
        # 检查文件是否创建
        checkpoint_path = os.path.join(
            self.checkpoint_manager.checkpoints_dir,
            "stage_0_checkpoint.json"
        )
        self.assertTrue(os.path.exists(checkpoint_path))

class TestCurriculumTrainerBasic(unittest.TestCase):
    """测试课程学习训练器 - 基础版"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
        
        self.temp_dir = tempfile.mkdtemp()
        self.base_config = Config()
        
        # 创建简化配置
        def simple_scenario(obstacle_tolerance=50.0):
            uavs = [UAV(1, [10, 10], 0, [50, 50], 100, [10, 20], 15)]
            targets = [Target(1, [100, 100], [30, 30], 100)]
            obstacles = []
            return uavs, targets, obstacles
        
        self.curriculum_config = CurriculumConfig(
            output_dir=self.temp_dir,
            num_workers=1,
            evaluation_frequency=5
        )
        
        # 使用简化的阶段配置
        self.curriculum_config.stages = [
            CurriculumStage(
                name="test_stage_1",
                description="测试阶段1",
                scenario_func=simple_scenario,
                scenario_params={"obstacle_tolerance": 20.0},
                max_episodes=5,
                success_threshold=0.5,
                min_episodes=2
            )
        ]
    
    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('run_curriculum_training.ray')
    @patch('run_curriculum_training.logger')
    def test_trainer_init_mocked(self, mock_logger, mock_ray):
        """测试训练器初始化（模拟Ray）"""
        mock_ray.is_initialized.return_value = False
        
        trainer = CurriculumTrainer(self.curriculum_config, self.base_config)
        
        self.assertEqual(trainer.current_stage_idx, 0)
        self.assertEqual(len(trainer.curriculum_config.stages), 1)
        self.assertTrue(os.path.exists(self.curriculum_config.output_dir))
    
    @patch('run_curriculum_training.ray')
    @patch('run_curriculum_training.logger')
    def test_env_config_creation(self, mock_logger, mock_ray):
        """测试环境配置创建"""
        mock_ray.is_initialized.return_value = False
        
        trainer = CurriculumTrainer(self.curriculum_config, self.base_config)
        stage = self.curriculum_config.stages[0]
        
        env_config = trainer._create_env_config(stage)
        
        # 检查基本结构
        required_keys = ["uavs", "targets", "obstacles", "config", "obs_mode"]
        for key in required_keys:
            self.assertIn(key, env_config)
        
        self.assertEqual(env_config["obs_mode"], "graph")
        self.assertIsInstance(env_config["uavs"], list)
        self.assertIsInstance(env_config["targets"], list)

class TestEnvironmentWrapperBasic(unittest.TestCase):
    """测试环境包装器 - 基础版"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_SUCCESS:
            self.skipTest("模块导入失败")
    
    def test_env_wrapper_creation(self):
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
        
        # 基本检查
        self.assertIsInstance(env, UAVTaskEnv)
        self.assertEqual(env.obs_mode, "graph")

def run_basic_functionality_test():
    """运行基本功能测试"""
    print("=" * 60)
    print("课程学习训练协调器 - 基本功能测试（最终版）")
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
        
        # 测试3: 训练器初始化测试（简化版）
        print("\n测试3: 训练器初始化...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config.output_dir = temp_dir
                
                # 使用模拟来避免Ray初始化
                with patch('run_curriculum_training.ray') as mock_ray, \
                     patch('run_curriculum_training.logger'):
                    mock_ray.is_initialized.return_value = False
                    trainer = CurriculumTrainer(config, base_config)
                    print("✓ 训练器初始化成功")
        except Exception as e:
            print(f"✗ 训练器初始化失败: {e}")
            return False
        
        # 测试4: 检查点管理器测试
        print("\n测试4: 检查点管理...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_manager = CheckpointManager(temp_dir)
                
                test_result = {
                    "stage_name": "test",
                    "episodes_trained": 10,
                    "success": True
                }
                
                # 使用模拟来避免日志输出
                with patch('run_curriculum_training.logger'):
                    checkpoint_manager.save_stage_checkpoint(0, test_result)
                
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
    print("课程学习训练协调器最终测试套件")
    print("=" * 50)
    
    # 运行基本功能测试
    basic_test_passed = run_basic_functionality_test()
    
    if not basic_test_passed:
        print("\n基本功能测试失败，跳过单元测试")
        return 1
    
    # 运行单元测试
    print("\n开始运行单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCurriculumStageBasic,
        TestCurriculumConfigBasic, 
        TestCheckpointManagerBasic,
        TestCurriculumTrainerBasic,
        TestEnvironmentWrapperBasic
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试，使用简化的输出
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout, buffer=True)
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
        return 0
    else:
        print("✗ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
