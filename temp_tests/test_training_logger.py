"""
训练日志记录器测试
测试TensorBoard集成、模型检查点保存和课程学习回调功能
"""

import os
import sys
import unittest
import tempfile
import shutil
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_logger import (
    CurriculumTensorBoardLogger, 
    ModelCheckpointManager, 
    CurriculumTrainingCallbacks,
    create_training_config_with_logging
)


class TestCurriculumTensorBoardLogger(unittest.TestCase):
    """测试课程学习TensorBoard日志记录器"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.logger = CurriculumTensorBoardLogger(self.test_dir, "test_experiment")
    
    def tearDown(self):
        """测试后清理"""
        self.logger.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.logger.experiment_name, "test_experiment")
        self.assertEqual(self.logger.current_stage, 0)
        self.assertIsNotNone(self.logger.writer)
    
    def test_log_stage_transition(self):
        """测试阶段切换记录"""
        self.logger.log_stage_transition(0, 1, 1000, "performance_threshold")
        
        # 检查记录是否正确保存
        self.assertEqual(len(self.logger.training_history["stage_transitions"]), 1)
        transition = self.logger.training_history["stage_transitions"][0]
        
        self.assertEqual(transition["from_stage"], 0)
        self.assertEqual(transition["to_stage"], 1)
        self.assertEqual(transition["step"], 1000)
        self.assertEqual(transition["reason"], "performance_threshold")
        self.assertEqual(self.logger.current_stage, 1)
    
    def test_log_rollback_event(self):
        """测试回退事件记录"""
        self.logger.log_rollback_event(2, 5000, 0.15, 0.1)
        
        # 检查回退事件记录
        self.assertEqual(len(self.logger.training_history["rollback_events"]), 1)
        rollback = self.logger.training_history["rollback_events"][0]
        
        self.assertEqual(rollback["stage"], 2)
        self.assertEqual(rollback["step"], 5000)
        self.assertEqual(rollback["performance_drop"], 0.15)
        self.assertEqual(rollback["threshold"], 0.1)
    
    def test_log_scale_invariant_metrics(self):
        """测试尺度不变指标记录"""
        metrics = {
            "per_agent_reward": 12.5,
            "normalized_completion_score": 0.85,
            "efficiency_metric": 0.42
        }
        
        self.logger.log_scale_invariant_metrics(metrics, 2000, 1, 5, 3)
        
        # 检查指标记录
        self.assertEqual(len(self.logger.training_history["metrics"]), 1)
        metric_record = self.logger.training_history["metrics"][0]
        
        self.assertEqual(metric_record["step"], 2000)
        self.assertEqual(metric_record["stage"], 1)
        self.assertEqual(metric_record["n_uavs"], 5)
        self.assertEqual(metric_record["n_targets"], 3)
        self.assertEqual(metric_record["per_agent_reward"], 12.5)
    
    def test_save_training_history(self):
        """测试训练历史保存"""
        # 添加一些测试数据
        self.logger.log_stage_transition(0, 1, 1000)
        self.logger.log_scale_invariant_metrics({"per_agent_reward": 10.0}, 1500, 1, 4, 2)
        
        # 保存历史
        self.logger.save_training_history()
        
        # 检查文件是否存在
        history_file = Path(self.test_dir) / "test_experiment_history.json"
        self.assertTrue(history_file.exists())
        
        # 检查文件内容
        import json
        with open(history_file, 'r', encoding='utf-8') as f:
            saved_history = json.load(f)
        
        self.assertEqual(len(saved_history["stage_transitions"]), 1)
        self.assertEqual(len(saved_history["metrics"]), 1)


class TestModelCheckpointManager(unittest.TestCase):
    """测试模型检查点管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelCheckpointManager(self.test_dir, max_checkpoints=3)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.manager.max_checkpoints, 3)
        self.assertEqual(len(self.manager.checkpoint_history), 0)
    
    def test_save_checkpoint(self):
        """测试检查点保存"""
        # 创建模拟模型状态
        model_state = {"layer1.weight": torch.randn(10, 5)}
        optimizer_state = {"lr": 0.001}
        metrics = {"normalized_completion_score": 0.85}
        
        # 保存检查点
        checkpoint_path = self.manager.save_checkpoint(
            model_state, optimizer_state, metrics, 1000, 1, is_best=True
        )
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # 检查历史记录
        self.assertEqual(len(self.manager.checkpoint_history), 1)
        checkpoint_info = self.manager.checkpoint_history[0]
        
        self.assertEqual(checkpoint_info["step"], 1000)
        self.assertEqual(checkpoint_info["stage"], 1)
        self.assertTrue(checkpoint_info["is_best"])
        self.assertEqual(checkpoint_info["metrics"]["normalized_completion_score"], 0.85)
    
    def test_load_checkpoint(self):
        """测试检查点加载"""
        # 先保存一个检查点
        model_state = {"layer1.weight": torch.randn(10, 5)}
        optimizer_state = {"lr": 0.001}
        metrics = {"normalized_completion_score": 0.85}
        
        checkpoint_path = self.manager.save_checkpoint(
            model_state, optimizer_state, metrics, 1000, 1
        )
        
        # 加载检查点
        loaded_data = self.manager.load_checkpoint(checkpoint_path)
        
        # 验证加载的数据
        self.assertIn("model_state_dict", loaded_data)
        self.assertIn("optimizer_state_dict", loaded_data)
        self.assertIn("metrics", loaded_data)
        self.assertEqual(loaded_data["step"], 1000)
        self.assertEqual(loaded_data["stage"], 1)
    
    def test_get_best_checkpoint(self):
        """测试获取最佳检查点"""
        # 保存多个检查点
        for i, score in enumerate([0.7, 0.85, 0.8]):
            model_state = {"layer1.weight": torch.randn(10, 5)}
            optimizer_state = {"lr": 0.001}
            metrics = {"normalized_completion_score": score}
            
            self.manager.save_checkpoint(
                model_state, optimizer_state, metrics, 
                i*1000, 1, is_best=(score == 0.85)
            )
        
        # 获取最佳检查点
        best_path = self.manager.get_best_checkpoint()
        self.assertIsNotNone(best_path)
        
        # 验证是最佳的
        loaded_data = self.manager.load_checkpoint(best_path)
        self.assertEqual(loaded_data["metrics"]["normalized_completion_score"], 0.85)
    
    def test_cleanup_old_checkpoints(self):
        """测试旧检查点清理"""
        # 保存超过最大数量的检查点
        for i in range(5):
            model_state = {"layer1.weight": torch.randn(10, 5)}
            optimizer_state = {"lr": 0.001}
            metrics = {"normalized_completion_score": 0.7 + i * 0.05}
            
            self.manager.save_checkpoint(
                model_state, optimizer_state, metrics, 
                i*1000, 1, is_best=(i == 4)
            )
        
        # 检查是否正确清理（保留最佳 + 最近的3个）
        non_best_count = len([cp for cp in self.manager.checkpoint_history if not cp["is_best"]])
        self.assertLessEqual(non_best_count, self.manager.max_checkpoints)


class TestCurriculumTrainingCallbacks(unittest.TestCase):
    """测试课程学习训练回调"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.callbacks = CurriculumTrainingCallbacks()
    
    def tearDown(self):
        """测试后清理"""
        if self.callbacks.logger:
            self.callbacks.logger.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('training_logger.CurriculumTensorBoardLogger')
    @patch('training_logger.ModelCheckpointManager')
    def test_on_algorithm_init(self, mock_checkpoint_manager, mock_logger):
        """测试算法初始化回调"""
        # 创建模拟算法
        mock_algorithm = Mock()
        mock_algorithm.config = {
            "log_dir": self.test_dir,
            "experiment_name": "test_experiment"
        }
        
        # 调用初始化回调
        self.callbacks.on_algorithm_init(algorithm=mock_algorithm)
        
        # 验证日志记录器和检查点管理器被创建
        mock_logger.assert_called_once_with(self.test_dir, "test_experiment")
        mock_checkpoint_manager.assert_called_once()
    
    def test_calculate_scale_invariant_metrics(self):
        """测试尺度不变指标计算"""
        # 创建模拟环境和回合
        mock_env = Mock()
        mock_env.n_active_uavs = 5
        mock_env.n_targets = 3
        mock_env.completed_targets = 2
        mock_env.total_flight_distance = 100.0
        mock_env.average_congestion = 0.1
        
        mock_episode = Mock()
        mock_episode.total_reward = 50.0
        
        # 计算指标
        metrics = self.callbacks._calculate_scale_invariant_metrics(mock_episode, mock_env)
        
        # 验证计算结果
        expected_per_agent_reward = 50.0 / 5  # 10.0
        expected_completion_rate = 2 / 3  # 0.667
        expected_normalized_score = expected_completion_rate * (1 - 0.1)  # 0.6
        expected_efficiency = 2 / 100.0  # 0.02
        
        self.assertAlmostEqual(metrics["per_agent_reward"], expected_per_agent_reward)
        self.assertAlmostEqual(metrics["normalized_completion_score"], expected_normalized_score, places=3)
        self.assertAlmostEqual(metrics["efficiency_metric"], expected_efficiency)
    
    def test_is_best_model(self):
        """测试最佳模型判断"""
        # 第一次调用应该返回True（没有历史记录）
        current_metrics = {"normalized_completion_score": 0.8}
        self.assertTrue(self.callbacks._is_best_model(current_metrics, 0))
        
        # 更好的性能应该返回True
        better_metrics = {"normalized_completion_score": 0.9}
        self.assertTrue(self.callbacks._is_best_model(better_metrics, 0))
        
        # 更差的性能应该返回False
        worse_metrics = {"normalized_completion_score": 0.7}
        self.assertFalse(self.callbacks._is_best_model(worse_metrics, 0))


class TestTrainingConfigCreation(unittest.TestCase):
    """测试训练配置创建"""
    
    def test_create_training_config_with_logging(self):
        """测试创建包含日志记录的训练配置"""
        base_config = {
            "env": "test_env",
            "num_workers": 4,
            "lr": 0.001
        }
        
        enhanced_config = create_training_config_with_logging(
            base_config, 
            log_dir="./test_logs",
            experiment_name="test_experiment"
        )
        
        # 验证基础配置保留
        self.assertEqual(enhanced_config["env"], "test_env")
        self.assertEqual(enhanced_config["num_workers"], 4)
        self.assertEqual(enhanced_config["lr"], 0.001)
        
        # 验证日志配置添加
        self.assertEqual(enhanced_config["log_dir"], "./test_logs")
        self.assertEqual(enhanced_config["experiment_name"], "test_experiment")
        self.assertEqual(enhanced_config["callbacks"], CurriculumTrainingCallbacks)
        
        # 验证TensorBoard配置
        self.assertIn("logger_config", enhanced_config)
        self.assertIn("checkpoint_freq", enhanced_config)


class TestIntegrationScenarios(unittest.TestCase):
    """集成测试场景"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_training_logging_workflow(self):
        """测试完整的训练日志记录工作流"""
        # 1. 初始化日志记录器
        logger = CurriculumTensorBoardLogger(self.test_dir, "integration_test")
        
        # 2. 初始化检查点管理器
        checkpoint_manager = ModelCheckpointManager(
            os.path.join(self.test_dir, "checkpoints")
        )
        
        # 3. 模拟训练过程
        for step in range(0, 1000, 100):
            stage = step // 300
            
            # 记录指标
            metrics = {
                "per_agent_reward": 10 + np.random.normal(0, 1),
                "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
                "efficiency_metric": 0.3 + np.random.uniform(-0.05, 0.1)
            }
            
            logger.log_scale_invariant_metrics(metrics, step, stage, 5, 3)
            
            # 模拟阶段切换
            if step in [300, 600, 900]:
                logger.log_stage_transition(stage-1, stage, step)
            
            # 保存检查点
            if step % 200 == 0:
                model_state = {"layer1.weight": torch.randn(10, 5)}
                optimizer_state = {"lr": 0.001}
                
                is_best = metrics["normalized_completion_score"] > 0.8
                checkpoint_manager.save_checkpoint(
                    model_state, optimizer_state, metrics, 
                    step, stage, is_best
                )
        
        # 4. 验证结果
        # 检查日志记录
        self.assertGreater(len(logger.training_history["metrics"]), 0)
        self.assertGreater(len(logger.training_history["stage_transitions"]), 0)
        
        # 检查检查点
        self.assertGreater(len(checkpoint_manager.checkpoint_history), 0)
        
        # 检查最佳检查点
        best_checkpoint = checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.assertTrue(os.path.exists(best_checkpoint))
        
        # 5. 清理
        logger.close()
        checkpoint_manager.save_checkpoint_history()
        
        print("✅ 完整训练日志记录工作流测试通过")


def run_training_logger_tests():
    """运行所有训练日志记录器测试"""
    print("🧪 开始训练日志记录器测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCurriculumTensorBoardLogger,
        TestModelCheckpointManager,
        TestCurriculumTrainingCallbacks,
        TestTrainingConfigCreation,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print(f"\n📊 测试结果摘要:")
    print(f"   总测试数: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ 失败的测试:")
        for failure in result.failures:
            print(f"   - {failure[0]}: {failure[1].split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 错误的测试:")
        for error in result.errors:
            print(f"   - {error[0]}: {error[1].split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_training_logger_tests()
    sys.exit(0 if success else 1)
