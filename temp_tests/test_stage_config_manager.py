"""
阶段配置管理器测试
测试课程学习阶段配置管理和最佳模型保存功能
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_config_manager import StageConfig, StageConfigManager


class TestStageConfig(unittest.TestCase):
    """测试阶段配置数据类"""
    
    def test_stage_config_creation(self):
        """测试阶段配置创建"""
        config = StageConfig(
            stage_id=1,
            n_uavs_range=(4, 6),
            n_targets_range=(3, 4),
            max_episodes=1500,
            success_threshold=0.75,
            fallback_threshold=0.55,
            learning_rate=0.0008,
            batch_size=128,
            exploration_noise=0.08,
            k_neighbors=6,
            description="测试阶段"
        )
        
        self.assertEqual(config.stage_id, 1)
        self.assertEqual(config.n_uavs_range, (4, 6))
        self.assertEqual(config.learning_rate, 0.0008)
        self.assertEqual(config.description, "测试阶段")
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        config = StageConfig(
            stage_id=0,
            n_uavs_range=(2, 3),
            n_targets_range=(1, 2),
            max_episodes=1000,
            success_threshold=0.8,
            fallback_threshold=0.6,
            learning_rate=0.001,
            batch_size=64,
            exploration_noise=0.1,
            k_neighbors=4
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["stage_id"], 0)
        self.assertEqual(config_dict["learning_rate"], 0.001)
        self.assertIn("n_uavs_range", config_dict)
    
    def test_from_dict_creation(self):
        """测试从字典创建配置"""
        config_data = {
            "stage_id": 2,
            "n_uavs_range": [8, 12],
            "n_targets_range": [5, 8],
            "max_episodes": 2000,
            "success_threshold": 0.7,
            "fallback_threshold": 0.5,
            "learning_rate": 0.0005,
            "batch_size": 256,
            "exploration_noise": 0.06,
            "k_neighbors": 8,
            "description": "高复杂度阶段"
        }
        
        config = StageConfig.from_dict(config_data)
        
        self.assertEqual(config.stage_id, 2)
        self.assertEqual(config.n_uavs_range, [8, 12])
        self.assertEqual(config.description, "高复杂度阶段")


class TestStageConfigManager(unittest.TestCase):
    """测试阶段配置管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.manager = StageConfigManager(self.test_dir)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertGreater(len(self.manager.stage_configs), 0)
        
        # 验证默认配置
        config_0 = self.manager.get_stage_config(0)
        self.assertIsNotNone(config_0)
        self.assertEqual(config_0.stage_id, 0)
        self.assertEqual(config_0.n_uavs_range, (2, 3))
    
    def test_get_stage_config(self):
        """测试获取阶段配置"""
        # 获取存在的配置
        config = self.manager.get_stage_config(1)
        self.assertIsNotNone(config)
        self.assertEqual(config.stage_id, 1)
        
        # 获取不存在的配置
        config = self.manager.get_stage_config(999)
        self.assertIsNone(config)
    
    def test_update_stage_config(self):
        """测试更新阶段配置"""
        # 更新存在的配置
        self.manager.update_stage_config(0, learning_rate=0.002, batch_size=128)
        
        config = self.manager.get_stage_config(0)
        self.assertEqual(config.learning_rate, 0.002)
        self.assertEqual(config.batch_size, 128)
        
        # 尝试更新不存在的配置（应该不会崩溃）
        self.manager.update_stage_config(999, learning_rate=0.001)
    
    def test_save_and_load_stage_config(self):
        """测试保存和加载阶段配置"""
        # 修改配置
        original_lr = self.manager.get_stage_config(0).learning_rate
        self.manager.update_stage_config(0, learning_rate=0.005)
        
        # 保存配置
        self.manager.save_stage_config(0)
        
        # 验证文件存在
        config_file = Path(self.test_dir) / "stage_0_config.json"
        self.assertTrue(config_file.exists())
        
        # 重置配置并重新加载
        self.manager.stage_configs[0].learning_rate = original_lr
        success = self.manager.load_stage_config(0)
        
        self.assertTrue(success)
        self.assertEqual(self.manager.get_stage_config(0).learning_rate, 0.005)
    
    def test_save_and_load_best_model(self):
        """测试保存和加载最佳模型"""
        # 创建模拟模型状态
        model_state = {"layer1.weight": torch.randn(10, 5)}
        performance_metrics = {"normalized_completion_score": 0.85}
        training_config = {"learning_rate": 0.001}
        
        # 保存最佳模型
        self.manager.save_best_model(1, model_state, performance_metrics, training_config)
        
        # 验证内存中的记录
        self.assertIn(1, self.manager.best_models)
        best_model_info = self.manager.best_models[1]
        self.assertEqual(best_model_info["stage_id"], 1)
        self.assertEqual(best_model_info["performance_metrics"]["normalized_completion_score"], 0.85)
        
        # 验证文件保存
        model_file = Path(self.test_dir) / "best_model_stage_1.pkl"
        self.assertTrue(model_file.exists())
        
        # 清除内存记录并重新加载
        del self.manager.best_models[1]
        loaded_model_info = self.manager.load_best_model(1)
        
        self.assertIsNotNone(loaded_model_info)
        self.assertEqual(loaded_model_info["stage_id"], 1)
        self.assertEqual(loaded_model_info["performance_metrics"]["normalized_completion_score"], 0.85)
    
    def test_record_stage_performance(self):
        """测试记录阶段性能"""
        performance_metrics = {
            "per_agent_reward": 12.5,
            "normalized_completion_score": 0.8,
            "efficiency_metric": 0.4
        }
        
        # 记录性能
        self.manager.record_stage_performance(0, performance_metrics, 100, 5000)
        
        # 验证记录
        self.assertIn(0, self.manager.stage_performance_history)
        history = self.manager.stage_performance_history[0]
        self.assertEqual(len(history), 1)
        
        record = history[0]
        self.assertEqual(record["episode"], 100)
        self.assertEqual(record["step"], 5000)
        self.assertEqual(record["per_agent_reward"], 12.5)
    
    def test_get_stage_performance_summary(self):
        """测试获取阶段性能摘要"""
        # 记录多个性能数据点
        for i in range(10):
            performance = {
                "per_agent_reward": 10 + i,
                "normalized_completion_score": 0.7 + i * 0.02,
                "efficiency_metric": 0.3 + i * 0.01
            }
            self.manager.record_stage_performance(0, performance, i, i*100)
        
        # 获取摘要
        summary = self.manager.get_stage_performance_summary(0)
        
        # 验证摘要内容
        self.assertEqual(summary["total_records"], 10)
        self.assertEqual(summary["first_episode"], 0)
        self.assertEqual(summary["last_episode"], 9)
        
        # 验证统计指标
        self.assertIn("per_agent_reward_mean", summary)
        self.assertIn("normalized_completion_score_max", summary)
        self.assertAlmostEqual(summary["per_agent_reward_mean"], 14.5)  # (10+19)/2
    
    def test_should_advance_stage(self):
        """测试阶段推进判断"""
        # 记录高性能数据（应该推进）
        for i in range(5):
            performance = {"normalized_completion_score": 0.85}  # 高于阈值0.8
            self.manager.record_stage_performance(0, performance, i, i*100)
        
        current_performance = {"normalized_completion_score": 0.85}
        should_advance = self.manager.should_advance_stage(0, current_performance, 3)
        self.assertTrue(should_advance)
        
        # 记录低性能数据（不应该推进）
        low_performance = {"normalized_completion_score": 0.7}  # 低于阈值
        should_advance = self.manager.should_advance_stage(0, low_performance, 3)
        self.assertFalse(should_advance)
    
    def test_should_fallback_stage(self):
        """测试阶段回退判断"""
        # 第0阶段不应该回退
        low_performance = {"normalized_completion_score": 0.5}
        should_fallback = self.manager.should_fallback_stage(0, low_performance, 3)
        self.assertFalse(should_fallback)
        
        # 记录低性能数据到阶段1（应该回退）
        for i in range(5):
            performance = {"normalized_completion_score": 0.5}  # 低于回退阈值0.55
            self.manager.record_stage_performance(1, performance, i, i*100)
        
        current_performance = {"normalized_completion_score": 0.5}
        should_fallback = self.manager.should_fallback_stage(1, current_performance, 3)
        self.assertTrue(should_fallback)
    
    def test_get_adaptive_learning_rate(self):
        """测试自适应学习率"""
        # 测试性能提升趋势
        improving_trend = [0.6, 0.65, 0.7, 0.75, 0.8]
        adaptive_lr = self.manager.get_adaptive_learning_rate(0, improving_trend)
        base_lr = self.manager.get_stage_config(0).learning_rate
        self.assertGreater(adaptive_lr, base_lr)  # 应该增加学习率
        
        # 测试性能下降趋势
        declining_trend = [0.8, 0.75, 0.7, 0.65, 0.6]
        adaptive_lr = self.manager.get_adaptive_learning_rate(0, declining_trend)
        self.assertLess(adaptive_lr, base_lr)  # 应该降低学习率
        
        # 测试稳定趋势
        stable_trend = [0.7, 0.7, 0.7, 0.7, 0.7]
        adaptive_lr = self.manager.get_adaptive_learning_rate(0, stable_trend)
        self.assertAlmostEqual(adaptive_lr, base_lr)  # 应该保持学习率
    
    def test_get_adaptive_k_neighbors(self):
        """测试自适应k近邻"""
        # 小规模场景
        k_small = self.manager.get_adaptive_k_neighbors(0, 3, 2)  # scale_factor = 6
        self.assertLessEqual(k_small, 2)  # k不应超过目标数量
        
        # 中等规模场景
        k_medium = self.manager.get_adaptive_k_neighbors(1, 5, 4)  # scale_factor = 20
        self.assertGreaterEqual(k_medium, 4)
        self.assertLessEqual(k_medium, 4)
        
        # 大规模场景
        k_large = self.manager.get_adaptive_k_neighbors(2, 10, 8)  # scale_factor = 80
        self.assertGreaterEqual(k_large, 6)
        self.assertLessEqual(k_large, 8)
    
    def test_export_and_import_configs(self):
        """测试配置导出和导入"""
        # 修改一些配置
        self.manager.update_stage_config(0, learning_rate=0.002)
        self.manager.update_stage_config(1, batch_size=256)
        
        # 导出配置
        export_file = self.manager.export_all_configs()
        self.assertTrue(os.path.exists(export_file))
        
        # 验证导出文件内容
        with open(export_file, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        self.assertIn("configs", exported_data)
        self.assertIn("export_timestamp", exported_data)
        self.assertEqual(exported_data["configs"]["0"]["learning_rate"], 0.002)
        
        # 重置配置并导入
        self.manager.stage_configs[0].learning_rate = 0.001
        success = self.manager.import_configs(export_file)
        
        self.assertTrue(success)
        self.assertEqual(self.manager.get_stage_config(0).learning_rate, 0.002)
    
    def test_get_stage_transition_recommendation(self):
        """测试阶段切换建议"""
        # 测试性能达标的情况
        high_performance = [
            {"normalized_completion_score": 0.85},
            {"normalized_completion_score": 0.87},
            {"normalized_completion_score": 0.89}
        ]
        
        # 先记录历史性能
        for i, perf in enumerate(high_performance):
            self.manager.record_stage_performance(0, perf, i, i*100)
        
        recommendation = self.manager.get_stage_transition_recommendation(0, high_performance)
        self.assertEqual(recommendation["action"], "advance")
        self.assertEqual(recommendation["target_stage"], 1)
        
        # 测试性能不足的情况
        low_performance = [{"normalized_completion_score": 0.5}] * 5
        for i, perf in enumerate(low_performance):
            self.manager.record_stage_performance(1, perf, i, i*100)
        
        recommendation = self.manager.get_stage_transition_recommendation(1, low_performance)
        self.assertEqual(recommendation["action"], "fallback")
        self.assertEqual(recommendation["target_stage"], 0)


def run_stage_config_tests():
    """运行所有阶段配置管理器测试"""
    print("⚙️ 开始阶段配置管理器测试...")
    
    # 创建测试套件
    test
