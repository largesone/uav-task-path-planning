"""
课程阶段与回退门限机制测试
验证课程学习阶段配置和回退机制的正确性
"""

import unittest
import tempfile
import shutil
import os
import torch
import numpy as np
from unittest.mock import Mock, patch

# 导入要测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_stages import CurriculumStages, StageConfig
from rollback_threshold_manager import RollbackThresholdManager, PerformanceMetrics
from model_state_manager import ModelStateManager

class TestCurriculumStages(unittest.TestCase):
    """测试课程阶段配置"""
    
    def setUp(self):
        self.curriculum = CurriculumStages()
    
    def test_stage_initialization(self):
        """测试阶段初始化"""
        self.assertEqual(len(self.curriculum.stages), 4)
        self.assertEqual(self.curriculum.current_stage_id, 0)
        
        # 验证第一阶段配置
        stage0 = self.curriculum.get_current_stage()
        self.assertEqual(stage0.stage_id, 0)
        self.assertEqual(stage0.n_uavs_range, (2, 3))
        self.assertEqual(stage0.n_targets_range, (1, 2))
    
    def test_stage_progression(self):
        """测试阶段推进"""
        # 推进到下一阶段
        success = self.curriculum.advance_to_next_stage()
        self.assertTrue(success)
        self.assertEqual(self.curriculum.current_stage_id, 1)
        
        # 验证第二阶段配置
        stage1 = self.curriculum.get_current_stage()
        self.assertEqual(stage1.n_uavs_range, (4, 6))
        self.assertEqual(stage1.n_targets_range, (3, 4))
    
    def test_stage_fallback(self):
        """测试阶段回退"""
        # 先推进到第二阶段
        self.curriculum.advance_to_next_stage()
        self.assertEqual(self.curriculum.current_stage_id, 1)
        
        # 回退到上一阶段
        success = self.curriculum.fallback_to_previous_stage()
        self.assertTrue(success)
        self.assertEqual(self.curriculum.current_stage_id, 0)
        
        # 第一阶段无法继续回退
        success = self.curriculum.fallback_to_previous_stage()
        self.assertFalse(success)
    
    def test_random_scenario_generation(self):
        """测试随机场景生成"""
        stage = self.curriculum.get_current_stage()
        
        # 生成多个随机场景，验证范围
        for _ in range(100):
            n_uavs, n_targets = stage.get_random_scenario_size()
            self.assertGreaterEqual(n_uavs, stage.n_uavs_range[0])
            self.assertLessEqual(n_uavs, stage.n_uavs_range[1])
            self.assertGreaterEqual(n_targets, stage.n_targets_range[0])
            self.assertLessEqual(n_targets, stage.n_targets_range[1])

class TestRollbackThresholdManager(unittest.TestCase):
    """测试回退门限机制"""
    
    def setUp(self):
        self.manager = RollbackThresholdManager(
            consecutive_evaluations_threshold=3,
            performance_drop_threshold=0.60,
            min_evaluations_before_rollback=5
        )
    
    def test_performance_update(self):
        """测试性能更新"""
        metrics = PerformanceMetrics()
        metrics.normalized_completion_score = 0.75
        metrics.per_agent_reward = 10.5
        metrics.efficiency_metric = 0.85
        
        self.manager.update_performance(0, metrics)
        
        # 验证性能历史记录
        self.assertIn(0, self.manager.stage_performance_history)
        self.assertEqual(len(self.manager.stage_performance_history[0]), 1)
        self.assertEqual(self.manager.evaluation_count, 1)
    
    def test_rollback_decision_insufficient_evaluations(self):
        """测试评估次数不足时的回退决策"""
        # 添加少量评估数据
        for i in range(3):
            metrics = PerformanceMetrics()
            metrics.normalized_completion_score = 0.4  # 低性能
            self.manager.update_performance(1, metrics)
        
        should_rollback, reason = self.manager.should_rollback(1)
        self.assertFalse(should_rollback)
        self.assertIn("评估次数不足", reason)
    
    def test_rollback_decision_performance_drop(self):
        """测试性能下降时的回退决策"""
        # 设置上一阶段的最终性能
        final_metrics = PerformanceMetrics()
        final_metrics.normalized_completion_score = 0.80
        self.manager.stage_final_performance[0] = final_metrics
        
        # 添加足够的评估数据，但性能较低
        for i in range(6):
            metrics = PerformanceMetrics()
            metrics.normalized_completion_score = 0.40  # 低于60%门限 (0.40/0.80 = 0.5 < 0.6)
            self.manager.update_performance(1, metrics)
        
        should_rollback, reason = self.manager.should_rollback(1)
        self.assertTrue(should_rollback)
        self.assertIn("连续", reason)
        self.assertIn("性能低于", reason)
    
    def test_rollback_record(self):
        """测试回退记录"""
        record = self.manager.record_rollback(
            from_stage=1,
            to_stage=0,
            reason="性能下降",
            learning_rate_adjustment=0.5
        )
        
        self.assertEqual(record["from_stage"], 1)
        self.assertEqual(record["to_stage"], 0)
        self.assertEqual(record["learning_rate_adjustment"], 0.5)
        self.assertEqual(len(self.manager.rollback_history), 1)
    
    def test_learning_rate_adjustment_calculation(self):
        """测试学习率调整计算"""
        # 第一次回退
        adjustment1 = self.manager.get_learning_rate_adjustment(1)
        self.assertEqual(adjustment1, 0.5)
        
        # 第二次回退
        adjustment2 = self.manager.get_learning_rate_adjustment(2)
        self.assertEqual(adjustment2, 0.25)
        
        # 多次回退后不低于0.1
        adjustment_many = self.manager.get_learning_rate_adjustment(10)
        self.assertEqual(adjustment_many, 0.1)

class TestModelStateManager(unittest.TestCase):
    """测试模型状态管理器"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelStateManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints_per_stage=3
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_save_and_load(self):
        """测试检查点保存和加载"""
        # 创建模拟的模型状态
        model_state = {"layer1.weight": torch.randn(10, 5)}
        optimizer_state = {"param_groups": [{"lr": 0.001}]}
        performance_metrics = {"normalized_completion_score": 0.75}
        
        # 保存检查点
        checkpoint_path = self.manager.save_checkpoint(
            stage_id=0,
            model_state=model_state,
            optimizer_state=optimizer_state,
            performance_metrics=performance_metrics,
            episode_count=1000,
            is_best=True
        )
        
        # 验证文件存在
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # 加载检查点
        loaded_data = self.manager.load_checkpoint(checkpoint_path)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["stage_id"], 0)
        self.assertEqual(loaded_data["episode_count"], 1000)
        self.assertTrue(loaded_data["is_best"])
    
    def test_best_checkpoint_tracking(self):
        """测试最佳检查点跟踪"""
        # 保存普通检查点
        self.manager.save_checkpoint(
            stage_id=0,
            model_state={},
            optimizer_state={},
            performance_metrics={"normalized_completion_score": 0.60},
            episode_count=500,
            is_best=False
        )
        
        # 保存最佳检查点
        best_path = self.manager.save_checkpoint(
            stage_id=0,
            model_state={},
            optimizer_state={},
            performance_metrics={"normalized_completion_score": 0.80},
            episode_count=1000,
            is_best=True
        )
        
        # 验证最佳检查点记录
        retrieved_best = self.manager.get_best_checkpoint_path(0)
        self.assertEqual(retrieved_best, best_path)
    
    def test_rollback_functionality(self):
        """测试回退功能"""
        # 保存阶段0的最佳检查点
        optimizer_state = {"param_groups": [{"lr": 0.001}]}
        best_path = self.manager.save_checkpoint(
            stage_id=0,
            model_state={"test": "data"},
            optimizer_state=optimizer_state,
            performance_metrics={"normalized_completion_score": 0.80},
            episode_count=1000,
            is_best=True
        )
        
        # 执行回退
        rollback_data = self.manager.rollback_to_previous_stage(
            current_stage_id=1,
            target_stage_id=0,
            learning_rate_adjustment=0.5
        )
        
        # 验证回退结果
        self.assertIsNotNone(rollback_data)
        self.assertEqual(rollback_data["stage_id"], 0)
        
        # 验证学习率调整
        adjusted_lr = rollback_data["optimizer_state_dict"]["param_groups"][0]["lr"]
        self.assertEqual(adjusted_lr, 0.0005)  # 0.001 * 0.5
    
    def test_checkpoint_cleanup(self):
        """测试检查点清理"""
        # 保存超过最大数量的检查点
        for i in range(5):
            self.manager.save_checkpoint(
                stage_id=0,
                model_state={},
                optimizer_state={},
                performance_metrics={"normalized_completion_score": 0.60 + i * 0.05},
                episode_count=100 * (i + 1),
                is_best=(i == 4)  # 最后一个是最佳
            )
        
        # 验证只保留了最大数量的检查点
        checkpoints = self.manager.stage_checkpoints[0]
        self.assertLessEqual(len(checkpoints), self.manager.max_checkpoints + 1)  # +1 for best

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.curriculum = CurriculumStages()
        self.rollback_manager = RollbackThresholdManager()
        self.model_manager = ModelStateManager(checkpoint_dir=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_complete_curriculum_workflow(self):
        """测试完整的课程学习工作流"""
        # 模拟第一阶段训练完成
        stage0 = self.curriculum.get_current_stage()
        
        # 记录第一阶段性能
        for i in range(10):
            metrics = PerformanceMetrics()
            metrics.normalized_completion_score = 0.70 + i * 0.01
            self.rollback_manager.update_performance(0, metrics)
        
        # 保存第一阶段最佳模型
        self.model_manager.save_checkpoint(
            stage_id=0,
            model_state={"stage0": "completed"},
            optimizer_state={"param_groups": [{"lr": 0.001}]},
            performance_metrics={"normalized_completion_score": 0.79},
            episode_count=5000,
            is_best=True
        )
        
        # 确定第一阶段最终性能
        self.rollback_manager.finalize_stage_performance(0)
        
        # 推进到第二阶段
        self.curriculum.advance_to_next_stage()
        stage1 = self.curriculum.get_current_stage()
        
        # 模拟第二阶段性能不佳
        for i in range(6):
            metrics = PerformanceMetrics()
            metrics.normalized_completion_score = 0.40  # 低于60%门限
            self.rollback_manager.update_performance(1, metrics)
        
        # 检查是否需要回退
        should_rollback, reason = self.rollback_manager.should_rollback(1)
        self.assertTrue(should_rollback)
        
        # 执行回退
        rollback_data = self.model_manager.rollback_to_previous_stage(1, 0, 0.5)
        self.assertIsNotNone(rollback_data)
        
        # 记录回退事件
        self.rollback_manager.record_rollback(1, 0, reason, 0.5)
        
        # 回退课程阶段
        self.curriculum.fallback_to_previous_stage()
        self.assertEqual(self.curriculum.current_stage_id, 0)

if __name__ == "__main__":
    # 配置日志
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2)
