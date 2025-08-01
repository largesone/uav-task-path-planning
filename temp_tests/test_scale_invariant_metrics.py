# -*- coding: utf-8 -*-
"""
测试文件: test_scale_invariant_metrics.py
验证尺度不变指标计算和TensorBoard集成的正确性
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllib_custom_callbacks import ScaleInvariantMetricsCallbacks, CurriculumLearningCallbacks
from scale_invariant_tensorboard_logger import ScaleInvariantTensorBoardLogger


class MockUAV:
    """模拟UAV对象"""
    def __init__(self, uav_id, resources, position):
        self.id = uav_id
        self.resources = np.array(resources)
        self.position = np.array(position)
        self.current_position = np.array(position)
        self.task_sequence = []


class MockTarget:
    """模拟目标对象"""
    def __init__(self, target_id, resources, position):
        self.id = target_id
        self.resources = np.array(resources)
        self.remaining_resources = np.array(resources)
        self.position = np.array(position)
        self.allocated_uavs = []


class MockEnvironment:
    """模拟环境对象"""
    def __init__(self, uavs, targets):
        self.uavs = uavs
        self.targets = targets


class MockEpisode:
    """模拟Episode对象"""
    def __init__(self):
        self.custom_metrics = {}
        self.total_reward = 0.0
        self.length = 0
        self.episode_id = "test_episode"


class TestScaleInvariantMetrics(unittest.TestCase):
    """测试尺度不变指标计算"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟环境
        self.uavs = [
            MockUAV(0, [100, 50], [0, 0]),
            MockUAV(1, [80, 60], [100, 0]),
            MockUAV(2, [0, 0], [200, 0])  # 无资源UAV
        ]
        
        self.targets = [
            MockTarget(0, [50, 30], [50, 50]),
            MockTarget(1, [40, 20], [150, 50])
        ]
        
        # 设置目标完成状态
        self.targets[0].remaining_resources = np.array([0, 0])  # 已完成
        self.targets[1].remaining_resources = np.array([20, 10])  # 未完成
        
        # 设置UAV分配
        self.targets[0].allocated_uavs = [(0, 0), (1, 0)]  # 2个UAV分配到目标0
        self.targets[1].allocated_uavs = [(1, 0)]  # 1个UAV分配到目标1
        
        self.env = MockEnvironment(self.uavs, self.targets)
        self.callbacks = ScaleInvariantMetricsCallbacks()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_per_agent_reward_calculation(self):
        """测试Per-Agent Reward计算"""
        episode = MockEpisode()
        episode.total_reward = 100.0
        
        # 初始化指标
        self.callbacks.on_episode_start(
            worker=None, base_env=None, policies=None, episode=episode
        )
        
        # 模拟环境
        class MockBaseEnv:
            def get_sub_environments(self):
                return [self.env]
        
        # 执行回合结束回调
        self.callbacks.on_episode_end(
            worker=None, base_env=MockBaseEnv(), policies=None, episode=episode
        )
        
        # 验证Per-Agent Reward计算
        # 活跃UAV数量应该是2（UAV 0和1有资源，UAV 2无资源）
        expected_per_agent_reward = 100.0 / 2  # total_reward / n_active_uavs
        actual_per_agent_reward = episode.custom_metrics["per_agent_reward"]
        
        self.assertAlmostEqual(actual_per_agent_reward, expected_per_agent_reward, places=4)
        print(f"✓ Per-Agent Reward计算正确: {actual_per_agent_reward:.4f}")
    
    def test_normalized_completion_score_calculation(self):
        """测试Normalized Completion Score计算"""
        episode = MockEpisode()
        
        # 初始化指标
        self.callbacks.on_episode_start(
            worker=None, base_env=None, policies=None, episode=episode
        )
        
        class MockBaseEnv:
            def get_sub_environments(self):
                return [self.env]
        
        # 执行回合结束回调
        self.callbacks.on_episode_end(
            worker=None, base_env=MockBaseEnv(), policies=None, episode=episode
        )
        
        # 验证Normalized Completion Score计算
        # 目标满足率 = 1/2 = 0.5 (1个目标完成，共2个目标)
        expected_satisfied_rate = 0.5
        
        # 拥堵指标计算较复杂，这里主要验证公式结构
        actual_ncs = episode.custom_metrics["normalized_completion_score"]
        actual_satisfied_rate = episode.custom_metrics["scale_invariant_metrics"]["satisfied_targets_rate"]
        
        self.assertAlmostEqual(actual_satisfied_rate, expected_satisfied_rate, places=4)
        self.assertGreaterEqual(actual_ncs, 0.0)
        self.assertLessEqual(actual_ncs, 1.0)
        
        print(f"✓ Normalized Completion Score计算正确: {actual_ncs:.4f}")
        print(f"  - 目标满足率: {actual_satisfied_rate:.4f}")
    
    def test_efficiency_metric_calculation(self):
        """测试Efficiency Metric计算"""
        episode = MockEpisode()
        
        # 设置UAV飞行距离
        self.uavs[0].current_position = np.array([50, 50])  # 飞行了约70.7单位
        self.uavs[1].current_position = np.array([150, 50])  # 飞行了约50单位
        
        # 初始化指标
        self.callbacks.on_episode_start(
            worker=None, base_env=None, policies=None, episode=episode
        )
        
        class MockBaseEnv:
            def get_sub_environments(self):
                return [self.env]
        
        # 执行回合结束回调
        self.callbacks.on_episode_end(
            worker=None, base_env=MockBaseEnv(), policies=None, episode=episode
        )
        
        # 验证Efficiency Metric计算
        completed_targets = 1  # 1个目标完成
        actual_efficiency = episode.custom_metrics["efficiency_metric"]
        
        self.assertGreater(actual_efficiency, 0.0)
        print(f"✓ Efficiency Metric计算正确: {actual_efficiency:.6f}")
    
    def test_congestion_metric_calculation(self):
        """测试拥堵指标计算"""
        # 创建高拥堵场景
        high_congestion_target = MockTarget(2, [100, 100], [300, 50])
        high_congestion_target.allocated_uavs = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]  # 5个分配
        
        env_with_congestion = MockEnvironment(self.uavs, self.targets + [high_congestion_target])
        
        congestion_metric = self.callbacks._calculate_congestion_metric(env_with_congestion)
        
        self.assertGreaterEqual(congestion_metric, 0.0)
        self.assertLessEqual(congestion_metric, 1.0)
        
        print(f"✓ 拥堵指标计算正确: {congestion_metric:.4f}")
    
    def test_metrics_trend_calculation(self):
        """测试指标趋势计算"""
        # 创建模拟回合历史
        episodes = [
            {"normalized_completion_score": 0.5},
            {"normalized_completion_score": 0.6},
            {"normalized_completion_score": 0.7},
            {"normalized_completion_score": 0.8},
            {"normalized_completion_score": 0.9}
        ]
        
        trend = self.callbacks._calculate_metrics_trend(episodes)
        
        # 应该是正趋势
        self.assertGreater(trend, 0.0)
        print(f"✓ 指标趋势计算正确: {trend:.4f} (正趋势)")
        
        # 测试负趋势
        declining_episodes = [
            {"normalized_completion_score": 0.9},
            {"normalized_completion_score": 0.7},
            {"normalized_completion_score": 0.5},
            {"normalized_completion_score": 0.3},
            {"normalized_completion_score": 0.1}
        ]
        
        declining_trend = self.callbacks._calculate_metrics_trend(declining_episodes)
        self.assertLess(declining_trend, 0.0)
        print(f"✓ 负趋势计算正确: {declining_trend:.4f} (负趋势)")


class TestCurriculumLearningCallbacks(unittest.TestCase):
    """测试课程学习回调函数"""
    
    def setUp(self):
        """测试前准备"""
        self.callbacks = CurriculumLearningCallbacks()
    
    def test_stage_advancement(self):
        """测试阶段推进"""
        initial_stage = self.callbacks.current_stage
        
        self.callbacks.advance_to_next_stage()
        
        self.assertEqual(self.callbacks.current_stage, initial_stage + 1)
        self.assertEqual(self.callbacks.stage_episode_count, 0)
        
        print(f"✓ 阶段推进测试通过: {initial_stage} → {self.callbacks.current_stage}")
    
    def test_stage_rollback(self):
        """测试阶段回退"""
        # 先推进到阶段1
        self.callbacks.advance_to_next_stage()
        current_stage = self.callbacks.current_stage
        
        self.callbacks.rollback_to_previous_stage()
        
        self.assertEqual(self.callbacks.current_stage, current_stage - 1)
        self.assertEqual(self.callbacks.stage_episode_count, 0)
        
        print(f"✓ 阶段回退测试通过: {current_stage} → {self.callbacks.current_stage}")
    
    def test_stage_completion_detection(self):
        """测试阶段完成检测"""
        stage_id = 0
        
        # 添加足够的高性能回合
        for i in range(15):
            self.callbacks.stage_performance_history[stage_id] = self.callbacks.stage_performance_history.get(stage_id, [])
            self.callbacks.stage_performance_history[stage_id].append({
                "normalized_completion_score": 0.85,  # 高于0.8阈值
                "episode_count": i + 1
            })
        
        is_completed = self.callbacks._is_stage_completed(stage_id)
        self.assertTrue(is_completed)
        
        print(f"✓ 阶段完成检测测试通过: 阶段{stage_id}已完成")


class TestScaleInvariantTensorBoardLogger(unittest.TestCase):
    """测试TensorBoard记录器"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ScaleInvariantTensorBoardLogger(self.temp_dir, "test_experiment")
    
    def tearDown(self):
        """测试后清理"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_metrics_logging(self):
        """测试指标记录"""
        metrics = {
            "per_agent_reward": 15.5,
            "normalized_completion_score": 0.75,
            "efficiency_metric": 0.002
        }
        
        scenario_info = {
            "n_uavs": 3,
            "n_targets": 2
        }
        
        stage_info = {
            "current_stage": 1,
            "stage_progress": 0.6
        }
        
        # 记录指标
        self.logger.log_scale_invariant_metrics(metrics, 100, scenario_info, stage_info)
        
        # 验证历史记录
        self.assertEqual(len(self.logger.metrics_history["per_agent_reward"]), 1)
        self.assertEqual(self.logger.metrics_history["per_agent_reward"][0], 15.5)
        
        print("✓ TensorBoard指标记录测试通过")
    
    def test_cross_scale_comparison(self):
        """测试跨尺度对比"""
        scale_data = {
            6: {"per_agent_reward": 10.0, "normalized_completion_score": 0.6},
            12: {"per_agent_reward": 8.0, "normalized_completion_score": 0.7},
            20: {"per_agent_reward": 6.0, "normalized_completion_score": 0.8}
        }
        
        self.logger.log_cross_scale_comparison(scale_data, 200)
        
        print("✓ 跨尺度对比记录测试通过")
    
    def test_summary_report_generation(self):
        """测试摘要报告生成"""
        # 添加一些测试数据
        for i in range(50):
            metrics = {
                "per_agent_reward": 10 + np.random.normal(0, 2),
                "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
                "efficiency_metric": 0.001 + np.random.uniform(0, 0.0005)
            }
            self.logger.log_scale_invariant_metrics(metrics, i)
        
        # 生成报告
        report_path = self.logger.create_training_summary_report()
        
        # 验证报告文件存在
        self.assertTrue(Path(report_path).exists())
        
        # 验证报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("尺度不变指标训练摘要报告", content)
            self.assertIn("Per-Agent Reward", content)
            self.assertIn("Normalized Completion Score", content)
            self.assertIn("Efficiency Metric", content)
        
        print(f"✓ 摘要报告生成测试通过: {report_path}")


def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("开始尺度不变指标系统综合测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestScaleInvariantMetrics))
    test_suite.addTest(unittest.makeSuite(TestCurriculumLearningCallbacks))
    test_suite.addTest(unittest.makeSuite(TestScaleInvariantTensorBoardLogger))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("🎉 所有测试通过！尺度不变指标系统实现正确。")
    else:
        print("❌ 部分测试失败，请检查实现。")
    
    exit(0 if success else 1)
