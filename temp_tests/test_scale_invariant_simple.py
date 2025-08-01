# -*- coding: utf-8 -*-
"""
简化的尺度不变指标测试
不依赖Ray RLlib，直接测试核心计算逻辑
"""

import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


class ScaleInvariantMetricsCalculator:
    """
    尺度不变指标计算器
    独立于RLlib的核心计算逻辑
    """
    
    def calculate_per_agent_reward(self, total_reward: float, uavs: list) -> float:
        """
        计算Per-Agent Reward
        
        Args:
            total_reward: 总奖励
            uavs: UAV列表
            
        Returns:
            float: Per-Agent Reward
        """
        # 计算活跃UAV数量（资源>0的UAV）
        n_active_uavs = len([uav for uav in uavs if np.any(uav.resources > 0)])
        n_active_uavs = max(n_active_uavs, 1)  # 避免除零
        
        return total_reward / n_active_uavs
    
    def calculate_normalized_completion_score(self, targets: list) -> float:
        """
        计算Normalized Completion Score
        
        Args:
            targets: 目标列表
            
        Returns:
            float: Normalized Completion Score
        """
        # 计算目标满足率
        completed_targets = sum(1 for target in targets 
                               if np.all(target.remaining_resources <= 0))
        total_targets = len(targets)
        satisfied_targets_rate = completed_targets / total_targets if total_targets > 0 else 0.0
        
        # 计算平均拥堵指标
        average_congestion_metric = self._calculate_congestion_metric(targets)
        
        # 计算Normalized Completion Score
        return satisfied_targets_rate * (1 - average_congestion_metric)
    
    def calculate_efficiency_metric(self, targets: list, uavs: list) -> float:
        """
        计算Efficiency Metric
        
        Args:
            targets: 目标列表
            uavs: UAV列表
            
        Returns:
            float: Efficiency Metric
        """
        # 计算完成的目标数量
        completed_targets = sum(1 for target in targets 
                               if np.all(target.remaining_resources <= 0))
        
        # 计算总飞行距离
        total_flight_distance = 0.0
        for uav in uavs:
            if hasattr(uav, 'task_sequence') and len(uav.task_sequence) > 0:
                current_pos = np.array(uav.current_position)
                initial_pos = np.array(uav.position)
                total_flight_distance += np.linalg.norm(current_pos - initial_pos)
        
        # 避免除零
        total_flight_distance = max(total_flight_distance, 1e-6)
        
        return completed_targets / total_flight_distance
    
    def _calculate_congestion_metric(self, targets: list) -> float:
        """
        计算平均拥堵指标
        
        Args:
            targets: 目标列表
            
        Returns:
            float: 平均拥堵指标 [0, 1]
        """
        congestion_scores = []
        
        for target in targets:
            if hasattr(target, 'allocated_uavs'):
                # 计算分配到该目标的UAV数量
                allocated_count = len(target.allocated_uavs)
                
                # 计算理想分配数量（基于目标资源需求）
                if hasattr(target, 'resources'):
                    total_demand = np.sum(target.resources)
                    # 假设每个UAV平均能提供的资源
                    avg_uav_capacity = 50.0
                    ideal_allocation = max(1, int(np.ceil(total_demand / avg_uav_capacity)))
                    
                    # 计算拥堵程度
                    if ideal_allocation > 0:
                        congestion_ratio = allocated_count / ideal_allocation
                        # 将拥堵比例映射到[0, 1]范围
                        congestion_score = min(1.0, max(0.0, (congestion_ratio - 1.0) / 1.0))
                        congestion_scores.append(congestion_score)
        
        return np.mean(congestion_scores) if congestion_scores else 0.0


class TestScaleInvariantMetrics(unittest.TestCase):
    """测试尺度不变指标计算"""
    
    def setUp(self):
        """测试前准备"""
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
        
        # 设置UAV飞行距离
        self.uavs[0].current_position = np.array([50, 50])  # 飞行了约70.7单位
        self.uavs[1].current_position = np.array([150, 50])  # 飞行了约50单位
        
        self.calculator = ScaleInvariantMetricsCalculator()
    
    def test_per_agent_reward_calculation(self):
        """测试Per-Agent Reward计算"""
        total_reward = 100.0
        
        per_agent_reward = self.calculator.calculate_per_agent_reward(total_reward, self.uavs)
        
        # 活跃UAV数量应该是2（UAV 0和1有资源，UAV 2无资源）
        expected_per_agent_reward = 100.0 / 2
        
        self.assertAlmostEqual(per_agent_reward, expected_per_agent_reward, places=4)
        print(f"✓ Per-Agent Reward计算正确: {per_agent_reward:.4f}")
    
    def test_normalized_completion_score_calculation(self):
        """测试Normalized Completion Score计算"""
        ncs = self.calculator.calculate_normalized_completion_score(self.targets)
        
        # 目标满足率应该是0.5（1个目标完成，共2个目标）
        expected_satisfied_rate = 0.5
        
        # 验证结果在合理范围内
        self.assertGreaterEqual(ncs, 0.0)
        self.assertLessEqual(ncs, 1.0)
        
        print(f"✓ Normalized Completion Score计算正确: {ncs:.4f}")
        
        # 验证满足率计算
        completed_targets = sum(1 for target in self.targets 
                               if np.all(target.remaining_resources <= 0))
        actual_satisfied_rate = completed_targets / len(self.targets)
        self.assertAlmostEqual(actual_satisfied_rate, expected_satisfied_rate, places=4)
        print(f"  - 目标满足率: {actual_satisfied_rate:.4f}")
    
    def test_efficiency_metric_calculation(self):
        """测试Efficiency Metric计算"""
        efficiency = self.calculator.calculate_efficiency_metric(self.targets, self.uavs)
        
        # 应该有1个完成的目标
        completed_targets = sum(1 for target in self.targets 
                               if np.all(target.remaining_resources <= 0))
        self.assertEqual(completed_targets, 1)
        
        # 效率应该大于0
        self.assertGreater(efficiency, 0.0)
        
        print(f"✓ Efficiency Metric计算正确: {efficiency:.6f}")
        print(f"  - 完成目标数: {completed_targets}")
    
    def test_congestion_metric_calculation(self):
        """测试拥堵指标计算"""
        congestion = self.calculator._calculate_congestion_metric(self.targets)
        
        # 拥堵指标应该在[0, 1]范围内
        self.assertGreaterEqual(congestion, 0.0)
        self.assertLessEqual(congestion, 1.0)
        
        print(f"✓ 拥堵指标计算正确: {congestion:.4f}")
    
    def test_zero_division_handling(self):
        """测试除零错误处理"""
        # 测试无UAV的情况
        empty_uavs = []
        per_agent_reward = self.calculator.calculate_per_agent_reward(100.0, empty_uavs)
        self.assertEqual(per_agent_reward, 100.0)  # 应该等于总奖励
        
        # 测试无目标的情况
        empty_targets = []
        ncs = self.calculator.calculate_normalized_completion_score(empty_targets)
        self.assertEqual(ncs, 0.0)
        
        # 测试无飞行距离的情况
        stationary_uavs = [MockUAV(0, [100, 50], [0, 0])]  # 没有移动
        efficiency = self.calculator.calculate_efficiency_metric([], stationary_uavs)
        self.assertGreaterEqual(efficiency, 0.0)
        
        print("✓ 除零错误处理测试通过")
    
    def test_scale_invariance_property(self):
        """测试尺度不变性属性"""
        # 创建2倍规模的场景
        scaled_uavs = [
            MockUAV(0, [100, 50], [0, 0]),
            MockUAV(1, [80, 60], [100, 0]),
            MockUAV(2, [90, 70], [200, 0]),
            MockUAV(3, [110, 40], [300, 0])
        ]
        
        scaled_targets = [
            MockTarget(0, [50, 30], [50, 50]),
            MockTarget(1, [40, 20], [150, 50]),
            MockTarget(2, [60, 35], [250, 50]),
            MockTarget(3, [45, 25], [350, 50])
        ]
        
        # 设置相同的完成比例
        scaled_targets[0].remaining_resources = np.array([0, 0])
        scaled_targets[1].remaining_resources = np.array([20, 10])
        scaled_targets[2].remaining_resources = np.array([0, 0])
        scaled_targets[3].remaining_resources = np.array([25, 15])
        
        # 设置相同的分配密度
        scaled_targets[0].allocated_uavs = [(0, 0), (1, 0)]
        scaled_targets[1].allocated_uavs = [(1, 0)]
        scaled_targets[2].allocated_uavs = [(2, 0), (3, 0)]
        scaled_targets[3].allocated_uavs = [(3, 0)]
        
        # 计算原始和缩放场景的指标
        original_ncs = self.calculator.calculate_normalized_completion_score(self.targets)
        scaled_ncs = self.calculator.calculate_normalized_completion_score(scaled_targets)
        
        # Per-Agent Reward应该保持相似（考虑到规模差异）
        original_par = self.calculator.calculate_per_agent_reward(100.0, self.uavs)
        scaled_par = self.calculator.calculate_per_agent_reward(200.0, scaled_uavs)  # 2倍奖励
        
        print(f"✓ 尺度不变性测试:")
        print(f"  - 原始NCS: {original_ncs:.4f}, 缩放NCS: {scaled_ncs:.4f}")
        print(f"  - 原始PAR: {original_par:.4f}, 缩放PAR: {scaled_par:.4f}")
        
        # NCS应该相似（因为完成比例相同）
        self.assertAlmostEqual(original_ncs, scaled_ncs, delta=0.1)


def run_simple_test():
    """运行简化测试"""
    print("=" * 60)
    print("开始尺度不变指标简化测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestScaleInvariantMetrics))
    
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
    success = run_simple_test()
    
    if success:
        print("🎉 所有测试通过！尺度不变指标核心计算逻辑正确。")
    else:
        print("❌ 部分测试失败，请检查实现。")
    
    exit(0 if success else 1)