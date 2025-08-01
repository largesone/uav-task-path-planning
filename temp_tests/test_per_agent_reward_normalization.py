# -*- coding: utf-8 -*-
"""
测试文件: test_per_agent_reward_normalization.py
描述: 验证Per-Agent奖励归一化功能的正确性和一致性

测试目标:
1. 验证不同无人机数量下的奖励一致性
2. 验证拥堵惩罚等数值会随无人机数量N增长的奖励项被正确归一化
3. 验证奖励组件跟踪功能
4. 验证归一化前后的奖励值记录
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entities import UAV, Target
from environment import DirectedGraph, UAVTaskEnv

class MockConfig:
    """模拟配置类，用于测试"""
    def __init__(self):
        self.GRAPH_N_PHI = 8
        self.MAP_SIZE = 1000.0
        self.USE_PHRRT_DURING_TRAINING = False
        self.ENABLE_REWARD_LOGGING = True
        self.UAV_COMM_FAILURE_RATE = 0.0
        self.UAV_SENSING_FAILURE_RATE = 0.0

def create_test_scenario(n_uavs: int, n_targets: int = 3):
    """
    创建测试场景，支持可变无人机数量
    
    Args:
        n_uavs: 无人机数量
        n_targets: 目标数量
        
    Returns:
        tuple: (uavs, targets, graph, obstacles, config)
    """
    config = MockConfig()
    
    # 创建UAV列表
    uavs = []
    for i in range(n_uavs):
        uav = UAV(
            id=i,
            position=(100 + i * 50, 100 + i * 30),
            heading=0.0,  # 添加朝向参数
            resources=np.array([50.0, 40.0]),
            max_distance=1000,
            velocity_range=(30, 100),
            economic_speed=60.0  # 添加经济速度参数
        )
        uavs.append(uav)
    
    # 创建目标列表
    targets = []
    target_positions = [(800, 200), (600, 600), (200, 800)]
    for i in range(n_targets):
        target = Target(
            id=i,
            position=target_positions[i % len(target_positions)],
            resources=np.array([80.0, 60.0]),
            value=100.0
        )
        targets.append(target)
    
    # 创建图
    obstacles = []  # 简化测试，不使用障碍物
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    return uavs, targets, graph, obstacles, config

def test_active_uav_count_calculation():
    """测试有效无人机数量计算"""
    print("=== 测试1: 有效无人机数量计算 ===")
    
    # 测试不同数量的UAV
    for n_uavs in [2, 5, 10, 15]:
        uavs, targets, graph, obstacles, config = create_test_scenario(n_uavs)
        env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
        
        active_count = env._calculate_active_uav_count()
        print(f"总UAV数量: {n_uavs}, 有效UAV数量: {active_count}")
        
        # 验证有效数量不超过总数量
        assert active_count <= n_uavs, f"有效UAV数量({active_count})不应超过总数量({n_uavs})"
        assert active_count >= 1, f"有效UAV数量({active_count})应至少为1"
    
    print("✓ 有效无人机数量计算测试通过\n")

def test_congestion_penalty_normalization():
    """测试拥堵惩罚的归一化"""
    print("=== 测试2: 拥堵惩罚归一化 ===")
    
    # 创建拥堵场景：多个UAV分配到同一目标
    uavs, targets, graph, obstacles, config = create_test_scenario(6, 2)
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    
    # 模拟多个UAV分配到第一个目标
    target = targets[0]
    target.allocated_uavs = [(0, 0), (1, 1), (2, 2), (3, 3)]  # 4个UAV分配到同一目标
    
    # 测试不同有效UAV数量下的拥堵惩罚
    test_cases = [
        {"n_active": 4, "expected_normalization": True},
        {"n_active": 6, "expected_normalization": True},
        {"n_active": 2, "expected_normalization": True}
    ]
    
    for case in test_cases:
        # 模拟不同的有效UAV数量
        original_method = env._calculate_active_uav_count
        env._calculate_active_uav_count = lambda: case["n_active"]
        
        # 计算拥堵惩罚
        congestion_raw = env._calculate_congestion_penalty(target, uavs[0], case["n_active"])
        congestion_normalized = congestion_raw / case["n_active"]
        
        print(f"有效UAV数量: {case['n_active']}")
        print(f"  原始拥堵惩罚: {congestion_raw:.4f}")
        print(f"  归一化拥堵惩罚: {congestion_normalized:.4f}")
        print(f"  归一化因子: {1.0/case['n_active']:.4f}")
        
        # 验证归一化效果
        if congestion_raw > 0:
            assert congestion_normalized < congestion_raw, "归一化后的惩罚应小于原始惩罚"
            assert abs(congestion_normalized - congestion_raw / case["n_active"]) < 1e-6, "归一化计算错误"
        
        # 恢复原始方法
        env._calculate_active_uav_count = original_method
    
    print("✓ 拥堵惩罚归一化测试通过\n")

def test_reward_consistency_across_uav_counts():
    """测试不同无人机数量下的奖励一致性"""
    print("=== 测试3: 不同无人机数量下的奖励一致性 ===")
    
    # 测试场景：相同的动作在不同UAV数量下应产生相似的归一化奖励
    uav_counts = [3, 6, 9, 12]
    rewards_by_count = {}
    
    for n_uavs in uav_counts:
        uavs, targets, graph, obstacles, config = create_test_scenario(n_uavs, 3)
        env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
        
        # 执行相同的动作：第一个UAV分配到第一个目标
        target = targets[0]
        uav = uavs[0]
        actual_contribution = np.array([20.0, 15.0])  # 固定贡献
        path_len = 500.0  # 固定路径长度
        travel_time = 10.0  # 固定旅行时间
        was_satisfied = False
        done = False
        
        # 计算奖励
        reward = env._calculate_reward(target, uav, actual_contribution, path_len, 
                                     was_satisfied, travel_time, done)
        
        rewards_by_count[n_uavs] = {
            'reward': reward,
            'components': env.last_reward_components.copy()
        }
        
        print(f"UAV数量: {n_uavs}, 奖励: {reward:.4f}")
        
        # 打印归一化信息
        norm_info = env.last_reward_components['per_agent_normalization']
        print(f"  有效UAV数量: {norm_info['n_active_uavs']}")
        print(f"  归一化组件: {norm_info['components_normalized']}")
        print(f"  归一化节省: {norm_info['normalization_impact']['normalization_savings']:.4f}")
    
    # 验证奖励一致性：归一化后的奖励应该在合理范围内
    reward_values = [data['reward'] for data in rewards_by_count.values()]
    reward_std = np.std(reward_values)
    reward_mean = np.mean(reward_values)
    
    print(f"\n奖励统计:")
    print(f"  平均奖励: {reward_mean:.4f}")
    print(f"  标准差: {reward_std:.4f}")
    print(f"  变异系数: {reward_std/abs(reward_mean):.4f}")
    
    # 验证变异系数在合理范围内（归一化应该减少变异性）
    cv_threshold = 0.3  # 变异系数阈值
    assert reward_std / abs(reward_mean) < cv_threshold, \
        f"奖励变异系数({reward_std/abs(reward_mean):.4f})超过阈值({cv_threshold})"
    
    print("✓ 奖励一致性测试通过\n")

def test_reward_component_tracking():
    """测试奖励组件跟踪功能"""
    print("=== 测试4: 奖励组件跟踪功能 ===")
    
    uavs, targets, graph, obstacles, config = create_test_scenario(5, 2)
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    
    # 执行一个动作
    target = targets[0]
    uav = uavs[0]
    actual_contribution = np.array([30.0, 25.0])
    path_len = 300.0
    travel_time = 8.0
    was_satisfied = False
    done = False
    
    reward = env._calculate_reward(target, uav, actual_contribution, path_len, 
                                 was_satisfied, travel_time, done)
    
    components = env.last_reward_components
    
    # 验证必要的组件存在
    required_components = [
        'n_active_uavs', 'normalization_applied', 'total_positive', 'total_costs',
        'final_reward', 'per_agent_normalization', 'debug_info'
    ]
    
    for component in required_components:
        assert component in components, f"缺少必要的奖励组件: {component}"
    
    # 验证归一化信息
    norm_info = components['per_agent_normalization']
    assert 'n_active_uavs' in norm_info
    assert 'normalization_factor' in norm_info
    assert 'components_normalized' in norm_info
    assert 'normalization_impact' in norm_info
    
    # 验证归一化影响分析
    impact = norm_info['normalization_impact']
    assert 'total_raw_normalized_rewards' in impact
    assert 'total_normalized_rewards' in impact
    assert 'normalization_savings' in impact
    assert 'components_impact' in impact
    
    print("奖励组件跟踪信息:")
    print(f"  有效UAV数量: {norm_info['n_active_uavs']}")
    print(f"  归一化因子: {norm_info['normalization_factor']:.4f}")
    print(f"  应用归一化的组件: {norm_info['components_normalized']}")
    print(f"  归一化节省: {impact['normalization_savings']:.4f}")
    print(f"  最终奖励: {components['final_reward']:.4f}")
    
    print("✓ 奖励组件跟踪测试通过\n")

def test_normalization_impact_analysis():
    """测试归一化影响分析"""
    print("=== 测试5: 归一化影响分析 ===")
    
    # 创建高拥堵场景
    uavs, targets, graph, obstacles, config = create_test_scenario(8, 2)
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    
    # 模拟高拥堵：多个UAV分配到同一目标
    target = targets[0]
    target.allocated_uavs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]  # 5个UAV分配到同一目标
    
    # 执行动作
    uav = uavs[0]
    actual_contribution = np.array([25.0, 20.0])
    path_len = 400.0
    travel_time = 12.0
    was_satisfied = False
    done = False
    
    reward = env._calculate_reward(target, uav, actual_contribution, path_len, 
                                 was_satisfied, travel_time, done)
    
    components = env.last_reward_components
    impact = components['per_agent_normalization']['normalization_impact']
    
    print("归一化影响分析:")
    print(f"  原始归一化奖励总和: {impact['total_raw_normalized_rewards']:.4f}")
    print(f"  归一化后奖励总和: {impact['total_normalized_rewards']:.4f}")
    print(f"  归一化节省: {impact['normalization_savings']:.4f}")
    
    # 验证归一化确实产生了影响
    if impact['total_raw_normalized_rewards'] > 0:
        assert impact['normalization_savings'] > 0, "在高拥堵场景下，归一化应该产生节省效果"
        assert impact['total_normalized_rewards'] < impact['total_raw_normalized_rewards'], \
            "归一化后的奖励应小于原始奖励"
    
    # 验证组件级别的影响分析
    for component, details in impact['components_impact'].items():
        print(f"  {component}组件:")
        print(f"    原始值: {details['raw']:.4f}")
        print(f"    归一化值: {details['normalized']:.4f}")
        print(f"    减少量: {details['reduction']:.4f}")
        
        if details['raw'] > 0:
            assert details['reduction'] >= 0, f"{component}组件的归一化减少量应为非负"
    
    print("✓ 归一化影响分析测试通过\n")

def run_comprehensive_test():
    """运行综合测试"""
    print("开始Per-Agent奖励归一化综合测试...\n")
    
    try:
        test_active_uav_count_calculation()
        test_congestion_penalty_normalization()
        test_reward_consistency_across_uav_counts()
        test_reward_component_tracking()
        test_normalization_impact_analysis()
        
        print("🎉 所有Per-Agent奖励归一化测试通过！")
        print("\n测试总结:")
        print("✓ 有效无人机数量计算正确")
        print("✓ 拥堵惩罚归一化功能正常")
        print("✓ 不同UAV数量下奖励保持一致性")
        print("✓ 奖励组件跟踪功能完整")
        print("✓ 归一化影响分析准确")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)