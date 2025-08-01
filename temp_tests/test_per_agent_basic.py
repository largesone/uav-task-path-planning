# -*- coding: utf-8 -*-
"""
简化的Per-Agent奖励归一化测试
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
        self.ENABLE_REWARD_LOGGING = False  # 关闭详细日志
        self.UAV_COMM_FAILURE_RATE = 0.0
        self.UAV_SENSING_FAILURE_RATE = 0.0

def create_simple_test_scenario():
    """创建简单的测试场景"""
    config = MockConfig()
    
    # 创建3个UAV
    uavs = []
    for i in range(3):
        uav = UAV(
            id=i,
            position=(100 + i * 100, 100),
            heading=0.0,
            resources=np.array([50.0, 40.0]),
            max_distance=1000,
            velocity_range=(30, 100),
            economic_speed=60.0
        )
        uavs.append(uav)
    
    # 创建2个目标
    targets = []
    for i in range(2):
        target = Target(
            id=i,
            position=(500 + i * 200, 300),
            resources=np.array([80.0, 60.0]),
            value=100.0
        )
        targets.append(target)
    
    # 创建图
    obstacles = []
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    return uavs, targets, graph, obstacles, config

def test_basic_functionality():
    """测试基本功能"""
    print("=== 基本功能测试 ===")
    
    uavs, targets, graph, obstacles, config = create_simple_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    
    # 测试有效UAV数量计算
    active_count = env._calculate_active_uav_count()
    print(f"有效UAV数量: {active_count}")
    assert active_count == 3, f"期望3个有效UAV，实际得到{active_count}"
    
    # 测试拥堵惩罚计算
    target = targets[0]
    uav = uavs[0]
    congestion_penalty = env._calculate_congestion_penalty(target, uav, active_count)
    print(f"拥堵惩罚: {congestion_penalty}")
    
    # 测试奖励计算
    actual_contribution = np.array([20.0, 15.0])
    path_len = 300.0
    travel_time = 8.0
    was_satisfied = False
    done = False
    
    reward = env._calculate_reward(target, uav, actual_contribution, path_len, 
                                 was_satisfied, travel_time, done)
    print(f"计算的奖励: {reward}")
    
    # 检查奖励组件
    components = env.last_reward_components
    print(f"奖励组件数量: {len(components)}")
    
    # 检查归一化信息
    if 'per_agent_normalization' in components:
        norm_info = components['per_agent_normalization']
        print(f"归一化信息: {norm_info['n_active_uavs']} 个有效UAV")
        print(f"应用归一化的组件: {norm_info['components_normalized']}")
    
    print("✓ 基本功能测试通过\n")

def test_different_uav_counts():
    """测试不同UAV数量的影响"""
    print("=== 不同UAV数量测试 ===")
    
    # 测试3个和6个UAV的场景
    for n_uavs in [3, 6]:
        config = MockConfig()
        
        # 创建UAV
        uavs = []
        for i in range(n_uavs):
            uav = UAV(
                id=i,
                position=(100 + i * 50, 100),
                heading=0.0,
                resources=np.array([50.0, 40.0]),
                max_distance=1000,
                velocity_range=(30, 100),
                economic_speed=60.0
            )
            uavs.append(uav)
        
        # 创建目标
        targets = []
        target = Target(
            id=0,
            position=(500, 300),
            resources=np.array([80.0, 60.0]),
            value=100.0
        )
        targets.append(target)
        
        # 创建环境
        obstacles = []
        graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
        env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
        
        # 计算奖励
        actual_contribution = np.array([20.0, 15.0])
        reward = env._calculate_reward(targets[0], uavs[0], actual_contribution, 300.0, 
                                     False, 8.0, False)
        
        print(f"UAV数量: {n_uavs}, 奖励: {reward:.4f}")
        
        # 检查归一化
        components = env.last_reward_components
        if 'per_agent_normalization' in components:
            norm_info = components['per_agent_normalization']
            print(f"  有效UAV: {norm_info['n_active_uavs']}")
            print(f"  归一化组件: {norm_info['components_normalized']}")
    
    print("✓ 不同UAV数量测试通过\n")

if __name__ == "__main__":
    print("开始Per-Agent奖励归一化基本测试...\n")
    
    try:
        test_basic_functionality()
        test_different_uav_counts()
        
        print("🎉 基本测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()