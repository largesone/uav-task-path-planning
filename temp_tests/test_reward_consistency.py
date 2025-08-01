# -*- coding: utf-8 -*-
"""
测试Per-Agent奖励归一化的一致性
验证不同无人机数量下的奖励一致性
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
        self.ENABLE_REWARD_LOGGING = False
        self.UAV_COMM_FAILURE_RATE = 0.0
        self.UAV_SENSING_FAILURE_RATE = 0.0

def create_test_scenario_with_congestion(n_uavs: int):
    """创建带拥堵的测试场景"""
    config = MockConfig()
    
    # 创建UAV列表
    uavs = []
    for i in range(n_uavs):
        uav = UAV(
            id=i,
            position=(100 + i * 20, 100 + i * 10),  # 让UAV更接近，增加拥堵
            heading=0.0,
            resources=np.array([50.0, 40.0]),
            max_distance=1000,
            velocity_range=(30, 100),
            economic_speed=60.0
        )
        uavs.append(uav)
    
    # 创建单个目标
    target = Target(
        id=0,
        position=(500, 300),
        resources=np.array([200.0, 150.0]),  # 大资源需求，需要多个UAV
        value=100.0
    )
    targets = [target]
    
    # 模拟多个UAV已分配到同一目标（制造拥堵）
    if n_uavs >= 4:
        target.allocated_uavs = [(0, 0), (1, 1), (2, 2), (3, 3)]
    elif n_uavs >= 2:
        target.allocated_uavs = [(0, 0), (1, 1)]
    
    # 创建图
    obstacles = []
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    return uavs, targets, graph, obstacles, config

def test_reward_consistency():
    """测试奖励一致性"""
    print("=== 奖励一致性测试 ===")
    
    # 测试不同的UAV数量
    uav_counts = [3, 6, 9, 12]
    rewards = []
    
    # 固定的测试参数
    test_params = {
        'actual_contribution': np.array([25.0, 20.0]),
        'path_len': 400.0,
        'travel_time': 10.0,
        'was_satisfied': False,
        'done': False
    }
    
    print("测试参数:")
    print(f"  贡献: {test_params['actual_contribution']}")
    print(f"  路径长度: {test_params['path_len']}")
    print(f"  旅行时间: {test_params['travel_time']}")
    print()
    
    for n_uavs in uav_counts:
        uavs, targets, graph, obstacles, config = create_test_scenario_with_congestion(n_uavs)
        env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
        
        # 计算奖励
        reward = env._calculate_reward(
            targets[0], uavs[0], 
            test_params['actual_contribution'],
            test_params['path_len'],
            test_params['was_satisfied'],
            test_params['travel_time'],
            test_params['done']
        )
        
        rewards.append(reward)
        
        # 获取奖励组件信息
        components = env.last_reward_components
        norm_info = components['per_agent_normalization']
        
        print(f"UAV数量: {n_uavs:2d}")
        print(f"  有效UAV: {norm_info['n_active_uavs']}")
        print(f"  奖励: {reward:.4f}")
        print(f"  归一化组件: {norm_info['components_normalized']}")
        
        # 显示归一化影响
        impact = norm_info['normalization_impact']
        if impact['normalization_savings'] > 0:
            print(f"  归一化节省: {impact['normalization_savings']:.4f}")
        
        print()
    
    # 分析奖励一致性
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    cv = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
    
    print("奖励一致性分析:")
    print(f"  奖励范围: {np.min(rewards):.4f} ~ {np.max(rewards):.4f}")
    print(f"  平均奖励: {mean_reward:.4f}")
    print(f"  标准差: {std_reward:.4f}")
    print(f"  变异系数: {cv:.4f}")
    
    # 验证一致性
    cv_threshold = 0.15  # 15%的变异系数阈值
    if cv < cv_threshold:
        print(f"✓ 奖励一致性良好 (CV={cv:.4f} < {cv_threshold})")
    else:
        print(f"⚠ 奖励一致性需要改进 (CV={cv:.4f} >= {cv_threshold})")
    
    return cv < cv_threshold

def test_normalization_effectiveness():
    """测试归一化效果"""
    print("\n=== 归一化效果测试 ===")
    
    # 创建高拥堵场景
    n_uavs = 8
    uavs, targets, graph, obstacles, config = create_test_scenario_with_congestion(n_uavs)
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    
    # 增加更多拥堵
    targets[0].allocated_uavs = [(i, i % 8) for i in range(6)]  # 6个UAV分配到同一目标
    
    # 计算奖励
    reward = env._calculate_reward(
        targets[0], uavs[0],
        np.array([30.0, 25.0]),
        500.0, False, 12.0, False
    )
    
    components = env.last_reward_components
    norm_info = components['per_agent_normalization']
    impact = norm_info['normalization_impact']
    
    print(f"高拥堵场景 (UAV数量: {n_uavs}):")
    print(f"  分配到目标的UAV数量: {len(targets[0].allocated_uavs)}")
    print(f"  有效UAV数量: {norm_info['n_active_uavs']}")
    print(f"  最终奖励: {reward:.4f}")
    print(f"  归一化组件: {norm_info['components_normalized']}")
    print(f"  归一化节省: {impact['normalization_savings']:.4f}")
    
    # 显示各组件的归一化效果
    for component, details in impact['components_impact'].items():
        print(f"  {component}:")
        print(f"    原始值: {details['raw']:.4f}")
        print(f"    归一化值: {details['normalized']:.4f}")
        print(f"    减少量: {details['reduction']:.4f}")
    
    # 验证归一化确实产生了效果
    has_normalization_effect = impact['normalization_savings'] > 0
    if has_normalization_effect:
        print("✓ 归一化产生了预期效果")
    else:
        print("⚠ 归一化效果不明显")
    
    return has_normalization_effect

def main():
    """主测试函数"""
    print("开始Per-Agent奖励归一化一致性测试...\n")
    
    try:
        # 测试奖励一致性
        consistency_ok = test_reward_consistency()
        
        # 测试归一化效果
        normalization_ok = test_normalization_effectiveness()
        
        print("\n" + "="*50)
        print("测试总结:")
        print(f"✓ 奖励一致性: {'通过' if consistency_ok else '需要改进'}")
        print(f"✓ 归一化效果: {'有效' if normalization_ok else '需要改进'}")
        
        if consistency_ok and normalization_ok:
            print("\n🎉 Per-Agent奖励归一化功能验证通过！")
            return True
        else:
            print("\n⚠ 部分测试需要改进")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)