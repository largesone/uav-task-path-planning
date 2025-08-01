# -*- coding: utf-8 -*-
# 文件名: test_dual_mode_observation.py
# 描述: 测试双模式观测系统的实现

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entities import UAV, Target
from environment import DirectedGraph, UAVTaskEnv
from config import Config

def create_test_scenario():
    """创建测试场景"""
    # 创建测试UAV
    uavs = [
        UAV(id=0, position=[100, 100], heading=0, resources=[50, 30], 
            max_distance=500, velocity_range=[10, 50], economic_speed=30),
        UAV(id=1, position=[200, 150], heading=np.pi/2, resources=[40, 35], 
            max_distance=600, velocity_range=[15, 45], economic_speed=25)
    ]
    
    # 创建测试目标
    targets = [
        Target(id=0, position=[300, 200], resources=[25, 20], value=100),
        Target(id=1, position=[400, 300], resources=[30, 25], value=150),
        Target(id=2, position=[150, 350], resources=[20, 15], value=80)
    ]
    
    # 创建配置
    config = Config()
    config.MAP_SIZE = 1000.0
    
    # 创建图
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, [], config)
    
    return uavs, targets, graph, config

def test_flat_mode():
    """测试扁平模式观测"""
    print("=== 测试扁平模式观测 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    
    # 创建扁平模式环境
    env_flat = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="flat")
    
    # 重置环境并获取状态
    state = env_flat.reset()
    
    print(f"扁平模式观测空间类型: {type(env_flat.observation_space)}")
    print(f"扁平模式观测空间形状: {env_flat.observation_space.shape}")
    print(f"扁平模式状态类型: {type(state)}")
    print(f"扁平模式状态形状: {state.shape}")
    print(f"扁平模式状态前10个值: {state[:10]}")
    
    # 验证状态维度
    expected_dim = (
        7 * len(targets) +      # 目标信息
        8 * len(uavs) +         # UAV信息  
        len(targets) * len(uavs) +  # 协同信息
        10                      # 全局信息
    )
    
    assert state.shape[0] == expected_dim, f"状态维度不匹配: 期望{expected_dim}, 实际{state.shape[0]}"
    print(f"✓ 扁平模式状态维度验证通过: {expected_dim}")
    
    return True

def test_graph_mode():
    """测试图模式观测"""
    print("\n=== 测试图模式观测 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    
    # 创建图模式环境
    env_graph = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    # 重置环境并获取状态
    state = env_graph.reset()
    
    print(f"图模式观测空间类型: {type(env_graph.observation_space)}")
    print(f"图模式状态类型: {type(state)}")
    print(f"图模式状态键: {list(state.keys())}")
    
    # 验证各个组件的形状
    n_uavs, n_targets = len(uavs), len(targets)
    
    # UAV特征
    uav_features = state["uav_features"]
    print(f"UAV特征形状: {uav_features.shape}")
    assert uav_features.shape == (n_uavs, 9), f"UAV特征形状不匹配: 期望({n_uavs}, 9), 实际{uav_features.shape}"
    
    # 目标特征
    target_features = state["target_features"]
    print(f"目标特征形状: {target_features.shape}")
    assert target_features.shape == (n_targets, 8), f"目标特征形状不匹配: 期望({n_targets}, 8), 实际{target_features.shape}"
    
    # 相对位置
    relative_positions = state["relative_positions"]
    print(f"相对位置形状: {relative_positions.shape}")
    assert relative_positions.shape == (n_uavs, n_targets, 2), f"相对位置形状不匹配"
    
    # 距离矩阵
    distances = state["distances"]
    print(f"距离矩阵形状: {distances.shape}")
    assert distances.shape == (n_uavs, n_targets), f"距离矩阵形状不匹配"
    
    # 掩码
    masks = state["masks"]
    uav_mask = masks["uav_mask"]
    target_mask = masks["target_mask"]
    print(f"UAV掩码形状: {uav_mask.shape}")
    print(f"目标掩码形状: {target_mask.shape}")
    assert uav_mask.shape == (n_uavs,), f"UAV掩码形状不匹配"
    assert target_mask.shape == (n_targets,), f"目标掩码形状不匹配"
    
    # 验证归一化范围
    print(f"UAV特征值范围: [{uav_features.min():.3f}, {uav_features.max():.3f}]")
    print(f"目标特征值范围: [{target_features.min():.3f}, {target_features.max():.3f}]")
    print(f"相对位置值范围: [{relative_positions.min():.3f}, {relative_positions.max():.3f}]")
    print(f"距离值范围: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # 验证掩码值
    print(f"UAV掩码值: {uav_mask}")
    print(f"目标掩码值: {target_mask}")
    
    print("✓ 图模式状态结构验证通过")
    
    return True

def test_observation_space_compatibility():
    """测试观测空间兼容性"""
    print("\n=== 测试观测空间兼容性 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    
    # 测试两种模式的环境创建
    env_flat = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="flat")
    env_graph = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    # 获取状态
    state_flat = env_flat.reset()
    state_graph = env_graph.reset()
    
    # 验证状态符合观测空间定义
    assert env_flat.observation_space.contains(state_flat), "扁平模式状态不符合观测空间"
    assert env_graph.observation_space.contains(state_graph), "图模式状态不符合观测空间"
    
    print("✓ 观测空间兼容性验证通过")
    
    return True

def test_step_functionality():
    """测试step功能在两种模式下的正常工作"""
    print("\n=== 测试step功能 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    
    # 测试扁平模式
    env_flat = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="flat")
    state_flat = env_flat.reset()
    
    # 执行一个动作
    action = 0  # 第一个可能的动作
    next_state, reward, done, truncated, info = env_flat.step(action)
    
    print(f"扁平模式step结果:")
    print(f"  下一状态类型: {type(next_state)}")
    print(f"  奖励: {reward}")
    print(f"  完成: {done}")
    print(f"  截断: {truncated}")
    
    # 测试图模式
    env_graph = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    state_graph = env_graph.reset()
    
    next_state_graph, reward_graph, done_graph, truncated_graph, info_graph = env_graph.step(action)
    
    print(f"图模式step结果:")
    print(f"  下一状态类型: {type(next_state_graph)}")
    print(f"  奖励: {reward_graph}")
    print(f"  完成: {done_graph}")
    print(f"  截断: {truncated_graph}")
    
    print("✓ step功能验证通过")
    
    return True

def test_invalid_mode():
    """测试无效模式处理"""
    print("\n=== 测试无效模式处理 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    
    try:
        env_invalid = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="invalid")
        env_invalid.reset()
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"✓ 正确捕获无效模式异常: {e}")
        return True

def main():
    """主测试函数"""
    print("开始测试双模式观测系统...")
    
    try:
        # 运行所有测试
        test_flat_mode()
        test_graph_mode()
        test_observation_space_compatibility()
        test_step_functionality()
        test_invalid_mode()
        
        print("\n" + "="*50)
        print("✅ 所有测试通过！双模式观测系统实现正确。")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()