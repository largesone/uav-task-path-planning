# -*- coding: utf-8 -*-
# 文件名: test_task2_verification.py
# 描述: 验证任务2的所有需求是否完全实现

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
            max_distance=600, velocity_range=[15, 45], economic_speed=25),
        UAV(id=2, position=[300, 250], heading=np.pi, resources=[0, 0],  # 无资源UAV
            max_distance=400, velocity_range=[12, 40], economic_speed=28)
    ]
    
    # 创建测试目标
    targets = [
        Target(id=0, position=[300, 200], resources=[25, 20], value=100),
        Target(id=1, position=[400, 300], resources=[30, 25], value=150),
        Target(id=2, position=[150, 350], resources=[20, 15], value=80)
    ]
    
    # 模拟一个目标已完成
    targets[2].remaining_resources = np.array([0, 0])
    
    # 创建配置
    config = Config()
    config.MAP_SIZE = 1000.0
    
    # 创建图
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, [], config)
    
    return uavs, targets, graph, config

def test_requirement_2_1():
    """测试需求2.1: 重构UAVTaskEnv._get_state方法，支持图模式状态字典输出"""
    print("=== 测试需求2.1: 图模式状态字典输出 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    state = env._get_state()
    
    # 验证返回类型是字典
    assert isinstance(state, dict), f"状态应该是字典类型，实际是{type(state)}"
    
    # 验证包含所有必需的键
    required_keys = ["uav_features", "target_features", "relative_positions", "distances", "masks"]
    for key in required_keys:
        assert key in state, f"状态字典缺少键: {key}"
    
    print("✓ 需求2.1验证通过: 图模式状态字典输出正确")
    return True

def test_requirement_2_2():
    """测试需求2.2: 实现uav_features和target_features，仅包含归一化的实体自身属性"""
    print("\n=== 测试需求2.2: 归一化实体特征 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    state = env._get_state()
    
    # 验证UAV特征
    uav_features = state["uav_features"]
    print(f"UAV特征形状: {uav_features.shape}")
    print(f"UAV特征值范围: [{uav_features.min():.3f}, {uav_features.max():.3f}]")
    
    # 验证特征都在[0,1]范围内（除了is_alive可能为0）
    assert np.all(uav_features >= 0.0), "UAV特征包含负值"
    assert np.all(uav_features <= 1.0), "UAV特征超过1.0"
    
    # 验证目标特征
    target_features = state["target_features"]
    print(f"目标特征形状: {target_features.shape}")
    print(f"目标特征值范围: [{target_features.min():.3f}, {target_features.max():.3f}]")
    
    # 验证特征都在[0,1]范围内
    assert np.all(target_features >= 0.0), "目标特征包含负值"
    assert np.all(target_features <= 1.0), "目标特征超过1.0"
    
    # 验证特征维度
    assert uav_features.shape[1] == 9, f"UAV特征维度应为9，实际为{uav_features.shape[1]}"
    assert target_features.shape[1] == 8, f"目标特征维度应为8，实际为{target_features.shape[1]}"
    
    print("✓ 需求2.2验证通过: 实体特征归一化正确")
    return True

def test_requirement_2_3():
    """测试需求2.3: 实现relative_positions键，存储归一化相对位置向量"""
    print("\n=== 测试需求2.3: 归一化相对位置向量 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    state = env._get_state()
    relative_positions = state["relative_positions"]
    
    print(f"相对位置形状: {relative_positions.shape}")
    print(f"相对位置值范围: [{relative_positions.min():.3f}, {relative_positions.max():.3f}]")
    
    # 验证形状
    n_uavs, n_targets = len(uavs), len(targets)
    expected_shape = (n_uavs, n_targets, 2)
    assert relative_positions.shape == expected_shape, f"相对位置形状应为{expected_shape}，实际为{relative_positions.shape}"
    
    # 验证值在[-1, 1]范围内
    assert np.all(relative_positions >= -1.0), "相对位置包含小于-1的值"
    assert np.all(relative_positions <= 1.0), "相对位置包含大于1的值"
    
    # 手动验证计算正确性
    map_size = config.MAP_SIZE
    for i, uav in enumerate(uavs):
        for j, target in enumerate(targets):
            expected_rel_pos = (np.array(target.position) - np.array(uav.current_position)) / map_size
            actual_rel_pos = relative_positions[i, j]
            
            assert np.allclose(actual_rel_pos, expected_rel_pos, atol=1e-6), \
                f"相对位置计算错误: UAV{i}->Target{j}, 期望{expected_rel_pos}, 实际{actual_rel_pos}"
    
    print("✓ 需求2.3验证通过: 相对位置向量计算正确")
    return True

def test_requirement_2_4():
    """测试需求2.4: 实现distances键，存储无人机与目标间的归一化距离"""
    print("\n=== 测试需求2.4: 归一化距离矩阵 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    state = env._get_state()
    distances = state["distances"]
    
    print(f"距离矩阵形状: {distances.shape}")
    print(f"距离值范围: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # 验证形状
    n_uavs, n_targets = len(uavs), len(targets)
    expected_shape = (n_uavs, n_targets)
    assert distances.shape == expected_shape, f"距离矩阵形状应为{expected_shape}，实际为{distances.shape}"
    
    # 验证值在[0, 1]范围内
    assert np.all(distances >= 0.0), "距离矩阵包含负值"
    assert np.all(distances <= 1.0), "距离矩阵包含大于1的值"
    
    # 手动验证计算正确性
    map_size = config.MAP_SIZE
    for i, uav in enumerate(uavs):
        for j, target in enumerate(targets):
            expected_dist = np.linalg.norm(
                np.array(target.position) - np.array(uav.current_position)
            ) / map_size
            expected_dist = min(expected_dist, 1.0)  # 限制在1.0以内
            actual_dist = distances[i, j]
            
            assert np.allclose(actual_dist, expected_dist, atol=1e-6), \
                f"距离计算错误: UAV{i}->Target{j}, 期望{expected_dist}, 实际{actual_dist}"
    
    print("✓ 需求2.4验证通过: 距离矩阵计算正确")
    return True

def test_requirement_2_5():
    """测试需求2.5: 实现masks键，包含uav_mask和target_mask用于标识有效实体"""
    print("\n=== 测试需求2.5: 有效实体掩码 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    state = env._get_state()
    masks = state["masks"]
    
    # 验证掩码字典结构
    assert isinstance(masks, dict), "掩码应该是字典类型"
    assert "uav_mask" in masks, "掩码字典缺少uav_mask"
    assert "target_mask" in masks, "掩码字典缺少target_mask"
    
    uav_mask = masks["uav_mask"]
    target_mask = masks["target_mask"]
    
    print(f"UAV掩码: {uav_mask}")
    print(f"目标掩码: {target_mask}")
    
    # 验证掩码形状
    assert uav_mask.shape == (len(uavs),), f"UAV掩码形状错误"
    assert target_mask.shape == (len(targets),), f"目标掩码形状错误"
    
    # 验证掩码值只能是0或1
    assert np.all(np.isin(uav_mask, [0, 1])), "UAV掩码包含非0/1值"
    assert np.all(np.isin(target_mask, [0, 1])), "目标掩码包含非0/1值"
    
    # 验证掩码逻辑正确性
    for i, uav in enumerate(uavs):
        expected_mask = 1 if np.any(uav.resources > 0) else 0
        assert uav_mask[i] == expected_mask, f"UAV{i}掩码错误: 期望{expected_mask}, 实际{uav_mask[i]}"
    
    for i, target in enumerate(targets):
        expected_mask = 1 if np.any(target.remaining_resources > 0) else 0
        assert target_mask[i] == expected_mask, f"Target{i}掩码错误: 期望{expected_mask}, 实际{target_mask[i]}"
    
    print("✓ 需求2.5验证通过: 有效实体掩码正确")
    return True

def test_scale_invariance():
    """测试尺度不变性"""
    print("\n=== 测试尺度不变性 ===")
    
    # 创建两个不同尺度的场景
    uavs1, targets1, graph1, config1 = create_test_scenario()
    config1.MAP_SIZE = 1000.0
    
    uavs2, targets2, graph2, config2 = create_test_scenario()
    config2.MAP_SIZE = 2000.0  # 双倍地图尺寸
    
    # 将第二个场景的所有位置放大2倍
    for uav in uavs2:
        uav.current_position = [pos * 2 for pos in uav.current_position]
        uav.position = [pos * 2 for pos in uav.position]
    for target in targets2:
        target.position = [pos * 2 for pos in target.position]
    
    # 重新创建图
    graph2 = DirectedGraph(uavs2, targets2, config2.GRAPH_N_PHI, [], config2)
    
    env1 = UAVTaskEnv(uavs1, targets1, graph1, [], config1, obs_mode="graph")
    env2 = UAVTaskEnv(uavs2, targets2, graph2, [], config2, obs_mode="graph")
    
    state1 = env1._get_state()
    state2 = env2._get_state()
    
    # 验证相对位置和距离应该相同（尺度不变）
    rel_pos_diff = np.abs(state1["relative_positions"] - state2["relative_positions"])
    dist_diff = np.abs(state1["distances"] - state2["distances"])
    
    print(f"相对位置最大差异: {rel_pos_diff.max():.6f}")
    print(f"距离最大差异: {dist_diff.max():.6f}")
    
    assert np.allclose(state1["relative_positions"], state2["relative_positions"], atol=1e-5), \
        "相对位置不具备尺度不变性"
    assert np.allclose(state1["distances"], state2["distances"], atol=1e-5), \
        "距离不具备尺度不变性"
    
    print("✓ 尺度不变性验证通过")
    return True

def test_robustness_masks():
    """测试鲁棒性掩码机制"""
    print("\n=== 测试鲁棒性掩码机制 ===")
    
    uavs, targets, graph, config = create_test_scenario()
    env = UAVTaskEnv(uavs, targets, graph, [], config, obs_mode="graph")
    
    state = env._get_state()
    
    # 验证UAV的is_alive位
    uav_features = state["uav_features"]
    for i, uav in enumerate(uavs):
        is_alive = uav_features[i, -1]  # 最后一位是is_alive
        expected_alive = 1.0 if np.any(uav.resources > 0) else 0.0
        assert is_alive == expected_alive, f"UAV{i}的is_alive位错误"
    
    # 验证目标的is_visible位
    target_features = state["target_features"]
    for i, target in enumerate(targets):
        is_visible = target_features[i, -1]  # 最后一位是is_visible
        expected_visible = 1.0 if np.any(target.remaining_resources > 0) else 0.0
        assert is_visible == expected_visible, f"Target{i}的is_visible位错误"
    
    print("✓ 鲁棒性掩码机制验证通过")
    return True

def main():
    """主测试函数"""
    print("开始验证任务2的所有需求...")
    
    try:
        # 验证所有需求
        test_requirement_2_1()
        test_requirement_2_2()
        test_requirement_2_3()
        test_requirement_2_4()
        test_requirement_2_5()
        
        # 额外测试
        test_scale_invariance()
        test_robustness_masks()
        
        print("\n" + "="*60)
        print("✅ 任务2的所有需求验证通过！")
        print("尺度不变的图模式状态输出已正确实现。")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()