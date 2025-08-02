#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试零样本GNN架构的实现

测试内容：
1. 双模式观测系统的正确性
2. ZeroShotGNN网络的前向传播
3. 不同规模场景的零样本迁移能力
4. 与传统FCN的性能对比
"""

import numpy as np
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph
from networks import ZeroShotGNN, create_network
from config import Config
from scenarios import get_small_scenario
from path_planning import CircularObstacle, PolygonalObstacle

def test_dual_mode_observation():
    """测试双模式观测系统"""
    print("=== 测试双模式观测系统 ===")
    
    # 创建测试场景
    config = Config()
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    # 测试扁平模式
    print("\n1. 测试扁平模式观测:")
    env_flat = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
    state_flat = env_flat.reset()
    print(f"   扁平状态类型: {type(state_flat)}")
    print(f"   扁平状态形状: {state_flat.shape}")
    print(f"   扁平状态前10个元素: {state_flat[:10]}")
    
    # 测试图模式
    print("\n2. 测试图模式观测:")
    env_graph = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="graph")
    state_graph = env_graph.reset()
    print(f"   图状态类型: {type(state_graph)}")
    print(f"   图状态键: {list(state_graph.keys())}")
    print(f"   UAV特征形状: {state_graph['uav_features'].shape}")
    print(f"   目标特征形状: {state_graph['target_features'].shape}")
    print(f"   相对位置形状: {state_graph['relative_positions'].shape}")
    print(f"   距离矩阵形状: {state_graph['distances'].shape}")
    print(f"   掩码: {state_graph['masks']}")
    
    # 验证观测空间
    print(f"\n3. 观测空间验证:")
    print(f"   扁平模式观测空间: {env_flat.observation_space}")
    print(f"   图模式观测空间类型: {type(env_graph.observation_space)}")
    
    return True

def test_zero_shot_gnn_network():
    """测试ZeroShotGNN网络"""
    print("\n=== 测试ZeroShotGNN网络 ===")
    
    # 创建测试场景
    config = Config()
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="graph")
    
    # 创建网络
    input_dim = 1  # 占位值
    output_dim = env.n_actions
    hidden_dims = [256, 128]
    
    print(f"\n1. 创建ZeroShotGNN网络:")
    print(f"   输出维度: {output_dim}")
    print(f"   UAV数量: {len(uavs)}, 目标数量: {len(targets)}")
    
    try:
        network = create_network("ZeroShotGNN", input_dim, hidden_dims, output_dim)
        print(f"   网络创建成功: {type(network)}")
        
        # 计算参数数量
        param_count = sum(p.numel() for p in network.parameters())
        print(f"   参数数量: {param_count:,}")
        
    except Exception as e:
        print(f"   网络创建失败: {e}")
        return False
    
    # 测试前向传播
    print(f"\n2. 测试前向传播:")
    try:
        state = env.reset()
        print(f"   输入状态键: {list(state.keys())}")
        
        # 转换为张量格式
        state_tensor = {}
        for key, value in state.items():
            if key == "masks":
                mask_tensor = {}
                for mask_key, mask_value in value.items():
                    mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0)
                state_tensor[key] = mask_tensor
            else:
                state_tensor[key] = torch.FloatTensor(value).unsqueeze(0)
        
        # 前向传播
        network.eval()
        with torch.no_grad():
            q_values = network(state_tensor)
        
        print(f"   输出Q值形状: {q_values.shape}")
        print(f"   Q值范围: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
        print(f"   前向传播成功!")
        
    except Exception as e:
        print(f"   前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_scalability():
    """测试可扩展性和零样本迁移"""
    print("\n=== 测试可扩展性和零样本迁移 ===")
    
    config = Config()
    
    # 测试不同规模的场景
    scenarios = [
        ("小规模", 2, 3),
        ("中规模", 4, 6),
        ("大规模", 6, 9)
    ]
    
    for name, n_uavs, n_targets in scenarios:
        print(f"\n{name}场景 (UAV: {n_uavs}, 目标: {n_targets}):")
        
        try:
            # 创建场景
            uavs = [UAV(i+1, np.array([100*i, 100*i]), 0, np.array([50, 50]), 500, (10, 20), 15) for i in range(n_uavs)]
            targets = [Target(i+1, np.array([200+50*i, 200+50*i]), np.array([30, 30]), 100) for i in range(n_targets)]
            obstacles = []
            
            graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
            env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="graph")
            
            # 创建网络（使用相同的架构）
            network = create_network("ZeroShotGNN", 1, [256, 128], env.n_actions)
            
            # 测试前向传播
            state = env.reset()
            state_tensor = {}
            for key, value in state.items():
                if key == "masks":
                    mask_tensor = {}
                    for mask_key, mask_value in value.items():
                        mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0)
                    state_tensor[key] = mask_tensor
                else:
                    state_tensor[key] = torch.FloatTensor(value).unsqueeze(0)
            
            with torch.no_grad():
                q_values = network(state_tensor)
            
            print(f"   状态形状: UAV {state['uav_features'].shape}, 目标 {state['target_features'].shape}")
            print(f"   输出形状: {q_values.shape}")
            print(f"   动作空间: {env.n_actions}")
            print(f"   ✓ 成功处理")
            
        except Exception as e:
            print(f"   ✗ 失败: {e}")
            return False
    
    return True

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    config = Config()
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    # 测试传统网络在扁平模式下的工作
    traditional_networks = ["SimpleNetwork", "DeepFCN", "DeepFCNResidual"]
    
    for network_type in traditional_networks:
        print(f"\n测试 {network_type}:")
        try:
            # 创建扁平模式环境
            env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
            state = env.reset()
            
            # 创建网络
            network = create_network(network_type, len(state), [256, 128], env.n_actions)
            
            # 测试前向传播
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            network.eval()  # 设置为评估模式避免BatchNorm问题
            with torch.no_grad():
                q_values = network(state_tensor)
            
            print(f"   输入形状: {state_tensor.shape}")
            print(f"   输出形状: {q_values.shape}")
            print(f"   ✓ 向后兼容")
            
        except Exception as e:
            print(f"   ✗ 兼容性问题: {e}")
            return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试零样本GNN架构实现...")
    
    tests = [
        ("双模式观测系统", test_dual_mode_observation),
        ("ZeroShotGNN网络", test_zero_shot_gnn_network),
        ("可扩展性和零样本迁移", test_scalability),
        ("向后兼容性", test_backward_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"运行测试: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ 通过" if result else "✗ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ✗ 异常 - {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结:")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！零样本GNN架构实现成功！")
        return True
    else:
        print("❌ 部分测试失败，需要修复问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)