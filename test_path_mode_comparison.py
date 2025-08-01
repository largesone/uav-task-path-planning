# -*- coding: utf-8 -*-
"""
路径模式对比测试脚本
对比简化欧几里得距离 vs 高精度PH-RRT路径规划的学习效果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_rl_solver import compare_path_modes, EnhancedRLSolver
from scenarios import get_simple_convergence_test_scenario as create_test_scenario
from config import Config
import os

def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def test_single_path_mode():
    """测试单个路径模式"""
    print("=== 测试单个路径模式 ===")
    
    config = Config()
    uavs, targets, obstacles = create_test_scenario()
    
    # 创建图结构
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 测试欧几里得距离模式
    print("\n1. 测试欧几里得距离模式...")
    euclidean_solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                       network_type='DeepFCN', use_ph_rrt=False)
    
    # 训练
    euclidean_history = euclidean_solver.train(episodes=100, log_interval=10)
    
    # 绘制训练历史
    euclidean_solver.plot_training_history('output/images/euclidean_training.png')
    
    # 测试PH-RRT模式
    print("\n2. 测试PH-RRT模式...")
    phrrt_solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                   network_type='DeepFCN', use_ph_rrt=True)
    
    # 训练
    phrrt_history = phrrt_solver.train(episodes=100, log_interval=10)
    
    # 绘制训练历史
    phrrt_solver.plot_training_history('output/images/phrrt_training.png')
    
    return euclidean_solver, phrrt_solver

def test_path_mode_comparison():
    """测试路径模式对比"""
    print("=== 路径模式对比实验 ===")
    
    config = Config()
    uavs, targets, obstacles = create_test_scenario()
    
    # 创建图结构
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 运行对比实验
    comparison_results, euclidean_solver, phrrt_solver = compare_path_modes(
        uavs, targets, graph, obstacles, config, episodes=200, network_type='DeepFCN'
    )
    
    return comparison_results, euclidean_solver, phrrt_solver

def test_network_comparison_with_path_modes():
    """测试不同网络结构在不同路径模式下的表现"""
    print("=== 网络结构与路径模式组合测试 ===")
    
    config = Config()
    uavs, targets, obstacles = create_test_scenario()
    
    # 创建图结构
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    network_types = ['DeepFCN', 'DeepFCN_Residual', 'GAT']
    path_modes = [False, True]  # False=欧几里得, True=PH-RRT
    
    results = {}
    
    for network_type in network_types:
        for use_ph_rrt in path_modes:
            print(f"\n测试网络: {network_type}, 路径模式: {'PH-RRT' if use_ph_rrt else 'Euclidean'}")
            
            solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                    network_type, use_ph_rrt)
            
            # 训练
            history = solver.train(episodes=150, log_interval=10)
            
            # 记录结果
            key = f"{network_type}_{'PH-RRT' if use_ph_rrt else 'Euclidean'}"
            results[key] = {
                'final_avg_reward': np.mean(history['episode_rewards'][-20:]),
                'max_reward': max(history['episode_rewards']),
                'convergence_episode': len(history['episode_rewards']),
                'final_epsilon': history['epsilon_history'][-1],
                'history': history
            }
            
            # 保存模型
            model_path = f"output/models/{key}_model.pth"
            solver.save_model(model_path)
            
            # 绘制训练历史
            plot_path = f"output/images/{key}_training.png"
            solver.plot_training_history(plot_path)
    
    # 绘制综合对比图
    plot_comprehensive_comparison(results)
    
    return results

def plot_comprehensive_comparison(results):
    """绘制综合对比图"""
    set_chinese_font()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 准备数据
    labels = list(results.keys())
    final_rewards = [results[key]['final_avg_reward'] for key in labels]
    max_rewards = [results[key]['max_reward'] for key in labels]
    convergence_episodes = [results[key]['convergence_episode'] for key in labels]
    
    # 最终平均奖励对比
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0, 0].bar(x, final_rewards, width, alpha=0.7)
    axes[0, 0].set_title('最终平均奖励对比')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    
    # 最大奖励对比
    axes[0, 1].bar(x, max_rewards, width, alpha=0.7, color='orange')
    axes[0, 1].set_title('最大奖励对比')
    axes[0, 1].set_ylabel('奖励')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    
    # 收敛轮次对比
    axes[1, 0].bar(x, convergence_episodes, width, alpha=0.7, color='green')
    axes[1, 0].set_title('收敛轮次对比')
    axes[1, 0].set_ylabel('轮次')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
    
    # 奖励曲线对比
    for key, result in results.items():
        history = result['history']
        axes[1, 1].plot(history['episode_rewards'], label=key, alpha=0.7)
    
    axes[1, 1].set_title('训练奖励曲线对比')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('奖励')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/images/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细结果
    print("\n=== 综合对比结果 ===")
    for key, result in results.items():
        print(f"\n{key}:")
        print(f"  最终平均奖励: {result['final_avg_reward']:.2f}")
        print(f"  最大奖励: {result['max_reward']:.2f}")
        print(f"  收敛轮次: {result['convergence_episode']}")
        print(f"  最终探索率: {result['final_epsilon']:.3f}")

def test_inference_comparison():
    """测试推理阶段两种路径模式的效果"""
    print("=== 推理阶段路径模式对比 ===")
    
    config = Config()
    uavs, targets, obstacles = create_test_scenario()
    
    # 创建图结构
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 训练两个模型
    euclidean_solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                       network_type='DeepFCN', use_ph_rrt=False)
    phrrt_solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                   network_type='DeepFCN', use_ph_rrt=True)
    
    # 训练
    print("训练欧几里得距离模型...")
    euclidean_solver.train(episodes=100)
    
    print("训练PH-RRT模型...")
    phrrt_solver.train(episodes=100)
    
    # 推理测试
    inference_results = {}
    
    for solver, mode in [(euclidean_solver, 'Euclidean'), (phrrt_solver, 'PH-RRT')]:
        print(f"\n测试{mode}模式推理效果...")
        
        # 重置环境
        solver._reset_environment()
        state = solver.get_state()
        episode_reward = 0
        episode_length = 0
        assignments = []
        
        while True:
            # 选择动作（推理模式）
            action = solver.select_action(state, training=False)
            
            # 执行动作
            next_state, reward, done, info = solver.step(action)
            
            # 记录分配
            target_idx, uav_idx, phi_idx = solver._action_to_assignment(action)
            assignments.append({
                'uav_id': uav_idx,
                'target_id': target_idx,
                'phi_idx': phi_idx,
                'reward': reward,
                'path_length': info.get('path_length', 0.0)
            })
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        inference_results[mode] = {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'assignments': assignments,
            'avg_path_length': np.mean([a['path_length'] for a in assignments]),
            'total_path_length': sum([a['path_length'] for a in assignments])
        }
    
    # 打印推理结果
    print("\n=== 推理结果对比 ===")
    for mode, result in inference_results.items():
        print(f"\n{mode}模式:")
        print(f"  总奖励: {result['total_reward']:.2f}")
        print(f"  任务数量: {result['episode_length']}")
        print(f"  平均路径长度: {result['avg_path_length']:.2f}")
        print(f"  总路径长度: {result['total_path_length']:.2f}")
    
    return inference_results

def main():
    """主函数"""
    print("开始路径模式对比实验...")
    
    # 创建输出目录
    os.makedirs('output/images', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    
    # 设置中文字体
    set_chinese_font()
    
    # 运行各种测试
    print("\n1. 测试单个路径模式...")
    euclidean_solver, phrrt_solver = test_single_path_mode()
    
    print("\n2. 测试路径模式对比...")
    comparison_results, _, _ = test_path_mode_comparison()
    
    print("\n3. 测试网络结构与路径模式组合...")
    network_results = test_network_comparison_with_path_modes()
    
    print("\n4. 测试推理阶段对比...")
    inference_results = test_inference_comparison()
    
    print("\n=== 实验完成 ===")
    print("所有结果已保存到 output/ 目录")

if __name__ == "__main__":
    main() 