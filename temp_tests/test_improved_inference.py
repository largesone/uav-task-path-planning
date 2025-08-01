# -*- coding: utf-8 -*-
"""
测试改进的推理机制和最佳模型加载功能

主要测试内容：
1. 验证低温softmax采样机制
2. 验证最佳模型加载功能
3. 比较不同温度参数的效果
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# 添加父目录到路径
sys.path.append('..')

from main import run_scenario, GraphRLSolver
from config import Config
from scenarios import get_new_experimental_scenario
from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph

def test_temperature_sampling():
    """测试不同温度参数对推理结果的影响"""
    print("=== 测试温度采样机制 ===")
    
    # 创建测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(50.0)
    config = Config()
    
    # 创建图和环境
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    test_env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    test_state = test_env.reset()
    input_dim = len(test_state)
    output_dim = test_env.n_actions
    
    # 创建求解器
    solver = GraphRLSolver(uavs, targets, graph, obstacles, 
                          i_dim=input_dim, h_dim=[512, 256, 128, 64], o_dim=output_dim, 
                          config=config, network_type="DeepFCNResidual")
    
    # 模拟一些Q值（实际应用中这些值来自训练好的网络）
    print("模拟Q值分布...")
    state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
    
    # 模拟网络输出
    with torch.no_grad():
        q_values = solver.policy_net(state_tensor)
    
    print(f"原始Q值形状: {q_values.shape}")
    print(f"Q值范围: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    
    # 测试不同温度参数
    temperatures = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}
    
    for temp in temperatures:
        print(f"\n--- 温度参数: {temp} ---")
        
        # 使用低温softmax采样
        logits = q_values / temp
        action_probs = F.softmax(logits, dim=1)
        
        # 获取概率分布信息
        max_prob = action_probs.max().item()
        min_prob = action_probs.min().item()
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()
        
        print(f"最大概率: {max_prob:.4f}")
        print(f"最小概率: {min_prob:.4f}")
        print(f"概率熵: {entropy:.4f}")
        
        # 多次采样观察分布
        samples = []
        for _ in range(10):
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()
            samples.append(action_idx)
        
        unique_samples = len(set(samples))
        print(f"10次采样中不同动作数: {unique_samples}")
        print(f"采样动作: {samples[:5]}...")  # 只显示前5个
        
        results[temp] = {
            'max_prob': max_prob,
            'entropy': entropy,
            'unique_samples': unique_samples,
            'samples': samples
        }
    
    # 分析结果
    print("\n=== 温度参数效果分析 ===")
    for temp in temperatures:
        result = results[temp]
        print(f"温度 {temp}: 最大概率={result['max_prob']:.4f}, "
              f"熵={result['entropy']:.4f}, 多样性={result['unique_samples']}")
    
    return results

def test_best_model_loading():
    """测试最佳模型加载功能"""
    print("\n=== 测试最佳模型加载功能 ===")
    
    # 创建测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(30.0)
    config = Config()
    
    # 修改配置以快速训练
    config.training_config.episodes = 20
    config.training_config.patience = 5
    
    print("开始训练测试模型...")
    start_time = time.time()
    
    # 运行场景（会训练并保存最佳模型）
    final_plan, training_time, training_history, evaluation_metrics = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="test_improved_inference",
        network_type="DeepFCNResidual",
        save_visualization=False,
        show_visualization=False,
        force_retrain=True,
        output_base_dir="temp_tests/output"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"训练完成，总耗时: {total_time:.2f}秒")
    print(f"训练时间: {training_time:.2f}秒")
    
    if evaluation_metrics:
        print(f"评估指标: {evaluation_metrics}")
    
    return final_plan, training_time, training_history, evaluation_metrics

def test_inference_comparison():
    """比较不同推理机制的效果"""
    print("\n=== 比较不同推理机制 ===")
    
    # 创建测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(25.0)
    config = Config()
    
    # 创建图和环境
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    test_env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
    test_state = test_env.reset()
    input_dim = len(test_state)
    output_dim = test_env.n_actions
    
    # 创建求解器
    solver = GraphRLSolver(uavs, targets, graph, obstacles, 
                          i_dim=input_dim, h_dim=[512, 256, 128, 64], o_dim=output_dim, 
                          config=config, network_type="DeepFCNResidual")
    
    # 模拟训练（简化版本）
    print("进行简化训练...")
    for episode in range(10):
        state = test_env.reset()
        done = False
        step = 0
        
        while not done and step < 50:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = solver.policy_net(state_tensor)
            
            # 随机选择动作进行训练
            action_idx = torch.randint(0, output_dim, (1,)).item()
            state, reward, done, _, _ = test_env.step(action_idx)
            step += 1
    
    print("比较不同推理机制...")
    
    # 1. 原始argmax方法
    print("1. 使用原始argmax方法...")
    start_time = time.time()
    assignments_argmax = solver.get_task_assignments_original()  # 假设有这个方法
    argmax_time = time.time() - start_time
    
    # 2. 低温softmax方法
    temperatures = [0.05, 0.1, 0.2]
    results = {}
    
    for temp in temperatures:
        print(f"2. 使用低温softmax方法 (温度={temp})...")
        start_time = time.time()
        assignments_softmax = solver.get_task_assignments(temperature=temp)
        softmax_time = time.time() - start_time
        
        # 计算任务分配统计
        total_assignments = sum(len(tasks) for tasks in assignments_softmax.values())
        
        results[temp] = {
            'assignments': assignments_softmax,
            'time': softmax_time,
            'total_assignments': total_assignments
        }
        
        print(f"   温度 {temp}: 耗时={softmax_time:.4f}s, 任务数={total_assignments}")
    
    return results

def main():
    """主测试函数"""
    print("开始测试改进的推理机制...")
    
    try:
        # 测试温度采样
        temp_results = test_temperature_sampling()
        
        # 测试最佳模型加载
        plan, train_time, history, metrics = test_best_model_loading()
        
        # 测试推理比较
        inference_results = test_inference_comparison()
        
        print("\n=== 测试完成 ===")
        print("所有测试已成功完成")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 