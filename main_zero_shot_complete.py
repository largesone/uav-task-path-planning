# -*- coding: utf-8 -*-
# 文件名: main_zero_shot_complete.py
# 描述: 完整的零样本迁移多无人机协同任务分配系统

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import json
import pickle
from typing import Dict, List, Tuple, Any

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 本地模块导入
from entities import UAV, Target
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario
from config import Config
from environment import UAVTaskEnv, DirectedGraph
from zero_shot_network import ZeroShotTransferNetwork
from zero_shot_environment import ZeroShotEnvironmentAdapter

def set_chinese_font():
    """设置matplotlib支持中文显示的字体"""
    try:
        plt.rcParams["font.family"] = "SimHei"
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception:
        try:
            plt.rcParams["font.family"] = "Microsoft YaHei"
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception:
            return False

class ZeroShotRLSolver:
    """零样本强化学习求解器"""
    
    def __init__(self, config, network_type="ZeroShotGNN"):
        self.config = config
        self.network_type = network_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建零样本网络
        self.policy_net = ZeroShotTransferNetwork(
            entity_feature_dim=64,
            hidden_dim=128,
            num_attention_heads=4,
            num_layers=2,
            max_entities=25,  # 最大支持10个UAV + 15个目标
            dropout=0.1
        ).to(self.device)
        
        self.target_net = ZeroShotTransferNetwork(
            entity_feature_dim=64,
            hidden_dim=128,
            num_attention_heads=4,
            num_layers=2,
            max_entities=25,
            dropout=0.1
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = deque(maxlen=config.MEMORY_CAPACITY)
        
        # 训练参数
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        # 当前环境适配器
        self.current_adapter = None
        
        print(f"[ZeroShotRLSolver] 初始化完成")
        print(f"  - 网络类型: {network_type}")
        print(f"  - 设备: {self.device}")
        print(f"  - 参数数量: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def set_environment(self, env_adapter: ZeroShotEnvironmentAdapter):
        """设置当前环境适配器"""
        self.current_adapter = env_adapter
    
    def select_action(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """选择动作"""
        if random.random() < self.epsilon:
            n_actions = self.current_adapter.n_actions if self.current_adapter else 48
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state_dict)
            self.policy_net.train()
            
            # 确保Q值维度与动作空间匹配
            if self.current_adapter and q_values.shape[1] > self.current_adapter.n_actions:
                q_values = q_values[:, :self.current_adapter.n_actions]
            
            return q_values.max(1)[1].view(1, 1)
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return None
        
        transitions = random.sample(self.memory, self.config.BATCH_SIZE)
        batch = list(zip(*transitions))
        
        # 分离状态字典的各个组件
        state_batch = self._merge_state_dicts(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = self._merge_state_dicts(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算目标Q值
        next_q_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_q_values[~done_batch] = self.target_net(next_state_batch).max(1)[0][~done_batch]
        
        expected_q_values = reward_batch + (self.config.GAMMA * next_q_values)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _merge_state_dicts(self, state_dict_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """合并状态字典列表 - 支持不同维度的状态"""
        if not state_dict_list:
            return {}
        
        merged = {}
        for key in state_dict_list[0].keys():
            tensors = [state_dict[key] for state_dict in state_dict_list]
            
            # 检查是否所有张量的形状都相同
            shapes = [tensor.shape for tensor in tensors]
            if all(shape == shapes[0] for shape in shapes):
                # 形状相同，直接拼接
                merged[key] = torch.cat(tensors, dim=0)
            else:
                # 形状不同，需要填充到相同大小
                max_shape = list(shapes[0])
                for shape in shapes[1:]:
                    for i in range(1, len(shape)):  # 跳过batch维度
                        max_shape[i] = max(max_shape[i], shape[i])
                
                # 填充所有张量到最大形状
                padded_tensors = []
                for tensor in tensors:
                    if list(tensor.shape) == max_shape:
                        padded_tensors.append(tensor)
                    else:
                        # 计算需要填充的大小
                        pad_sizes = []
                        for i in range(len(tensor.shape) - 1, 0, -1):  # 从最后一维开始
                            pad_size = max_shape[i] - tensor.shape[i]
                            pad_sizes.extend([0, pad_size])
                        
                        # 使用零填充
                        padded_tensor = torch.nn.functional.pad(tensor, pad_sizes, value=0)
                        padded_tensors.append(padded_tensor)
                
                merged[key] = torch.cat(padded_tensors, dim=0)
        
        return merged
    
    def train_on_scenario(self, scenario_name: str, episodes: int = 200) -> Tuple[float, List[float]]:
        """在特定场景上训练"""
        print(f"在场景 {scenario_name} 上训练 {episodes} 轮")
        
        # 创建场景
        uavs, targets, obstacles = self._create_scenario(scenario_name)
        
        # 创建环境适配器
        config = Config()
        graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
        base_env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
        adapter = ZeroShotEnvironmentAdapter(base_env)
        self.set_environment(adapter)
        
        start_time = time.time()
        episode_rewards = []
        
        for i_episode in range(episodes):
            state_dict = adapter.reset()
            episode_reward = 0
            
            for step in range(100):  # 限制每轮最大步数
                action = self.select_action(state_dict)
                next_state_dict, reward, done, truncated, _ = adapter.step(action.item())
                episode_reward += reward
                
                # 存储经验
                self.memory.append((
                    state_dict,
                    action,
                    torch.tensor([reward], device=self.device),
                    next_state_dict,
                    done
                ))
                
                # 优化模型
                if len(self.memory) >= self.config.BATCH_SIZE:
                    self.optimize_model()
                
                state_dict = next_state_dict
                
                if done or truncated:
                    break
            
            # 更新目标网络
            if i_episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            episode_rewards.append(episode_reward)
            
            if i_episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"  Episode {i_episode}, 平均奖励: {avg_reward:.2f}, 探索率: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        print(f"场景 {scenario_name} 训练完成，耗时: {training_time:.2f}秒")
        
        return training_time, episode_rewards
    
    def _create_scenario(self, scenario_name: str):
        """创建场景"""
        if scenario_name == "small":
            return get_small_scenario(obstacle_tolerance=50.0)
        elif scenario_name == "balanced":
            return get_balanced_scenario(obstacle_tolerance=50.0)
        elif scenario_name == "complex":
            return get_complex_scenario()
        else:
            return get_small_scenario(obstacle_tolerance=50.0)
    
    def get_task_assignments(self, scenario_name: str) -> Dict[int, List[Tuple[int, int]]]:
        """获取任务分配"""
        # 创建场景
        uavs, targets, obstacles = self._create_scenario(scenario_name)
        
        # 创建环境适配器
        config = Config()
        graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
        base_env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
        adapter = ZeroShotEnvironmentAdapter(base_env)
        self.set_environment(adapter)
        
        self.policy_net.eval()
        state_dict = adapter.reset()
        assignments = {u.id: [] for u in uavs}
        
        for step in range(len(targets) * len(uavs)):
            with torch.no_grad():
                q_values = self.policy_net(state_dict)
                action_idx = q_values.max(1)[1].item()
            
            # 解码动作
            uav_idx, target_idx, direction_idx = adapter._decode_action(action_idx)
            
            if uav_idx < len(uavs) and target_idx < len(targets):
                uav_id = uavs[uav_idx].id
                target_id = targets[target_idx].id
                assignments[uav_id].append((target_id, direction_idx))
            
            next_state_dict, _, done, _, _ = adapter.step(action_idx)
            if done:
                break
            
            state_dict = next_state_dict
        
        return assignments

def enhanced_evaluate_plan(task_assignments: Dict[int, List[Tuple[int, int]]], 
                          uavs: List[UAV], 
                          targets: List[Target]) -> Dict[str, float]:
    """增强的计划评估函数"""
    if not task_assignments or not any(task_assignments.values()):
        return {
            'completion_rate': 0.0,
            'total_assignments': 0,
            'resource_utilization': 0.0,
            'target_coverage': 0.0,
            'efficiency_score': 0.0
        }
    
    # 基础统计
    total_assignments = sum(len(assignments) for assignments in task_assignments.values())
    assigned_targets = set()
    assigned_uavs = set()
    
    for uav_id, assignments in task_assignments.items():
        if assignments:
            assigned_uavs.add(uav_id)
            for target_id, _ in assignments:
                assigned_targets.add(target_id)
    
    # 计算各项指标
    target_coverage = len(assigned_targets) / len(targets) if targets else 0
    uav_utilization = len(assigned_uavs) / len(uavs) if uavs else 0
    
    # 资源匹配度评估
    resource_match_score = 0.0
    if targets and uavs:
        total_demand = sum(np.sum(target.resources) for target in targets)
        total_supply = sum(np.sum(uav.resources) for uav in uavs)
        resource_match_score = min(1.0, total_supply / total_demand) if total_demand > 0 else 1.0
    
    # 综合完成率
    completion_rate = (target_coverage * 0.4 + uav_utilization * 0.3 + resource_match_score * 0.3)
    
    # 效率分数
    efficiency_score = total_assignments / (len(uavs) * len(targets)) if (uavs and targets) else 0
    
    return {
        'completion_rate': completion_rate,
        'total_assignments': total_assignments,
        'resource_utilization': uav_utilization,
        'target_coverage': target_coverage,
        'efficiency_score': efficiency_score,
        'resource_match_score': resource_match_score
    }

def enhanced_visualize_results(task_assignments: Dict[int, List[Tuple[int, int]]], 
                              uavs: List[UAV], 
                              targets: List[Target], 
                              obstacles: List, 
                              scenario_name: str, 
                              training_time: float,
                              evaluation_metrics: Dict[str, float],
                              network_type: str = "ZeroShotGNN"):
    """增强的结果可视化"""
    set_chinese_font()
    
    # 创建更大的图形，包含详细信息
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # === 左侧：任务分配可视化 ===
    # 绘制障碍物
    for obs in obstacles:
        if hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center, obs.radius, color='gray', alpha=0.3)
            ax1.add_patch(circle)
        elif hasattr(obs, 'vertices'):
            from matplotlib.patches import Polygon
            polygon = Polygon(obs.vertices, color='gray', alpha=0.3)
            ax1.add_patch(polygon)
    
    # 绘制UAV
    uav_colors = plt.cm.Set3(np.linspace(0, 1, len(uavs)))
    for i, uav in enumerate(uavs):
        ax1.scatter(uav.position[0], uav.position[1], c=[uav_colors[i]], 
                   marker='s', s=150, label=f'UAV{uav.id}' if i < 5 else "", 
                   edgecolors='black', linewidth=2)
        ax1.annotate(f'UAV{uav.id}', (uav.position[0], uav.position[1]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # 显示UAV资源信息
        resource_text = f"R:[{uav.resources[0]:.0f},{uav.resources[1]:.0f}]"
        ax1.annotate(resource_text, (uav.position[0], uav.position[1]), 
                    xytext=(5, -15), textcoords='offset points', fontsize=8)
    
    # 绘制目标
    for target in targets:
        ax1.scatter(target.position[0], target.position[1], c='red', 
                   marker='o', s=150, alpha=0.8, edgecolors='black', linewidth=2)
        ax1.annotate(f'T{target.id}', (target.position[0], target.position[1]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # 显示目标需求信息
        demand_text = f"D:[{target.resources[0]:.0f},{target.resources[1]:.0f}]"
        ax1.annotate(demand_text, (target.position[0], target.position[1]), 
                    xytext=(5, -15), textcoords='offset points', fontsize=8)
    
    # 绘制任务分配连线
    for i, (uav_id, assignments) in enumerate(task_assignments.items()):
        uav = next((u for u in uavs if u.id == uav_id), None)
        if not uav or not assignments:
            continue
            
        color = uav_colors[i % len(uav_colors)]
        
        for j, (target_id, direction_idx) in enumerate(assignments):
            target = next((t for t in targets if t.id == target_id), None)
            if not target:
                continue
                
            # 绘制连线
            ax1.plot([uav.position[0], target.position[0]], 
                    [uav.position[1], target.position[1]], 
                    color=color, alpha=0.7, linewidth=2 + j * 0.5)
            
            # 在连线中点添加方向标记
            mid_x = (uav.position[0] + target.position[0]) / 2
            mid_y = (uav.position[1] + target.position[1]) / 2
            ax1.annotate(f'D{direction_idx}', (mid_x, mid_y), 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8))
    
    ax1.set_title(f'{scenario_name} - {network_type} 任务分配结果\n训练时间: {training_time:.2f}秒', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('X坐标 (m)', fontsize=12)
    ax1.set_ylabel('Y坐标 (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # === 右侧：评估指标可视化 ===
    metrics_names = ['完成率', '目标覆盖率', 'UAV利用率', '资源匹配度', '效率分数']
    metrics_values = [
        evaluation_metrics['completion_rate'],
        evaluation_metrics['target_coverage'],
        evaluation_metrics['resource_utilization'],
        evaluation_metrics['resource_match_score'],
        evaluation_metrics['efficiency_score']
    ]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    metrics_values += metrics_values[:1]  # 闭合数据
    
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, metrics_values, 'o-', linewidth=2, color='blue', alpha=0.7)
    ax2.fill(angles, metrics_values, alpha=0.25, color='blue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_names, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title(f'性能评估雷达图\n总分配数: {evaluation_metrics["total_assignments"]}', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # 添加数值标签
    for angle, value, name in zip(angles[:-1], metrics_values[:-1], metrics_names):
        ax2.annotate(f'{value:.3f}', (angle, value), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, ha='center')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('output', exist_ok=True)
    save_path = f'output/{scenario_name}_{network_type}_enhanced_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"增强结果图已保存至: {save_path}")
    
    plt.show()

def run_zero_shot_scenario(scenario_name: str, episodes: int = 200) -> Dict[str, Any]:
    """运行零样本场景"""
    print(f"=" * 60)
    print(f"运行零样本场景: {scenario_name}")
    print(f"=" * 60)
    
    # 创建配置和求解器
    config = Config()
    solver = ZeroShotRLSolver(config, "ZeroShotGNN")
    
    # 训练
    training_time, episode_rewards = solver.train_on_scenario(scenario_name, episodes)
    
    # 获取任务分配
    task_assignments = solver.get_task_assignments(scenario_name)
    
    # 创建场景用于评估
    uavs, targets, obstacles = solver._create_scenario(scenario_name)
    
    # 评估
    evaluation_metrics = enhanced_evaluate_plan(task_assignments, uavs, targets)
    
    print(f"\n评估结果:")
    print(f"  完成率: {evaluation_metrics['completion_rate']:.3f}")
    print(f"  目标覆盖率: {evaluation_metrics['target_coverage']:.3f}")
    print(f"  UAV利用率: {evaluation_metrics['resource_utilization']:.3f}")
    print(f"  总分配数: {evaluation_metrics['total_assignments']}")
    
    # 可视化
    enhanced_visualize_results(task_assignments, uavs, targets, obstacles, 
                              scenario_name, training_time, evaluation_metrics, "ZeroShotGNN")
    
    return {
        'scenario_name': scenario_name,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'task_assignments': task_assignments,
        'evaluation_metrics': evaluation_metrics,
        'uavs': uavs,
        'targets': targets,
        'obstacles': obstacles
    }

def run_zero_shot_transfer_experiment():
    """运行零样本迁移实验"""
    print("=" * 80)
    print("零样本迁移实验")
    print("=" * 80)
    
    # 创建求解器
    config = Config()
    solver = ZeroShotRLSolver(config, "ZeroShotGNN")
    
    # 定义训练和测试场景
    training_scenarios = ["small", "balanced"]
    test_scenarios = ["small", "balanced", "complex"]
    
    results = {}
    
    # 阶段1：在小规模场景上训练
    print("\n阶段1：在训练场景上训练网络")
    for scenario in training_scenarios:
        print(f"\n训练场景: {scenario}")
        training_time, episode_rewards = solver.train_on_scenario(scenario, episodes=150)
        results[f"training_{scenario}"] = {
            'training_time': training_time,
            'episode_rewards': episode_rewards
        }
    
    # 阶段2：在所有场景上测试（包括未见过的复杂场景）
    print("\n阶段2：零样本迁移测试")
    for scenario in test_scenarios:
        print(f"\n测试场景: {scenario}")
        
        # 获取任务分配（不进行额外训练）
        task_assignments = solver.get_task_assignments(scenario)
        
        # 创建场景用于评估
        uavs, targets, obstacles = solver._create_scenario(scenario)
        
        # 评估
        evaluation_metrics = enhanced_evaluate_plan(task_assignments, uavs, targets)
        
        print(f"  完成率: {evaluation_metrics['completion_rate']:.3f}")
        print(f"  目标覆盖率: {evaluation_metrics['target_coverage']:.3f}")
        print(f"  UAV利用率: {evaluation_metrics['resource_utilization']:.3f}")
        
        # 可视化
        enhanced_visualize_results(task_assignments, uavs, targets, obstacles, 
                                  f"{scenario}_transfer", 0, evaluation_metrics, "ZeroShotGNN")
        
        results[f"test_{scenario}"] = {
            'task_assignments': task_assignments,
            'evaluation_metrics': evaluation_metrics,
            'scenario_info': {
                'num_uavs': len(uavs),
                'num_targets': len(targets),
                'num_obstacles': len(obstacles)
            }
        }
    
    # 生成迁移能力报告
    generate_transfer_report(results)
    
    return results

def generate_transfer_report(results: Dict[str, Any]):
    """生成迁移能力报告"""
    print("\n" + "=" * 60)
    print("零样本迁移能力报告")
    print("=" * 60)
    
    # 提取测试结果
    test_results = {k: v for k, v in results.items() if k.startswith('test_')}
    
    print(f"{'场景':<15} {'完成率':<10} {'目标覆盖':<10} {'UAV利用':<10} {'效率分数':<10}")
    print("-" * 60)
    
    for scenario_key, result in test_results.items():
        scenario_name = scenario_key.replace('test_', '')
        metrics = result['evaluation_metrics']
        
        print(f"{scenario_name:<15} {metrics['completion_rate']:<10.3f} "
              f"{metrics['target_coverage']:<10.3f} {metrics['resource_utilization']:<10.3f} "
              f"{metrics['efficiency_score']:<10.3f}")
    
    # 计算平均性能
    avg_completion = np.mean([r['evaluation_metrics']['completion_rate'] for r in test_results.values()])
    avg_coverage = np.mean([r['evaluation_metrics']['target_coverage'] for r in test_results.values()])
    avg_utilization = np.mean([r['evaluation_metrics']['resource_utilization'] for r in test_results.values()])
    
    print("-" * 60)
    print(f"{'平均性能':<15} {avg_completion:<10.3f} {avg_coverage:<10.3f} {avg_utilization:<10.3f}")
    
    # 保存报告
    os.makedirs('output', exist_ok=True)
    report_path = 'output/zero_shot_transfer_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                    for key, val in v.items() if key != 'task_assignments'  # 跳过复杂对象
                }
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n迁移能力报告已保存至: {report_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='零样本迁移多无人机协同任务分配系统')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'transfer'],
                       help='运行模式：single(单场景) 或 transfer(迁移实验)')
    parser.add_argument('--scenario', type=str, default='small',
                       choices=['small', 'balanced', 'complex'],
                       help='场景类型（仅在single模式下有效）')
    parser.add_argument('--episodes', type=int, default=200,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("零样本迁移多无人机协同任务分配系统")
    print("=" * 80)
    
    try:
        if args.mode == 'single':
            # 单场景模式
            result = run_zero_shot_scenario(args.scenario, args.episodes)
            
            print(f"\n执行完成!")
            print(f"训练时间: {result['training_time']:.2f}秒")
            print(f"最终平均奖励: {np.mean(result['episode_rewards'][-10:]):.2f}")
            
        elif args.mode == 'transfer':
            # 迁移实验模式
            results = run_zero_shot_transfer_experiment()
            
            print(f"\n迁移实验完成!")
            print(f"测试了 {len([k for k in results.keys() if k.startswith('test_')])} 个场景")
            
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
