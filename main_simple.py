# -*- coding: utf-8 -*-
# 文件名: main_simple.py
# 描述: 简化版本的多无人机协同任务分配与路径规划系统

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 本地模块导入
from entities import UAV, Target
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario
from config import Config
from networks import create_network
from environment import UAVTaskEnv, DirectedGraph

def set_chinese_font():
    """设置matplotlib支持中文显示的字体"""
    try:
        plt.rcParams["font.family"] = "SimHei"
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception:
        return False

class SimpleRLSolver:
    """简化的强化学习求解器"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, network_type="SimpleNetwork"):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.network_type = network_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建简化网络（不使用BatchNorm）
        self.policy_net = self._create_simple_network(i_dim, h_dim, o_dim).to(self.device)
        self.target_net = self._create_simple_network(i_dim, h_dim, o_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = deque(maxlen=config.MEMORY_CAPACITY)
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        # 环境
        self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
        
        # 动作映射
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}
        self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
    
    def _create_simple_network(self, input_dim, hidden_dims, output_dim):
        """创建简化的网络（不使用BatchNorm）"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def select_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.env.n_actions)]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.max(1)[1].view(1, 1)
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        
        transitions = random.sample(self.memory, self.config.BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_states_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_q_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_q_values[~done_batch] = self.target_net(next_states_batch[~done_batch]).max(1)[0]
        
        expected_q_values = reward_batch + (self.config.GAMMA * next_q_values)
        
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes):
        """训练模型"""
        print(f"开始训练 {self.network_type} 网络，训练轮数: {episodes}")
        start_time = time.time()
        
        episode_rewards = []
        
        for i_episode in range(episodes):
            state = self.env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            episode_reward = 0
            
            for step in range(100):  # 限制每轮最大步数
                action = self.select_action(state_tensor)
                next_state, reward, done, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                self.memory.append((
                    state_tensor,
                    action,
                    torch.tensor([reward], device=self.device),
                    next_state_tensor,
                    done
                ))
                
                if len(self.memory) >= self.config.BATCH_SIZE:
                    self.optimize_model()
                
                state_tensor = next_state_tensor
                
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
                print(f"Episode {i_episode}, 平均奖励: {avg_reward:.2f}, 探索率: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        return training_time, episode_rewards
    
    def get_task_assignments(self):
        """获取任务分配"""
        self.policy_net.eval()
        state = self.env.reset()
        assignments = {u.id: [] for u in self.env.uavs}
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        for step in range(len(self.env.targets) * len(self.env.uavs)):
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
            
            # 将动作索引转换为具体动作
            n_u, n_p = len(self.env.uavs), self.graph.n_phi
            t_idx = action_idx // (n_u * n_p)
            u_idx = (action_idx % (n_u * n_p)) // n_p
            p_idx = action_idx % n_p
            
            if t_idx < len(self.env.targets) and u_idx < len(self.env.uavs):
                target_id = self.env.targets[t_idx].id
                uav_id = self.env.uavs[u_idx].id
                assignments[uav_id].append((target_id, p_idx))
            
            next_state, _, done, _, _ = self.env.step(action_idx)
            if done:
                break
            
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        return assignments

def simple_evaluate_plan(task_assignments, uavs, targets):
    """简单的计划评估函数"""
    if not task_assignments or not any(task_assignments.values()):
        return {
            'completion_rate': 0.0,
            'total_assignments': 0
        }
    
    total_assignments = sum(len(assignments) for assignments in task_assignments.values())
    completion_rate = min(1.0, total_assignments / len(targets))
    
    return {
        'completion_rate': completion_rate,
        'total_assignments': total_assignments
    }

def visualize_results(task_assignments, uavs, targets, obstacles, scenario_name, training_time):
    """可视化结果"""
    set_chinese_font()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制障碍物
    for obs in obstacles:
        if hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center, obs.radius, color='gray', alpha=0.3)
            ax.add_patch(circle)
    
    # 绘制UAV
    for uav in uavs:
        ax.scatter(uav.position[0], uav.position[1], c='blue', marker='s', s=100, label='UAV' if uav == uavs[0] else "")
        ax.annotate(f'UAV{uav.id}', (uav.position[0], uav.position[1]), xytext=(5, 5), textcoords='offset points')
    
    # 绘制目标
    for target in targets:
        ax.scatter(target.position[0], target.position[1], c='red', marker='o', s=100, label='Target' if target == targets[0] else "")
        ax.annotate(f'T{target.id}', (target.position[0], target.position[1]), xytext=(5, 5), textcoords='offset points')
    
    # 绘制任务分配连线
    colors = ['green', 'orange', 'purple', 'brown', 'pink']
    for i, (uav_id, assignments) in enumerate(task_assignments.items()):
        uav = next(u for u in uavs if u.id == uav_id)
        color = colors[i % len(colors)]
        
        for target_id, _ in assignments:
            target = next(t for t in targets if t.id == target_id)
            ax.plot([uav.position[0], target.position[0]], 
                   [uav.position[1], target.position[1]], 
                   color=color, alpha=0.6, linewidth=2)
    
    ax.set_title(f'{scenario_name} - 任务分配结果\n训练时间: {training_time:.2f}秒')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 保存图片
    os.makedirs('output', exist_ok=True)
    save_path = f'output/{scenario_name}_simple_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"结果图已保存至: {save_path}")
    
    plt.show()

def run_simple_scenario(scenario_name, network_type="SimpleNetwork", episodes=200):
    """运行简化场景"""
    print(f"运行场景: {scenario_name}")
    print(f"网络类型: {network_type}")
    
    # 创建配置
    config = Config()
    
    # 加载场景
    if scenario_name == "small":
        uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    elif scenario_name == "balanced":
        uavs, targets, obstacles = get_balanced_scenario(obstacle_tolerance=50.0)
    elif scenario_name == "complex":
        uavs, targets, obstacles = get_complex_scenario()
    else:
        print(f"未知场景: {scenario_name}，使用小规模场景")
        uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    
    print(f"场景信息: {len(uavs)} UAV, {len(targets)} 目标, {len(obstacles)} 障碍物")
    
    # 创建图
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    # 创建求解器
    test_env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
    test_state = test_env.reset()
    
    i_dim = len(test_state)
    h_dim = [256, 128]
    o_dim = test_env.n_actions
    
    print(f"网络维度: 输入{i_dim}, 隐藏{h_dim}, 输出{o_dim}")
    
    solver = SimpleRLSolver(uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, network_type)
    
    # 训练
    training_time, episode_rewards = solver.train(episodes)
    
    # 获取任务分配
    task_assignments = solver.get_task_assignments()
    
    # 评估
    evaluation_metrics = simple_evaluate_plan(task_assignments, uavs, targets)
    
    print(f"评估结果:")
    print(f"  完成率: {evaluation_metrics['completion_rate']:.3f}")
    print(f"  总分配数: {evaluation_metrics['total_assignments']}")
    
    # 可视化
    visualize_results(task_assignments, uavs, targets, obstacles, scenario_name, training_time)
    
    return {
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'task_assignments': task_assignments,
        'evaluation_metrics': evaluation_metrics
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='简化版多无人机协同任务分配系统')
    parser.add_argument('--network', type=str, default='SimpleNetwork',
                       choices=['SimpleNetwork', 'DeepFCN', 'DeepFCNResidual'],
                       help='网络类型')
    parser.add_argument('--scenario', type=str, default='small',
                       choices=['small', 'balanced', 'complex'],
                       help='场景类型')
    parser.add_argument('--episodes', type=int, default=200,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("简化版多无人机协同任务分配系统")
    print("=" * 60)
    
    try:
        result = run_simple_scenario(args.scenario, args.network, args.episodes)
        
        print(f"\n执行完成!")
        print(f"训练时间: {result['training_time']:.2f}秒")
        print(f"最终平均奖励: {np.mean(result['episode_rewards'][-10:]):.2f}")
        
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