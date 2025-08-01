# -*- coding: utf-8 -*-
"""
增强的强化学习求解器
支持两种路径模式对比：简化欧几里得距离 vs 高精度PH-RRT路径规划
重点优化：死锁检测、资源满足、避免零贡献、协作奖励
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import time
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from path_planning import PHCurveRRTPlanner, RRTPlanner
from entities import UAV, Target
from config import Config

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedNetwork(nn.Module):
    """增强的网络结构，支持路径模式对比"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, network_type='DeepFCN'):
        super(EnhancedNetwork, self).__init__()
        self.network_type = network_type
        
        if network_type == 'DeepFCN':
            self._build_deep_fcn(input_dim, hidden_dims, output_dim, dropout)
        elif network_type == 'DeepFCN_Residual':
            self._build_residual_fcn(input_dim, hidden_dims, output_dim, dropout)
        elif network_type == 'GAT':
            self._build_gat_network(input_dim, hidden_dims, output_dim, dropout)
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")
    
    def _build_deep_fcn(self, input_dim, hidden_dims, output_dim, dropout):
        """构建深度全连接网络"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.extend([
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _build_residual_fcn(self, input_dim, hidden_dims, output_dim, dropout):
        """构建残差全连接网络"""
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        
        self.residual_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dims[i+1])
            )
            self.residual_layers.append(layer)
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self._init_weights()
    
    def _build_gat_network(self, input_dim, hidden_dims, output_dim, dropout):
        """构建图注意力网络"""
        # 简化的GAT实现
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        self.attention_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            attention_layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dims[i+1])
            )
            self.attention_layers.append(attention_layer)
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        if self.network_type == 'DeepFCN':
            return self.network(x)
        elif self.network_type == 'DeepFCN_Residual':
            return self._forward_residual(x)
        elif self.network_type == 'GAT':
            return self._forward_gat(x)
    
    def _forward_residual(self, x):
        """残差网络前向传播"""
        x = torch.relu(self.input_bn(self.input_layer(x)))
        
        for layer in self.residual_layers:
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + residual
        
        return self.output_layer(x)
    
    def _forward_gat(self, x):
        """GAT网络前向传播"""
        x = torch.relu(self.input_projection(x))
        
        for layer in self.attention_layers:
            x = layer(x)
        
        return self.output_layer(x)

class PathPlanningModule:
    """路径规划模块，支持两种模式对比"""
    
    def __init__(self, config: Config):
        self.config = config
        self.path_cache = {}  # 路径缓存
        self.distance_cache = {}  # 距离缓存
    
    def calculate_distance(self, start_pos, end_pos, start_heading, end_heading, 
                          obstacles, use_ph_rrt=False):
        """计算两点间距离，支持两种模式"""
        cache_key = (tuple(start_pos), tuple(end_pos), use_ph_rrt)
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        if use_ph_rrt:
            # 使用高精度PH-RRT路径规划
            distance = self._calculate_ph_rrt_distance(start_pos, end_pos, 
                                                     start_heading, end_heading, obstacles)
        else:
            # 使用简化欧几里得距离
            distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        
        self.distance_cache[cache_key] = distance
        return distance
    
    def _calculate_ph_rrt_distance(self, start_pos, end_pos, start_heading, 
                                  end_heading, obstacles):
        """使用PH-RRT计算路径距离"""
        try:
            planner = PHCurveRRTPlanner(start_pos, end_pos, start_heading, 
                                       end_heading, obstacles, self.config)
            result = planner.plan()
            
            if result is not None:
                path, length = result
                return length
            else:
                # 如果PH-RRT失败，回退到欧几里得距离
                return np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        except Exception as e:
            print(f"PH-RRT路径规划失败: {e}")
            return np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    
    def get_path(self, start_pos, end_pos, start_heading, end_heading, 
                 obstacles, use_ph_rrt=False):
        """获取完整路径"""
        cache_key = (tuple(start_pos), tuple(end_pos), use_ph_rrt)
        
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if use_ph_rrt:
            try:
                planner = PHCurveRRTPlanner(start_pos, end_pos, start_heading, 
                                           end_heading, obstacles, self.config)
                result = planner.plan()
                
                if result is not None:
                    path, length = result
                    self.path_cache[cache_key] = (path, length)
                    return path, length
                else:
                    # 回退到直线路径
                    path = np.array([start_pos, end_pos])
                    length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
                    self.path_cache[cache_key] = (path, length)
                    return path, length
            except Exception as e:
                print(f"PH-RRT路径规划失败: {e}")
                path = np.array([start_pos, end_pos])
                length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
                self.path_cache[cache_key] = (path, length)
                return path, length
        else:
            # 简化模式：直线路径
            path = np.array([start_pos, end_pos])
            length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            self.path_cache[cache_key] = (path, length)
            return path, length

class DeadlockDetector:
    """死锁检测器"""
    
    def __init__(self):
        self.deadlock_history = []
        self.max_history_length = 10
    
    def detect_deadlock(self, uavs, targets, step_count, max_steps):
        """检测死锁情况"""
        # 检查是否有UAV和目标都无法进行有效分配
        valid_assignments = 0
        total_possible_assignments = 0
        
        for uav in uavs:
            for target in targets:
                if not np.all(uav.resources <= 0) and not np.all(target.remaining_resources <= 0):
                    total_possible_assignments += 1
                    # 检查是否有有效的资源贡献
                    potential_contribution = np.minimum(uav.resources, target.remaining_resources)
                    if np.any(potential_contribution > 0):
                        valid_assignments += 1
        
        # 如果没有任何有效分配，可能存在死锁
        deadlock_ratio = 1.0 - (valid_assignments / max(total_possible_assignments, 1))
        
        # 记录死锁历史
        self.deadlock_history.append(deadlock_ratio)
        if len(self.deadlock_history) > self.max_history_length:
            self.deadlock_history.pop(0)
        
        # 检测死锁：连续多步都没有有效分配
        if len(self.deadlock_history) >= 3:
            recent_deadlock_avg = np.mean(self.deadlock_history[-3:])
            if recent_deadlock_avg > 0.8:  # 80%以上的分配都无效
                return True, deadlock_ratio
        
        return False, deadlock_ratio

class EnhancedRLSolver:
    """增强的强化学习求解器，支持路径模式对比"""
    
    def __init__(self, uavs, targets, graph, obstacles, config: Config, 
                 network_type='DeepFCN', use_ph_rrt=False):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.use_ph_rrt = use_ph_rrt
        self.network_type = network_type
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 路径规划模块
        self.path_planner = PathPlanningModule(config)
        
        # 死锁检测器
        self.deadlock_detector = DeadlockDetector()
        
        # 计算状态和动作空间维度
        self.state_dim = self._calculate_state_dim()
        self.action_dim = len(uavs) * len(targets) * graph.n_phi
        
        # 创建网络
        hidden_dims = [256, 128, 64] if network_type == 'DeepFCN' else [128, 64]
        self.policy_net = EnhancedNetwork(self.state_dim, hidden_dims, self.action_dim, 
                                        network_type=network_type).to(self.device)
        self.target_net = EnhancedNetwork(self.state_dim, hidden_dims, self.action_dim, 
                                        network_type=network_type).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                   lr=config.LEARNING_RATE)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # 训练参数
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'loss_history': [],
            'path_mode': 'PH-RRT' if use_ph_rrt else 'Euclidean',
            'deadlock_events': [],
            'resource_satisfaction': []
        }
    
    def _calculate_state_dim(self):
        """计算状态空间维度"""
        # 获取资源维度
        uav_resource_dim = len(self.uavs[0].resources) if self.uavs else 0
        target_resource_dim = len(self.targets[0].resources) if self.targets else 0
        
        # UAV状态: 位置(2) + 资源(uav_resource_dim) + 任务序列长度(1)
        uav_state_dim = 2 + uav_resource_dim + 1
        
        # 目标状态: 位置(2) + 剩余资源(target_resource_dim) + 已分配UAV数量(1)
        target_state_dim = 2 + target_resource_dim + 1
        
        # 全局状态: 目标完成度(1) + 总资源利用率(1) + 平均距离(1) + 死锁风险(1) = 4
        global_state_dim = 4
        
        return len(self.uavs) * uav_state_dim + len(self.targets) * target_state_dim + global_state_dim
    
    def get_state(self):
        """获取当前状态"""
        state = []
        
        # UAV状态
        for uav in self.uavs:
            uav_state = [
                uav.current_position[0], uav.current_position[1],  # 位置
                *uav.resources,  # 资源
                len(uav.task_sequence)  # 任务序列长度
            ]
            state.extend(uav_state)
        
        # 目标状态
        for target in self.targets:
            target_state = [
                target.position[0], target.position[1],  # 位置
                *target.remaining_resources,  # 剩余资源
                len(target.allocated_uavs)  # 已分配UAV数量
            ]
            state.extend(target_state)
        
        # 全局状态
        total_targets = len(self.targets)
        completed_targets = sum(1 for t in self.targets if np.all(t.remaining_resources <= 0))
        completion_ratio = completed_targets / total_targets if total_targets > 0 else 0
        
        total_resources = sum(np.sum(uav.resources) for uav in self.uavs)
        initial_resources = sum(np.sum(uav.initial_resources) for uav in self.uavs)
        resource_utilization = 1 - (total_resources / initial_resources) if initial_resources > 0 else 0
        
        # 计算平均距离
        total_distance = 0
        count = 0
        for uav in self.uavs:
            for target in self.targets:
                if not np.all(target.remaining_resources <= 0):
                    distance = self.path_planner.calculate_distance(
                        uav.current_position, target.position, 
                        uav.heading, 0, self.obstacles, self.use_ph_rrt
                    )
                    total_distance += distance
                    count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        
        # 死锁风险检测
        is_deadlocked, deadlock_ratio = self.deadlock_detector.detect_deadlock(
            self.uavs, self.targets, 0, 1000
        )
        
        global_state = [completion_ratio, resource_utilization, avg_distance, deadlock_ratio]
        state.extend(global_state)
        
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randrange(self.action_dim)
        
        # 使用策略网络选择动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 在推理时设置BatchNorm为评估模式
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            # 恢复训练模式
            if training:
                self.policy_net.train()
            return q_values.argmax().item()
    
    def _action_to_assignment(self, action):
        """将动作索引转换为任务分配"""
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        n_phi = self.graph.n_phi
        
        target_idx = action // (n_uavs * n_phi)
        remaining = action % (n_uavs * n_phi)
        uav_idx = remaining // n_phi
        phi_idx = remaining % n_phi
        
        return target_idx, uav_idx, phi_idx
    
    def _is_valid_action(self, target, uav, phi_idx):
        """检查动作是否有效"""
        if np.all(uav.resources <= 0):
            return False
        if np.all(target.remaining_resources <= 0):
            return False
        if (uav.id, phi_idx) in target.allocated_uavs:
            return False
        return True
    
    def _check_zero_contribution(self, uav, target):
        """检查是否会产生零贡献"""
        potential_contribution = np.minimum(uav.resources, target.remaining_resources)
        return np.all(potential_contribution <= 0)
    
    def _calculate_collaboration_bonus(self, target, uav):
        """计算协作奖励"""
        # 如果多个UAV协作完成同一个目标，给予奖励
        if len(target.allocated_uavs) > 1:
            return self.config.COLLABORATION_BONUS
        return 0
    
    def step(self, action):
        """执行一步动作"""
        target_idx, uav_idx, phi_idx = self._action_to_assignment(action)
        target = self.targets[target_idx]
        uav = self.uavs[uav_idx]
        
        # 检查动作有效性
        if not self._is_valid_action(target, uav, phi_idx):
            return self.get_state(), self.config.INVALID_ACTION_PENALTY, True, {
                'invalid_action': True,
                'reason': 'invalid_assignment'
            }
        
        # 检查零贡献情况
        if self._check_zero_contribution(uav, target):
            return self.get_state(), self.config.ZERO_CONTRIBUTION_PENALTY, True, {
                'invalid_action': True,
                'reason': 'zero_contribution'
            }
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 计算路径长度（使用配置的路径模式）
        path_len = self.path_planner.calculate_distance(
            uav.current_position, target.position,
            uav.heading, phi_idx * (2 * np.pi / self.config.GRAPH_N_PHI),
            self.obstacles, self.use_ph_rrt
        )
        
        travel_time = path_len / uav.velocity_range[1]
        
        # 计算实际贡献
        actual_contribution = np.minimum(uav.resources, target.remaining_resources)
        
        # 更新状态
        uav.resources = uav.resources.astype(np.float64) - actual_contribution.astype(np.float64)
        target.remaining_resources = target.remaining_resources.astype(np.float64) - actual_contribution.astype(np.float64)
        
        if uav_idx not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav_idx, phi_idx))
        uav.task_sequence.append((target_idx, phi_idx))
        uav.current_position = target.position
        uav.heading = phi_idx * (2 * np.pi / self.config.GRAPH_N_PHI)
        
        # 检查是否完成所有目标
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        done = bool(total_satisfied == total_targets)
        
        # 计算奖励
        reward = self._calculate_enhanced_reward(target, uav, actual_contribution, path_len, 
                                               was_satisfied, travel_time, done)
        
        info = {
            'target_id': int(target_idx),
            'uav_id': int(uav_idx),
            'phi_idx': int(phi_idx),
            'actual_contribution': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': bool(done),
            'path_mode': 'PH-RRT' if self.use_ph_rrt else 'Euclidean'
        }
        
        return self.get_state(), reward, done, info
    
    def _calculate_enhanced_reward(self, target, uav, actual_contribution, path_len, 
                                  was_satisfied, travel_time, done):
        """计算增强的分层奖励函数"""
        
        # ===== 第一层：任务完成奖励（最高优先级） =====
        if done:
            # 巨大的任务完成奖励
            return 20000.0
        
        # ===== 第二层：强惩罚机制（最高优先级） =====
        # 检查是否触发了强惩罚条件
        is_deadlocked, _ = self.deadlock_detector.detect_deadlock(self.uavs, self.targets, 0, 1000)
        if is_deadlocked:
            return -500.0  # 死锁强惩罚
        
        # 检查无效动作
        if np.all(actual_contribution <= 0):
            return -500.0  # 零贡献强惩罚
        
        # ===== 第三层：核心层奖励（基础奖励） =====
        core_reward = 0.0
        
        # 目标完成奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        if new_satisfied:
            core_reward += 1500.0  # 新完成目标的奖励
        
        # 实际贡献奖励
        contribution_reward = np.sum(actual_contribution) * 5.0
        core_reward += contribution_reward
        
        # 如果核心层奖励为负或零，直接返回
        if core_reward <= 0:
            return core_reward
        
        # ===== 第四层：效率层奖励（仅在核心层为正时计算） =====
        efficiency_reward = 0.0
        
        # 边际效用递减奖励
        target_initial_total = np.sum(target.resources)
        target_remaining_before = np.sum(target.remaining_resources + actual_contribution)
        target_remaining_after = np.sum(target.remaining_resources)
        completion_ratio_before = 1.0 - (target_remaining_before / target_initial_total)
        completion_ratio_after = 1.0 - (target_remaining_after / target_initial_total)
        completion_improvement = completion_ratio_after - completion_ratio_before
        
        marginal_utility = completion_improvement * (1.0 - completion_ratio_before)
        marginal_reward = marginal_utility * 1000.0
        efficiency_reward += marginal_reward
        
        # 资源效率奖励
        resource_efficiency = np.sum(actual_contribution) / np.sum(uav.resources + actual_contribution)
        efficiency_bonus = resource_efficiency * 500.0
        efficiency_reward += efficiency_bonus
        
        # 协作奖励
        collaboration_bonus = self._calculate_collaboration_bonus(target, uav)
        efficiency_reward += collaboration_bonus
        
        # ===== 第五层：成本惩罚（仅在核心层为正时计算） =====
        cost_penalty = 0.0
        
        # 距离惩罚（根据路径模式调整）
        distance_penalty_factor = 0.05 if self.use_ph_rrt else 0.1
        distance_penalty = -path_len * distance_penalty_factor * 0.1
        cost_penalty += distance_penalty
        
        # 时间惩罚
        time_penalty = -travel_time * 10.0
        cost_penalty += time_penalty
        
        # ===== 总奖励计算 =====
        total_reward = core_reward + efficiency_reward + cost_penalty
        
        return float(total_reward)
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*batch))
        
        # 转换为张量
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 
                                         self.config.max_grad_norm)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes, patience=50, log_interval=10, model_save_path=None):
        """训练模型"""
        best_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(episodes):
            # 重置环境
            self._reset_environment()
            state = self.get_state()
            episode_reward = 0
            episode_length = 0
            deadlock_events = 0
            resource_satisfaction = 0
            
            while True:
                # 选择动作
                action = self.select_action(state, training=True)
                
                # 执行动作
                next_state, reward, done, info = self.step(action)
                
                # 记录死锁事件
                if info.get('reason') in ['invalid_assignment', 'zero_contribution']:
                    deadlock_events += 1
                
                # 记录资源满足情况
                if info.get('actual_contribution', 0) > 0:
                    resource_satisfaction += 1
                
                # 存储经验
                self.store_experience(state, action, reward, next_state, done)
                
                # 优化模型
                loss = self.optimize_model()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 更新目标网络
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 记录历史
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_history'].append(self.epsilon)
            self.training_history['deadlock_events'].append(deadlock_events)
            self.training_history['resource_satisfaction'].append(resource_satisfaction / max(episode_length, 1))
            if 'loss' in locals():
                self.training_history['loss_history'].append(loss)
            
            # 打印日志
            if episode % log_interval == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-log_interval:])
                avg_deadlock = np.mean(self.training_history['deadlock_events'][-log_interval:])
                avg_satisfaction = np.mean(self.training_history['resource_satisfaction'][-log_interval:])
                print(f"Episode {episode}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Deadlock Events: {avg_deadlock:.2f}, "
                      f"Resource Satisfaction: {avg_satisfaction:.3f}, "
                      f"Path Mode: {self.training_history['path_mode']}")
            
            # 早停检查
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                if model_save_path:
                    self.save_model(model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，在episode {episode}")
                    break
        
        return self.training_history
    
    def _reset_environment(self):
        """重置环境状态"""
        # 重置UAV状态
        for uav in self.uavs:
            uav.resources = uav.initial_resources.copy()
            uav.task_sequence = []
            uav.current_position = uav.position.copy()
            uav.heading = 0.0
        
        # 重置目标状态
        for target in self.targets:
            target.remaining_resources = target.resources.copy()
            target.allocated_uavs = []
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'network_type': self.network_type,
            'use_ph_rrt': self.use_ph_rrt
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
    
    def get_task_assignments(self):
        """获取任务分配结果"""
        assignments = []
        for uav in self.uavs:
            for target_idx, phi_idx in uav.task_sequence:
                target = self.targets[target_idx]
                assignments.append({
                    'uav_id': uav.id,
                    'target_id': target.id,
                    'phi_idx': phi_idx,
                    'target_position': target.position,
                    'uav_position': uav.current_position
                })
        return assignments
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title(f'Episode Rewards ({self.training_history["path_mode"]})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 探索率曲线
        axes[0, 1].plot(self.training_history['epsilon_history'])
        axes[0, 1].set_title('Epsilon Decay')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        
        # 死锁事件
        if self.training_history['deadlock_events']:
            axes[0, 2].plot(self.training_history['deadlock_events'])
            axes[0, 2].set_title('Deadlock Events')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Deadlock Count')
        
        # 资源满足率
        if self.training_history['resource_satisfaction']:
            axes[1, 0].plot(self.training_history['resource_satisfaction'])
            axes[1, 0].set_title('Resource Satisfaction Rate')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Satisfaction Rate')
        
        # 损失曲线
        if self.training_history['loss_history']:
            axes[1, 1].plot(self.training_history['loss_history'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
        
        # 平均奖励曲线
        window_size = 50
        if len(self.training_history['episode_rewards']) >= window_size:
            avg_rewards = []
            for i in range(window_size, len(self.training_history['episode_rewards'])):
                avg_rewards.append(np.mean(self.training_history['episode_rewards'][i-window_size:i]))
            axes[1, 2].plot(avg_rewards)
            axes[1, 2].set_title(f'Average Reward (Window={window_size})')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Average Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def compare_path_modes(uavs, targets, graph, obstacles, config, 
                      episodes=500, network_type='DeepFCN'):
    """对比两种路径模式的学习效果"""
    
    print("开始路径模式对比实验...")
    
    # 创建两种模式的求解器
    euclidean_solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                       network_type, use_ph_rrt=False)
    phrrt_solver = EnhancedRLSolver(uavs, targets, graph, obstacles, config, 
                                   network_type, use_ph_rrt=True)
    
    # 训练两种模式
    print("训练欧几里得距离模式...")
    euclidean_history = euclidean_solver.train(episodes)
    
    print("训练PH-RRT模式...")
    phrrt_history = phrrt_solver.train(episodes)
    
    # 对比结果
    comparison_results = {
        'euclidean': {
            'final_avg_reward': np.mean(euclidean_history['episode_rewards'][-50:]),
            'max_reward': max(euclidean_history['episode_rewards']),
            'convergence_episode': len(euclidean_history['episode_rewards']),
            'final_epsilon': euclidean_history['epsilon_history'][-1],
            'avg_deadlock_events': np.mean(euclidean_history['deadlock_events']),
            'avg_resource_satisfaction': np.mean(euclidean_history['resource_satisfaction'])
        },
        'phrrt': {
            'final_avg_reward': np.mean(phrrt_history['episode_rewards'][-50:]),
            'max_reward': max(phrrt_history['episode_rewards']),
            'convergence_episode': len(phrrt_history['episode_rewards']),
            'final_epsilon': phrrt_history['epsilon_history'][-1],
            'avg_deadlock_events': np.mean(phrrt_history['deadlock_events']),
            'avg_resource_satisfaction': np.mean(phrrt_history['resource_satisfaction'])
        }
    }
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 奖励对比
    axes[0, 0].plot(euclidean_history['episode_rewards'], label='Euclidean', alpha=0.7)
    axes[0, 0].plot(phrrt_history['episode_rewards'], label='PH-RRT', alpha=0.7)
    axes[0, 0].set_title('Reward Comparison')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    
    # 探索率对比
    axes[0, 1].plot(euclidean_history['epsilon_history'], label='Euclidean', alpha=0.7)
    axes[0, 1].plot(phrrt_history['epsilon_history'], label='PH-RRT', alpha=0.7)
    axes[0, 1].set_title('Epsilon Decay Comparison')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].legend()
    
    # 死锁事件对比
    axes[0, 2].plot(euclidean_history['deadlock_events'], label='Euclidean', alpha=0.7)
    axes[0, 2].plot(phrrt_history['deadlock_events'], label='PH-RRT', alpha=0.7)
    axes[0, 2].set_title('Deadlock Events Comparison')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Deadlock Count')
    axes[0, 2].legend()
    
    # 资源满足率对比
    axes[1, 0].plot(euclidean_history['resource_satisfaction'], label='Euclidean', alpha=0.7)
    axes[1, 0].plot(phrrt_history['resource_satisfaction'], label='PH-RRT', alpha=0.7)
    axes[1, 0].set_title('Resource Satisfaction Comparison')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Satisfaction Rate')
    axes[1, 0].legend()
    
    # 平均奖励对比
    window_size = 50
    if len(euclidean_history['episode_rewards']) >= window_size:
        euclidean_avg = []
        phrrt_avg = []
        for i in range(window_size, min(len(euclidean_history['episode_rewards']), 
                                       len(phrrt_history['episode_rewards']))):
            euclidean_avg.append(np.mean(euclidean_history['episode_rewards'][i-window_size:i]))
            phrrt_avg.append(np.mean(phrrt_history['episode_rewards'][i-window_size:i]))
        
        axes[1, 1].plot(euclidean_avg, label='Euclidean', alpha=0.7)
        axes[1, 1].plot(phrrt_avg, label='PH-RRT', alpha=0.7)
        axes[1, 1].set_title(f'Average Reward Comparison (Window={window_size})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].legend()
    
    # 性能指标对比
    metrics = ['Final Avg Reward', 'Max Reward', 'Deadlock Events', 'Resource Satisfaction']
    euclidean_values = [comparison_results['euclidean']['final_avg_reward'],
                       comparison_results['euclidean']['max_reward'],
                       comparison_results['euclidean']['avg_deadlock_events'],
                       comparison_results['euclidean']['avg_resource_satisfaction']]
    phrrt_values = [comparison_results['phrrt']['final_avg_reward'],
                   comparison_results['phrrt']['max_reward'],
                   comparison_results['phrrt']['avg_deadlock_events'],
                   comparison_results['phrrt']['avg_resource_satisfaction']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, euclidean_values, width, label='Euclidean', alpha=0.7)
    axes[1, 2].bar(x + width/2, phrrt_values, width, label='PH-RRT', alpha=0.7)
    axes[1, 2].set_title('Performance Metrics Comparison')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics, rotation=45)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('output/images/enhanced_path_mode_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印对比结果
    print("\n=== 增强路径模式对比结果 ===")
    print(f"欧几里得距离模式:")
    print(f"  最终平均奖励: {comparison_results['euclidean']['final_avg_reward']:.2f}")
    print(f"  最大奖励: {comparison_results['euclidean']['max_reward']:.2f}")
    print(f"  平均死锁事件: {comparison_results['euclidean']['avg_deadlock_events']:.2f}")
    print(f"  平均资源满足率: {comparison_results['euclidean']['avg_resource_satisfaction']:.3f}")
    
    print(f"\nPH-RRT模式:")
    print(f"  最终平均奖励: {comparison_results['phrrt']['final_avg_reward']:.2f}")
    print(f"  最大奖励: {comparison_results['phrrt']['max_reward']:.2f}")
    print(f"  平均死锁事件: {comparison_results['phrrt']['avg_deadlock_events']:.2f}")
    print(f"  平均资源满足率: {comparison_results['phrrt']['avg_resource_satisfaction']:.3f}")
    
    return comparison_results, euclidean_solver, phrrt_solver

if __name__ == "__main__":
    # 测试代码
    from scenarios import get_simple_convergence_test_scenario as create_test_scenario
    
    config = Config()
    uavs, targets, obstacles = create_test_scenario()
    from main import DirectedGraph
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles)
    
    # 运行增强的路径模式对比实验
    comparison_results, euclidean_solver, phrrt_solver = compare_path_modes(
        uavs, targets, graph, obstacles, config, episodes=300
    )