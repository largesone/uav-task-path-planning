# -*- coding: utf-8 -*-
# 文件名: environment.py
# 描述: 定义强化学习的环境，包括场景的有向图表示和任务环境本身。

import numpy as np
import itertools
from scipy.spatial.distance import cdist
from typing import Union, Dict, Any, Literal
import gymnasium as gym
from gymnasium import spaces

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner

# =============================================================================
# section 3: 场景建模与强化学习环境
# =============================================================================

class DirectedGraph:
    """(已修订) 使用numpy高效构建和管理任务场景的有向图"""
    def __init__(self, uavs, targets, n_phi, obstacles, config):
        self.uavs, self.targets, self.config = uavs, targets, config
        self.n_phi = n_phi
        self.n_uavs, self.n_targets = len(uavs), len(targets)
        self.uav_positions = np.array([u.position for u in uavs])
        self.target_positions = np.array([t.position for t in targets])
        
        self.nodes = uavs + targets
        self.node_positions = np.vstack([self.uav_positions, self.target_positions])
        self.node_map = {node.id: i for i, node in enumerate(self.nodes)}

        self.dist_matrix = self._calculate_distances(obstacles)
        self.adj_matrix = self._build_adjacency_matrix()
        self.phi_matrix = self._calculate_phi_matrix()

    def _calculate_distances(self, obstacles):
        """计算所有节点间的距离，可选地使用PH-RRT处理障碍物"""
        dist_matrix = cdist(self.node_positions, self.node_positions)
        if hasattr(self.config, 'USE_PHRRT_DURING_TRAINING') and self.config.USE_PHRRT_DURING_TRAINING and obstacles:
            for i, j in itertools.product(range(len(self.nodes)), repeat=2):
                if i == j: continue
                p1, p2 = self.node_positions[i], self.node_positions[j]
                planner = PHCurveRRTPlanner(p1, p2, 0, 0, obstacles, self.config)
                path_info = planner.plan()
                if path_info: dist_matrix[i, j] = path_info[1]
        return dist_matrix

    def _build_adjacency_matrix(self):
        """构建邻接矩阵，UAV可以飞到任何目标，目标之间不能互飞"""
        adj = np.zeros((len(self.nodes), len(self.nodes)))
        adj[:self.n_uavs, self.n_uavs:] = 1
        return adj

    def _calculate_phi_matrix(self):
        """(已修订) 高效计算所有节点对之间的相对方向分区(phi值)"""
        delta = self.node_positions[:, np.newaxis, :] - self.node_positions[np.newaxis, :, :]
        angles = np.arctan2(delta[..., 1], delta[..., 0])
        phi_matrix = np.floor((angles % (2 * np.pi)) / (2 * np.pi / self.config.GRAPH_N_PHI))
        return phi_matrix.astype(int)

    def get_dist(self, from_node_id, to_node_id):
        """获取两个节点间的距离"""
        return self.dist_matrix[self.node_map[from_node_id], self.node_map[to_node_id]]

class UAVTaskEnv:
    """
    (已修订) 无人机协同任务分配的强化学习环境
    
    支持双模式观测系统：
    - "flat" 模式：传统扁平向量观测，确保FCN向后兼容性
    - "graph" 模式：结构化图观测，支持TransformerGNN架构和可变数量实体
    """
    def __init__(self, uavs, targets, graph, obstacles, config, obs_mode: Literal["flat", "graph"] = "flat"):
        """
        初始化UAV任务环境
        
        Args:
            uavs: UAV实体列表
            targets: 目标实体列表  
            graph: 有向图对象
            obstacles: 障碍物列表
            config: 配置对象
            obs_mode: 观测模式，"flat"为扁平向量模式，"graph"为图结构模式
        """
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.obs_mode = obs_mode
        self.step_count = 0
        self.max_steps = len(targets) * len(uavs) * 2
        self.invalid_action_penalty = -75.0
        
        # 计算动作空间大小
        self.n_actions = len(targets) * len(uavs) * self.graph.n_phi
        
        # 动态创建观测空间
        self.observation_space = self._create_observation_space()
        
        # 定义动作空间
        self.action_space = spaces.Discrete(self.n_actions)
    
    def _create_observation_space(self) -> spaces.Space:
        """
        动态观测空间创建的工厂模式
        
        根据obs_mode参数创建相应的观测空间：
        - "flat": 扁平向量观测空间，确保FCN向后兼容性
        - "graph": 字典结构观测空间，支持可变数量实体
        
        Returns:
            gym.spaces.Space: 对应模式的观测空间
        """
        if self.obs_mode == "flat":
            return self._create_flat_observation_space()
        elif self.obs_mode == "graph":
            return self._create_graph_observation_space()
        else:
            raise ValueError(f"不支持的观测模式: {self.obs_mode}。支持的模式: ['flat', 'graph']")
    
    def _create_flat_observation_space(self) -> spaces.Box:
        """
        创建扁平向量观测空间，维持现有实现的向后兼容性
        
        状态组成：
        - 目标信息：position(2) + resources(2) + value(1) + remaining_resources(2) = 7 * n_targets
        - UAV信息：position(2) + heading(1) + resources(2) + max_distance(1) + velocity_range(2) = 8 * n_uavs  
        - 协同信息：分配状态 = 1 * n_targets * n_uavs
        - 全局信息：10个全局状态特征
        
        Returns:
            spaces.Box: 扁平向量观测空间
        """
        n_targets = len(self.targets)
        n_uavs = len(self.uavs)
        
        # 计算状态维度
        target_dim = 7 * n_targets  # 每个目标7个特征
        uav_dim = 8 * n_uavs        # 每个UAV 8个特征
        collaboration_dim = n_targets * n_uavs  # 协同分配状态
        global_dim = 10             # 全局状态特征
        
        total_dim = target_dim + uav_dim + collaboration_dim + global_dim
        
        # 创建观测空间，使用合理的边界值
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def _create_graph_observation_space(self) -> spaces.Dict:
        """
        创建图结构观测空间，支持可变数量实体
        
        图模式状态结构：
        - uav_features: [N_uav, uav_feature_dim] - UAV实体特征（归一化）
        - target_features: [N_target, target_feature_dim] - 目标实体特征（归一化）
        - relative_positions: [N_uav, N_target, 2] - 归一化相对位置向量
        - distances: [N_uav, N_target] - 归一化距离矩阵
        - masks: 有效实体掩码字典
        
        Returns:
            spaces.Dict: 图结构观测空间
        """
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # UAV特征维度：position(2) + heading(1) + resources_ratio(2) + max_distance_norm(1) + 
        #              velocity_norm(2) + is_alive(1) = 9
        uav_feature_dim = 9
        
        # 目标特征维度：position(2) + resources_ratio(2) + value_norm(1) + 
        #              remaining_ratio(2) + is_visible(1) = 8  
        target_feature_dim = 8
        
        return spaces.Dict({
            # UAV实体特征矩阵 [N_uav, uav_feature_dim]
            "uav_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_uavs, uav_feature_dim),
                dtype=np.float32
            ),
            
            # 目标实体特征矩阵 [N_target, target_feature_dim]
            "target_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_targets, target_feature_dim),
                dtype=np.float32
            ),
            
            # 相对位置矩阵 [N_uav, N_target, 2] - 归一化相对位置向量
            "relative_positions": spaces.Box(
                low=-1.0, high=1.0,
                shape=(n_uavs, n_targets, 2),
                dtype=np.float32
            ),
            
            # 距离矩阵 [N_uav, N_target] - 归一化距离
            "distances": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_uavs, n_targets),
                dtype=np.float32
            ),
            
            # 掩码字典
            "masks": spaces.Dict({
                # UAV有效性掩码 [N_uav] - 1表示有效，0表示无效
                "uav_mask": spaces.Box(
                    low=0, high=1,
                    shape=(n_uavs,),
                    dtype=np.int32
                ),
                
                # 目标有效性掩码 [N_target] - 1表示有效，0表示无效
                "target_mask": spaces.Box(
                    low=0, high=1,
                    shape=(n_targets,),
                    dtype=np.int32
                )
            })
        })
        
    def reset(self):
        """重置环境"""
        for uav in self.uavs:
            uav.reset()
        for target in self.targets:
            target.reset()
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> Union[np.ndarray, Dict[str, Any]]:
        """
        获取当前状态，根据obs_mode返回不同格式
        
        Returns:
            Union[np.ndarray, Dict]: 
                - "flat"模式：扁平向量状态
                - "graph"模式：结构化图状态字典
        """
        if self.obs_mode == "flat":
            return self._get_flat_state()
        elif self.obs_mode == "graph":
            return self._get_graph_state()
        else:
            raise ValueError(f"不支持的观测模式: {self.obs_mode}")
    
    def _get_flat_state(self) -> np.ndarray:
        """
        获取扁平向量状态，维持现有实现的向后兼容性
        
        Returns:
            np.ndarray: 扁平向量状态
        """
        state = []
        
        # 目标信息
        for target in self.targets:
            target_state = [
                target.position[0], target.position[1],
                target.resources[0], target.resources[1],
                target.value,
                target.remaining_resources[0], target.remaining_resources[1]
            ]
            state.extend(target_state)
        
        # UAV信息
        for uav in self.uavs:
            uav_state = [
                uav.current_position[0], uav.current_position[1],
                uav.heading,
                uav.resources[0], uav.resources[1],
                uav.max_distance,
                uav.velocity_range[0], uav.velocity_range[1]
            ]
            state.extend(uav_state)
        
        # 协同信息
        for target in self.targets:
            for uav in self.uavs:
                is_assigned = any(
                    (uav.id, phi_idx) in target.allocated_uavs 
                    for phi_idx in range(self.graph.n_phi)
                )
                state.append(1.0 if is_assigned else 0.0)
        
        # 全局状态信息
        total_targets = len(self.targets)
        completed_targets = sum(
            1 for target in self.targets 
            if np.all(target.remaining_resources <= 0)
        )
        completion_rate = completed_targets / total_targets if total_targets > 0 else 0.0
        
        global_state = [
            self.step_count,
            completion_rate,
            len([u for u in self.uavs if np.any(u.resources > 0)]),
            sum(np.sum(target.remaining_resources) for target in self.targets),
            sum(np.sum(uav.resources) for uav in self.uavs),
            completed_targets,
            total_targets,
            self.max_steps - self.step_count,
            np.mean([uav.heading for uav in self.uavs]),
            np.std([uav.heading for uav in self.uavs])
        ]
        state.extend(global_state)
        
        return np.array(state, dtype=np.float32)
    
    def _get_graph_state(self) -> Dict[str, Any]:
        """
        获取图结构状态，支持TransformerGNN架构
        
        实现尺度不变的状态表示：
        - 移除绝对坐标，使用归一化相对位置
        - 实体特征仅包含归一化的自身属性
        - 添加鲁棒性掩码机制，支持通信/感知失效场景
        - 使用固定维度确保批处理兼容性
        
        Returns:
            Dict[str, Any]: 图结构状态字典
        """
        # 使用固定的最大数量，确保维度一致性
        max_uavs = getattr(self.config, 'MAX_UAVS', 10)
        max_targets = getattr(self.config, 'MAX_TARGETS', 15)
        
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # 计算地图尺度用于归一化（假设地图为正方形）
        map_size = getattr(self.config, 'MAP_SIZE', 1000.0)
        
        # === 1. UAV特征矩阵 [max_uavs, uav_feature_dim] ===
        uav_features = np.zeros((max_uavs, 9), dtype=np.float32)
        
        for i, uav in enumerate(self.uavs):
            # 归一化位置 [0, 1]
            norm_pos = np.array(uav.current_position) / map_size
            
            # 归一化朝向 [0, 1]
            norm_heading = uav.heading / (2 * np.pi)
            
            # 资源比例 [0, 1]
            initial_resources = getattr(uav, 'initial_resources', uav.resources + 1e-6)
            resource_ratio = uav.resources / (initial_resources + 1e-6)
            
            # 归一化最大距离 [0, 1]
            norm_max_distance = uav.max_distance / map_size
            
            # 归一化速度范围 [0, 1]
            max_velocity = 100.0  # 假设最大速度
            norm_velocity = np.array(uav.velocity_range) / max_velocity
            
            # 鲁棒性掩码：is_alive位（0/1），标识无人机通信/感知状态
            is_alive = self._calculate_uav_alive_status(uav, i)
            
            uav_features[i] = [
                norm_pos[0], norm_pos[1],           # 归一化位置 (2)
                norm_heading,                       # 归一化朝向 (1)
                resource_ratio[0], resource_ratio[1], # 资源比例 (2)
                norm_max_distance,                  # 归一化最大距离 (1)
                norm_velocity[0], norm_velocity[1], # 归一化速度 (2)
                is_alive                           # 存活状态 (1)
            ]
        
        # === 2. 目标特征矩阵 [max_targets, target_feature_dim] ===
        target_features = np.zeros((max_targets, 8), dtype=np.float32)
        
        for i, target in enumerate(self.targets):
            # 归一化位置 [0, 1]
            norm_pos = np.array(target.position) / map_size
            
            # 资源比例 [0, 1]
            initial_resources = target.resources + 1e-6
            resource_ratio = target.resources / initial_resources
            
            # 归一化价值 [0, 1]（假设最大价值为1000）
            max_value = 1000.0
            norm_value = min(target.value / max_value, 1.0)
            
            # 剩余资源比例 [0, 1]
            remaining_ratio = target.remaining_resources / initial_resources
            
            # 鲁棒性掩码：is_visible位（0/1），标识目标可见性状态
            is_visible = self._calculate_target_visibility_status(target, i)
            
            target_features[i] = [
                norm_pos[0], norm_pos[1],                    # 归一化位置 (2)
                resource_ratio[0], resource_ratio[1],        # 资源比例 (2)
                norm_value,                                  # 归一化价值 (1)
                remaining_ratio[0], remaining_ratio[1],      # 剩余资源比例 (2)
                is_visible                                   # 可见性状态 (1)
            ]
        
        # === 3. 相对位置矩阵 [max_uavs, max_targets, 2] ===
        relative_positions = np.zeros((max_uavs, max_targets, 2), dtype=np.float32)
        
        for i, uav in enumerate(self.uavs):
            for j, target in enumerate(self.targets):
                # 计算相对位置向量 (pos_target - pos_uav)
                rel_pos = np.array(target.position) - np.array(uav.current_position)
                # 归一化到 [-1, 1] 范围
                relative_positions[i, j] = rel_pos / map_size
        
        # === 4. 距离矩阵 [max_uavs, max_targets] ===
        distances = np.zeros((max_uavs, max_targets), dtype=np.float32)
        
        for i, uav in enumerate(self.uavs):
            for j, target in enumerate(self.targets):
                # 计算欧几里得距离并归一化
                dist = np.linalg.norm(
                    np.array(target.position) - np.array(uav.current_position)
                )
                distances[i, j] = min(dist / map_size, 1.0)  # 归一化到 [0, 1]
        
        # === 5. 增强的掩码字典 ===
        masks = self._calculate_robust_masks()
        
        # 构建图状态字典
        graph_state = {
            "uav_features": uav_features,
            "target_features": target_features,
            "relative_positions": relative_positions,
            "distances": distances,
            "masks": masks
        }
        
        return graph_state

    def step(self, action):
        """执行一步动作"""
        self.step_count += 1
        
        # 转换动作
        target_idx, uav_idx, phi_idx = self._action_to_assignment(action)
        target = self.targets[target_idx]
        uav = self.uavs[uav_idx]
        
        # 检查动作有效性
        if not self._is_valid_action(target, uav, phi_idx):
            return self._get_state(), self.invalid_action_penalty, False, False, {
                'invalid_action': True,
                'reason': 'invalid_assignment'
            }
        
        # 计算实际贡献
        actual_contribution = np.minimum(uav.resources, target.remaining_resources)

        # 检查是否有实际贡献
        if np.all(actual_contribution <= 0):
            return self._get_state(), self.invalid_action_penalty, False, False, {
                'invalid_action': True,
                'reason': 'no_contribution'
            }
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 计算路径长度
        path_len = np.linalg.norm(uav.current_position - target.position)
        travel_time = path_len / uav.velocity_range[1]
        
        # 更新状态
        uav.resources = uav.resources.astype(np.float64) - actual_contribution.astype(np.float64)
        target.remaining_resources = target.remaining_resources.astype(np.float64) - actual_contribution.astype(np.float64)
        
        if uav_idx not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav_idx, phi_idx))
        uav.task_sequence.append((target_idx, phi_idx))
        uav.current_position = target.position
        uav.heading = phi_idx * (2 * np.pi / self.graph.n_phi)
        
        # 检查是否完成所有目标
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        done = bool(total_satisfied == total_targets)
        
        # 计算奖励
        reward = self._calculate_reward(target, uav, actual_contribution, path_len, 
                                      was_satisfied, travel_time, done)
        
        # 检查是否超时
        truncated = self.step_count >= self.max_steps
        
        # 构建信息字典
        info = {
            'target_id': int(target_idx),
            'uav_id': int(uav_idx),
            'phi_idx': int(phi_idx),
            'actual_contribution': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': bool(done)
        }
        
        return self._get_state(), reward, done, truncated, info

    def _action_to_assignment(self, action):
        """将动作索引转换为任务分配 - 修复版本，添加边界检查"""
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        n_phi = self.graph.n_phi
        
        # 确保动作在有效范围内
        max_valid_action = n_targets * n_uavs * n_phi - 1
        if action > max_valid_action:
            print(f"警告: 动作 {action} 超出有效范围 [0, {max_valid_action}]，调整为模运算结果")
            action = action % (max_valid_action + 1)
        
        target_idx = action // (n_uavs * n_phi)
        remaining = action % (n_uavs * n_phi)
        uav_idx = remaining // n_phi
        phi_idx = remaining % n_phi
        
        # 再次验证索引边界
        target_idx = min(target_idx, n_targets - 1)
        uav_idx = min(uav_idx, n_uavs - 1)
        phi_idx = min(phi_idx, n_phi - 1)
        
        # 确保索引非负
        target_idx = max(0, target_idx)
        uav_idx = max(0, uav_idx)
        phi_idx = max(0, phi_idx)
        
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

    def calculate_simplified_reward(self, target, uav, actual_contribution, path_len, 
                                was_satisfied, travel_time, done):
        """
        简化的奖励函数，重点关注目标资源满足和死锁避免
        
        Args:
            target: 目标对象
            uav: UAV对象
            actual_contribution: 实际资源贡献
            path_len: 路径长度
            was_satisfied: 之前是否已满足目标
            travel_time: 旅行时间
            done: 是否完成所有目标
            
        Returns:
            float: 归一化的奖励值
        """
        # 1. 任务完成奖励 (最高优先级)
        if done:
            return 10.0  # 归一化后的最高奖励
        
        # 2. 目标满足奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 5.0 if new_satisfied else 0.0
        
        # 3. 资源贡献奖励 (核心奖励)
        # 计算贡献比例而不是绝对值
        target_initial_total = np.sum(target.resources)
        contribution_ratio = np.sum(actual_contribution) / target_initial_total if target_initial_total > 0 else 0
        contribution_reward = contribution_ratio * 3.0  # 最高3分
        
        # 4. 零贡献惩罚 (避免死锁)
        if np.all(actual_contribution <= 0):
            return -5.0  # 严重惩罚零贡献动作
        
        # 5. 距离惩罚 (简化版)
        # 使用相对距离而不是绝对距离
        max_distance = 1000.0  # 假设的最大距离
        distance_ratio = min(path_len / max_distance, 1.0)
        distance_penalty = -distance_ratio * 1.0  # 最多-1分
        
        # 总奖励 (归一化到[-5, 10]范围)
        total_reward = target_completion_reward + contribution_reward + distance_penalty
        
        return float(total_reward)
    
    def _calculate_reward(self, target, uav, actual_contribution, path_len, 
                         was_satisfied, travel_time, done):
        """
        Per-Agent归一化奖励函数 - 解决尺度漂移问题
        
        核心设计理念:
        1. 巨大的正向奖励作为核心激励
        2. 所有成本作为正奖励的动态百分比减项
        3. 塑形奖励引导探索
        4. **Per-Agent归一化**: 识别与无人机数量相关的奖励项，除以当前有效无人机数量
        5. 移除所有硬编码的巨大惩罚值
        
        奖励结构:
        - 任务完成奖励: 100.0 (核心正向激励)
        - 资源贡献奖励: 10.0-50.0 (基于贡献比例)
        - 塑形奖励: 0.1-2.0 (接近目标、协作等)
        - 动态成本: 正奖励的3-8%作为减项
        - **归一化处理**: 拥堵惩罚等与UAV数量相关的奖励项按N_active归一化
        """
        
        # ===== 第一部分: 计算当前有效无人机数量 (Per-Agent归一化基础) =====
        n_active_uavs = self._calculate_active_uav_count()
        
        # ===== 第二部分: 计算所有正向奖励 =====
        positive_rewards = 0.0
        reward_components = {
            'n_active_uavs': n_active_uavs,  # 记录当前有效无人机数量用于调试
            'normalization_applied': []      # 记录哪些奖励项应用了归一化
        }
        
        # 1. 任务完成的巨大正向奖励 (核心激励)
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = now_satisfied and not was_satisfied
        
        if new_satisfied:
            task_completion_reward = 100.0  # 巨大的任务完成奖励
            positive_rewards += task_completion_reward
            reward_components['task_completion'] = task_completion_reward
        
        # 2. 资源贡献奖励 (基于实际贡献的正向激励)
        contribution_reward = 0.0
        if np.sum(actual_contribution) > 0:
            target_initial_total = np.sum(target.resources)
            if target_initial_total > 0:
                # 计算贡献比例
                contribution_ratio = np.sum(actual_contribution) / target_initial_total
                
                # 基础贡献奖励: 10-50分
                base_contribution = 10.0 + 40.0 * contribution_ratio
                
                # 边际效用奖励: 对小贡献也给予鼓励
                marginal_utility = 15.0 * np.sqrt(contribution_ratio)
                
                # 高效贡献奖励: 对大比例贡献给予额外奖励
                efficiency_bonus = 0.0
                if contribution_ratio > 0.3:
                    efficiency_bonus = 10.0 * (contribution_ratio - 0.3)
                
                contribution_reward = base_contribution + marginal_utility + efficiency_bonus
                positive_rewards += contribution_reward
                reward_components['contribution'] = contribution_reward
        
        # 3. 塑形奖励 - 引导探索和协作
        shaping_rewards = 0.0
        
        # 3.1 接近目标的塑形奖励
        approach_reward = self._calculate_approach_reward(uav, target)
        shaping_rewards += approach_reward
        reward_components['approach_shaping'] = approach_reward
        
        # 3.2 首次接触目标奖励
        if len(target.allocated_uavs) == 1 and target.allocated_uavs[0][0] == uav.id:
            first_contact_reward = 5.0
            shaping_rewards += first_contact_reward
            reward_components['first_contact'] = first_contact_reward
        
        # 3.3 协作塑形奖励 (Per-Agent归一化)
        collaboration_reward_raw = self._calculate_collaboration_reward(target, uav)
        # 协作奖励与UAV数量相关，需要归一化
        collaboration_reward = collaboration_reward_raw / n_active_uavs
        shaping_rewards += collaboration_reward
        reward_components['collaboration_raw'] = collaboration_reward_raw
        reward_components['collaboration_normalized'] = collaboration_reward
        reward_components['normalization_applied'].append('collaboration')
        
        # 3.4 全局完成进度奖励
        global_progress_reward = self._calculate_global_progress_reward()
        shaping_rewards += global_progress_reward
        reward_components['global_progress'] = global_progress_reward
        
        positive_rewards += shaping_rewards
        
        # ===== 第三部分: 动态尺度成本计算 (包含Per-Agent归一化) =====
        total_costs = 0.0
        
        # 确保有最小正向奖励基数，避免除零
        reward_base = max(positive_rewards, 1.0)
        
        # 1. 距离成本 - 正向奖励的3-5%
        distance_cost_ratio = 0.03 + 0.02 * min(1.0, path_len / 3000.0)  # 3%-5%
        distance_cost_raw = reward_base * distance_cost_ratio
        total_costs += distance_cost_raw
        reward_components['distance_cost'] = -distance_cost_raw
        
        # 2. 时间成本 - 正向奖励的2-3%
        time_cost_ratio = 0.02 + 0.01 * min(1.0, travel_time / 60.0)  # 2%-3%
        time_cost_raw = reward_base * time_cost_ratio
        total_costs += time_cost_raw
        reward_components['time_cost'] = -time_cost_raw
        
        # 3. 拥堵惩罚 (新增 - 与UAV数量直接相关，需要Per-Agent归一化)
        congestion_penalty_raw = self._calculate_congestion_penalty(target, uav, n_active_uavs)
        congestion_penalty_normalized = congestion_penalty_raw / n_active_uavs
        total_costs += congestion_penalty_normalized
        reward_components['congestion_penalty_raw'] = -congestion_penalty_raw
        reward_components['congestion_penalty_normalized'] = -congestion_penalty_normalized
        if congestion_penalty_raw > 0:
            reward_components['normalization_applied'].append('congestion_penalty')
        
        # 4. 资源效率成本 - 如果贡献效率低
        efficiency_cost = 0.0
        if np.sum(actual_contribution) > 0:
            # 计算资源利用效率
            uav_capacity = np.sum(uav.resources)
            if uav_capacity > 0:
                utilization_ratio = np.sum(actual_contribution) / uav_capacity
                if utilization_ratio < 0.5:  # 利用率低于50%
                    efficiency_cost_ratio = 0.02 * (0.5 - utilization_ratio)  # 最多2%
                    efficiency_cost = reward_base * efficiency_cost_ratio
                    total_costs += efficiency_cost
                    reward_components['efficiency_cost'] = -efficiency_cost
        
        # ===== 第四部分: 特殊情况处理 =====
        
        # 零贡献的温和引导 (不再是硬编码的巨大惩罚)
        if np.sum(actual_contribution) <= 0:
            # 给予最小的基础奖励，但增加成本比例
            if positive_rewards == 0:
                positive_rewards = 0.5  # 最小基础奖励
                reward_components['base_reward'] = 0.5
            
            # 增加无效行动成本 (正向奖励的10%)
            ineffective_cost = positive_rewards * 0.1
            total_costs += ineffective_cost
            reward_components['ineffective_cost'] = -ineffective_cost
        
        # 全局任务完成的超级奖励
        if done:
            all_targets_satisfied = all(np.all(t.remaining_resources <= 0) for t in self.targets)
            if all_targets_satisfied:
                global_completion_reward = 200.0  # 超级完成奖励
                positive_rewards += global_completion_reward
                reward_components['global_completion'] = global_completion_reward
        
        # ===== 第五部分: 最终奖励计算与归一化总结 =====
        final_reward = positive_rewards - total_costs
        
        # 温和的奖励范围限制 (不再硬性裁剪)
        final_reward = np.clip(final_reward, -10.0, 300.0)
        
        # 记录详细的奖励组成 (增强版 - 支持Per-Agent归一化监控)
        reward_components.update({
            'total_positive': positive_rewards,
            'total_costs': total_costs,
            'final_reward': final_reward,
            'target_id': target.id,
            'uav_id': uav.id,
            'contribution_amount': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': done,
            
            # Per-Agent归一化相关信息
            'per_agent_normalization': {
                'n_active_uavs': n_active_uavs,
                'total_uavs': len(self.uavs),
                'normalization_factor': 1.0 / n_active_uavs,
                'components_normalized': reward_components['normalization_applied'],
                'normalization_impact': self._calculate_normalization_impact(reward_components)
            },
            
            # 调试信息
            'debug_info': {
                'step_count': self.step_count,
                'allocated_uavs_to_target': len(target.allocated_uavs),
                'target_remaining_resources': float(np.sum(target.remaining_resources)),
                'uav_remaining_resources': float(np.sum(uav.resources))
            }
        })
        
        # 保存最新的奖励组成用于调试和监控
        self.last_reward_components = reward_components
        
        # 如果启用了详细日志记录，输出归一化信息
        if getattr(self.config, 'ENABLE_REWARD_LOGGING', False):
            self._log_reward_components(reward_components)
        
        return float(final_reward)
    
    def _calculate_approach_reward(self, uav, target):
        """
        计算接近目标的塑形奖励
        
        核心思想: 如果无人机相比上一步更接近任何未完成的目标，给予微小正奖励
        这解决了目标过远导致的探索初期无正反馈问题
        """
        approach_reward = 0.0
        
        # 获取当前位置到目标的距离
        current_distance = np.linalg.norm(np.array(uav.current_position) - np.array(target.position))
        
        # 检查是否有历史位置记录
        if hasattr(uav, 'previous_position') and uav.previous_position is not None:
            previous_distance = np.linalg.norm(np.array(uav.previous_position) - np.array(target.position))
            
            # 如果更接近目标
            if current_distance < previous_distance:
                # 计算接近程度
                distance_improvement = previous_distance - current_distance
                max_improvement = 100.0  # 假设的最大改进距离
                
                # 基础接近奖励: 0.1-1.0
                base_approach = 0.1 + 0.9 * min(1.0, distance_improvement / max_improvement)
                
                # 距离越近，奖励越高
                proximity_bonus = 0.0
                if current_distance < 500.0:  # 在500米内
                    proximity_factor = (500.0 - current_distance) / 500.0
                    proximity_bonus = 0.5 * proximity_factor
                
                approach_reward = base_approach + proximity_bonus
        
        # 更新位置历史
        uav.previous_position = uav.current_position.copy()
        
        return approach_reward
    
    def _calculate_uav_alive_status(self, uav, uav_index):
        """
        计算无人机的存活状态（鲁棒性掩码）
        
        Args:
            uav: UAV对象
            uav_index: UAV索引
            
        Returns:
            float: 存活状态 (0.0 或 1.0)
        """
        # 基础存活检查：资源是否耗尽
        if np.all(uav.resources <= 0):
            return 0.0
        
        # 可以在这里添加更复杂的存活逻辑，如：
        # - 通信失效概率
        # - 传感器故障概率
        # - 距离过远导致的信号丢失
        
        return 1.0
    
    def _calculate_target_visibility_status(self, target, target_index):
        """
        计算目标的可见性状态（鲁棒性掩码）
        
        Args:
            target: 目标对象
            target_index: 目标索引
            
        Returns:
            float: 可见性状态 (0.0 或 1.0)
        """
        # 基础可见性检查：目标是否已完成
        if np.all(target.remaining_resources <= 0):
            return 0.0
        
        # 可以在这里添加更复杂的可见性逻辑，如：
        # - 天气条件影响
        # - 障碍物遮挡
        # - 传感器范围限制
        
        return 1.0
    
    def _calculate_robust_masks(self):
        """
        计算增强的掩码字典，支持鲁棒性场景
        
        Returns:
            dict: 包含各种掩码的字典
        """
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # UAV有效性掩码
        uav_mask = np.ones(n_uavs, dtype=np.int32)
        for i, uav in enumerate(self.uavs):
            uav_mask[i] = int(self._calculate_uav_alive_status(uav, i))
        
        # 目标有效性掩码
        target_mask = np.ones(n_targets, dtype=np.int32)
        for i, target in enumerate(self.targets):
            target_mask[i] = int(self._calculate_target_visibility_status(target, i))
        
        return {
            "uav_mask": uav_mask,
            "target_mask": target_mask
        }
    
    def _calculate_active_uav_count(self):
        """
        计算当前有效无人机数量（用于Per-Agent归一化）
        
        Returns:
            int: 有效无人机数量
        """
        active_count = 0
        for uav in self.uavs:
            if np.any(uav.resources > 0):  # 至少有一种资源大于0
                active_count += 1
        return max(active_count, 1)  # 确保至少为1，避免除零错误
    
    def _calculate_congestion_penalty(self, target, uav, n_active_uavs):
        """
        计算拥堵惩罚（与UAV数量相关，需要归一化）
        
        Args:
            target: 目标对象
            uav: UAV对象
            n_active_uavs: 当前有效无人机数量
            
        Returns:
            float: 拥堵惩罚值
        """
        # 计算分配到同一目标的无人机数量
        uavs_on_target = len(target.allocated_uavs)
        
        # 如果多个无人机分配到同一目标，产生拥堵惩罚
        if uavs_on_target > 1:
            # 惩罚与分配的无人机数量成正比
            congestion_factor = (uavs_on_target - 1) / n_active_uavs
            base_penalty = 2.0  # 基础惩罚
            return base_penalty * congestion_factor
        
        return 0.0
    
    def _calculate_global_progress_reward(self):
        """
        计算全局完成进度奖励
        
        Returns:
            float: 全局进度奖励
        """
        if not self.targets:
            return 0.0
        
        # 计算总体完成进度
        total_initial_resources = sum(np.sum(target.resources) for target in self.targets)
        total_remaining_resources = sum(np.sum(target.remaining_resources) for target in self.targets)
        
        if total_initial_resources <= 0:
            return 0.0
        
        progress_ratio = (total_initial_resources - total_remaining_resources) / total_initial_resources
        
        # 给予渐进式奖励
        if progress_ratio > 0.8:
            return 2.0 * (progress_ratio - 0.8) / 0.2  # 80%-100%时给予最高2分
        elif progress_ratio > 0.5:
            return 1.0 * (progress_ratio - 0.5) / 0.3  # 50%-80%时给予最高1分
        else:
            return 0.5 * progress_ratio / 0.5  # 0%-50%时给予最高0.5分
    
    def _calculate_normalization_impact(self, reward_components):
        """
        计算归一化对奖励的影响
        
        Args:
            reward_components: 奖励组成字典
            
        Returns:
            dict: 归一化影响分析
        """
        impact = {}
        
        # 计算归一化前后的差异
        for component in reward_components.get('normalization_applied', []):
            raw_key = f"{component}_raw"
            normalized_key = f"{component}_normalized"
            
            if raw_key in reward_components and normalized_key in reward_components:
                raw_value = reward_components[raw_key]
                normalized_value = reward_components[normalized_key]
                impact[component] = {
                    'raw': raw_value,
                    'normalized': normalized_value,
                    'difference': raw_value - normalized_value,
                    'reduction_ratio': (raw_value - normalized_value) / raw_value if raw_value != 0 else 0
                }
        
        return impact
    
    def _log_reward_components(self, reward_components):
        """
        记录详细的奖励组成信息（用于调试）
        
        Args:
            reward_components: 奖励组成字典
        """
        print(f"[奖励详情] Step {self.step_count}")
        print(f"  有效UAV数量: {reward_components['n_active_uavs']}")
        print(f"  最终奖励: {reward_components['final_reward']:.3f}")
        print(f"  正向奖励总计: {reward_components['total_positive']:.3f}")
        print(f"  成本总计: {reward_components['total_costs']:.3f}")
        
        # 输出归一化信息
        if reward_components['normalization_applied']:
            print(f"  归一化组件: {reward_components['normalization_applied']}")
            for component, impact in reward_components['per_agent_normalization']['normalization_impact'].items():
                print(f"    {component}: {impact['raw']:.3f} -> {impact['normalized']:.3f} (减少 {impact['reduction_ratio']:.1%})")

    def _calculate_collaboration_reward(self, target, uav):
        """
        计算协作塑形奖励
        
        鼓励合理的协作，避免过度集中或过度分散
        """
        collaboration_reward = 0.0
        
        # 获取当前分配到该目标的UAV数量
        current_uav_count = len(target.allocated_uavs)
        
        if current_uav_count > 0:
            # 计算目标的资源需求量
            target_demand = np.sum(target.resources)
            
            # 估算理想的UAV数量 (基于资源需求)
            avg_uav_capacity = 50.0  # 假设平均UAV容量
            ideal_uav_count = max(1, min(4, int(np.ceil(target_demand / avg_uav_capacity))))
            
            # 协作效率奖励
            if current_uav_count <= ideal_uav_count:
                # 理想协作范围内
                efficiency_factor = 1.0 - abs(current_uav_count - ideal_uav_count) / ideal_uav_count
                collaboration_reward = 1.0 * efficiency_factor
            else:
                # 过度协作，递减奖励
                over_collaboration_penalty = (current_uav_count - ideal_uav_count) * 0.2
                collaboration_reward = max(0.2, 1.0 - over_collaboration_penalty)
            
            # 多样性奖励: 如果UAV来自不同起始位置
            if current_uav_count > 1:
                diversity_bonus = 0.3  # 基础多样性奖励
                collaboration_reward += diversity_bonus
        
        return collaboration_reward
    
    def _calculate_global_progress_reward(self):
        """
        计算全局进度塑形奖励
        
        基于整体任务完成进度给予奖励，鼓励系统性进展
        """
        if not self.targets:
            return 0.0
        
        # 计算全局完成率
        total_demand = sum(np.sum(target.resources) for target in self.targets)
        total_remaining = sum(np.sum(target.remaining_resources) for target in self.targets)
        
        if total_demand <= 0:
            return 0.0
        
        completion_rate = (total_demand - total_remaining) / total_demand
        
        # 基于完成率的进度奖励
        progress_reward = 0.0
        
        # 里程碑奖励
        milestones = [0.25, 0.5, 0.75, 0.9]
        milestone_rewards = [0.5, 1.0, 1.5, 2.0]
        
        for milestone, reward in zip(milestones, milestone_rewards):
            if completion_rate >= milestone:
                # 检查是否刚达到这个里程碑
                if not hasattr(self, '_milestone_reached'):
                    self._milestone_reached = set()
                
                if milestone not in self._milestone_reached:
                    self._milestone_reached.add(milestone)
                    progress_reward += reward
        
        # 连续进度奖励 (平滑的进度激励)
        smooth_progress = 0.2 * completion_rate
        progress_reward += smooth_progress
        
        return progress_reward
    
    def _calculate_active_uav_count(self) -> int:
        """
        计算当前有效无人机数量，用于Per-Agent奖励归一化
        
        有效无人机定义：
        - 拥有剩余资源 (resources > 0)
        - 通信/感知系统正常 (is_alive = 1.0)
        
        Returns:
            int: 当前有效无人机数量 N_active
        """
        active_count = 0
        
        for i, uav in enumerate(self.uavs):
            # 检查是否有剩余资源
            has_resources = np.any(uav.resources > 0)
            
            # 检查通信/感知状态
            is_alive = self._calculate_uav_alive_status(uav, i)
            
            # 只有同时满足资源和通信条件的UAV才算有效
            if has_resources and is_alive >= 0.5:  # is_alive >= 0.5 表示至少部分功能正常
                active_count += 1
        
        # 确保至少有1个有效UAV，避免除零错误
        return max(active_count, 1)
    
    def _calculate_congestion_penalty(self, target, uav, n_active_uavs: int) -> float:
        """
        计算拥堵惩罚 - 与无人机数量相关的惩罚项
        
        拥堵惩罚的核心思想：
        1. 当多个UAV同时分配到同一目标时，产生拥堵
        2. 拥堵程度与分配到该目标的UAV数量成正比
        3. 该惩罚项会随着总UAV数量增加而增长，因此需要归一化
        
        Args:
            target: 目标对象
            uav: 当前UAV对象
            n_active_uavs: 当前有效无人机数量
            
        Returns:
            float: 拥堵惩罚值 (原始值，调用方负责归一化)
        """
        congestion_penalty = 0.0
        
        # 1. 目标拥堵惩罚：分配到同一目标的UAV过多
        allocated_uav_count = len(target.allocated_uavs)
        if allocated_uav_count > 1:
            # 计算理想分配数量
            target_demand = np.sum(target.resources)
            avg_uav_capacity = 50.0  # 假设平均UAV容量
            ideal_allocation = max(1, min(3, int(np.ceil(target_demand / avg_uav_capacity))))
            
            if allocated_uav_count > ideal_allocation:
                # 过度分配惩罚，随UAV数量线性增长
                over_allocation = allocated_uav_count - ideal_allocation
                congestion_penalty += over_allocation * 2.0  # 每个多余UAV惩罚2分
        
        # 2. 全局拥堵惩罚：系统整体UAV密度过高
        if n_active_uavs > len(self.targets) * 2:  # 如果UAV数量超过目标数量的2倍
            density_factor = n_active_uavs / (len(self.targets) * 2)
            global_congestion = (density_factor - 1.0) * 1.5  # 密度超标惩罚
            congestion_penalty += global_congestion
        
        # 3. 局部拥堵惩罚：计算当前UAV周围的拥堵情况
        local_congestion = self._calculate_local_congestion(uav, target)
        congestion_penalty += local_congestion
        
        return max(congestion_penalty, 0.0)  # 确保惩罚值非负
    
    def _calculate_local_congestion(self, uav, target) -> float:
        """
        计算局部拥堵情况
        
        Args:
            uav: 当前UAV对象
            target: 目标对象
            
        Returns:
            float: 局部拥堵惩罚值
        """
        local_congestion = 0.0
        congestion_radius = 200.0  # 拥堵检测半径
        
        # 统计在拥堵半径内的其他UAV数量
        nearby_uavs = 0
        for other_uav in self.uavs:
            if other_uav.id != uav.id:
                distance = np.linalg.norm(
                    np.array(other_uav.current_position) - np.array(uav.current_position)
                )
                if distance < congestion_radius:
                    nearby_uavs += 1
        
        # 如果附近UAV过多，产生拥堵惩罚
        if nearby_uavs > 2:  # 超过2个邻近UAV就算拥堵
            local_congestion = (nearby_uavs - 2) * 0.5  # 每个多余邻近UAV惩罚0.5分
        
        return local_congestion
    
    def _calculate_normalization_impact(self, reward_components: dict) -> dict:
        """
        计算归一化对奖励的影响程度
        
        Args:
            reward_components: 奖励组成字典
            
        Returns:
            dict: 归一化影响分析
        """
        impact = {
            'total_raw_normalized_rewards': 0.0,
            'total_normalized_rewards': 0.0,
            'normalization_savings': 0.0,
            'components_impact': {}
        }
        
        # 计算协作奖励的归一化影响
        if 'collaboration_raw' in reward_components and 'collaboration_normalized' in reward_components:
            raw_collab = reward_components['collaboration_raw']
            norm_collab = reward_components['collaboration_normalized']
            impact['total_raw_normalized_rewards'] += raw_collab
            impact['total_normalized_rewards'] += norm_collab
            impact['components_impact']['collaboration'] = {
                'raw': raw_collab,
                'normalized': norm_collab,
                'reduction': raw_collab - norm_collab
            }
        
        # 计算拥堵惩罚的归一化影响
        if 'congestion_penalty_raw' in reward_components and 'congestion_penalty_normalized' in reward_components:
            raw_congestion = abs(reward_components['congestion_penalty_raw'])
            norm_congestion = abs(reward_components['congestion_penalty_normalized'])
            impact['total_raw_normalized_rewards'] += raw_congestion
            impact['total_normalized_rewards'] += norm_congestion
            impact['components_impact']['congestion_penalty'] = {
                'raw': raw_congestion,
                'normalized': norm_congestion,
                'reduction': raw_congestion - norm_congestion
            }
        
        # 计算总的归一化节省
        impact['normalization_savings'] = impact['total_raw_normalized_rewards'] - impact['total_normalized_rewards']
        
        return impact
    
    def _log_reward_components(self, reward_components: dict):
        """
        记录奖励组成的详细日志，用于调试和监控
        
        Args:
            reward_components: 奖励组成字典
        """
        normalization_info = reward_components['per_agent_normalization']
        
        print(f"[Step {self.step_count}] Per-Agent奖励归一化详情:")
        print(f"  有效UAV数量: {normalization_info['n_active_uavs']}/{normalization_info['total_uavs']}")
        print(f"  归一化因子: {normalization_info['normalization_factor']:.4f}")
        print(f"  应用归一化的组件: {normalization_info['components_normalized']}")
        
        impact = normalization_info['normalization_impact']
        if impact['normalization_savings'] > 0:
            print(f"  归一化节省: {impact['normalization_savings']:.4f}")
            for component, details in impact['components_impact'].items():
                print(f"    {component}: {details['raw']:.4f} -> {details['normalized']:.4f} "
                      f"(减少 {details['reduction']:.4f})")
        
        print(f"  最终奖励: {reward_components['final_reward']:.4f}")
        print()
    
    def _calculate_uav_alive_status(self, uav, uav_idx: int) -> float:
        """
        计算UAV的存活状态，考虑通信/感知失效情况
        
        鲁棒性掩码机制的核心组件，用于标识无人机的通信/感知状态。
        在实际部署中，这可以基于：
        - 通信链路质量
        - 传感器状态
        - 电池电量
        - 系统健康状态
        
        Args:
            uav: UAV实体对象
            uav_idx: UAV索引
            
        Returns:
            float: 存活状态 (0.0=失效, 1.0=正常)
        """
        # 基础存活检查：是否有剩余资源
        has_resources = np.any(uav.resources > 0)
        if not has_resources:
            return 0.0
        
        # 模拟通信失效场景（可配置的失效概率）
        communication_failure_rate = getattr(self.config, 'UAV_COMM_FAILURE_RATE', 0.0)
        if communication_failure_rate > 0:
            # 使用确定性的伪随机数，基于step_count和uav_idx确保可复现性
            failure_seed = (self.step_count * 31 + uav_idx * 17) % 1000
            failure_prob = failure_seed / 1000.0
            if failure_prob < communication_failure_rate:
                return 0.0
        
        # 模拟感知系统失效（基于距离和环境复杂度）
        sensing_failure_rate = getattr(self.config, 'UAV_SENSING_FAILURE_RATE', 0.0)
        if sensing_failure_rate > 0:
            # 计算环境复杂度因子（障碍物密度、目标密度等）
            complexity_factor = self._calculate_environment_complexity(uav)
            adjusted_failure_rate = sensing_failure_rate * complexity_factor
            
            sensing_seed = (self.step_count * 23 + uav_idx * 19) % 1000
            sensing_prob = sensing_seed / 1000.0
            if sensing_prob < adjusted_failure_rate:
                return 0.0
        
        # 模拟电池电量影响的通信能力
        battery_threshold = getattr(self.config, 'UAV_LOW_BATTERY_THRESHOLD', 0.1)
        if hasattr(uav, 'battery_level'):
            if uav.battery_level < battery_threshold:
                # 低电量时通信能力下降，但不完全失效
                return 0.3
        
        # 模拟系统过载导致的响应延迟
        system_load = len([u for u in self.uavs if np.any(u.resources > 0)])
        max_concurrent_uavs = getattr(self.config, 'MAX_CONCURRENT_UAVS', 20)
        if system_load > max_concurrent_uavs:
            # 系统过载时，部分UAV可能响应延迟
            overload_factor = (system_load - max_concurrent_uavs) / max_concurrent_uavs
            if (uav_idx + self.step_count) % system_load < overload_factor * system_load:
                return 0.5  # 部分功能受限
        
        return 1.0  # 正常状态
    
    def _calculate_target_visibility_status(self, target, target_idx: int) -> float:
        """
        计算目标的可见性状态，考虑感知范围和环境遮挡
        
        鲁棒性掩码机制的核心组件，用于标识目标的可见性状态。
        在实际部署中，这可以基于：
        - 传感器感知范围
        - 环境遮挡（建筑物、地形）
        - 天气条件
        - 目标特性（大小、反射率等）
        
        Args:
            target: 目标实体对象
            target_idx: 目标索引
            
        Returns:
            float: 可见性状态 (0.0=不可见, 1.0=完全可见)
        """
        # 基础可见性检查：目标是否还有剩余资源
        has_remaining_resources = np.any(target.remaining_resources > 0)
        if not has_remaining_resources:
            return 0.0
        
        # 计算最近UAV到目标的距离
        min_distance = float('inf')
        closest_uav_alive = False
        
        for i, uav in enumerate(self.uavs):
            if np.any(uav.resources > 0):  # UAV仍然活跃
                dist = np.linalg.norm(
                    np.array(target.position) - np.array(uav.current_position)
                )
                if dist < min_distance:
                    min_distance = dist
                    # 检查最近的UAV是否处于正常通信状态
                    closest_uav_alive = self._calculate_uav_alive_status(uav, i) > 0.5
        
        # 如果没有活跃的UAV，目标不可见
        if min_distance == float('inf') or not closest_uav_alive:
            return 0.0
        
        # 基于距离的可见性衰减
        max_sensing_range = getattr(self.config, 'MAX_SENSING_RANGE', 1000.0)
        if min_distance > max_sensing_range:
            return 0.0
        
        # 距离衰减函数：近距离完全可见，远距离逐渐衰减
        distance_visibility = max(0.0, 1.0 - (min_distance / max_sensing_range) ** 2)
        
        # 模拟环境遮挡影响
        occlusion_rate = getattr(self.config, 'TARGET_OCCLUSION_RATE', 0.0)
        if occlusion_rate > 0:
            # 基于目标位置和环境复杂度计算遮挡概率
            occlusion_seed = (self.step_count * 37 + target_idx * 41) % 1000
            occlusion_prob = occlusion_seed / 1000.0
            
            # 环境复杂度影响遮挡概率
            env_complexity = self._calculate_target_environment_complexity(target)
            adjusted_occlusion_rate = occlusion_rate * env_complexity
            
            if occlusion_prob < adjusted_occlusion_rate:
                distance_visibility *= 0.2  # 遮挡时可见性大幅下降
        
        # 模拟天气条件影响
        weather_visibility = getattr(self.config, 'WEATHER_VISIBILITY_FACTOR', 1.0)
        distance_visibility *= weather_visibility
        
        # 模拟目标特性影响（大小、反射率等）
        target_detectability = getattr(target, 'detectability_factor', 1.0)
        distance_visibility *= target_detectability
        
        # 确保返回值在[0, 1]范围内
        return float(np.clip(distance_visibility, 0.0, 1.0))
    
    def _calculate_environment_complexity(self, uav) -> float:
        """
        计算UAV周围环境的复杂度因子
        
        用于调整通信/感知失效概率。环境越复杂，失效概率越高。
        
        Args:
            uav: UAV实体对象
            
        Returns:
            float: 环境复杂度因子 [0.5, 2.0]
        """
        complexity = 1.0
        
        # 障碍物密度影响
        if hasattr(self, 'obstacles') and self.obstacles:
            nearby_obstacles = 0
            search_radius = 200.0  # 搜索半径
            
            for obstacle in self.obstacles:
                if hasattr(obstacle, 'position'):
                    dist = np.linalg.norm(
                        np.array(uav.current_position) - np.array(obstacle.position)
                    )
                    if dist < search_radius:
                        nearby_obstacles += 1
            
            # 障碍物密度因子
            obstacle_density = nearby_obstacles / max(1, len(self.obstacles))
            complexity += obstacle_density * 0.5
        
        # UAV密度影响（通信干扰）
        nearby_uavs = 0
        interference_radius = 150.0
        
        for other_uav in self.uavs:
            if other_uav.id != uav.id and np.any(other_uav.resources > 0):
                dist = np.linalg.norm(
                    np.array(uav.current_position) - np.array(other_uav.current_position)
                )
                if dist < interference_radius:
                    nearby_uavs += 1
        
        # UAV密度因子
        uav_density = nearby_uavs / max(1, len(self.uavs) - 1)
        complexity += uav_density * 0.3
        
        # 目标密度影响（感知负载）
        nearby_targets = 0
        sensing_radius = 300.0
        
        for target in self.targets:
            if np.any(target.remaining_resources > 0):
                dist = np.linalg.norm(
                    np.array(uav.current_position) - np.array(target.position)
                )
                if dist < sensing_radius:
                    nearby_targets += 1
        
        # 目标密度因子
        target_density = nearby_targets / max(1, len(self.targets))
        complexity += target_density * 0.2
        
        # 限制复杂度因子范围
        return float(np.clip(complexity, 0.5, 2.0))
    
    def _calculate_target_environment_complexity(self, target) -> float:
        """
        计算目标周围环境的复杂度因子
        
        用于调整目标遮挡概率。环境越复杂，遮挡概率越高。
        
        Args:
            target: 目标实体对象
            
        Returns:
            float: 环境复杂度因子 [0.5, 2.0]
        """
        complexity = 1.0
        
        # 障碍物遮挡影响
        if hasattr(self, 'obstacles') and self.obstacles:
            nearby_obstacles = 0
            occlusion_radius = 100.0  # 遮挡影响半径
            
            for obstacle in self.obstacles:
                if hasattr(obstacle, 'position'):
                    dist = np.linalg.norm(
                        np.array(target.position) - np.array(obstacle.position)
                    )
                    if dist < occlusion_radius:
                        nearby_obstacles += 1
            
            # 障碍物遮挡因子
            occlusion_density = nearby_obstacles / max(1, len(self.obstacles))
            complexity += occlusion_density * 0.8
        
        # 其他目标的干扰影响
        nearby_targets = 0
        interference_radius = 80.0
        
        for other_target in self.targets:
            if (other_target.id != target.id and 
                np.any(other_target.remaining_resources > 0)):
                dist = np.linalg.norm(
                    np.array(target.position) - np.array(other_target.position)
                )
                if dist < interference_radius:
                    nearby_targets += 1
        
        # 目标干扰因子
        target_interference = nearby_targets / max(1, len(self.targets) - 1)
        complexity += target_interference * 0.3
        
        # 限制复杂度因子范围
        return float(np.clip(complexity, 0.5, 2.0))
    
    def _calculate_robust_masks(self) -> Dict[str, np.ndarray]:
        """
        计算增强的鲁棒性掩码，结合is_alive和is_visible位
        
        掩码机制的核心功能：
        1. 基础有效性掩码：基于资源状态
        2. 通信/感知掩码：基于is_alive和is_visible位
        3. 组合掩码：为TransformerGNN提供失效节点屏蔽能力
        4. 使用固定维度确保批处理兼容性
        
        Returns:
            Dict[str, np.ndarray]: 包含多层掩码的字典
        """
        # 使用固定的最大数量，确保维度一致性
        max_uavs = getattr(self.config, 'MAX_UAVS', 10)
        max_targets = getattr(self.config, 'MAX_TARGETS', 15)
        
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # === 基础有效性掩码 ===
        # UAV基础掩码：基于资源状态，使用固定维度
        uav_resource_mask = np.zeros(max_uavs, dtype=np.int32)
        for i, uav in enumerate(self.uavs):
            uav_resource_mask[i] = 1 if np.any(uav.resources > 0) else 0
        
        # 目标基础掩码：基于剩余资源状态，使用固定维度
        target_resource_mask = np.zeros(max_targets, dtype=np.int32)
        for i, target in enumerate(self.targets):
            target_resource_mask[i] = 1 if np.any(target.remaining_resources > 0) else 0
        
        # === 通信/感知掩码 ===
        # UAV通信掩码：基于is_alive位，使用固定维度
        uav_communication_mask = np.zeros(max_uavs, dtype=np.int32)
        for i, uav in enumerate(self.uavs):
            uav_communication_mask[i] = 1 if self._calculate_uav_alive_status(uav, i) > 0.5 else 0
        
        # 目标可见性掩码：基于is_visible位，使用固定维度
        target_visibility_mask = np.zeros(max_targets, dtype=np.int32)
        for i, target in enumerate(self.targets):
            target_visibility_mask[i] = 1 if self._calculate_target_visibility_status(target, i) > 0.5 else 0
        
        # === 组合掩码（用于TransformerGNN） ===
        # UAV有效掩码：同时满足资源和通信条件
        uav_effective_mask = uav_resource_mask & uav_communication_mask
        
        # 目标有效掩码：同时满足资源和可见性条件
        target_effective_mask = target_resource_mask & target_visibility_mask
        
        # === 交互掩码 ===
        # UAV-目标交互掩码 [max_uavs, max_targets]：标识哪些UAV-目标对可以进行有效交互
        interaction_mask = np.zeros((max_uavs, max_targets), dtype=np.int32)
        
        for i in range(n_uavs):
            for j in range(n_targets):
                # 只有当UAV有效且目标有效时，才能进行交互
                if uav_effective_mask[i] == 1 and target_effective_mask[j] == 1:
                    # 额外检查距离约束
                    uav = self.uavs[i]
                    target = self.targets[j]
                    dist = np.linalg.norm(
                        np.array(target.position) - np.array(uav.current_position)
                    )
                    max_interaction_range = getattr(self.config, 'MAX_INTERACTION_RANGE', 2000.0)
                    
                    if dist <= max_interaction_range:
                        interaction_mask[i, j] = 1
        
        # 构建完整的掩码字典
        masks = {
            # 基础掩码（向后兼容）
            "uav_mask": uav_effective_mask,
            "target_mask": target_effective_mask,
            
            # 详细掩码（用于调试和分析）
            "uav_resource_mask": uav_resource_mask,
            "uav_communication_mask": uav_communication_mask,
            "target_resource_mask": target_resource_mask,
            "target_visibility_mask": target_visibility_mask,
            
            # 交互掩码（用于TransformerGNN的注意力计算）
            "interaction_mask": interaction_mask,
            
            # 统计信息（用于监控和调试）
            "active_uav_count": np.sum(uav_effective_mask),
            "visible_target_count": np.sum(target_effective_mask),
            "total_interactions": np.sum(interaction_mask)
        }
        
        return masks