# -*- coding: utf-8 -*-
# 文件名: main.py
# 描述: 多无人机协同任务分配与路径规划的最终集成算法。
#      包含了从环境定义、强化学习求解、路径规划到结果可视化的完整流程。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from collections import deque, defaultdict
import os
import time
import pickle
import random
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F

# --- 本地模块导入 ---
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_small_scenario, get_complex_scenario, get_new_experimental_scenario, get_complex_scenario_v4, get_strategic_trap_scenario
from config import Config
from evaluate import evaluate_plan

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # 如果detach方法不可用，跳过编码设置
        pass

# 初始化配置类
config = Config()

# =============================================================================
# section 1: 全局辅助函数与字体设置
# =============================================================================
def set_chinese_font(preferred_fonts=None, manual_font_path=None):
    """
    (增强版) 设置matplotlib支持中文显示的字体，以避免乱码和警告。

    Args:
        preferred_fonts (list, optional): 优先尝试的字体名称列表。 Defaults to None.
        manual_font_path (str, optional): 手动指定的字体文件路径，具有最高优先级。 Defaults to None.

    Returns:
        bool: 是否成功设置字体。
    """
    if manual_font_path and os.path.exists(manual_font_path):
        try:
            font_prop = FontProperties(fname=manual_font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功加载手动指定的字体: {manual_font_path}")
            return True
        except Exception as e:
            print(f"加载手动指定字体失败: {e}")
    
    if preferred_fonts is None:
        preferred_fonts = ['Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'KaiTi', 'FangSong']
    
    try:
        for font in preferred_fonts:
            if findfont(FontProperties(family=font)):
                plt.rcParams["font.family"] = font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"已自动设置中文字体为: {font}")
                return True
    except Exception:
        pass
    
    print("警告: 自动或手动设置中文字体失败。图片中的中文可能显示为方框。")
    return False

# 文件名: main.py
# ... (main.py 文件其他部分无改动) ...

# =============================================================================
# section 3: 核心业务逻辑 - 实体与环境
# =============================================================================
class DirectedGraph:
    """
    (最终修复版) 构建任务场景的有向图表示。
    此版本已更新，可以在构建图时感知障碍物，为所有算法提供更真实的距离估算。
    """
    def __init__(self, uavs: List[UAV], targets: List[Target], n_phi: int, obstacles: List = []):
        """
        构造函数。
        
        Args:
            uavs (List[UAV]): 无人机列表。
            targets (List[Target]): 目标列表。
            n_phi (int): 离散化角度数量。
            obstacles (List, optional): 场景中的障碍物列表。 Defaults to [].
        """
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi
        self.obstacles = obstacles
        self.phi_set = [2 * np.pi * i / n_phi for i in range(n_phi)]
        self.vertices = self._create_vertices()
        all_vertices_flat = sum(self.vertices['UAVs'].values(), []) + sum(self.vertices['Targets'].values(), [])
        self.vertex_to_idx = {v: i for i, v in enumerate(all_vertices_flat)}
        self.edges = self._create_edges()
        self.adjacency_matrix = self._create_adjacency_matrix()

    def _create_vertices(self) -> Dict:
        return {'UAVs': {u.id: [(-hash(u.id), None)] for u in self.uavs}, 'Targets': {t.id: [(hash(t.id), p) for p in self.phi_set] for t in self.targets}}

    def _create_edges(self) -> List:
        edges = []
        uav_vs = sum(self.vertices['UAVs'].values(), [])
        target_vs = sum(self.vertices['Targets'].values(), [])
        for uav_v in uav_vs:
            for target_v in target_vs:
                edges.append((uav_v, target_v))
        for t1_v in target_vs:
            for t2_v in target_vs:
                if t1_v[0] != t2_v[0]:
                    edges.append((t1_v, t2_v))
        return edges

    def _create_adjacency_matrix(self) -> np.ndarray:
        """
        在计算距离时，增加直线碰撞检测。如果两点间直线路径穿过障碍物，则距离为无穷大。
        """
        n = len(self.vertex_to_idx)
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0)
        
        pos_cache = {**{next(iter(self.vertices['UAVs'][u.id])): u.position for u in self.uavs}, 
                     **{v: t.position for t in self.targets for v in self.vertices['Targets'][t.id]}}

        for start_v, end_v in self.edges:
            p1, p2 = pos_cache[start_v], pos_cache[end_v]
            
            has_collision = False
            if self.obstacles:
                for obs in self.obstacles:
                    if obs.check_line_segment_collision(p1, p2):
                        has_collision = True
                        break
            
            if has_collision:
                adj[self.vertex_to_idx[start_v], self.vertex_to_idx[end_v]] = np.inf
            else:
                adj[self.vertex_to_idx[start_v], self.vertex_to_idx[end_v]] = np.linalg.norm(p2 - p1)
                
        return adj



class UAVTaskEnv:
    """强化学习环境：定义状态、动作、奖励和转移逻辑"""
    def __init__(self, uavs, targets, graph, obstacles, config):
        self.uavs, self.targets, self.graph, self.obstacles, self.config = uavs, targets, graph, obstacles, config
        self.load_balance_penalty = config.LOAD_BALANCE_PENALTY
        self.alliance_bonus = 100.0  # 协作奖励大幅提升
        self.use_phrrt_in_training = config.USE_PHRRT_DURING_TRAINING
        # 新增：无效行动惩罚参数
        self.invalid_action_penalty = -50.0  # 无效分配的重惩罚
        self.marginal_utility_threshold = 0.8  # 边际效用递减阈值
        self.crowding_penalty_factor = 0.5  # 拥挤惩罚因子
        self.reset()
    
    def reset(self):
        for uav in self.uavs: uav.reset()
        for target in self.targets: target.reset()
        return self._get_state()
    
    def _get_state(self):
        """
        增强版状态表示，包含协同信息和边际效用信息
        """
        state = []
        
        # 目标信息：位置、剩余资源、总资源
        for t in self.targets: 
            state.extend([*t.position, *t.remaining_resources, *t.resources])
        
        # 无人机信息：位置、资源、航向、距离
        for u in self.uavs: 
            state.extend([*u.current_position, *u.resources, u.heading, u.current_distance])
        
        # === 新增：协同信息 ===
        for t in self.targets:
            # 目标拥挤度：已分配无人机数量
            allocated_uav_count = len(t.allocated_uavs)
            state.append(allocated_uav_count)
            
            # 目标完成度：剩余资源比例
            completion_ratio = 1.0 - (np.sum(t.remaining_resources) / np.sum(t.resources))
            state.append(completion_ratio)
            
            # 目标边际效用：接近完成时的效率递减
            marginal_utility = 1.0
            if completion_ratio > self.marginal_utility_threshold:
                # 当完成度超过阈值时，边际效用递减
                excess_ratio = (completion_ratio - self.marginal_utility_threshold) / (1.0 - self.marginal_utility_threshold)
                marginal_utility = 1.0 - (excess_ratio * self.crowding_penalty_factor)
            state.append(marginal_utility)
            
            # 目标资源需求紧迫度：剩余资源与总资源的比例
            urgency_ratio = np.sum(t.remaining_resources) / np.sum(t.resources)
            state.append(urgency_ratio)
        
        # === 新增：全局协同信息 ===
        # 总体完成度
        total_completion = sum(1.0 - (np.sum(t.remaining_resources) / np.sum(t.resources)) for t in self.targets) / len(self.targets)
        state.append(total_completion)
        
        # 资源分配均衡度
        uav_utilizations = []
        for u in self.uavs:
            initial_total = np.sum(u.initial_resources)
            current_total = np.sum(u.resources)
            if initial_total > 0:
                utilization = 1.0 - (current_total / initial_total)
                uav_utilizations.append(utilization)
        
        if uav_utilizations:
            balance_score = 1.0 - np.std(uav_utilizations)  # 标准差越小，均衡度越高
            state.append(balance_score)
        else:
            state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        target_id, uav_id, phi_idx = action
        target = next((t for t in self.targets if t.id == target_id), None)
        uav = next((u for u in self.uavs if u.id == uav_id), None)
        
        if not target or not uav: 
            return self._get_state(), -100, True, {}
        
        # 计算实际贡献
        actual_contribution = np.minimum(target.remaining_resources, uav.resources)
        
        # === 增强：无效行动检测和惩罚 ===
        if np.all(actual_contribution <= 0): 
            # 无效分配：目标已完成或无人机资源不足
            # 增加更详细的惩罚信息
            target_completion = 1.0 - (np.sum(target.remaining_resources) / np.sum(target.resources))
            uav_resources = np.sum(uav.resources)
            
            # 根据具体情况调整惩罚力度
            if target_completion >= 1.0:
                # 目标已完成，惩罚更重
                penalty = self.invalid_action_penalty * 1.5
                reason = 'target_already_completed'
            elif uav_resources <= 0:
                # 无人机无资源，惩罚更重
                penalty = self.invalid_action_penalty * 1.5
                reason = 'uav_no_resources'
            else:
                # 其他无效分配
                penalty = self.invalid_action_penalty
                reason = 'no_contribution'
            
            return self._get_state(), penalty, False, {
                'invalid_reason': reason,
                'target_completion': target_completion,
                'uav_resources': uav_resources,
                'penalty_multiplier': penalty / self.invalid_action_penalty
            }
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 计算路径长度（保留原有设计）
        if self.use_phrrt_in_training:
            start_heading = uav.heading if not uav.task_sequence else self.graph.phi_set[uav.task_sequence[-1][1]]
            planner = PHCurveRRTPlanner(uav.current_position, target.position, start_heading, self.graph.phi_set[phi_idx], self.obstacles, self.config)
            plan_result = planner.plan()
            path_len = plan_result[1] if plan_result else np.linalg.norm(uav.current_position - target.position)
        else:
            path_len = np.linalg.norm(uav.current_position - target.position)
        
        # 计算旅行时间
        travel_time = path_len / uav.velocity_range[1]
        
        # 更新状态
        uav.resources = uav.resources.astype(np.float64) - actual_contribution.astype(np.float64)
        target.remaining_resources = target.remaining_resources.astype(np.float64) - actual_contribution.astype(np.float64)
        if uav_id not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav_id, phi_idx))
        uav.task_sequence.append((target_id, phi_idx))
        uav.current_position = target.position
        uav.heading = self.graph.phi_set[phi_idx]
        
        # 检查是否完成所有目标 - 确保返回Python bool类型
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        total_targets = len(self.targets)
        done = bool(total_satisfied == total_targets)  # 强制转换为Python bool
        
        # === 精细化奖励函数设计 ===
        
        # 1. 目标完成奖励（最高优先级）
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 0
        if new_satisfied:
            target_completion_reward = 1500  # 增加新完成目标的奖励
        
        # 2. 边际效用递减奖励（新增）
        target_initial_total = np.sum(target.resources)
        target_remaining_before = np.sum(target.remaining_resources + actual_contribution)
        target_remaining_after = np.sum(target.remaining_resources)
        completion_ratio_before = 1.0 - (target_remaining_before / target_initial_total)
        completion_ratio_after = 1.0 - (target_remaining_after / target_initial_total)
        completion_improvement = completion_ratio_after - completion_ratio_before
        
        # 边际效用递减：当目标接近完成时，奖励递减
        marginal_utility_factor = 1.0
        if completion_ratio_after > self.marginal_utility_threshold:
            excess_ratio = (completion_ratio_after - self.marginal_utility_threshold) / (1.0 - self.marginal_utility_threshold)
            marginal_utility_factor = 1.0 - (excess_ratio * self.crowding_penalty_factor)
        
        completion_progress_reward = completion_improvement * 800 * marginal_utility_factor
        
        # 3. 资源满足度奖励（考虑边际效用）
        resource_satisfaction_ratio = np.sum(actual_contribution) / np.sum(target.remaining_resources + actual_contribution)
        resource_satisfaction_reward = resource_satisfaction_ratio * 200 * marginal_utility_factor
        
        # 4. 协作奖励（增强，但考虑拥挤效应）
        collaboration_bonus = 0
        if len(target.allocated_uavs) > 1:  # 多无人机协作
            # 协作奖励，但考虑拥挤效应
            crowding_factor = 1.0
            if len(target.allocated_uavs) > 2:  # 超过2个无人机时开始拥挤
                crowding_factor = max(0.5, 1.0 - (len(target.allocated_uavs) - 2) * 0.2)
            collaboration_bonus = self.alliance_bonus * len(target.allocated_uavs) * 0.5 * crowding_factor
        
        # 5. 效率奖励（新增）- 精细化设计
        efficiency_reward = 0
        if travel_time > 0:
            efficiency_reward = 100 / (travel_time + 1)  # 时间越短奖励越高
        
        # 6. 负载均衡奖励（轻微）
        load_balance_reward = 0
        if len(target.allocated_uavs) > 1:
            uav_contributions = []
            for uav_id, _ in target.allocated_uavs:
                uav = next((u for u in self.uavs if u.id == uav_id), None)
                if uav:
                    contribution = np.sum(np.minimum(target.resources, uav.initial_resources))
                    uav_contributions.append(contribution)
            if uav_contributions:
                std_contribution = np.std(uav_contributions)
                load_balance_reward = max(0, int(50 - std_contribution))  # 贡献越均衡奖励越高
        
        # 7. 路径惩罚（轻微）
        path_penalty = -travel_time * 0.1  # 轻微惩罚长路径
        
        # 8. 新增：拥挤惩罚（当目标过于拥挤时）
        crowding_penalty = 0
        if len(target.allocated_uavs) > 3:  # 超过3个无人机时惩罚
            crowding_penalty = -(len(target.allocated_uavs) - 3) * 30
        
        # 9. 新增：效率导向奖励 - 与总任务完成时间成反比
        completion_bonus = 0
        if done:
            # 计算总任务完成时间（简化版）
            total_time = sum(len(uav.task_sequence) for uav in self.uavs)
            if total_time > 0:
                completion_bonus = 50000 / total_time  # 完成时间越短奖励越高
        
        # 10. 新增：资源节约奖励
        resource_conservation_bonus = 0
        if done:
            total_remaining_resources = sum(np.sum(uav.resources) for uav in self.uavs)
            total_initial_resources = sum(np.sum(uav.initial_resources) for uav in self.uavs)
            if total_initial_resources > 0:
                conservation_ratio = total_remaining_resources / total_initial_resources
                resource_conservation_bonus = conservation_ratio * 1000  # 剩余资源越多奖励越高
        
        # 11. 新增：协同失败惩罚
        sync_failure_penalty = 0
        if len(target.allocated_uavs) > 1:
            # 简化版协同时间差预估
            uav_times = []
            for uav_id, _ in target.allocated_uavs:
                uav = next((u for u in self.uavs if u.id == uav_id), None)
                if uav:
                    uav_times.append(len(uav.task_sequence))
            if len(uav_times) > 1:
                time_diff = max(uav_times) - min(uav_times)
                if time_diff > 3:  # 时间差过大时惩罚
                    sync_failure_penalty = -time_diff * 50
        
        # 综合奖励
        total_reward = (
            target_completion_reward +      # 目标完成（最高权重）
            completion_progress_reward +    # 渐进完成（考虑边际效用）
            resource_satisfaction_reward +  # 资源满足（考虑边际效用）
            collaboration_bonus +           # 协作奖励（考虑拥挤）
            efficiency_reward +             # 效率奖励
            load_balance_reward +           # 负载均衡
            path_penalty +                  # 路径惩罚
            crowding_penalty +              # 拥挤惩罚（新增）
            completion_bonus +              # 效率导向奖励（新增）
            resource_conservation_bonus +   # 资源节约奖励（新增）
            sync_failure_penalty            # 协同失败惩罚（新增）
        )
        
        # 奖励缩放 - 将大奖励缩放到合理范围
        if hasattr(self.config, 'reward_scaling_factor') and self.config.reward_scaling_factor > 0:
            total_reward = total_reward / self.config.reward_scaling_factor
        else:
            # 默认奖励缩放
            total_reward = total_reward / 1000.0  # 将大奖励缩放到合理范围
        
        # 限制奖励范围，防止梯度爆炸
        min_reward = getattr(self.config, 'min_reward', -10.0)
        max_reward = getattr(self.config, 'max_reward', 10.0)
        total_reward = np.clip(total_reward, min_reward, max_reward)
        
        return self._get_state(), total_reward, done, {
            'marginal_utility_factor': marginal_utility_factor,
            'crowding_factor': len(target.allocated_uavs),
            'completion_ratio': completion_ratio_after,
            'actual_contribution': np.sum(actual_contribution),
            'completion_bonus': completion_bonus,
            'resource_conservation_bonus': resource_conservation_bonus,
            'sync_failure_penalty': sync_failure_penalty
        }


# =============================================================================
# section 4: 强化学习求解器 - 优化版网络架构
# =============================================================================

class OptimizedDeepFCN(nn.Module):
    """优化后的深度全连接网络 - 解决梯度爆炸问题"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(OptimizedDeepFCN, self).__init__()
        
        # 限制网络宽度，避免过宽层
        max_width = 128  # 减少最大宽度
        hidden_dims = [min(dim, max_width) for dim in hidden_dims]
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm，更稳定
                nn.GELU(),  # 使用GELU激活函数，更稳定
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化 - 使用更保守的初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，但降低gain值
                nn.init.xavier_uniform_(module.weight, gain=0.3)  # 降低gain值
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

class OptimizedDeepFCN_Residual(nn.Module):
    """优化后的残差深度全连接网络"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(OptimizedDeepFCN_Residual, self).__init__()
        
        # 限制网络宽度
        max_width = 128
        hidden_dims = [min(dim, max_width) for dim in hidden_dims]
        
        self.layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 主层
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
            
            # 残差投影层
            if i == 0:  # 第一层需要投影输入维度
                residual_proj = nn.Linear(input_dim, hidden_dim)
            else:
                residual_proj = nn.Linear(hidden_dims[i-1], hidden_dim)
            self.residual_projections.append(residual_proj)
            
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        residual_input = x
        
        for i, (layer, residual_proj) in enumerate(zip(self.layers, self.residual_projections)):
            # 主路径
            main_output = layer(x)
            
            # 残差路径
            if i == 0:
                residual_output = residual_proj(residual_input)
            else:
                residual_output = residual_proj(x)
            
            # 残差连接
            x = main_output + residual_output
            
            # 激活函数
            x = F.gelu(x)
        
        return self.output_layer(x)

class OptimizedGNN(nn.Module):
    """优化后的图神经网络 - 解决梯度爆炸问题"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(OptimizedGNN, self).__init__()
        
        # 减少嵌入维度，避免过大的参数
        self.embedding_dim = 64  # 从128减少到64
        self.num_heads = 2       # 从4减少到2
        
        # 实体编码器 - 使用更保守的架构
        self.uav_encoder = nn.Sequential(
            nn.Linear(6, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(8, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 注意力机制 - 使用更稳定的配置
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 全局上下文编码器 - 简化架构
        self.global_context_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 动作解码器 - 使用更保守的架构
        policy_input_dim = self.embedding_dim * 2 + 16
        self.action_decoder = nn.Sequential(
            nn.Linear(policy_input_dim, 64),  # 减少宽度
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),  # 减少宽度
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _extract_entity_features(self, state):
        """提取实体特征"""
        # 动态特征提取，适应不同的状态维度
        state_dim = state.size(-1)
        
        # 假设状态向量包含：目标信息 + 无人机信息 + 协同信息
        # 为了简化，我们取前几个特征作为示例
        if state_dim >= 14:
            uav_features = state[:6]  # 前6个特征为UAV特征
            target_features = state[6:14]  # 接下来8个特征为目标特征
        else:
            # 如果状态维度不够，使用零填充
            uav_features = torch.zeros(6, device=state.device)
            target_features = torch.zeros(8, device=state.device)
            if state_dim >= 6:
                uav_features[:state_dim] = state[:state_dim]
            if state_dim >= 14:
                target_features[:8] = state[6:14]
        
        return uav_features, target_features
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 提取实体特征
        uav_features, target_features = self._extract_entity_features(x[0])
        
        # 编码实体
        uav_embedding = self.uav_encoder(uav_features.unsqueeze(0))
        target_embedding = self.target_encoder(target_features.unsqueeze(0))
        
        # 注意力机制
        combined_embeddings = torch.cat([uav_embedding, target_embedding], dim=1)
        attended_embeddings, _ = self.attention(
            combined_embeddings, combined_embeddings, combined_embeddings
        )
        
        # 全局上下文
        global_context = self.global_context_encoder(target_embedding.mean(dim=1))
        
        # 合并特征
        policy_input = torch.cat([
            uav_embedding.squeeze(),
            attended_embeddings.mean(dim=1),
            global_context.squeeze()
        ])
        
        # 输出Q值
        q_values = self.action_decoder(policy_input)
        
        # 扩展到批次维度
        q_values = q_values.unsqueeze(0).expand(batch_size, -1)
        
        return q_values

# =============================================================================
# section 4: 强化学习求解器 - 简化版深度全连接网络
# =============================================================================
class DeepFCN(nn.Module):
    """深度全连接网络 - 基础版本"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(DeepFCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 构建网络层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        # 通过隐藏层
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output


class DeepFCN_Residual(nn.Module):
    """深度全连接网络 - 带残差连接版本，优化损失监控"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(DeepFCN_Residual, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 构建网络层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # 添加隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 残差连接的投影层（当维度不匹配时）
        self.residual_projections = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            if current_dim != hidden_dim:
                self.residual_projections.append(nn.Linear(current_dim, hidden_dim))
            else:
                self.residual_projections.append(nn.Identity())
            current_dim = hidden_dim
        
        # 损失监控参数
        self.loss_history = []
        self.gradient_norms = []
        self.layer_activations = []
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重 - 使用更稳定的初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Kaiming初始化，更适合ReLU激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _monitor_layer_activations(self, x, layer_name):
        """监控层激活值"""
        with torch.no_grad():
            activation_norm = torch.norm(x).item()
            self.layer_activations.append({
                'layer': layer_name,
                'norm': activation_norm,
                'mean': x.mean().item(),
                'std': x.std().item()
            })
    
    def forward(self, x):
        """前向传播 - 带残差连接和监控"""
        residual = x
        
        # 通过隐藏层
        for i, layer in enumerate(self.layers):
            # 主路径
            out = layer(x)
            
            # 监控激活值
            self._monitor_layer_activations(out, f'layer_{i}')
            
            # 残差连接
            if i < len(self.residual_projections):
                residual_proj = self.residual_projections[i](residual)
                out = out + residual_proj
            
            x = out
            residual = out  # 更新残差连接
        
        # 输出层
        output = self.output_layer(x)
        
        return output
    
    def get_monitoring_info(self):
        """获取监控信息"""
        return {
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'layer_activations': self.layer_activations
        }
    
    def clear_monitoring_data(self):
        """清除监控数据"""
        self.loss_history.clear()
        self.gradient_norms.clear()
        self.layer_activations.clear()


class GNN(nn.Module):
    """图神经网络 - 增强版实体嵌入与注意力机制"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(GNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 增强的网络参数
        self.embedding_dim = 128  # 增加嵌入维度 (从64到128)
        self.num_heads = 4        # 增加注意力头数 (从2到4)
        
        # 计算实体特征维度 - 修复版本
        # 假设状态向量平均分配给UAV和目标
        entity_feature_dim = input_dim // 2
        
        # 第一步：实体分别编码 - 增强版
        self.uav_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, 256),  # 增加隐藏层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        # UAV编码器的残差连接
        self.uav_residual = nn.Linear(entity_feature_dim, self.embedding_dim)
        
        self.target_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, 256),  # 增加隐藏层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        # 目标编码器的残差连接
        self.target_residual = nn.Linear(entity_feature_dim, self.embedding_dim)
        
        # 第二步：增强的注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,  # 增加到4个头
            dropout=dropout,
            batch_first=True
        )
        
        # 全局上下文编码器
        self.global_context_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32)
        )
        
        # 第三步：增强的动作Q值解码器
        # 计算正确的输入维度
        policy_input_dim = self.embedding_dim * 2 + 32  # uav_embedding + context + g_context
        self.action_decoder = nn.Sequential(
            nn.Linear(policy_input_dim, 256),  # 使用正确的输入维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)  # 每个UAV对每个目标的Q值
        )
        
        # 动作解码器的残差连接
        self.action_residual = nn.Linear(policy_input_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重 - 使用更稳定的初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Kaiming初始化，更适合ReLU激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _extract_entity_features(self, state):
        """从状态向量中提取实体特征 - 修复版本"""
        batch_size = state.shape[0]
        feature_dim = state.shape[1]
        
        # 动态分割状态向量
        # 假设状态向量前半部分是UAV相关，后半部分是目标相关
        split_point = feature_dim // 2
        
        uav_features = state[:, :split_point]
        target_features = state[:, split_point:]
        
        return uav_features, target_features
    
    def forward(self, x):
        """前向传播 - 增强版实体嵌入与注意力机制"""
        batch_size = x.shape[0]
        
        # 处理BatchNorm在单样本时的问题
        if batch_size == 1:
            # 单样本推理时，临时切换到eval模式
            training_mode = self.training
            self.eval()
        else:
            training_mode = None
        
        try:
            # 第一步：实体分别编码
            uav_features, target_features = self._extract_entity_features(x)
            
            # 编码UAV嵌入 - 带残差连接
            uav_embedding_main = self.uav_encoder(uav_features)  # [batch, embedding_dim]
            uav_embedding_residual = self.uav_residual(uav_features)  # [batch, embedding_dim]
            uav_embedding = uav_embedding_main + uav_embedding_residual  # 残差连接
            
            # 编码目标嵌入 - 带残差连接
            target_embedding_main = self.target_encoder(target_features)  # [batch, embedding_dim]
            target_embedding_residual = self.target_residual(target_features)  # [batch, embedding_dim]
            target_embedding = target_embedding_main + target_embedding_residual  # 残差连接
            
            # 扩展为序列形式用于注意力计算
            uav_embedding = uav_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
            target_embedding = target_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
            
            # 第二步：增强的注意力关系聚合
            # 使用UAV嵌入作为query，目标嵌入作为key和value
            attn_output, _ = self.attention(
                query=uav_embedding,
                key=target_embedding,
                value=target_embedding
            )
            
            context_vector = attn_output.squeeze(1)  # [batch, embedding_dim]
            
            # 第三步：全局上下文感知 - 新增
            # 对所有目标嵌入进行平均池化，得到全局上下文
            global_context = torch.mean(target_embedding, dim=1)  # [batch, embedding_dim]
            g_context = self.global_context_encoder(global_context)  # [batch, 32]
            
            # 第四步：增强的动作Q值解码
            # 拼接UAV嵌入、上下文向量和全局上下文
            policy_input = torch.cat([
                uav_embedding.squeeze(1),  # [batch, embedding_dim]
                context_vector,             # [batch, embedding_dim]
                g_context                   # [batch, 32]
            ], dim=1)  # [batch, embedding_dim*2 + 32]
            
            # 检查维度匹配
            expected_dim = self.embedding_dim * 2 + 32  # 128*2 + 32 = 288
            actual_dim = policy_input.shape[1]
            if actual_dim != expected_dim:
                print(f"⚠️  维度不匹配: 实际={actual_dim}, 期望={expected_dim}")
                # 调整action_decoder的输入维度
                if hasattr(self, '_action_decoder_fixed'):
                    action_decoder = self._action_decoder_fixed
                    action_residual = self._action_residual_fixed
                else:
                    # 创建新的action_decoder
                    self._action_decoder_fixed = nn.Sequential(
                        nn.Linear(actual_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(256),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(128),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(64),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.BatchNorm1d(32),
                        nn.Linear(32, 1)
                    )
                    self._action_residual_fixed = nn.Linear(actual_dim, 1)
                    action_decoder = self._action_decoder_fixed
                    action_residual = self._action_residual_fixed
            else:
                action_decoder = self.action_decoder
                action_residual = self.action_residual
            
            # 解码Q值 - 带残差连接
            q_value_main = action_decoder(policy_input)  # [batch, 1]
            q_value_residual = action_residual(policy_input)  # [batch, 1]
            q_value = q_value_main + q_value_residual  # 残差连接
            
            # 扩展到完整的动作空间 - 修复维度匹配问题
            # 确保输出维度与预期一致
            if batch_size == 1:
                # 单样本推理
                expanded_q_values = q_value.expand(1, self.output_dim)
            else:
                # 批量训练
                expanded_q_values = q_value.expand(batch_size, self.output_dim)
            
            return expanded_q_values
            
        finally:
            # 恢复训练模式
            if training_mode is not None:
                if training_mode:
                    self.train()
                else:
                    self.eval()


class GraphRLSolver:
    """强化学习求解器 - 支持多种网络架构"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, network_type='DeepFCN'):
        """初始化强化学习求解器"""
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络参数
        self.input_dim = i_dim
        self.output_dim = o_dim
        
        # 修复hidden_dims参数处理
        if isinstance(h_dim, list):
            # 如果传入的是列表，直接使用
            hidden_dims = h_dim
        else:
            # 如果传入的是单个数字，构建三层隐藏层
            hidden_dims = [h_dim, h_dim // 2, h_dim // 4]
        
        self.hidden_dims = hidden_dims
        
        # 创建环境
        self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config)
        
        # 根据网络类型创建网络
        if network_type == 'DeepFCN':
            self.q_network = DeepFCN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'DeepFCN_Residual':
            self.q_network = DeepFCN_Residual(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'GNN':
            self.q_network = GNN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'OptimizedDeepFCN':
            self.q_network = OptimizedDeepFCN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'OptimizedDeepFCN_Residual':
            self.q_network = OptimizedDeepFCN_Residual(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'OptimizedGNN':
            self.q_network = OptimizedGNN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")
        
        # 创建目标网络
        if network_type == 'DeepFCN':
            self.target_network = DeepFCN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'DeepFCN_Residual':
            self.target_network = DeepFCN_Residual(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'GNN':
            self.target_network = GNN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'OptimizedDeepFCN':
            self.target_network = OptimizedDeepFCN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'OptimizedDeepFCN_Residual':
            self.target_network = OptimizedDeepFCN_Residual(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        elif network_type == 'OptimizedGNN':
            self.target_network = OptimizedGNN(self.input_dim, self.hidden_dims, self.output_dim).to(self.device)
        
        # 复制权重
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 训练参数
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        
        # 优化器 - 根据网络类型选择不同的优化策略
        if network_type.startswith('Optimized'):
            # 优化网络使用更保守的设置，但保持合理的学习率
            self.optimizer = torch.optim.AdamW(
                self.q_network.parameters(), 
                lr=0.0005,  # 适中的学习率，避免过低
                weight_decay=1e-4,  # 适度的权重衰减
                betas=(0.9, 0.999), 
                eps=1e-8
            )
        else:
            # 原始网络使用原有设置
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        
        # 经验回放
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # 训练历史
        self.train_history = {
            'episode_rewards': [],
            'episode_losses': [],
            'epsilon_values': [],
            'learning_rates': [],
            'gradient_norms': []
        }
        
        # 早停检测参数
        self.min_episodes_before_detection = getattr(config, 'min_episodes_before_detection', 100)
        self.reward_history = []
        self.loss_history = []
        self.epsilon_history = []
        
        # 自适应参数
        self.adaptive_window_adjustment = getattr(config, 'adaptive_window_adjustment', True)
        self.adaptive_threshold_adjustment = getattr(config, 'adaptive_threshold_adjustment', True)
        self.detection_window = getattr(config, 'detection_window', 50)
        self.improvement_threshold = getattr(config, 'improvement_threshold', 0.01)
        self.stability_threshold = getattr(config, 'stability_threshold', 0.05)
        self.best_reward = float('-inf')
        self.best_loss = float('inf')
        self.reward_plateau_count = 0
        self.loss_plateau_count = 0
        self.gradient_norm_history = []
        
        # 干预系统参数
        self.intervention_applied = False
        self.max_interventions = getattr(config, 'max_interventions', 5)
        self.intervention_epsilon_boost = getattr(config, 'intervention_epsilon_boost', 0.2)
        self.intervention_lr_reduction = getattr(config, 'intervention_lr_reduction', 0.5)
        self.intervention_memory_refresh = getattr(config, 'intervention_memory_refresh', True)
        self.training_metrics = {
            'intervention_history': [],
            'early_stop_history': []
        }
        
        # 停滞检测参数
        self.reward_plateau_count = 0
        self.loss_plateau_count = 0
        
        print(f"使用设备: {self.device}")
        print(f"网络架构: {network_type}")
        print(f"网络结构: 输入维度={self.input_dim}, 隐藏层={self.hidden_dims}, 输出维度={self.output_dim}")
    
    def _action_to_index(self, a):
        """将动作元组转换为索引"""
        target_id, uav_id, phi_idx = a
        target_idx = next(i for i, t in enumerate(self.targets) if t.id == target_id)
        uav_idx = next(i for i, u in enumerate(self.uavs) if u.id == uav_id)
        return target_idx * len(self.uavs) * len(self.graph.phi_set) + uav_idx * len(self.graph.phi_set) + phi_idx
    
    def _index_to_action(self, i):
        """将索引转换为动作元组"""
        n_phi = len(self.graph.phi_set)
        target_idx = i // (len(self.uavs) * n_phi)
        uav_idx = (i % (len(self.uavs) * n_phi)) // n_phi
        phi_idx = i % n_phi
        return (self.targets[target_idx].id, self.uavs[uav_idx].id, phi_idx)
    
    def replay(self):
        """经验回放 - 增强版损失监控"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return None
        
        # 采样批次
        batch = random.sample(self.memory, self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量 - 修复数据类型转换问题
        states = torch.FloatTensor(states).to(self.device)
        
        # 修复actions的维度问题
        if isinstance(actions[0], tuple):
            # 如果actions是元组，需要转换为索引
            action_indices = [self._action_to_index(action) for action in actions]
            actions = torch.LongTensor(action_indices).to(self.device)
        else:
            # 如果actions已经是索引
            actions = torch.LongTensor(actions).to(self.device)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # 确保dones是Python bool类型，然后转换为tensor
        dones = [bool(done) for done in dones]  # 强制转换为Python bool
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states)
        
        # 确保actions的维度正确
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # 添加维度以匹配gather操作
        
        current_q_values = current_q_values.gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.GAMMA * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 梯度裁剪
        if self.config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 计算梯度范数
        grad_norm = self._calculate_gradient_norm()
        
        # 优化器步进
        self.optimizer.step()
        
        # === 增强的损失监控 ===
        loss_value = loss.item()
        
        # 记录损失历史
        if hasattr(self, 'loss_history'):
            self.loss_history.append(loss_value)
        else:
            self.loss_history = [loss_value]
        
        # 记录梯度范数
        if hasattr(self, 'gradient_norm_history'):
            self.gradient_norm_history.append(grad_norm)
        else:
            self.gradient_norm_history = [grad_norm]
        
        # 损失爆炸检测
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            loss_mean = np.mean(recent_losses)
            loss_std = np.std(recent_losses)
            
            # 检测异常损失
            if loss_value > loss_mean + 3 * loss_std:
                print(f"警告: 损失异常高 ({loss_value:.4f} > {loss_mean + 3 * loss_std:.4f})")
                
                # 自适应调整学习率
                if hasattr(self, 'config') and hasattr(self.config, 'use_adaptive_learning_rate'):
                    if self.config.use_adaptive_learning_rate:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        new_lr = current_lr * 0.8  # 降低学习率
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"自适应学习率调整: {current_lr:.6f} -> {new_lr:.6f}")
        
        # 梯度消失/爆炸检测
        if grad_norm > 10.0:
            print(f"警告: 梯度范数过大 ({grad_norm:.4f})")
        elif grad_norm < 1e-6:
            print(f"警告: 梯度范数过小 ({grad_norm:.4f})")
        
        # 网络特定监控
        if hasattr(self.q_network, 'get_monitoring_info'):
            monitoring_info = self.q_network.get_monitoring_info()
            
            # 监控层激活值
            if 'layer_activations' in monitoring_info:
                for activation_info in monitoring_info['layer_activations']:
                    if activation_info['norm'] > 100:
                        print(f"警告: 层 {activation_info['layer']} 激活范数过大 ({activation_info['norm']:.4f})")
                    elif activation_info['norm'] < 1e-6:
                        print(f"警告: 层 {activation_info['layer']} 激活范数过小 ({activation_info['norm']:.4f})")
        
        return loss_value
    
    def _get_valid_action_mask(self):
        """获取有效动作掩码"""
        valid_actions = []
        for t in self.targets:
            for u in self.uavs:
                for phi_idx in range(len(self.graph.phi_set)):
                    # 检查是否有有效贡献
                    actual_contribution = np.minimum(t.remaining_resources, u.resources)
                    if np.any(actual_contribution > 0):
                        valid_actions.append((t.id, u.id, phi_idx))
        return valid_actions
    
    def _select_action(self, state, valid_actions):
        """选择动作"""
        if not valid_actions:
            return None
        
        # 探索：随机选择
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 利用：选择最大Q值的动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 切换到评估模式以避免BatchNorm问题
        self.q_network.eval()
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # 只考虑有效动作
            valid_indices = [self._action_to_index(a) for a in valid_actions]
            valid_q_values = q_values[0, valid_indices]
            
            # 选择最大Q值的动作
            max_idx = torch.argmax(valid_q_values).item()
            selected_action = valid_actions[max_idx]
        
        # 切换回训练模式
        self.q_network.train()
        
        return selected_action

    def save_model(self, path):
        """保存模型"""
        torch.save(self.q_network.state_dict(), path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"模型已从 {path} 加载")
            return True
        return False
    
    def _calculate_gradient_norm(self):
        """计算当前梯度的范数"""
        total_norm = 0.0
        for p in self.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _fix_q_values_for_inference(self, q_values, valid_actions):
        """修复推理阶段的Q值中的无效值"""
        # 检查并修复NaN和inf值
        if torch.isnan(q_values).any() or torch.isinf(q_values).any():
            print("⚠️  检测到Q值包含NaN或inf，正在修复...")
            
            # 将NaN和inf替换为0
            q_values = torch.where(torch.isnan(q_values), torch.zeros_like(q_values), q_values)
            q_values = torch.where(torch.isinf(q_values), torch.zeros_like(q_values), q_values)
            
            # 确保有效动作的Q值不为0
            valid_indices = torch.where(valid_actions)[0]
            if len(valid_indices) > 0:
                # 给有效动作分配小的正值
                q_values[valid_indices] = torch.rand(len(valid_indices)) * 0.1
        
        return q_values
    
    def safe_probability_calculation(self, qs, valid_mask, inference_temperature, run):
        """安全的概率计算"""
        # 限制Q值范围
        qs_clipped = torch.clamp(qs, min=-10.0, max=10.0)
        probs = torch.softmax(qs_clipped / inference_temperature, dim=0)
        
        # 检查概率是否有效
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"⚠️  推理轮次 {run+1}: 概率计算出现无效值，使用随机选择")
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) > 0:
                return valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
            else:
                return 0
        
        # 确保概率和为1
        probs = probs / probs.sum()
        return torch.multinomial(probs, 1).item()
    
    def _detect_early_stopping(self, episode, avg_reward, avg_loss):
        """自适应多维闭环早熟检测 - 增强版"""
        if episode < self.min_episodes_before_detection:
            return False, {}
        
        # 更新历史记录
        self.reward_history.append(avg_reward)
        self.loss_history.append(avg_loss)
        self.epsilon_history.append(self.epsilon)
        
        # 计算梯度范数
        if len(self.reward_history) > 1:
            grad_norm = self._calculate_gradient_norm()
            self.gradient_norm_history.append(grad_norm)
        
        # 损失爆炸检测 - 新增
        if hasattr(self.config, 'loss_explosion_threshold') and avg_loss > self.config.loss_explosion_threshold:
            print(f"警告: 检测到损失爆炸! 当前损失: {avg_loss:.4f}, 阈值: {self.config.loss_explosion_threshold}")
            return True, {'detection_type': 'loss_explosion', 'loss_value': avg_loss}
        
        # 损失停滞检测 - 新增
        if hasattr(self.config, 'loss_stagnation_threshold') and len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_std = np.std(recent_losses)
            if loss_std < self.config.loss_stagnation_threshold:
                print(f"警告: 检测到损失停滞! 损失标准差: {loss_std:.6f}")
                return True, {'detection_type': 'loss_stagnation', 'loss_std': loss_std}
        
        # 自适应调整检测窗口
        if self.adaptive_window_adjustment and episode > 200:
            # 根据训练进度动态调整窗口大小
            progress_ratio = episode / self.config.EPISODES
            self.detection_window = max(30, min(100, int(50 * (1 + progress_ratio))))
        
        # 自适应调整阈值
        if self.adaptive_threshold_adjustment and episode > 150:
            # 根据训练进度动态调整阈值
            progress_ratio = episode / self.config.EPISODES
            self.improvement_threshold = max(0.005, min(0.02, 0.01 * (1 + progress_ratio)))
            self.stability_threshold = max(0.03, min(0.08, 0.05 * (1 + progress_ratio)))
        
        # 多维度检测指标
        detection_results = {
            'reward_plateau': False,
            'loss_plateau': False,
            'gradient_stagnation': False,
            'exploration_deficiency': False,
            'overall_stagnation': False
        }
        
        # 1. 奖励停滞检测
        if len(self.reward_history) >= self.detection_window:
            recent_rewards = self.reward_history[-self.detection_window:]
            recent_max = max(recent_rewards)
            if recent_max <= self.best_reward * (1 + self.improvement_threshold):
                detection_results['reward_plateau'] = True
                self.reward_plateau_count += 1
            else:
                self.reward_plateau_count = 0
        
        # 2. 损失停滞检测
        if len(self.loss_history) >= self.detection_window:
            recent_losses = self.loss_history[-self.detection_window:]
            recent_min = min(recent_losses)
            if recent_min >= self.best_loss * (1 - self.improvement_threshold):
                detection_results['loss_plateau'] = True
                self.loss_plateau_count += 1
            else:
                self.loss_plateau_count = 0
        
        # 3. 梯度停滞检测
        if len(self.gradient_norm_history) >= self.detection_window:
            recent_grads = self.gradient_norm_history[-self.detection_window:]
            grad_std = np.std(recent_grads)
            grad_mean = np.mean(recent_grads)
            if grad_std < grad_mean * self.stability_threshold:
                detection_results['gradient_stagnation'] = True
        
        # 4. 探索不足检测
        if len(self.epsilon_history) >= self.detection_window:
            recent_epsilon = self.epsilon_history[-self.detection_window:]
            epsilon_std = np.std(recent_epsilon)
            if epsilon_std < 0.01:  # 探索率变化很小
                detection_results['exploration_deficiency'] = True
        
        # 5. 综合停滞检测
        stagnation_count = sum(detection_results.values())
        if stagnation_count >= 2:  # 至少两个维度出现停滞
            detection_results['overall_stagnation'] = True
        
        # 判断是否需要干预
        needs_intervention = (
            detection_results['overall_stagnation'] or
            self.reward_plateau_count >= 3 or
            self.loss_plateau_count >= 3
        )
        
        return needs_intervention, detection_results
    
    def _apply_early_stopping_intervention(self, detection_results):
        """应用早熟干预措施 - 增强版"""
        if self.intervention_applied or len(self.training_metrics['intervention_history']) >= self.max_interventions:
            return False
        
        print(f"\n=== 检测到早熟问题，应用干预措施 ===")
        print(f"检测结果: {detection_results}")
        
        # 1. 提升探索率
        old_epsilon = self.epsilon
        self.epsilon = min(1.0, self.epsilon + self.intervention_epsilon_boost)
        print(f"探索率提升: {old_epsilon:.3f} -> {self.epsilon:.3f}")
        
        # 2. 自适应学习率调整 - 增强版
        old_lr = self.optimizer.param_groups[0]['lr']
        if hasattr(self.config, 'use_adaptive_learning_rate') and self.config.use_adaptive_learning_rate:
            # 根据检测类型调整学习率
            if 'loss_explosion' in str(detection_results):
                # 损失爆炸时大幅降低学习率
                new_lr = old_lr * getattr(self.config, 'lr_reduction_factor', 0.5)
                print(f"损失爆炸检测: 学习率大幅降低")
            elif 'loss_stagnation' in str(detection_results):
                # 损失停滞时适度降低学习率
                new_lr = old_lr * getattr(self.config, 'lr_reduction_factor', 0.7)
                print(f"损失停滞检测: 学习率适度降低")
            else:
                # 其他情况适度降低学习率
                new_lr = old_lr * self.intervention_lr_reduction
            
            # 确保学习率在合理范围内
            min_lr = getattr(self.config, 'min_learning_rate', 1e-6)
            max_lr = getattr(self.config, 'max_learning_rate', 1e-3)
            new_lr = max(min_lr, min(max_lr, new_lr))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"学习率调整: {old_lr:.6f} -> {new_lr:.6f}")
        else:
            # 原有的学习率降低逻辑
            new_lr = old_lr * self.intervention_lr_reduction
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 3. 刷新记忆库（部分）
        if len(self.memory) > 0:
            refresh_size = int(len(self.memory) * self.intervention_memory_refresh)
            for _ in range(refresh_size):
                if len(self.memory) > 0:
                    self.memory.popleft()
            print(f"记忆库刷新: 移除 {refresh_size} 个旧经验")
        
        # 4. 重置停滞计数器
        self.reward_plateau_count = 0
        self.loss_plateau_count = 0
        
        # 5. 记录干预历史
        intervention_record = {
            'episode': len(self.reward_history),
            'detection_results': detection_results,
            'old_epsilon': old_epsilon,
            'new_epsilon': self.epsilon,
            'old_lr': old_lr,
            'new_lr': new_lr,
            'timestamp': time.time()
        }
        self.training_metrics['intervention_history'].append(intervention_record)
        
        self.intervention_applied = True
        print(f"干预措施应用完成")
        return True
    def train(self, episodes, patience, log_interval, model_save_path):
        """训练强化学习模型 - 改进版自适应训练系统"""
        print(f"开始训练，总轮次: {episodes}")
        
        # 训练历史记录
        episode_rewards = []
        episode_losses = []
        intervention_history = []
        
        # 早停检测相关
        best_reward = float('-inf')
        patience_counter = 0
        early_stopping_triggered = False
        
        # 自适应训练参数
        current_lr = self.optimizer.param_groups[0]['lr']
        current_epsilon = self.epsilon
        
        # === 改进的自适应训练系统参数 ===
        use_adaptive_intervention = True  # 是否启用自适应干预
        intervention_cooldown = 50        # 干预冷却期（轮次）
        last_intervention_episode = -intervention_cooldown  # 上次干预轮次
        consecutive_problem_episodes = 0  # 连续问题轮次计数
        problem_threshold = 3             # 触发干预的连续问题轮次阈值
        
        for episode in tqdm(range(episodes), desc="训练进度"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            while True:
                # 获取有效动作掩码
                valid_actions = self._get_valid_action_mask()
                if not any(valid_actions):
                    break
                
                # 选择动作
                action = self._select_action(state, valid_actions)
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                self.memory.append((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                steps += 1
                
                # 训练网络
                if len(self.memory) >= self.config.BATCH_SIZE:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss += loss
                
                if done:
                    break
            
            # 记录历史
            episode_rewards.append(episode_reward)
            if episode_loss > 0:
                episode_losses.append(episode_loss / steps)
            else:
                episode_losses.append(0.0)
            
            # 实时更新训练历史
            self.train_history['episode_rewards'] = episode_rewards
            self.train_history['episode_losses'] = episode_losses
            self.train_history['epsilon_values'].append(self.epsilon)
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 记录梯度范数
            if len(episode_rewards) > 1:
                grad_norm = self._calculate_gradient_norm()
                self.train_history['gradient_norms'].append(grad_norm)
            
            # === 改进的自适应训练系统 ===
            # 1. 默认行为：经典指数衰减
            old_epsilon = self.epsilon
            self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)
            
            # 2. 自适应干预检测（仅在启用时）
            intervention_applied = False
            if use_adaptive_intervention and len(episode_rewards) >= 20:
                needs_intervention, detection_results = self._detect_early_stopping(
                    episode, 
                    np.mean(episode_rewards[-20:]), 
                    np.mean(episode_losses[-20:]) if episode_losses else 0
                )
                
                # 检查是否在冷却期内
                in_cooldown = (episode - last_intervention_episode) < intervention_cooldown
                
                if needs_intervention and not in_cooldown:
                    consecutive_problem_episodes += 1
                    
                    # 连续问题达到阈值时触发干预
                    if consecutive_problem_episodes >= problem_threshold:
                        intervention_applied = self._apply_improved_intervention(
                            detection_results, episode, old_epsilon
                        )
                        
                        if intervention_applied:
                            last_intervention_episode = episode
                            consecutive_problem_episodes = 0
                            
                            # 记录干预历史
                            intervention_history.append({
                                'episode': episode,
                                'detection_results': detection_results,
                                'intervention_applied': True,
                                'old_epsilon': old_epsilon,
                                'new_epsilon': self.epsilon,
                                'consecutive_problems': consecutive_problem_episodes,
                                'note': '自适应干预触发'
                            })
                            
                            print(f"轮次 {episode}: 自适应干预触发")
                            print(f"  检测结果: {detection_results}")
                            print(f"  探索率调整: {old_epsilon:.4f} -> {self.epsilon:.4f}")
                            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    # 重置连续问题计数
                    consecutive_problem_episodes = 0
                    
                    # 记录检测但不干预的情况
                    if needs_intervention and in_cooldown:
                        intervention_history.append({
                            'episode': episode,
                            'detection_results': detection_results,
                            'intervention_applied': False,
                            'note': f'检测到问题但在冷却期内（冷却期剩余: {intervention_cooldown - (episode - last_intervention_episode)}轮）'
                        })
            
            # 定期保存模型
            if (episode + 1) % 100 == 0:
                self.save_model(model_save_path)
            
            # 早停检查
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停触发，轮次 {episode}")
                break
        
        # 保存最终模型
        self.save_model(model_save_path)
        
        # 生成增强收敛图
        self._plot_enhanced_convergence(model_save_path)
        
        # 生成奖励曲线报告
        self._generate_reward_curve_report(model_save_path)
        
        # 保存训练历史到self.train_history
        self.train_history['episode_rewards'] = episode_rewards
        self.train_history['episode_losses'] = episode_losses
        self.train_history['intervention_history'] = intervention_history
        self.train_history['early_stopping_triggered'] = early_stopping_triggered
        self.train_history['final_epsilon'] = current_epsilon
        self.train_history['final_lr'] = current_lr
        
        # 保存训练历史到文件
        history_data = {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'intervention_history': intervention_history,
            'early_stopping_triggered': early_stopping_triggered,
            'final_epsilon': current_epsilon,
            'final_lr': current_lr
        }
        
        history_path = model_save_path.replace('.pth', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history_data, f)
        
        print(f"训练完成，总轮次: {len(episode_rewards)}")
        print(f"最佳奖励: {best_reward:.2f}")
        print(f"最终探索率: {current_epsilon:.4f}")
        print(f"最终学习率: {current_lr:.6f}")
        if intervention_history:
            print(f"干预次数: {len([h for h in intervention_history if h['intervention_applied']])}")
        
        return episode_rewards, episode_losses

    def _apply_improved_intervention(self, detection_results, episode, old_epsilon):
        """应用改进的干预措施 - 更智能的干预策略"""
        if self.intervention_applied or len(self.training_metrics['intervention_history']) >= self.max_interventions:
            return False
        
        print(f"\n=== 应用改进的干预措施 ===")
        print(f"检测结果: {detection_results}")
        
        # 根据检测结果类型选择不同的干预策略
        intervention_strategy = self._select_intervention_strategy(detection_results)
        
        # 1. 智能探索率调整
        old_epsilon = self.epsilon
        epsilon_boost = intervention_strategy['epsilon_boost']
        self.epsilon = min(1.0, self.epsilon + epsilon_boost)
        print(f"探索率调整: {old_epsilon:.4f} -> {self.epsilon:.4f} (提升: {epsilon_boost:.3f})")
        
        # 2. 自适应学习率调整
        old_lr = self.optimizer.param_groups[0]['lr']
        lr_adjustment = intervention_strategy['lr_adjustment']
        new_lr = old_lr * lr_adjustment
        
        # 确保学习率在合理范围内
        min_lr = getattr(self.config, 'min_learning_rate', 1e-6)
        max_lr = getattr(self.config, 'max_learning_rate', 1e-3)
        new_lr = max(min_lr, min(max_lr, new_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"学习率调整: {old_lr:.6f} -> {new_lr:.6f} (调整因子: {lr_adjustment:.2f})")
        
        # 3. 记忆库刷新（可选）
        if intervention_strategy.get('refresh_memory', False):
            if len(self.memory) > 0:
                refresh_size = int(len(self.memory) * 0.1)  # 刷新10%
                for _ in range(refresh_size):
                    if len(self.memory) > 0:
                        self.memory.popleft()
                print(f"记忆库刷新: 移除 {refresh_size} 个旧经验")
        
        # 4. 重置停滞计数器
        self.reward_plateau_count = 0
        self.loss_plateau_count = 0
        
        # 5. 记录干预历史
        intervention_record = {
            'episode': episode,
            'detection_results': detection_results,
            'intervention_strategy': intervention_strategy,
            'old_epsilon': old_epsilon,
            'new_epsilon': self.epsilon,
            'old_lr': old_lr,
            'new_lr': new_lr,
            'timestamp': time.time()
        }
        self.training_metrics['intervention_history'].append(intervention_record)
        
        self.intervention_applied = True
        print(f"干预措施应用完成")
        return True
    
    def _select_intervention_strategy(self, detection_results):
        """根据检测结果选择干预策略"""
        # 默认策略：温和干预
        strategy = {
            'epsilon_boost': 0.1,      # 探索率提升
            'lr_adjustment': 0.8,      # 学习率调整
            'refresh_memory': False     # 是否刷新记忆库
        }
        
        # 根据检测结果调整策略
        if 'loss_explosion' in str(detection_results):
            # 损失爆炸：强烈干预
            strategy.update({
                'epsilon_boost': 0.3,
                'lr_adjustment': 0.5,
                'refresh_memory': True
            })
        elif 'loss_stagnation' in str(detection_results):
            # 损失停滞：中等干预
            strategy.update({
                'epsilon_boost': 0.2,
                'lr_adjustment': 0.7,
                'refresh_memory': False
            })
        elif 'reward_plateau' in str(detection_results):
            # 奖励停滞：轻微干预
            strategy.update({
                'epsilon_boost': 0.15,
                'lr_adjustment': 0.85,
                'refresh_memory': False
            })
        elif 'gradient_stagnation' in str(detection_results):
            # 梯度停滞：中等干预
            strategy.update({
                'epsilon_boost': 0.2,
                'lr_adjustment': 0.75,
                'refresh_memory': True
            })
        
        return strategy

    def _plot_enhanced_convergence(self, model_save_path):
        """绘制增强的训练收敛情况图表，包含早熟检测和干预信息"""
        # 创建保存路径
        save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('增强训练收敛分析 - 包含早熟检测与干预', fontsize=16, fontweight='bold')
        
        # 1. 奖励收敛图
        ax1 = axes[0, 0]
        rewards = self.train_history['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, 'b-', alpha=0.6, label='每轮奖励')
        
        # 添加移动平均线
        window_size = min(50, len(rewards))
        if window_size > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = range(window_size, len(rewards) + 1)
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
        
        # 标记早熟检测点
        if hasattr(self, 'early_stop_history') and self.early_stop_history:
            for ep, detected in self.early_stop_history:
                if detected:
                    ax1.axvline(x=ep, color='red', linestyle='--', alpha=0.7, label='早熟检测' if ep == self.early_stop_history[0][0] else "")
        
        # 标记干预点
        if hasattr(self, 'intervention_history') and self.intervention_history:
            for ep, intervention_type in self.intervention_history:
                ax1.axvline(x=ep, color='orange', linestyle=':', alpha=0.7, label=f'干预({intervention_type})' if ep == self.intervention_history[0][0] else "")
        
        ax1.set_title('奖励收敛曲线')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('奖励值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失收敛图
        ax2 = axes[0, 1]
        if 'episode_losses' in self.train_history and self.train_history['episode_losses']:
            losses = self.train_history['episode_losses']
            loss_episodes = range(1, len(losses) + 1)
            ax2.plot(loss_episodes, losses, 'g-', alpha=0.6, label='每轮损失')
            
            # 添加移动平均线
            if len(losses) >= window_size:
                moving_loss_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                moving_loss_episodes = range(window_size, len(losses) + 1)
                ax2.plot(moving_loss_episodes, moving_loss_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
            
            ax2.set_title('损失收敛曲线')
            ax2.set_xlabel('训练轮次')
            ax2.set_ylabel('损失值')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无损失数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('损失收敛曲线')
        
        # 3. 梯度范数变化
        ax3 = axes[1, 0]
        if 'gradient_norms' in self.train_history and self.train_history['gradient_norms']:
            grad_norms = self.train_history['gradient_norms']
            grad_episodes = range(1, len(grad_norms) + 1)
            ax3.plot(grad_episodes, grad_norms, 'purple', alpha=0.6, label='梯度范数')
            
            # 添加移动平均线
            if len(grad_norms) >= window_size:
                moving_grad_avg = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
                moving_grad_episodes = range(window_size, len(grad_norms) + 1)
                ax3.plot(moving_grad_episodes, moving_grad_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
            
            ax3.set_title('梯度范数变化')
            ax3.set_xlabel('训练轮次')
            ax3.set_ylabel('梯度范数')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无梯度数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('梯度范数变化')
        
        # 4. 探索率变化
        ax4 = axes[1, 1]
        if 'epsilon_values' in self.train_history and self.train_history['epsilon_values']:
            epsilons = self.train_history['epsilon_values']
            epsilon_episodes = range(1, len(epsilons) + 1)
            ax4.plot(epsilon_episodes, epsilons, 'orange', alpha=0.6, label='探索率')
            ax4.set_title('探索率变化')
            ax4.set_xlabel('训练轮次')
            ax4.set_ylabel('探索率')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无探索率数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('探索率变化')
        
        # 5. 早熟检测指标
        ax5 = axes[2, 0]
        if hasattr(self, 'early_stop_metrics') and self.early_stop_metrics:
            metrics = self.early_stop_metrics
            episodes = range(1, len(metrics) + 1)
            
            # 绘制多维度指标
            for metric_name, values in metrics.items():
                if isinstance(values, (list, np.ndarray)) and len(values) == len(episodes):
                    ax5.plot(episodes, values, alpha=0.6, label=metric_name)
            
            ax5.set_title('早熟检测指标')
            ax5.set_xlabel('训练轮次')
            ax5.set_ylabel('指标值')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, '无早熟检测数据', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('早熟检测指标')
        
        # 6. 干预历史统计
        ax6 = axes[2, 1]
        if hasattr(self, 'intervention_history') and self.intervention_history:
            intervention_types = [intervention[1] for intervention in self.intervention_history]
            intervention_counts = {}
            for intervention_type in intervention_types:
                intervention_counts[intervention_type] = intervention_counts.get(intervention_type, 0) + 1
            
            if intervention_counts:
                types = list(intervention_counts.keys())
                counts = list(intervention_counts.values())
                ax6.bar(types, counts, color=['red', 'orange', 'yellow'])
                ax6.set_title('干预类型统计')
                ax6.set_xlabel('干预类型')
                ax6.set_ylabel('干预次数')
                ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, '无干预数据', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('干预类型统计')
        
        plt.tight_layout()
        
        # 保存图片
        convergence_path = model_save_path.replace('.pth', '_enhanced_convergence.png')
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"增强收敛分析图已保存至: {convergence_path}")
        
        # 生成奖励曲线详细报告
        self._generate_reward_curve_report(model_save_path)
    
    def _generate_reward_curve_report(self, model_save_path):
        """生成奖励曲线详细报告"""
        save_dir = os.path.dirname(model_save_path)
        report_path = model_save_path.replace('.pth', '_reward_curve_report.txt')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("强化学习训练奖励曲线分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 奖励统计
        if 'episode_rewards' in self.train_history and self.train_history['episode_rewards']:
            rewards = self.train_history['episode_rewards']
            report_lines.append("奖励统计:")
            report_lines.append(f"  总训练轮次: {len(rewards)}")
            report_lines.append(f"  最高奖励: {max(rewards):.2f}")
            report_lines.append(f"  最低奖励: {min(rewards):.2f}")
            report_lines.append(f"  平均奖励: {np.mean(rewards):.2f}")
            report_lines.append(f"  奖励标准差: {np.std(rewards):.2f}")
            report_lines.append(f"  最终奖励: {rewards[-1]:.2f}")
        else:
            report_lines.append("奖励统计: 无奖励数据")
        
        # 奖励趋势分析
        if 'episode_rewards' in self.train_history and self.train_history['episode_rewards']:
            rewards = self.train_history['episode_rewards']
            if len(rewards) > 10:
                recent_rewards = rewards[-10:]
                early_rewards = rewards[:10]
                recent_avg = np.mean(recent_rewards)
                early_avg = np.mean(early_rewards)
                improvement = (recent_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else 0
                report_lines.append(f"  奖励改进: {improvement:.2f}%")
            
            # 收敛性分析
            if len(rewards) > 50:
                last_50 = rewards[-50:]
                first_50 = rewards[:50]
                convergence_ratio = np.std(last_50) / np.std(first_50) if np.std(first_50) > 0 else 0
                report_lines.append(f"  收敛性指标: {convergence_ratio:.3f} (越小越稳定)")
        
        # 如果没有奖励数据，跳过趋势分析
        
        report_lines.append("")
        
        # 早熟检测统计
        if hasattr(self, 'early_stop_history') and self.early_stop_history:
            report_lines.append("早熟检测统计:")
            total_detections = sum(1 for _, detected in self.early_stop_history if detected)
            report_lines.append(f"  总检测次数: {len(self.early_stop_history)}")
            report_lines.append(f"  早熟检测次数: {total_detections}")
            report_lines.append(f"  早熟检测率: {total_detections/len(self.early_stop_history)*100:.1f}%")
        
        report_lines.append("")
        
        # 干预统计
        if hasattr(self, 'intervention_history') and self.intervention_history:
            report_lines.append("干预统计:")
            intervention_types = {}
            for _, intervention_type in self.intervention_history:
                intervention_types[intervention_type] = intervention_types.get(intervention_type, 0) + 1
            
            for intervention_type, count in intervention_types.items():
                report_lines.append(f"  {intervention_type}: {count}次")
        
        report_lines.append("")
        
        # 训练建议
        report_lines.append("训练建议:")
        if 'episode_rewards' in self.train_history:
            rewards = self.train_history['episode_rewards']
            if len(rewards) > 0:
                final_reward = rewards[-1]
                max_reward = max(rewards)
                
                if final_reward < max_reward * 0.8:
                    report_lines.append("  - 当前奖励远低于历史最佳，建议调整学习率或增加训练轮次")
                
                if len(rewards) > 100 and np.std(rewards[-50:]) < np.std(rewards[:50]) * 0.1:
                    report_lines.append("  - 奖励变化很小，可能已收敛，建议停止训练")
                
                if hasattr(self, 'intervention_history') and len(self.intervention_history) > 3:
                    report_lines.append("  - 干预次数较多，建议调整早熟检测参数")
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"奖励曲线分析报告已保存至: {report_path}")

    def get_task_assignments(self):
        """多次推理取最优值的任务分配方法"""
        self.q_network.eval()
        
        # 获取多次推理的参数
        n_inference_runs = getattr(self.config, 'RL_N_INFERENCE_RUNS', 10)
        inference_temperature = getattr(self.config, 'RL_INFERENCE_TEMPERATURE', 0.1)
        
        best_assignments = None
        best_reward = float('-inf')
        
        print(f"开始多次推理优化 (推理次数: {n_inference_runs})")
        
        for run in range(n_inference_runs):
            # 重置环境
            state = self.env.reset()
            assignments = {u.id: [] for u in self.env.uavs}
            done, step = False, 0
            total_reward = 0
            
            while not done and step < len(self.env.targets) * len(self.env.uavs):
                with torch.no_grad():
                    valid_actions = self._get_valid_action_mask()
                    if not valid_actions:  # 检查列表是否为空
                        break
                    
                    # 添加温度参数以增加探索性
                    qs = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                    
                    # 修复：处理Q值中的无效值
                    qs = self._fix_q_values_for_inference(qs, valid_actions)
                    
                    # 使用温度参数进行softmax采样
                    if inference_temperature > 0:
                        action_idx = self.safe_probability_calculation(qs, valid_actions, inference_temperature, run)
                    else:
                        # 直接选择最大Q值
                        valid_indices = [self._action_to_index(a) for a in valid_actions]
                        valid_q_values = qs[valid_indices]
                        
                        if len(valid_q_values) > 0:
                            max_idx = torch.argmax(valid_q_values).item()
                            action_idx = valid_indices[max_idx]
                        else:
                            action_idx = 0
                
                action = self._index_to_action(action_idx)
                assignments[action[1]].append((action[0], action[2]))
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                step += 1
            
            # 评估当前推理结果
            if total_reward > best_reward:
                best_reward = total_reward
                best_assignments = assignments.copy()
                print(f"推理轮次 {run+1}: 发现更好的分配方案 (奖励: {best_reward:.2f})")
        
        self.q_network.train()
        print(f"多次推理完成，最优奖励: {best_reward:.2f}")
        return best_assignments
    




# =============================================================================
# section 5: 核心业务流程
# =============================================================================
def _plan_single_leg(args):
    uav_id, start_pos, target_pos, start_heading, end_heading, obstacles, config = args
    planner = PHCurveRRTPlanner(start_pos, target_pos, start_heading, end_heading, obstacles, config); return uav_id, planner.plan()
# 此函数非常关键，直接觉得算法能够解除死锁、保证经济同步速度，而且速度还很快，协同分配资源的效率也很高。
def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config) -> Tuple[defaultdict, dict]:
    """(已更新) 计算经济同步速度，并返回未完成的任务以进行死锁检测。"""
    # 转换任务数据结构并补充资源消耗
    final_plan = defaultdict(list)
    uav_status = {u.id: {'pos': u.position, 'free_at': 0.0, 'heading': u.heading} for u in uavs}
    remaining_tasks = {uav_id: list(tasks) for uav_id, tasks in task_assignments.items()}; task_step_counter = defaultdict(lambda: 1)
    while any(v for v in remaining_tasks.values()):
        next_target_groups = defaultdict(list)
        for uav_id, tasks in remaining_tasks.items():
            if tasks: next_target_groups[tasks[0][0]].append({'uav_id': uav_id, 'phi_idx': tasks[0][1]})
        if not next_target_groups: break
        group_arrival_times = []
        for target_id, uav_infos in next_target_groups.items():
            target = next((t for t in targets if t.id == target_id), None)
            if not target: continue
            path_planners = {}
            for uav_info in uav_infos:
                uav_id = uav_info['uav_id']; args = (uav_id, uav_status[uav_id]['pos'], target.position, uav_status[uav_id]['heading'], graph.phi_set[uav_info['phi_idx']], obstacles, config)
                _, plan_result = _plan_single_leg(args)
                if plan_result: path_planners[uav_id] = {'path_points': plan_result[0], 'distance': plan_result[1]}
            time_windows = []
            for uav_info in uav_infos:
                uav_id = uav_info['uav_id']
                if uav_id not in path_planners: continue
                uav = next((u for u in uavs if u.id == uav_id), None)
                if not uav: continue
                distance = path_planners[uav_id]['distance']; free_at = uav_status[uav_id]['free_at']; t_min = free_at + (distance / uav.velocity_range[1]); t_max = free_at + (distance / uav.velocity_range[0]) if uav.velocity_range[0] > 0 else float('inf')
                t_econ = free_at + (distance / uav.economic_speed)
                time_windows.append({'uav_id': uav_id, 'phi_idx': uav_info['phi_idx'], 't_min': t_min, 't_max': t_max, 't_econ': t_econ})
            if not time_windows: continue
            sync_start = max(tw['t_min'] for tw in time_windows); sync_end = min(tw['t_max'] for tw in time_windows); is_feasible = sync_start <= sync_end + 1e-6
            final_sync_time = np.clip(np.median([tw['t_econ'] for tw in time_windows]), sync_start, sync_end) if is_feasible else sync_start
            group_arrival_times.append({'target_id': target_id, 'arrival_time': final_sync_time, 'uav_infos': time_windows, 'is_feasible': is_feasible, 'path_planners': path_planners})
        if not group_arrival_times: break
        next_event = min(group_arrival_times, key=lambda x: x['arrival_time']); target_pos = next(t.position for t in targets if t.id == next_event['target_id'])
        for uav_info in next_event['uav_infos']:
            uav_id = uav_info['uav_id']
            if uav_id not in next_event['path_planners']: continue
            uav, plan_data = next(u for u in uavs if u.id == uav_id), next_event['path_planners'][uav_id]; travel_time = next_event['arrival_time'] - uav_status[uav_id]['free_at']
            speed = (plan_data['distance'] / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
            final_plan[uav_id].append({'target_id': next_event['target_id'], 'start_pos': uav_status[uav_id]['pos'], 'speed': np.clip(speed, uav.velocity_range[0], uav.velocity_range[1]), 'arrival_time': next_event['arrival_time'], 'step': task_step_counter[uav_id], 'is_sync_feasible': next_event['is_feasible'], 'phi_idx': uav_info['phi_idx'], 'path_points': plan_data['path_points'], 'distance': plan_data['distance']})
            task_step_counter[uav_id] += 1; uav_status[uav_id].update(pos=target_pos, free_at=next_event['arrival_time'], heading=graph.phi_set[uav_info['phi_idx']])
            if remaining_tasks.get(uav_id): remaining_tasks[uav_id].pop(0)
    # 应用协同贪婪资源分配策略以匹配可视化逻辑
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 按事件分组任务
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            # 将无人机ID和任务引用存入对应的事件组
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    # 按时间顺序处理事件
    for event_key in sorted(events.keys()):
        arrival_time, target_id = event_key
        collaborating_tasks = events[event_key]
        target_remaining = temp_target_resources[target_id].copy()
        
        # 分配资源
        for item in collaborating_tasks:
            uav_id = item['uav_id']
            task = item['task_ref']
            uav_resources = temp_uav_resources[uav_id]
            
            if np.all(target_remaining < 1e-6):
                task['resource_cost'] = np.zeros_like(uav_resources)
                continue
                
            contribution = np.minimum(target_remaining, uav_resources)
            task['resource_cost'] = contribution
            temp_uav_resources[uav_id] -= contribution
            target_remaining -= contribution
            
        temp_target_resources[target_id] = target_remaining

    return final_plan, remaining_tasks



# [新增] 从批处理测试器导入评估函数，用于对单个方案进行性能评估
from evaluate import evaluate_plan

def calibrate_resource_assignments(task_assignments, uavs, targets):
    """
    校准资源分配，移除无效的任务分配。
    
    Args:
        task_assignments: 原始任务分配
        uavs: 无人机列表
        targets: 目标列表
    
    Returns:
        校准后的任务分配
    """
    print("正在校准资源分配...")
    
    # 创建资源状态副本
    uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    target_needs = {t.id: t.resources.copy().astype(float) for t in targets}
    
    # 按时间顺序处理任务分配（这里简化处理，按无人机ID顺序）
    calibrated_assignments = {u.id: [] for u in uavs}
    
    for uav_id in sorted(task_assignments.keys()):
        uav_tasks = task_assignments[uav_id]
        for target_id, phi_idx in uav_tasks:
            # 检查目标是否还需要资源
            if not np.any(target_needs[target_id] > 1e-6):
                print(f"警告: UAV {uav_id} 被分配到已满足的目标 {target_id}，跳过此分配")
                continue
            
            # 检查无人机是否还有资源
            if not np.any(uav_resources[uav_id] > 1e-6):
                print(f"警告: UAV {uav_id} 资源已耗尽，跳过后续分配")
                break
            
            # 计算实际贡献
            contribution = np.minimum(uav_resources[uav_id], target_needs[target_id])
            
            # 只有当有实际贡献时才保留此分配
            if np.any(contribution > 1e-6):
                calibrated_assignments[uav_id].append((target_id, phi_idx))
                uav_resources[uav_id] -= contribution
                target_needs[target_id] -= contribution
                print(f"UAV {uav_id} -> 目标 {target_id}: 贡献 {contribution}")
            else:
                print(f"警告: UAV {uav_id} 对目标 {target_id} 无有效贡献，跳过此分配")
    
    # 统计校准结果
    original_count = sum(len(tasks) for tasks in task_assignments.values())
    calibrated_count = sum(len(tasks) for tasks in calibrated_assignments.values())
    removed_count = original_count - calibrated_count
    
    print(f"资源分配校准完成:")
    print(f"  原始分配数量: {original_count}")
    print(f"  校准后数量: {calibrated_count}")
    print(f"  移除无效分配: {removed_count}")
    
    return calibrated_assignments


def visualize_task_assignments(final_plan, uavs, targets, obstacles, config, scenario_name, training_time, plan_generation_time,
                             save_plot=True, show_plot=False, save_report=False, deadlocked_tasks=None, evaluation_metrics=None):
    """(已更新并修复资源计算bug) 可视化任务分配方案。"""
    
    # [增加协同事件分析] 在报告中加入事件说明，解释资源竞争
    report_content = f'"""---------- {scenario_name} 执行报告 ----------\n\n'

    # [二次修复] 采用"协同贪婪"策略精确模拟资源消耗
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 1. 按"事件"（同一时间、同一目标）对所有步骤进行分组
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            # 将无人机ID和任务引用存入对应的事件组
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    # 2. 按时间顺序（事件发生的顺序）对事件进行排序
    sorted_event_keys = sorted(events.keys())

    # [新增] 协同事件日志
    collaboration_log = "\n\n 协同事件日志 (揭示资源竞争):\n ------------------------------------\n"

    # 3. 按事件顺序遍历，处理每个协作事件
    for event_key in sorted_event_keys:
        arrival_time, target_id = event_key
        collaborating_steps = events[event_key]
        
        target_remaining_need_before = temp_target_resources[target_id].copy()
        collaboration_log += f" * 事件: 在 t={arrival_time:.2f}s, 无人机(UAVs) {', '.join([str(s['uav_id']) for s in collaborating_steps])} 到达 目标 {target_id}\n"
        collaboration_log += f"   - 目标初始需求: {target_remaining_need_before}\n"

        # 4. 在事件内部，让每个协作者依次、尽力地贡献资源
        for step in collaborating_steps:
            uav_id = step['uav_id']
            task = step['task_ref']

            uav_available_resources = temp_uav_resources[uav_id]
            actual_contribution = np.minimum(target_remaining_need_before, uav_available_resources)
            
            if np.all(actual_contribution < 1e-6):
                task['resource_cost'] = np.zeros_like(uav_available_resources)
                collaboration_log += f"     - UAV {uav_id} 尝试贡献，但目标需求已满足。贡献: [0. 0.]\n"
                continue

            temp_uav_resources[uav_id] -= actual_contribution
            target_remaining_need_before -= actual_contribution
            task['resource_cost'] = actual_contribution
            collaboration_log += f"     - UAV {uav_id} 贡献 {actual_contribution}, 剩余资源 {temp_uav_resources[uav_id]}\n"
            
        temp_target_resources[target_id] = target_remaining_need_before
        collaboration_log += f"   - 事件结束，目标剩余需求: {target_remaining_need_before}\n\n"

    """(已更新并修复资源计算bug) 可视化任务分配方案。"""
    
    # [二次修复] 采用"协同贪婪"策略精确模拟资源消耗
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

    # 1. 按"事件"（同一时间、同一目标）对所有步骤进行分组
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            # 将无人机ID和任务引用存入对应的事件组
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    # 2. 按时间顺序（事件发生的顺序）对事件进行排序
    sorted_event_keys = sorted(events.keys())

    # 3. 按事件顺序遍历，处理每个协作事件
    for event_key in sorted_event_keys:
        arrival_time, target_id = event_key
        collaborating_steps = events[event_key]
        
        target_remaining_need = temp_target_resources[target_id].copy()
        
        # 4. 在事件内部，让每个协作者依次、尽力地贡献资源
        for step in collaborating_steps:
            uav_id = step['uav_id']
            task = step['task_ref']

            if not np.any(target_remaining_need > 1e-6):
                task['resource_cost'] = np.zeros_like(temp_uav_resources[uav_id])
                continue

            uav_available_resources = temp_uav_resources[uav_id]
            actual_contribution = np.minimum(target_remaining_need, uav_available_resources)
            
            temp_uav_resources[uav_id] -= actual_contribution
            target_remaining_need -= actual_contribution
            
            task['resource_cost'] = actual_contribution
            
        temp_target_resources[target_id] = target_remaining_need


    # --- 后续的可视化和报告生成逻辑将使用上面计算出的精确 resource_cost ---
    fig, ax = plt.subplots(figsize=(22, 14)); ax.set_facecolor("#f0f0f0");
    for obs in obstacles: obs.draw(ax)

    target_collaborators_details = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in sorted(tasks, key=lambda x: x['step']):
            target_id = task['target_id']
            resource_cost = task.get('resource_cost', np.zeros_like(uavs[0].resources))
            target_collaborators_details[target_id].append({'uav_id': uav_id, 'arrival_time': task['arrival_time'], 'resource_cost': resource_cost})

    summary_text = ""
    if targets:
        satisfied_targets_count = 0; resource_types = len(targets[0].resources) if targets else 2
        total_demand_all = np.sum([t.resources for t in targets], axis=0)

        # --- [修订] 修复当 final_plan 为空时 np.sum 的计算错误 ---
        all_resource_costs = [d['resource_cost'] for details in target_collaborators_details.values() for d in details]
        if not all_resource_costs:
            total_contribution_all_for_summary = np.zeros(resource_types)
        else:
            total_contribution_all_for_summary = np.sum(all_resource_costs, axis=0)
        # --- 修订结束 ---

        for t in targets:
            current_target_contribution_sum = np.sum([d['resource_cost'] for d in target_collaborators_details.get(t.id, [])], axis=0)
            if np.all(current_target_contribution_sum >= t.resources - 1e-5): satisfied_targets_count += 1
        num_targets = len(targets); satisfaction_rate_percent = (satisfied_targets_count / num_targets * 100) if num_targets > 0 else 100
        total_demand_safe = total_demand_all.copy(); total_demand_safe[total_demand_safe == 0] = 1e-6
        overall_completion_rate_percent = np.mean(np.minimum(total_contribution_all_for_summary, total_demand_all) / total_demand_safe) * 100
        summary_text = (f"总体资源满足情况:\n--------------------------\n- 总需求: {np.array2string(total_demand_all, formatter={'float_kind':lambda x: '%.0f' % x})}\n- 总贡献: {np.array2string(total_contribution_all_for_summary, formatter={'float_kind':lambda x: '%.1f' % x})}\n- 已满足目标: {satisfied_targets_count} / {num_targets} ({satisfaction_rate_percent:.1f}%)\n- 资源完成率: {overall_completion_rate_percent:.1f}%")
    
    # ... (函数其余的可视化和报告生成代码未变) ...
    ax.scatter([u.position[0] for u in uavs], [u.position[1] for u in uavs], c='blue', marker='s', s=150, label='无人机起点', zorder=5, edgecolors='black')
    for u in uavs:
        ax.annotate(f"UAV{u.id}", (u.position[0], u.position[1]), fontsize=12, fontweight='bold', xytext=(0, -25), textcoords='offset points', ha='center', va='top')
        ax.annotate(f"初始: {np.array2string(u.initial_resources, formatter={'float_kind': lambda x: f'{x:.0f}'})}", (u.position[0], u.position[1]), fontsize=8, xytext=(15, 10), textcoords='offset points', ha='left', color='navy')
    ax.scatter([t.position[0] for t in targets], [t.position[1] for t in targets], c='red', marker='o', s=150, label='目标', zorder=5, edgecolors='black')
    for t in targets:
        demand_str = np.array2string(t.resources, formatter={'float_kind': lambda x: "%.0f" % x}); annotation_text = f"目标 {t.id}\n总需求: {demand_str}\n------------------"
        total_contribution = np.sum([d['resource_cost'] for d in target_collaborators_details.get(t.id, [])], axis=0)
        details_text = sorted(target_collaborators_details.get(t.id, []), key=lambda x: x['arrival_time'])
        if not details_text: annotation_text += "\n未分配无人机"
        else:
            for detail in details_text: annotation_text += f"\nUAV {detail['uav_id']} (T:{detail['arrival_time']:.1f}s) 贡献:{np.array2string(detail['resource_cost'], formatter={'float_kind': lambda x: '%.1f' % x})}"
        if np.all(total_contribution >= t.resources - 1e-5):
            satisfaction_str, bbox_color = "[OK] 需求满足", 'lightgreen'
        else:
            satisfaction_str, bbox_color = "[NG] 资源不足", 'mistyrose'
        annotation_text += f"\n------------------\n状态: {satisfaction_str}"
        ax.annotate(f"T{t.id}", (t.position[0], t.position[1]), fontsize=12, fontweight='bold', xytext=(0, 18), textcoords='offset points', ha='center', va='bottom')
        ax.annotate(annotation_text, (t.position[0], t.position[1]), fontsize=7, xytext=(15, -15), textcoords='offset points', ha='left', va='top', bbox=dict(boxstyle='round,pad=0.4', fc=bbox_color, ec='black', alpha=0.9, lw=0.5), zorder=8)

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(uavs))) if uavs else []; uav_color_map = {u.id: colors[i] for i, u in enumerate(uavs)}
    for uav_id, tasks in final_plan.items():
        uav_color = uav_color_map.get(uav_id, 'gray'); temp_resources = next(u for u in uavs if u.id == uav_id).initial_resources.copy().astype(float)
        for task in sorted(tasks, key=lambda x: x['step']):
            path_points = task.get('path_points')
            if path_points is not None and len(path_points) > 1:
                ax.plot(path_points[:, 0], path_points[:, 1], color=uav_color, linestyle='-' if task['is_sync_feasible'] else '--', linewidth=2, alpha=0.9, zorder=3)
                num_pos = path_points[int(len(path_points) * 0.45)]; ax.text(num_pos[0], num_pos[1], str(task['step']), color='white', backgroundcolor=uav_color, ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='circle,pad=0.2', fc=uav_color, ec='none'), zorder=4)
                resource_cost = task.get('resource_cost', np.zeros_like(temp_resources))
                temp_resources -= resource_cost
                resource_annotation_pos = path_points[int(len(path_points) * 0.85)]; remaining_res_str = f"R: {np.array2string(temp_resources.clip(0), formatter={'float_kind': lambda x: f'{x:.0f}'})}"
                ax.text(resource_annotation_pos[0], resource_annotation_pos[1], remaining_res_str, color=uav_color, backgroundcolor='white', ha='center', va='center', fontsize=7, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=uav_color, alpha=0.8, lw=0.5), zorder=7)

    deadlock_summary_text = ""
    if deadlocked_tasks and any(deadlocked_tasks.values()):
        deadlock_summary_text += "!!! 死锁检测 !!!\n--------------------------\n以下无人机未能完成其任务序列，可能陷入死锁：\n"
        for uav_id, tasks in deadlocked_tasks.items():
            if tasks: deadlock_summary_text += f"- UAV {uav_id}: 等待执行 -> {' -> '.join([f'T{t[0]}' for t in tasks])}\n"
        deadlock_summary_text += ("-"*30) + "\n\n"
    report_header = f"---------- {scenario_name} 执行报告 ----------\n\n" + deadlock_summary_text
    if summary_text: report_header += summary_text + "\n" + ("-"*30) + "\n\n"
    
    # 添加评估指标到报告中
    if evaluation_metrics:
        report_header += "评估指标:\n--------------------------\n"
        for key, value in evaluation_metrics.items():
            # 特殊处理带归一化的指标
            if key in ['completion_rate', 'satisfied_targets_rate', 'sync_feasibility_rate', 'load_balance_score', 'resource_utilization_rate']:
                norm_value = evaluation_metrics.get(f'norm_{key}', 'N/A')
                if isinstance(norm_value, float):
                    print(f"  - {key}: {value:.4f} (归一化: {norm_value:.4f})")
                else:
                    print(f"  - {key}: {value:.4f} (归一化: {norm_value})")
        print("-" * 20)
    
    report_body_image = ""; report_body_file = ""
    for uav in uavs:
        uav_header = f"* 无人机 {uav.id} (初始资源: {np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: f'{x:.0f}'})})\n"; report_body_image += uav_header; report_body_file += uav_header
        details = sorted(final_plan.get(uav.id, []), key=lambda x: x['step'])
        if not details: no_task_str = "  - 未分配任何任务\n"; report_body_image += no_task_str; report_body_file += no_task_str
        else:
            temp_resources_report = uav.initial_resources.copy().astype(float)
            for detail in details:
                resource_cost = detail.get('resource_cost', np.zeros_like(temp_resources_report))
                temp_resources_report -= resource_cost
                sync_status = "" if detail['is_sync_feasible'] else " (警告: 无法同步)"
                common_report_part = f"  {detail['step']}. 飞向目标 {detail['target_id']}{sync_status}:\n"; common_report_part += f"     - 飞行距离: {detail.get('distance', 0):.2f} m, 速度: {detail['speed']:.2f} m/s, 到达时间点: {detail['arrival_time']:.2f} s\n"
                common_report_part += f"     - 消耗资源: {np.array2string(resource_cost, formatter={'float_kind': lambda x: '%.1f' % x})}\n"; common_report_part += f"     - 剩余资源: {np.array2string(temp_resources_report.clip(0), formatter={'float_kind': lambda x: f'{x:.1f}'})}\n"
                report_body_image += common_report_part; report_body_file += common_report_part
                # 根据用户要求，报告中不输出路径点
                # path_points = detail.get('path_points')
                # if path_points is not None and len(path_points) > 0:
                #     points_per_line = 4; path_str_lines = []; line_buffer = []
                #     for p in path_points:
                #         line_buffer.append(f"({p[0]:.0f}, {p[1]:.0f})")
                #         if len(line_buffer) >= points_per_line: path_str_lines.append(" -> ".join(line_buffer)); line_buffer = []
                #     if line_buffer: path_str_lines.append(" -> ".join(line_buffer))
                #     report_body_file += "     - 路径坐标:\n"
                #     for line in path_str_lines: report_body_file += f"          {line}\n"
        report_body_image += "\n"; report_body_file += "\n"
    
    final_report_for_image = report_header + report_body_image; final_report_for_file = report_header + report_body_file
    plt.subplots_adjust(right=0.75); fig.text(0.77, 0.95, final_report_for_image, transform=plt.gcf().transFigure, ha="left", va="top", fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.9))
    
    train_mode_str = '高精度' if config.USE_PHRRT_DURING_TRAINING else '快速近似'
    
    # 处理training_time格式化问题
    if isinstance(training_time, (tuple, list)):
        actual_episodes = len(training_time[0]) if training_time and len(training_time) > 0 else 0
        estimated_time = actual_episodes * 0.13  # 基于观察到的每轮约0.13秒
        training_time_str = f"{estimated_time:.2f}s ({actual_episodes}轮)"
    else:
        training_time_str = f"{training_time:.2f}s"
    
    title_text = (
        f"多无人机任务分配与路径规划 - {scenario_name}\n"
        f"UAV: {len(uavs)}, 目标: {len(targets)}, 障碍: {len(obstacles)} | 模式: {train_mode_str}\n"
        f"模型训练耗时: {training_time_str} | 方案生成耗时: {plan_generation_time:.2f}s"
    )
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)

    ax.set_xlabel("X坐标 (m)", fontsize=14); ax.set_ylabel("Y坐标 (m)", fontsize=14); ax.legend(loc="lower left"); ax.grid(True, linestyle='--', alpha=0.5, zorder=0); ax.set_aspect('equal', adjustable='box')
    
    if save_plot:
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
        base_filename = f"{clean_scenario_name}_{timestamp}"
        img_filepath = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(img_filepath, dpi=300)
        print(f"结果图已保存至: {img_filepath}")
        
        if save_report:
            report_dir = "output/reports"
            os.makedirs(report_dir, exist_ok=True)
            report_filepath = os.path.join(report_dir, f"{base_filename}.txt")
            try:
                with open(report_filepath, 'w', encoding='utf-8') as f:
                    f.write(final_report_for_file)
                print(f"任务分配报告已保存至: {report_filepath}")
            except Exception as e:
                print(f"错误：无法保存任务报告至 {report_filepath}. 原因: {e}")
                
    if show_plot:
        plt.show()
    plt.close(fig) # 确保无论是否显示，图形对象都被关闭以释放内存

# =============================================================================
# section 6: (新增) 辅助函数 & 主流程控制
# =============================================================================
import hashlib

def get_config_hash(config):
    """根据关键配置参数生成直观的配置标识字符串"""
    return (
        f"lr{config.LEARNING_RATE}_g{config.GAMMA}_"
        f"eps{config.EPSILON_END}-{config.EPSILON_DECAY}_"
        f"upd{config.TARGET_UPDATE_FREQ}_bs{config.BATCH_SIZE}_"
        f"phi{config.GRAPH_N_PHI}_phrrt{int(config.USE_PHRRT_DURING_TRAINING)}_"
        f"steps{config.EPISODES}"
    )

def _find_latest_checkpoint(model_path: str) -> Optional[str]:
    """查找最新的检查点文件"""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.startswith('model_checkpoint_ep_') and file.endswith('.pth'):
            checkpoint_files.append(os.path.join(model_dir, file))
    
    if not checkpoint_files:
        return None
    
    # 按文件名中的轮次数排序
    checkpoint_files.sort(key=lambda x: int(x.split('_ep_')[1].split('.')[0]))
    return checkpoint_files[-1] if checkpoint_files else None

def _get_trained_episodes(model_path: str) -> int:
    """获取已训练的轮次数"""
    try:
        # 尝试从训练历史文件中获取
        history_dir = os.path.dirname(model_path)
        history_files = [f for f in os.listdir(history_dir) if f.startswith('training_history_')]
        
        if history_files:
            # 从最新的历史文件中提取轮次数
            latest_history = max(history_files, key=lambda x: os.path.getmtime(os.path.join(history_dir, x)))
            with open(os.path.join(history_dir, latest_history), 'rb') as f:
                import pickle
                history = pickle.load(f)
                return len(history.get('episode_rewards', []))
        
        # 如果无法从历史文件获取，尝试从检查点文件名获取
        checkpoint_path = _find_latest_checkpoint(model_path)
        if checkpoint_path:
            episode_str = checkpoint_path.split('_ep_')[1].split('.')[0]
            return int(episode_str)
        
        return 0
    except Exception as e:
        print(f"获取已训练轮次数时出错: {e}")
        return 0

def run_scenario(config, base_uavs, base_targets, obstacles, scenario_name, 
                 save_visualization=True, show_visualization=True, save_report=False,
                 force_retrain=False, incremental_training=False, network_type='DeepFCN'):
    """运行场景 - 支持多种网络架构"""
    print(f"=========================")
    print(f"   Running: {scenario_name} (Config Hash: {get_config_hash(config)})")
    print(f"=========================")
    
    # 创建图结构
    graph = DirectedGraph(base_uavs, base_targets, config.GRAPH_N_PHI, obstacles)
    
    # 创建临时环境来获取实际的状态维度
    temp_env = UAVTaskEnv(base_uavs, base_targets, graph, obstacles, config)
    temp_state = temp_env.reset()
    
    # 使用实际的状态维度
    i_dim = len(temp_state)  # 实际状态维度
    h_dim = 256  # 隐藏层维度
    o_dim = len(base_uavs) * len(base_targets) * config.GRAPH_N_PHI  # 输出维度
    
    # 创建求解器 - 支持网络类型选择
    solver = GraphRLSolver(base_uavs, base_targets, graph, obstacles, i_dim, h_dim, o_dim, config, network_type)
    
    # 模型保存路径
    model_dir = os.path.join("output/models", scenario_name)
    os.makedirs(model_dir, exist_ok=True)
    config_hash = get_config_hash(config)
    model_path = os.path.join(model_dir, f"{config_hash}/model.pth")
    
    # 检查是否需要训练
    if force_retrain or not os.path.exists(model_path):
        print(f"开始新的训练... ({model_path})")
        
        # 创建模型目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 训练模型
        training_time = solver.train(episodes=config.EPISODES, patience=config.PATIENCE,
                                   log_interval=config.LOG_INTERVAL, model_save_path=model_path)
        
        # 保存模型
        solver.save_model(model_path)
    else:
        print(f"加载已训练的模型... ({model_path})")
        solver.load_model(model_path)
        training_time = 0.0
    
    # 生成任务分配方案
    print("----- [阶段 2: 生成最终方案 ({scenario_name} @ {config_hash})] -----")
    plan_start_time = time.time()
    
    # 多次推理优化
    print(f"开始多次推理优化 (推理次数: {config.RL_N_INFERENCE_RUNS})")
    task_assignments = solver.get_task_assignments()
    plan_generation_time = time.time() - plan_start_time
    
    # 校准资源分配
    calibrated_assignments = calibrate_resource_assignments(task_assignments, base_uavs, base_targets)
    
    # 计算经济同步速度
    final_plan, remaining_tasks = calculate_economic_sync_speeds(
        calibrated_assignments, base_uavs, base_targets, graph, obstacles, config
    )
    
    # 评估方案质量
    print("----- [阶段 3: 方案质量评估 ({scenario_name})] -----")
    evaluation_metrics = evaluate_plan(final_plan, base_uavs, base_targets, remaining_tasks)
    
    # 可视化结果
    visualize_task_assignments(
        final_plan, base_uavs, base_targets, obstacles, config, scenario_name,
        training_time, plan_generation_time, save_visualization, show_visualization, 
        save_report, remaining_tasks, evaluation_metrics
    )
    
    return final_plan, evaluation_metrics


def get_simple_scenario(obstacle_tolerance):
    """创建简单场景 - 2UAV-2Target"""
    uavs = [
        UAV(1, (100, 100), 0, [100, 100], 500, (10, 20), 15),
        UAV(2, (200, 200), 0, [100, 100], 500, (10, 20), 15)
    ]
    
    targets = [
        Target(1, (300, 300), [80, 80], 10),
        Target(2, (400, 400), [80, 80], 10)
    ]
    
    obstacles = []
    
    print("简单场景已创建:")
    print(f"  - 无人机数量: {len(uavs)}")
    print(f"  - 目标数量: {len(targets)}")
    print(f"  - 障碍物数量: {len(obstacles)}")
    print(f"  - 场景特点: 简单任务分配，无障碍物")
    
    return uavs, targets, obstacles


def main():
    """主函数，用于单独运行和调试一个默认场景。"""
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    config = Config()
    
    # 检查是否启用批量测试
    if config.BATCH_TEST_SCENARIOS:
        print("检测到批量测试模式，启动批量测试...")
        from batch_test_networks import run_batch_test
        run_batch_test()
        return
    
    # 检查是否启用网络结构对比模式
    if config.NETWORK_COMPARISON_MODE:
        print("检测到网络结构对比模式...")
        config.EPISODES = 1500
        complex_uavs, complex_targets, complex_obstacles = get_strategic_trap_scenario(config.OBSTACLE_TOLERANCE)
        
        # 测试不同的网络架构
        network_types = ['DeepFCN', 'DeepFCN_Residual', 'GNN']
        
        for network_type in network_types:
            print(f"\n{'='*60}")
            print(f"测试网络架构: {network_type}")
            print(f"{'='*60}")
            
            try:
                run_scenario(config, complex_uavs, complex_targets, complex_obstacles, 
                            f"网络架构测试-{network_type}", show_visualization=False, save_report=True,
                            force_retrain=True, incremental_training=False, network_type=network_type)
            except Exception as e:
                print(f"网络架构 {network_type} 测试失败: {e}")
                continue
        return
    
    # 测试自适应训练系统
    config.USE_ADAPTIVE_TRAINING = False  # 启用自适应训练
    
    # 设置简化的训练参数，用于收敛性测试
    config.EPISODES = 50  # 减少轮次，便于快速测试
    config.training_config.learning_rate = 0.0001  # 降低学习率
    config.training_config.batch_size = 16  # 减小批次大小
    config.training_config.reward_scaling_factor = 0.1  # 降低奖励缩放
    config.USE_PHRRT_DURING_TRAINING = False  # 关闭PH-RRT以加快训练
    
    config.EPISODES = 1500
    complex_uavs, complex_targets, complex_obstacles = get_strategic_trap_scenario(config.OBSTACLE_TOLERANCE)
    
    # 使用配置中指定的网络类型
    network_type = config.NETWORK_TYPE
    print(f"使用配置的网络类型: {network_type}")
    
    run_scenario(config, complex_uavs, complex_targets, complex_obstacles, 
                f"网络架构测试-{network_type}", show_visualization=False, save_report=True,
                force_retrain=True, incremental_training=False, network_type=network_type)

if __name__ == "__main__":
    main()




