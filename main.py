# -*- coding: utf-8 -*-
# 文件名: main_simplified.py
# 描述: 多无人机协同任务分配与路径规划的简化版本。
#      移除了重复的网络结构定义，保留核心功能。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from collections import deque, defaultdict
import os
import time
import pickle
import random
import json
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
import glob

# TensorBoard支持 - 可选依赖 
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("警告: TensorBoard未安装，将跳过TensorBoard功能")
    print("安装命令: pip install tensorboard")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# --- 本地模块导入 ---
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario, get_new_experimental_scenario, get_complex_scenario_v4, get_strategic_trap_scenario
from config import Config
from evaluate import evaluate_plan
from networks import create_network, get_network_info
from environment import UAVTaskEnv, DirectedGraph

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

# JSON序列化辅助函数
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # 处理任何其他numpy标量类型
        return obj.item()
    else:
        return obj

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
                # print(f"已自动设置中文字体为: {font}")  # 移除字体设置输出
                return True
    except Exception:
        pass
    
    # print("警告: 自动或手动设置中文字体失败。图片中的中文可能显示为方框。")  # 简化输出
    return False

# =============================================================================
# section 2: 核心业务逻辑 - 强化学习求解器
# =============================================================================

# =============================================================================
# section 3: 强化学习求解器
# =============================================================================

# =============================================================================
# section 3.5: 优先经验回放缓冲区 (Prioritized Experience Replay)
# =============================================================================
class SumTree:
    """
    Sum Tree数据结构，用于高效的优先级采样
    支持O(log n)的更新和采样操作
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级的二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储实际经验数据
        self.write = 0  # 写入指针
        self.n_entries = 0  # 当前存储的经验数量
    
    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """根据累积优先级检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """返回总优先级"""
        return self.tree[0]
    
    def add(self, p, data):
        """添加新经验"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        """更新优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        """根据累积优先级获取经验"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        # 检查数据索引是否有效
        if dataIdx < 0 or dataIdx >= len(self.data) or self.data[dataIdx] is None:
            return (idx, self.tree[idx], None)
        
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    核心思想：
    - 根据TD误差分配优先级，误差越大优先级越高
    - 使用重要性采样权重修正非均匀采样的偏差
    - 通过α控制优先级的影响程度，β控制重要性采样的强度
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数，0表示均匀采样，1表示完全按优先级采样
            beta_start: 重要性采样权重的初始值
            beta_frames: β从beta_start线性增长到1.0的帧数
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # 防止优先级为0
        self.max_priority = 1.0  # 最大优先级
    
    def beta(self):
        """计算当前的β值（重要性采样权重强度）"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """添加新经验，使用最大优先级"""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        采样一个批次的经验
        
        Returns:
            batch: 经验批次
            indices: 在树中的索引
            weights: 重要性采样权重
        """
        batch = []
        indices = []
        priorities = []
        
        # 计算采样区间
        segment = self.tree.total() / batch_size
        
        # 尝试采样，如果失败则重试
        max_attempts = batch_size * 3  # 最多尝试3倍的次数
        attempts = 0
        
        while len(batch) < batch_size and attempts < max_attempts:
            i = len(batch)  # 当前需要采样的索引
            
            # 在对应区间内随机采样
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # 获取经验
            (idx, p, data) = self.tree.get(s)
            
            # 验证数据格式并添加到批次中
            if data is not None and isinstance(data, (tuple, list)) and len(data) == 5:
                batch.append(data)
                indices.append(idx)
                priorities.append(p)
            
            attempts += 1
        
        # 计算重要性采样权重
        if len(priorities) > 0:
            # 计算概率
            probs = np.array(priorities) / self.tree.total()
            # 计算权重
            weights = (len(batch) * probs) ** (-self.beta())
            # 归一化权重
            weights = weights / weights.max()
        else:
            weights = np.ones(len(batch))
        
        self.frame += 1
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            # 确保优先级为正数
            priority = abs(priority) + self.epsilon
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            # 应用α指数
            priority = priority ** self.alpha
            # 更新树中的优先级
            self.tree.update(idx, priority)
    
    def __len__(self):
        """返回当前存储的经验数量"""
        return self.tree.n_entries

# =============================================================================
# section 4: 简化的强化学习求解器
# =============================================================================
class GraphRLSolver:
    """简化的基于图的强化学习求解器 - 增强版本，支持TensorBoard监控"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, network_type="SimpleNetwork", tensorboard_dir=None, obs_mode="flat"):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.network_type = network_type
        self.obs_mode = obs_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 保存输出目录用于模型加载
        self.output_dir = os.path.dirname(tensorboard_dir) if tensorboard_dir else None
        
        # TensorBoard支持 - 安全初始化
        self.tensorboard_dir = tensorboard_dir
        self.writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(tensorboard_dir)
                print(f"TensorBoard日志将保存至: {tensorboard_dir}")
            except Exception as e:
                print(f"TensorBoard初始化失败: {e}")
        elif tensorboard_dir and not TENSORBOARD_AVAILABLE:
            print("TensorBoard未安装，跳过日志记录")
        
        # 创建网络
        self.policy_net = create_network(network_type, i_dim, h_dim, o_dim).to(self.device)
        self.target_net = create_network(network_type, i_dim, h_dim, o_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # 使用优先经验回放缓冲区替代普通deque
        self.use_per = config.training_config.use_prioritized_replay
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=config.MEMORY_CAPACITY,
                alpha=config.training_config.per_alpha,
                beta_start=config.training_config.per_beta_start,
                beta_frames=config.training_config.per_beta_frames
            )
            print(f"  - 优先经验回放: 启用 (α={config.training_config.per_alpha}, β_start={config.training_config.per_beta_start})")
        else:
            self.memory = deque(maxlen=config.MEMORY_CAPACITY)
            print("  - 优先经验回放: 禁用")
        self.epsilon = config.training_config.epsilon_start
        
        # 环境 - 使用指定的观测模式
        self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode=obs_mode)
        
        # 重新计算实际的输入和输出维度，确保与环境一致
        actual_state = self.env.reset()
        actual_output_dim = self.env.n_actions
        
        if obs_mode == "flat":
            actual_input_dim = len(actual_state)
            print(f"[GraphRLSolver] 扁平模式维度校正:")
            print(f"  - 传入的输入维度: {i_dim}, 实际输入维度: {actual_input_dim}")
            print(f"  - 传入的输出维度: {o_dim}, 实际输出维度: {actual_output_dim}")
            
            # 使用实际维度而不是传入的维度
            if actual_input_dim != i_dim or actual_output_dim != o_dim:
                print(f"  - 检测到维度不匹配，使用实际维度创建网络")
                i_dim = actual_input_dim
                o_dim = actual_output_dim
        else:  # graph mode
            print(f"[GraphRLSolver] 图模式初始化:")
            print(f"  - 状态结构: {list(actual_state.keys())}")
            print(f"  - UAV特征形状: {actual_state['uav_features'].shape}")
            print(f"  - 目标特征形状: {actual_state['target_features'].shape}")
            print(f"  - 输出维度: {actual_output_dim}")
            # 对于图模式，使用传入的维度（占位值）
            o_dim = actual_output_dim
        
        # 动作映射
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}
        self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
        
        # 训练参数 - 进一步优化的超参数
        self.epsilon_decay = 0.995  # 适中的衰减率，平衡探索与利用
        self.epsilon_min = 0.15     # 提高最小探索率，确保持续探索
        
        # 高级DQN技术
        self.use_double_dqn = True  # 启用Double DQN
        self.use_dueling_dqn = True # 启用Dueling DQN架构
        self.use_grad_clip = True   # 启用梯度裁剪
        self.use_prioritized_replay = True  # 启用优先经验回放
        
        # 学习率调整
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        self.grad_clip_norm = 0.5   # 更严格的梯度裁剪阈值
        
        # 训练统计
        self.step_count = 0
        self.update_count = 0
        
        print(f"初始化完成: {network_type} 网络")
        print(f"  - Double DQN: {'启用' if self.use_double_dqn else '禁用'}")
        print(f"  - 梯度裁剪: {'启用' if self.use_grad_clip else '禁用'}")
        print(f"  - 探索率衰减: {self.epsilon_decay}")
    
    def _action_to_index(self, a):
        """将动作转换为索引"""
        t_idx, u_idx, p_idx = self.target_id_map[a[0]], self.uav_id_map[a[1]], a[2]
        return t_idx * (len(self.env.uavs) * self.graph.n_phi) + u_idx * self.graph.n_phi + p_idx
    
    def _index_to_action(self, i):
        """将索引转换为动作"""
        n_u, n_p = len(self.env.uavs), self.graph.n_phi
        t_idx, u_idx, p_idx = i // (n_u * n_p), (i % (n_u * n_p)) // n_p, i % n_p
        return (self.env.targets[t_idx].id, self.env.uavs[u_idx].id, p_idx)
    
    def _merge_graph_states(self, state_list):
        """
        合并图结构状态列表为批次
        
        Args:
            state_list: 图结构状态列表
            
        Returns:
            dict: 合并后的批次状态字典
        """
        if not state_list:
            return {}
        
        # 获取第一个状态的结构
        first_state = state_list[0]
        merged_state = {}
        
        # 合并每个键的值
        for key in first_state.keys():
            if key == "masks":
                # 处理掩码字典
                merged_masks = {}
                for mask_key in first_state[key].keys():
                    mask_tensors = [state[key][mask_key] for state in state_list]
                    merged_masks[mask_key] = torch.cat(mask_tensors, dim=0)
                merged_state[key] = merged_masks
            else:
                # 处理其他张量
                tensors = [state[key] for state in state_list]
                merged_state[key] = torch.cat(tensors, dim=0)
        
        return merged_state
    
    def _prepare_state_tensor(self, state):
        """
        将状态转换为张量格式 - 支持双模式
        
        Args:
            state: 状态（扁平向量或图结构字典）
            
        Returns:
            torch.Tensor or dict: 处理后的状态张量
        """
        if self.obs_mode == "flat":
            # 扁平模式：转换为张量
            return torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:  # graph mode
            # 图模式：转换字典中的每个组件为张量
            state_tensor = {}
            for key, value in state.items():
                if key == "masks":
                    # 处理掩码字典
                    mask_tensor = {}
                    for mask_key, mask_value in value.items():
                        mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                    state_tensor[key] = mask_tensor
                else:
                    # 处理其他张量
                    state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            return state_tensor
    
    def select_action(self, state):
        """使用Epsilon-Greedy策略选择动作 - 支持双模式状态，修复版本"""
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.env.n_actions)]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            # 临时切换到eval模式避免BatchNorm问题
            self.policy_net.eval()
            
            # 根据观测模式处理状态
            if self.obs_mode == "flat":
                # 扁平模式：直接传入状态张量
                q_values = self.policy_net(state)
            else:  # graph mode
                # 图模式：传入图结构字典
                q_values = self.policy_net(state)
            
            self.policy_net.train()
            
            # 修复：确保Q值维度与动作空间匹配
            if q_values.shape[1] > self.env.n_actions:
                q_values = q_values[:, :self.env.n_actions]
            elif q_values.shape[1] < self.env.n_actions:
                # 如果Q值维度不足，填充最小值
                padding = torch.full((q_values.shape[0], self.env.n_actions - q_values.shape[1]), 
                                   float('-inf'), device=q_values.device)
                q_values = torch.cat([q_values, padding], dim=1)
            
            action_idx = q_values.max(1)[1].view(1, 1)
            
            # 验证动作有效性
            validated_action = self._validate_action(action_idx.item())
            return torch.tensor([[validated_action]], device=self.device, dtype=torch.long)
    
    def _validate_action(self, action_idx):
        """验证动作索引是否有效"""
        if action_idx >= self.env.n_actions:
            # 如果动作超出范围，使用模运算调整
            valid_action = action_idx % self.env.n_actions
            if hasattr(self, '_action_warning_count'):
                self._action_warning_count += 1
            else:
                self._action_warning_count = 1
                
            # 只在前几次显示警告，避免日志过多
            if self._action_warning_count <= 5:
                print(f"警告: 动作 {action_idx} 超出范围 [0, {self.env.n_actions-1}]，调整为 {valid_action}")
            return valid_action
        return action_idx
    
    def optimize_model(self):
        """
        从经验回放池中采样并优化模型 - 支持优先经验回放(PER)
        
        核心改进:
        1. 支持PER的带权重采样
        2. 计算TD误差并更新优先级
        3. 使用重要性采样权重修正偏差
        """
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        
        # 根据是否使用PER选择不同的采样策略
        if self.use_per:
            # PER采样：获取经验、索引和重要性采样权重
            transitions, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # 标准随机采样
            transitions = random.sample(self.memory, self.config.BATCH_SIZE)
            indices = None
            weights = torch.ones(self.config.BATCH_SIZE).to(self.device)
        
        # 解包批次数据
        # 检查是否有足够的有效数据
        if not transitions or len(transitions) < self.config.BATCH_SIZE:
            # 如果数据不足，跳过这次优化
            return None
        
        batch = list(zip(*transitions))
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)
        
        # 根据观测模式处理状态批次
        if self.obs_mode == "flat":
            # 扁平模式：直接拼接状态张量
            state_batch = torch.cat(batch[0])
            next_states_batch = torch.cat(batch[3])
        else:  # graph mode
            # 图模式：合并字典结构的状态
            state_batch = self._merge_graph_states(batch[0])
            next_states_batch = self._merge_graph_states(batch[3])
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算目标Q值
        next_q_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 使用策略网络选择动作，目标网络评估动作
                next_actions = self.policy_net(next_states_batch).max(1)[1].unsqueeze(1)
                
                # 对于图模式，需要特殊处理
                if self.obs_mode == "graph":
                    # 计算所有状态的Q值，然后只使用未完成的
                    all_next_q_values = self.target_net(next_states_batch)
                    selected_q_values = all_next_q_values.gather(1, next_actions).squeeze(1)
                    next_q_values[~done_batch] = selected_q_values[~done_batch]
                else:
                    # 扁平模式的原有逻辑
                    next_q_values[~done_batch] = self.target_net(next_states_batch[~done_batch]).gather(1, next_actions[~done_batch]).squeeze(1)
            else:
                # 标准DQN
                if self.obs_mode == "graph":
                    # 计算所有状态的Q值，然后只使用未完成的
                    all_next_q_values = self.target_net(next_states_batch)
                    max_q_values = all_next_q_values.max(1)[0]
                    next_q_values[~done_batch] = max_q_values[~done_batch]
                else:
                    # 扁平模式的原有逻辑
                    next_q_values[~done_batch] = self.target_net(next_states_batch[~done_batch]).max(1)[0]
        
        expected_q_values = reward_batch + (self.config.GAMMA * next_q_values)
        
        # 计算TD误差（用于更新优先级）
        td_errors = (current_q_values.squeeze() - expected_q_values).detach()
        
        # 计算加权损失
        if self.use_per:
            # 使用重要性采样权重修正损失
            elementwise_loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none')
            loss = (elementwise_loss.squeeze() * weights).mean()
        else:
            # 标准损失
            loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        # 检查损失是否为NaN或无穷大
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: 损失为NaN或无穷大 ({loss.item()})，跳过此次更新")
            return None
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度是否包含NaN
        has_nan_grad = False
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("警告: 梯度包含NaN或无穷大，跳过此次更新")
            return None
        
        # 梯度裁剪
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        self.update_count += 1
        
        # 更新PER优先级
        if self.use_per and indices is not None:
            # 使用TD误差的绝对值作为新的优先级
            priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # TensorBoard记录
        if self.writer:
            # 记录基础指标
            self.writer.add_scalar('Training/Loss', loss.item(), self.update_count)
            self.writer.add_scalar('Training/Mean_Q_Value', current_q_values.mean().item(), self.update_count)
            self.writer.add_scalar('Training/Mean_TD_Error', td_errors.abs().mean().item(), self.update_count)
            
            # 记录PER相关指标
            if self.use_per:
                self.writer.add_scalar('Training/PER_Beta', self.memory.beta(), self.update_count)
                self.writer.add_scalar('Training/Mean_IS_Weight', weights.mean().item(), self.update_count)
                self.writer.add_scalar('Training/Max_Priority', self.memory.max_priority, self.update_count)
            
            # 记录梯度信息
            total_norm = 0
            for p in self.policy_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.writer.add_scalar('Training/Gradient_Norm', total_norm, self.update_count)
        
        # 软更新目标网络
        if self.update_count % self.config.TARGET_UPDATE_FREQ == 0:
            tau = 0.01  # 软更新系数
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
        
        # 自适应学习率调度
        if self.update_count % 500 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.95, 1e-5)
        
        return loss.item()
        
        # 记录训练指标到TensorBoard
        if self.writer:
            self.writer.add_scalar('Training/Loss', loss.item(), self.update_count)
            self.writer.add_scalar('Training/Q_Value_Mean', current_q_values.mean().item(), self.update_count)
            self.writer.add_scalar('Training/Q_Value_Std', current_q_values.std().item(), self.update_count)
            self.writer.add_scalar('Training/Target_Q_Mean', expected_q_values.mean().item(), self.update_count)
            self.writer.add_scalar('Training/Reward_Mean', reward_batch.mean().item(), self.update_count)
        
        return loss.item()
    
    def train(self, episodes, patience, log_interval, model_save_path):
        """
        训练模型 - 增强版本，支持TensorBoard监控和基于平均奖励的最佳模型保存
        
        Args:
            episodes (int): 训练轮数
            patience (int): 早停耐心值
            log_interval (int): 日志记录间隔
            model_save_path (str): 模型保存路径
        
        Returns:
            float: 训练耗时
        """
        start_time = time.time()
        best_reward = float('-inf')
        best_avg_reward = float('-inf')  # 新增：最佳平均奖励
        patience_counter = 0
        
        # 初始化训练历史记录
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_values = []
        self.completion_rates = []
        self.episode_steps = []
        self.memory_usage = []
        
        # 简化训练开始信息
        param_count = sum(p.numel() for p in self.policy_net.parameters())
        print(f"初始化 {self.network_type} 网络 (参数: {param_count:,}, 设备: {self.device})")
        
        for i_episode in tqdm(range(episodes), desc=f"训练进度 [{self.network_type}]"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            episode_step_count = 0
            
            for step in range(self.env.max_steps):
                # 根据观测模式处理状态
                state_tensor = self._prepare_state_tensor(state)
                action = self.select_action(state_tensor)
                
                next_state, reward, done, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                episode_step_count += 1
                self.step_count += 1
                
                # 准备下一状态张量
                next_state_tensor = self._prepare_state_tensor(next_state)
                
                # 添加经验到回放缓冲区
                reward_tensor = torch.tensor([reward], device=self.device)
                
                if self.use_per:
                    self.memory.push(
                        state_tensor,
                        action,
                        reward_tensor,
                        next_state_tensor,
                        done
                    )
                else:
                    self.memory.append((
                        state_tensor,
                        action,
                        reward_tensor,
                        next_state_tensor,
                        done
                    ))
                
                # 优化模型（每步都尝试优化）
                if len(self.memory) >= self.config.BATCH_SIZE:
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                
                state = next_state
                
                if done or truncated:
                    break
            
            # 更新目标网络
            if i_episode % self.config.training_config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if self.writer:
                    self.writer.add_scalar('Training/Target_Network_Update', 1, i_episode)
            
            # 衰减探索率 - 使用调整后的参数
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon * self.epsilon_decay)
            
            # 记录训练历史
            avg_episode_loss = episode_loss / max(loss_count, 1)
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_episode_loss)
            self.epsilon_values.append(self.epsilon)
            self.episode_steps.append(episode_step_count)
            self.memory_usage.append(len(self.memory))
            
            # 计算完成率 - 优化版本，综合考虑目标满足数量、总体资源满足率、资源利用率
            if self.env.targets:
                # 1. 目标满足数量比例
                completed_targets = sum(1 for target in self.env.targets if np.all(target.remaining_resources <= 0))
                target_satisfaction_rate = completed_targets / len(self.env.targets)
                
                # 2. 总体资源满足率
                total_demand = sum(np.sum(target.resources) for target in self.env.targets)
                total_remaining = sum(np.sum(target.remaining_resources) for target in self.env.targets)
                resource_satisfaction_rate = (total_demand - total_remaining) / total_demand if total_demand > 0 else 0
                
                # 3. 资源利用率
                total_initial_supply = sum(np.sum(uav.initial_resources) for uav in self.env.uavs)
                total_current_supply = sum(np.sum(uav.resources) for uav in self.env.uavs)
                resource_utilization_rate = (total_initial_supply - total_current_supply) / total_initial_supply if total_initial_supply > 0 else 0
                
                # 综合完成率 = 目标满足率(50%) + 资源满足率(30%) + 资源利用率(20%)
                completion_rate = (target_satisfaction_rate * 0.5 + 
                                 resource_satisfaction_rate * 0.3 + 
                                 resource_utilization_rate * 0.2)
            else:
                completion_rate = 0
            
            self.completion_rates.append(completion_rate)
            
            # 增强的TensorBoard记录
            if self.writer:
                # 基础指标
                self.writer.add_scalar('Episode/Reward', episode_reward, i_episode)
                self.writer.add_scalar('Episode/Loss', avg_episode_loss, i_episode)
                self.writer.add_scalar('Episode/Epsilon', self.epsilon, i_episode)
                self.writer.add_scalar('Episode/Completion_Rate', completion_rate, i_episode)
                self.writer.add_scalar('Episode/Steps', episode_step_count, i_episode)
                self.writer.add_scalar('Episode/Memory_Usage', len(self.memory), i_episode)
                
                # 移动平均指标
                if len(self.episode_rewards) >= 20:
                    recent_reward_avg = np.mean(self.episode_rewards[-20:])
                    recent_completion_avg = np.mean(self.completion_rates[-20:])
                    self.writer.add_scalar('Episode/Reward_MA20', recent_reward_avg, i_episode)
                    self.writer.add_scalar('Episode/Completion_Rate_MA20', recent_completion_avg, i_episode)
                
                # 收敛性指标
                if len(self.episode_rewards) >= 50:
                    recent_std = np.std(self.episode_rewards[-50:])
                    overall_std = np.std(self.episode_rewards)
                    stability_ratio = recent_std / overall_std if overall_std > 0 else 0
                    self.writer.add_scalar('Convergence/Stability_Ratio', stability_ratio, i_episode)
                    self.writer.add_scalar('Convergence/Recent_Std', recent_std, i_episode)
                
                # 早停相关指标
                self.writer.add_scalar('Training/Patience_Counter', patience_counter, i_episode)
                self.writer.add_scalar('Training/Best_Reward', best_reward, i_episode)
                
                # 网络权重和梯度分析
                if i_episode % (log_interval * 2) == 0:
                    for name, param in self.policy_net.named_parameters():
                        if param.grad is not None:
                            # 检查参数是否为空，避免TensorBoard错误
                            if param.numel() > 0 and not torch.isnan(param).any() and not torch.isinf(param).any():
                                try:
                                    self.writer.add_histogram(f'Weights/{name}', param, i_episode)
                                except ValueError:
                                    pass  # 跳过空直方图
                            if param.grad.numel() > 0 and not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                                try:
                                    self.writer.add_histogram(f'Gradients/{name}', param.grad, i_episode)
                                    self.writer.add_scalar(f'Gradients/{name}_norm', param.grad.norm(), i_episode)
                                except ValueError:
                                    pass  # 跳过空直方图
                
                # 学习率记录（如果使用调度器）
                if hasattr(self, 'optimizer'):
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Training/Learning_Rate', current_lr, i_episode)
            
            # 简化的训练进度输出 - 仅在关键节点输出
            if i_episode % (log_interval * 5) == 0 and i_episode > 0:  # 减少输出频率
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_completion = np.mean(self.completion_rates[-log_interval:])
                
                print(f"训练进度 - Episode {i_episode:4d}: 平均奖励 {avg_reward:8.2f}, 完成率 {avg_completion:6.3f}")
            
            # 改进的早停检查 - 基于资源满足率和训练进度
            current_completion_rate = completion_rate
            
            # 1. 基于平均奖励的最佳模型保存（新增）
            if len(self.episode_rewards) >= log_interval:
                # 计算最近N轮的平均奖励
                recent_avg_reward = np.mean(self.episode_rewards[-log_interval:])
                
                # 如果平均奖励超过历史最佳，保存最佳模型
                if recent_avg_reward > best_avg_reward:
                    best_avg_reward = recent_avg_reward
                    
                    # 保存最优模型（覆盖之前的最优模型）
                    saved_path = self._save_best_model(model_save_path)
                    
                    if self.writer:
                        self.writer.add_scalar('Training/Best_Avg_Reward', best_avg_reward, i_episode)
                    print(f"Episode {i_episode}: 新的最佳平均奖励 {best_avg_reward:.2f} (最近{log_interval}轮)")
                    print(f"  已保存最佳模型: {os.path.basename(saved_path)}")
            
            # 2. 传统单轮奖励早停（保留但提高阈值）
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                # 注意：不再保存单轮最佳模型，只保存平均奖励最佳模型
                if self.writer:
                    self.writer.add_scalar('Training/Best_Reward', best_reward, i_episode)
            else:
                patience_counter += 1
            
            # 2. 新增：基于资源满足率的早停准则
            should_early_stop = False
            early_stop_reason = ""
            
            # 确保至少训练总轮次的20%
            min_training_episodes = max(int(episodes * 0.2), 100)
            
            if i_episode >= min_training_episodes:
                # 检查最近50轮的平均完成率
                if len(self.completion_rates) >= 50:
                    recent_completion_avg = np.mean(self.completion_rates[-50:])
                    
                    # 如果资源满足率持续高于95%，可以早停
                    if recent_completion_avg >= 0.95:
                        should_early_stop = True
                        early_stop_reason = f"资源满足率达标 (平均: {recent_completion_avg:.3f})"
                    
                    # 检查收敛性：最近50轮的标准差很小
                    recent_completion_std = np.std(self.completion_rates[-50:])
                    if recent_completion_std < 0.02 and recent_completion_avg >= 0.85:
                        should_early_stop = True
                        early_stop_reason = f"完成率收敛 (平均: {recent_completion_avg:.3f}, 标准差: {recent_completion_std:.4f})"
            
            # 传统早停（提高patience阈值，避免过早停止）
            if patience_counter >= patience * 2:  # 将patience阈值翻倍
                should_early_stop = True
                early_stop_reason = f"奖励无改进超过 {patience * 2} 轮"
            
            if should_early_stop:
                print(f"早停触发于第 {i_episode} 回合: {early_stop_reason}")
                print(f"最佳奖励: {best_reward:.2f}, 最终完成率: {current_completion_rate:.3f}")
                break
        
        training_time = time.time() - start_time
        
        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()
        
        print(f"\n训练完成 - 耗时: {training_time:.2f}秒")
        print(f"训练统计:")
        print(f"  总回合数: {len(self.episode_rewards)}")
        print(f"  最佳单轮奖励: {best_reward:.2f}")
        print(f"  最佳平均奖励: {best_avg_reward:.2f} (最近{log_interval}轮)")
        print(f"  最终完成率: {self.completion_rates[-1]:.3f} (综合目标满足、资源满足、资源利用率)")
        print(f"  最终探索率: {self.epsilon:.4f} (随机动作概率)")
        
        # 生成详细的收敛性分析
        self.generate_enhanced_convergence_analysis(model_save_path, i_episode)
        
        return training_time
    
    def get_convergence_metrics(self):
        """获取收敛性指标"""
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        
        # 计算收敛性指标
        metrics = {
            'final_reward': float(rewards[-1]),
            'max_reward': float(np.max(rewards)),
            'mean_reward': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_improvement': float(rewards[-1] - rewards[0]) if len(rewards) > 1 else 0.0
        }
        
        # 收敛稳定性分析
        if len(rewards) > 100:
            recent_rewards = rewards[-50:]
            early_rewards = rewards[:50]
            
            metrics.update({
                'recent_mean': float(np.mean(recent_rewards)),
                'recent_std': float(np.std(recent_rewards)),
                'early_mean': float(np.mean(early_rewards)),
                'stability_ratio': float(np.std(recent_rewards) / np.std(rewards)) if np.std(rewards) > 0 else 0.0,
                'improvement_ratio': float((np.mean(recent_rewards) - np.mean(early_rewards)) / abs(np.mean(early_rewards))) if np.mean(early_rewards) != 0 else 0.0
            })
            
            # 判断收敛状态
            if metrics['stability_ratio'] < 0.3:
                metrics['convergence_status'] = 'converged'
            elif metrics['stability_ratio'] < 0.6:
                metrics['convergence_status'] = 'partially_converged'
            else:
                metrics['convergence_status'] = 'unstable'
        
        # 添加完成率相关指标
        if hasattr(self, 'completion_rates') and self.completion_rates:
            completion_rates = np.array(self.completion_rates)
            metrics.update({
                'final_completion_rate': float(completion_rates[-1]),
                'max_completion_rate': float(np.max(completion_rates)),
                'mean_completion_rate': float(np.mean(completion_rates)),
                'completion_rate_std': float(np.std(completion_rates)),
                'completion_improvement': float(completion_rates[-1] - completion_rates[0]) if len(completion_rates) > 1 else 0.0
            })
            
            # 完成率收敛性分析
            if len(completion_rates) > 50:
                recent_completion = completion_rates[-50:]
                metrics.update({
                    'recent_completion_mean': float(np.mean(recent_completion)),
                    'recent_completion_std': float(np.std(recent_completion)),
                    'completion_stability': float(np.std(recent_completion) / np.std(completion_rates)) if np.std(completion_rates) > 0 else 0.0
                })
        
        return metrics
    
    def generate_enhanced_convergence_analysis(self, model_save_path, final_episode):
        """生成增强的收敛性分析图表和报告"""
        if not self.episode_rewards:
            return
        
        # 设置中文字体 - 修复中文乱码问题
        set_chinese_font()
        
        # 创建多子图布局
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{self.network_type} 网络训练收敛性分析', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 1. 奖励曲线和移动平均
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.6, color='lightblue', label='原始奖励')
        if len(self.episode_rewards) > 20:
            moving_avg = np.convolve(self.episode_rewards, np.ones(20)/20, mode='valid')
            ax1.plot(range(20, len(self.episode_rewards) + 1), moving_avg, 
                    color='red', linewidth=2, label='移动平均(20)')
        ax1.set_title('奖励收敛曲线')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('奖励值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失曲线
        ax2 = axes[0, 1]
        if self.episode_losses:
            ax2.plot(episodes, self.episode_losses, color='orange', alpha=0.7)
            if len(self.episode_losses) > 20:
                loss_moving_avg = np.convolve(self.episode_losses, np.ones(20)/20, mode='valid')
                ax2.plot(range(20, len(self.episode_losses) + 1), loss_moving_avg, 
                        color='red', linewidth=2, label='移动平均(20)')
        ax2.set_title('损失收敛曲线')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 完成率曲线
        ax3 = axes[1, 0]
        if self.completion_rates:
            ax3.plot(episodes, self.completion_rates, color='green', alpha=0.7, label='完成率')
            if len(self.completion_rates) > 20:
                completion_moving_avg = np.convolve(self.completion_rates, np.ones(20)/20, mode='valid')
                ax3.plot(range(20, len(self.completion_rates) + 1), completion_moving_avg, 
                        color='red', linewidth=2, label='移动平均(20)')
            ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='早停阈值(0.95)')
            ax3.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='收敛阈值(0.85)')
        ax3.set_title('完成率收敛曲线')
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('完成率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 探索率衰减
        ax4 = axes[1, 1]
        if self.epsilon_values:
            ax4.plot(episodes, self.epsilon_values, color='purple', alpha=0.8)
        ax4.set_title('探索率衰减曲线')
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('探索率 (ε)')
        ax4.grid(True, alpha=0.3)
        
        # 5. 收敛性指标分析
        ax5 = axes[2, 0]
        if len(self.episode_rewards) > 100:
            # 计算滑动窗口标准差
            window_size = 50
            rolling_std = []
            for i in range(window_size, len(self.episode_rewards) + 1):
                window_std = np.std(self.episode_rewards[i-window_size:i])
                rolling_std.append(window_std)
            
            ax5.plot(range(window_size, len(self.episode_rewards) + 1), rolling_std, 
                    color='brown', alpha=0.8, label=f'滑动标准差({window_size})')
            ax5.axhline(y=np.std(self.episode_rewards) * 0.3, color='red', 
                       linestyle='--', alpha=0.7, label='收敛阈值')
        ax5.set_title('收敛稳定性分析')
        ax5.set_xlabel('训练轮次')
        ax5.set_ylabel('标准差')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 训练统计摘要
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # 获取收敛指标
        convergence_metrics = self.get_convergence_metrics()
        
        # 创建统计文本
        stats_text = f"""训练统计摘要:
        
总训练轮次: {final_episode}
最终奖励: {convergence_metrics.get('final_reward', 0):.2f}
最大奖励: {convergence_metrics.get('max_reward', 0):.2f}
平均奖励: {convergence_metrics.get('mean_reward', 0):.2f}
奖励改进: {convergence_metrics.get('reward_improvement', 0):.2f}

最终完成率: {convergence_metrics.get('final_completion_rate', 0):.3f}
平均完成率: {convergence_metrics.get('mean_completion_rate', 0):.3f}
完成率改进: {convergence_metrics.get('completion_improvement', 0):.3f}

收敛状态: {convergence_metrics.get('convergence_status', 'unknown')}
稳定性比率: {convergence_metrics.get('stability_ratio', 0):.3f}
改进比率: {convergence_metrics.get('improvement_ratio', 0):.3f}

最终探索率: {self.epsilon:.4f}
内存使用: {len(self.memory) if hasattr(self, 'memory') else 0}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        convergence_path = model_save_path.replace('.pth', '_enhanced_convergence.png')
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"增强收敛分析图已保存至: {convergence_path}")
        
        # 生成详细的收敛性报告
        self.generate_convergence_report(model_save_path, convergence_metrics, final_episode)
    
    def generate_convergence_report(self, model_save_path, metrics, final_episode):
        """生成详细的收敛性文字报告"""
        report_path = model_save_path.replace('.pth', '_convergence_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.network_type} 网络收敛性分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本训练信息
            f.write("训练基本信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"网络类型: {self.network_type}\n")
            f.write(f"训练轮次: {final_episode}\n")
            f.write(f"设备: {self.device}\n")
            f.write(f"参数数量: {sum(p.numel() for p in self.policy_net.parameters()):,}\n\n")
            
            # 收敛性分析
            f.write("收敛性分析:\n")
            f.write("-" * 30 + "\n")
            f.write(f"收敛状态: {metrics.get('convergence_status', 'unknown')}\n")
            f.write(f"稳定性比率: {metrics.get('stability_ratio', 0):.4f} (< 0.3 为收敛)\n")
            f.write(f"改进比率: {metrics.get('improvement_ratio', 0):.4f}\n\n")
            
            # 奖励分析
            f.write("奖励分析:\n")
            f.write("-" * 30 + "\n")
            f.write(f"最终奖励: {metrics.get('final_reward', 0):.2f}\n")
            f.write(f"最大奖励: {metrics.get('max_reward', 0):.2f}\n")
            f.write(f"平均奖励: {metrics.get('mean_reward', 0):.2f}\n")
            f.write(f"奖励标准差: {metrics.get('reward_std', 0):.2f}\n")
            f.write(f"奖励改进: {metrics.get('reward_improvement', 0):.2f}\n")
            
            if 'recent_mean' in metrics:
                f.write(f"最近50轮平均: {metrics['recent_mean']:.2f}\n")
                f.write(f"最近50轮标准差: {metrics['recent_std']:.2f}\n")
                f.write(f"早期50轮平均: {metrics['early_mean']:.2f}\n")
            f.write("\n")
            
            # 完成率分析
            if 'final_completion_rate' in metrics:
                f.write("完成率分析:\n")
                f.write("-" * 30 + "\n")
                f.write(f"最终完成率: {metrics['final_completion_rate']:.3f}\n")
                f.write(f"最大完成率: {metrics['max_completion_rate']:.3f}\n")
                f.write(f"平均完成率: {metrics['mean_completion_rate']:.3f}\n")
                f.write(f"完成率标准差: {metrics['completion_rate_std']:.3f}\n")
                f.write(f"完成率改进: {metrics['completion_improvement']:.3f}\n")
                
                if 'recent_completion_mean' in metrics:
                    f.write(f"最近50轮完成率: {metrics['recent_completion_mean']:.3f}\n")
                    f.write(f"完成率稳定性: {metrics['completion_stability']:.3f}\n")
                f.write("\n")
            
            # 训练建议
            f.write("训练建议:\n")
            f.write("-" * 30 + "\n")
            
            convergence_status = metrics.get('convergence_status', 'unknown')
            if convergence_status == 'converged':
                f.write("✓ 训练已收敛，模型性能稳定\n")
            elif convergence_status == 'partially_converged':
                f.write("⚠ 部分收敛，建议继续训练或调整超参数\n")
            else:
                f.write("✗ 训练不稳定，建议:\n")
                f.write("  - 降低学习率\n")
                f.write("  - 增加训练轮次\n")
                f.write("  - 调整网络结构\n")
            
            stability_ratio = metrics.get('stability_ratio', 1.0)
            if stability_ratio > 0.6:
                f.write("⚠ 奖励波动较大，建议:\n")
                f.write("  - 使用更平滑的探索策略\n")
                f.write("  - 增加批次大小\n")
                f.write("  - 添加奖励平滑机制\n")
            
            final_completion = metrics.get('final_completion_rate', 0)
            if final_completion < 0.8:
                f.write("⚠ 完成率较低，建议:\n")
                f.write("  - 调整奖励函数\n")
                f.write("  - 增加网络容量\n")
                f.write("  - 优化环境设计\n")
        
        print(f"收敛性报告已保存至: {report_path}")

    def save_model(self, path):
        """保存模型 - 增强版本，包含训练信息"""
        # 提取路径信息
        dir_path = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        # 构建包含训练信息的文件名
        training_info = {
            'episodes': getattr(self, 'final_episode', 0),
            'high_precision': getattr(self.config, 'HIGH_PRECISION_DISTANCE', False),
            'network_type': self.network_type,
            'epsilon': round(self.epsilon, 4),
            'timestamp': time.strftime("%Y%m%d-%H%M%S")
        }
        
        # 构建新的文件名
        info_str = f"ep{training_info['episodes']}"
        if training_info['high_precision']:
            info_str += "_hp"
        info_str += f"_eps{training_info['epsilon']}"
        info_str += f"_{training_info['timestamp']}"
        
        new_filename = f"{base_name}_{info_str}.pth"
        new_path = os.path.join(dir_path, new_filename)
        
        # 保存模型
        torch.save(self.policy_net.state_dict(), new_path)
        
        # 保存训练信息
        info_path = new_path.replace('.pth', '_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        return new_path
    
    def _cleanup_old_best_models(self, base_path):
        """清理旧的最佳模型文件，只保留最新的"""
        try:
            dir_path = os.path.dirname(base_path)
            base_name = os.path.splitext(os.path.basename(base_path))[0]
            
            # 查找所有相关的最佳模型文件
            pattern = f"{base_name}_best_avg_*.pth"
            old_models = []
            
            for filename in os.listdir(dir_path):
                if filename.startswith(f"{base_name}_best_avg_") and filename.endswith('.pth'):
                    full_path = os.path.join(dir_path, filename)
                    old_models.append(full_path)
            
            # 删除旧的最佳模型文件（保留最新的3个）
            if len(old_models) > 3:
                old_models.sort(key=os.path.getctime)
                for old_model in old_models[:-3]:  # 保留最新的3个
                    try:
                        os.remove(old_model)
                        # 同时删除对应的info文件
                        info_file = old_model.replace('.pth', '_info.json')
                        if os.path.exists(info_file):
                            os.remove(info_file)
                        print(f"  已清理旧模型: {os.path.basename(old_model)}")
                    except Exception as e:
                        print(f"  清理模型失败 {os.path.basename(old_model)}: {e}")
                        
        except Exception as e:
            print(f"清理旧模型时出错: {e}")
    
    def _save_best_model(self, model_save_path):
        """保存最佳模型，只保留一个最优模型文件"""
        try:
            # 根据网络类型决定保存路径
            if self.network_type == "ZeroShotGNN":
                # ZeroShotGNN保存到全局目录
                model_dir = "output"
                best_model_path = os.path.join(model_dir, f"best_{self.network_type}_model.pth")
            else:
                # 其他网络保存到场景目录
                dir_path = os.path.dirname(model_save_path)
                base_name = os.path.splitext(os.path.basename(model_save_path))[0]
                # 提取场景名称
                if "_" in base_name:
                    parts = base_name.split("_")
                    scenario_name = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
                    best_model_path = os.path.join(dir_path, f"best_{self.network_type}_{scenario_name}_model.pth")
                else:
                    best_model_path = os.path.join(dir_path, f"best_{self.network_type}_model.pth")
            
            # 删除旧的最佳模型（如果存在）
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
                info_file = best_model_path.replace('.pth', '_info.json')
                if os.path.exists(info_file):
                    os.remove(info_file)
            
            # 保存新的最佳模型
            torch.save(self.policy_net.state_dict(), best_model_path)
            
            # 保存模型信息
            model_info = {
                'network_type': self.network_type,
                'episodes': getattr(self, 'final_episode', 0),
                'epsilon': round(self.epsilon, 4),
                'timestamp': time.strftime("%Y%m%d-%H%M%S"),
                'best_avg_reward': getattr(self, 'best_avg_reward', 0)
            }
            
            info_path = best_model_path.replace('.pth', '_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            return best_model_path
            
        except Exception as e:
            print(f"保存最佳模型失败: {e}")
            return model_save_path
    
    def load_model(self, path):
        """加载模型 - 增强版本，支持信息文件和维度检查"""
        try:
            # 先加载模型检查维度兼容性
            checkpoint = torch.load(path, map_location=self.device)
            
            # 检查模型维度兼容性
            current_state_dict = self.policy_net.state_dict()
            
            # 检查关键层的维度
            dimension_mismatch = False
            mismatch_details = []
            for key in checkpoint.keys():
                if key in current_state_dict:
                    if checkpoint[key].shape != current_state_dict[key].shape:
                        mismatch_details.append(f"{key}: 保存的模型 {checkpoint[key].shape} vs 当前模型 {current_state_dict[key].shape}")
                        dimension_mismatch = True
            
            if dimension_mismatch:
                print(f"模型 {os.path.basename(path)} 与当前网络结构不兼容，跳过加载")
                for detail in mismatch_details:
                    print(f"  维度不匹配 - {detail}")
                return None
            
            # 维度兼容，加载模型
            self.policy_net.load_state_dict(checkpoint)
            
            # 尝试加载训练信息
            info_path = path.replace('.pth', '_info.json')
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        training_info = json.load(f)
                    print(f"已加载兼容模型: {os.path.basename(path)}")
                    print(f"  训练轮次: {training_info.get('episodes', 'unknown')}")
                    print(f"  高精度距离: {'是' if training_info.get('high_precision', False) else '否'}")
                    print(f"  最终探索率: {training_info.get('epsilon', 'unknown')}")
                    return training_info
                except Exception as e:
                    print(f"加载训练信息失败: {e}")
            
            print(f"已加载兼容模型: {os.path.basename(path)} (无训练信息)")
            return {}
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    def _load_best_model(self):
        """加载最佳模型以修复NaN问题"""
        try:
            # 查找最佳模型文件
            output_dir = getattr(self, 'output_dir', 'output')
            if os.path.exists(output_dir):
                best_models = []
                for filename in os.listdir(output_dir):
                    if filename.endswith('.pth') and 'best_avg' in filename:
                        full_path = os.path.join(output_dir, filename)
                        best_models.append(full_path)
                
                if best_models:
                    # 选择最新的最佳模型
                    latest_model = max(best_models, key=os.path.getctime)
                    checkpoint = torch.load(latest_model, map_location=self.device)
                    self.policy_net.load_state_dict(checkpoint)
                    print(f"已重新加载最佳模型: {os.path.basename(latest_model)}")
                    return True
        except Exception as e:
            print(f"重新加载最佳模型失败: {e}")
        return False
    
    def get_task_assignments(self, temperature=0.1):
        """
        获取任务分配 - 改进版本，使用带低温的softmax采样
        
        Args:
            temperature (float): 温度参数，控制采样的随机性。值越小，决策越确定性。
                              建议范围：0.05-0.3，默认0.1
        
        Returns:
            dict: 任务分配结果
        """
        self.policy_net.eval()
        state = self.env.reset()
        assignments = {u.id: [] for u in self.env.uavs}
        done, step = False, 0
        
        while not done and step < len(self.env.targets) * len(self.env.uavs):
            state_tensor = self._prepare_state_tensor(state)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            
            # 数值稳定性处理
            q_values = torch.clamp(q_values, min=-1e6, max=1e6)  # 防止极值
            
            # 检查Q值是否包含NaN或无穷大
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                print("警告: Q值包含NaN或无穷大，重新加载最佳模型")
                # 尝试重新加载最佳模型
                try:
                    self._load_best_model()
                    q_values = self.policy_net(state_tensor)
                    q_values = torch.clamp(q_values, min=-1e6, max=1e6)
                except:
                    pass
                
                # 如果仍然有问题，使用随机动作
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    action_idx = random.randint(0, q_values.size(1) - 1)
                else:
                    action_idx = q_values.argmax(dim=1).item()
            else:
                # 使用带低温的softmax进行采样，替代简单的argmax
                # 低温确保大部分概率集中在最优动作上，但仍保留微小的随机性
                logits = q_values / max(temperature, 1e-8)  # 防止除零
                
                # 防止数值溢出
                logits = torch.clamp(logits, min=-50, max=50)
                action_probs = F.softmax(logits, dim=1)
                
                # 检查概率是否有效
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    print("警告: 概率分布包含NaN或无穷大，使用贪婪策略")
                    action_idx = q_values.argmax(dim=1).item()
                else:
                    # 确保概率和为1（数值稳定性）
                    action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
                    
                    # 从概率分布中采样动作
                    try:
                        action_dist = torch.distributions.Categorical(action_probs)
                        action_idx = action_dist.sample().item()
                    except ValueError as e:
                        print(f"警告: 采样失败 {e}，使用贪婪策略")
                        action_idx = q_values.argmax(dim=1).item()
            
            action = self._index_to_action(action_idx)
            target_id, uav_id, _ = action
            
            uav = self.env.uavs[uav_id - 1]
            target = self.env.targets[target_id - 1]
            
            if np.all(uav.resources <= 0):
                step += 1
                continue
            
            assignments[uav_id].append((target_id, action[2]))
            state, _, done, _, _ = self.env.step(action_idx)
            step += 1
        
        return assignments

# =============================================================================
# section 5: 核心功能函数
# =============================================================================
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
                # print(f"警告: UAV {uav_id} 被分配到已满足的目标 {target_id}，跳过此分配")  # 简化输出
                continue
            
            # 检查无人机是否还有资源
            if not np.any(uav_resources[uav_id] > 1e-6):
                # print(f"警告: UAV {uav_id} 资源已耗尽，跳过后续分配")  # 简化输出
                break
            
            # 计算实际贡献
            contribution = np.minimum(uav_resources[uav_id], target_needs[target_id])
            
            # 只有当有实际贡献时才保留此分配
            if np.any(contribution > 1e-6):
                calibrated_assignments[uav_id].append((target_id, phi_idx))
                uav_resources[uav_id] -= contribution
                target_needs[target_id] -= contribution
                # print(f"UAV {uav_id} -> 目标 {target_id}: 贡献 {contribution}")  # 简化输出
            else:
                # print(f"警告: UAV {uav_id} 对目标 {target_id} 无有效贡献，跳过此分配")  # 简化输出
                pass
    
    # 统计校准结果
    original_count = sum(len(tasks) for tasks in task_assignments.values())
    calibrated_count = sum(len(tasks) for tasks in calibrated_assignments.values())
    removed_count = original_count - calibrated_count
    
    print(f"资源分配校准完成:")
    print(f"  原始分配数量: {original_count}")
    print(f"  校准后数量: {calibrated_count}")
    print(f"  移除无效分配: {removed_count}")
    
    return calibrated_assignments

def plot_training_curves(training_history, output_dir, timestamp):
    """绘制训练曲线"""
    if not training_history or not training_history.get('episode_rewards'):
        return
    
    set_chinese_font()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(len(training_history['episode_rewards']))
    
    # 奖励曲线
    ax1.plot(episodes, training_history['episode_rewards'], 'b-', alpha=0.7)
    if len(training_history['episode_rewards']) > 50:
        # 添加移动平均线
        window = min(50, len(training_history['episode_rewards']) // 10)
        moving_avg = np.convolve(training_history['episode_rewards'], 
                               np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(training_history['episode_rewards'])), 
                moving_avg, 'r-', linewidth=2, label=f'{window}轮移动平均')
        ax1.legend()
    ax1.set_title('训练奖励曲线')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('奖励')
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    if training_history.get('episode_losses'):
        ax2.plot(episodes, training_history['episode_losses'], 'g-', alpha=0.7)
        ax2.set_title('训练损失曲线')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失')
        ax2.grid(True, alpha=0.3)
    
    # 探索率曲线
    if training_history.get('epsilon_values'):
        ax3.plot(episodes, training_history['epsilon_values'], 'm-', alpha=0.7)
        ax3.set_title('探索率变化')
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('探索率')
        ax3.grid(True, alpha=0.3)
    
    # 完成率曲线
    if training_history.get('completion_rates'):
        ax4.plot(episodes, training_history['completion_rates'], 'c-', alpha=0.7)
        ax4.set_title('任务完成率')
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('完成率')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    curves_path = os.path.join(output_dir, f"training_curves_{timestamp}.png")
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存至: {curves_path}")
    plt.close()

def simple_evaluate_plan(task_assignments, uavs, targets, deadlocked_tasks=None):
    """修复后的计划评估函数 - 基于实际任务完成情况和资源分配"""
    if not task_assignments or not any(task_assignments.values()):
        return {
            'completion_rate': 0.0,
            'satisfied_targets_rate': 0.0,
            'total_reward_score': 0.0,
            'marginal_utility_score': 0.0,
            'resource_efficiency_score': 0.0,
            'distance_cost_score': 0.0,
            'actual_completion_rate': 0.0,
            'resource_satisfaction_rate': 0.0
        }
    
    # 重置目标资源状态以计算实际完成情况
    temp_targets = []
    for target in targets:
        temp_target = type('TempTarget', (), {'id': target.id, 'position': target.position, 
                                           'resources': target.resources.astype(float).copy(), 
                                           'remaining_resources': target.resources.astype(float).copy()})()
        temp_targets.append(temp_target)
    
    # 计算实际资源贡献
    total_contribution = 0.0
    total_available = 0.0
    total_distance = 0.0
    
    for uav in uavs:
        total_available += np.sum(uav.resources)
        uav_remaining = uav.initial_resources.copy().astype(float)
        
        if uav.id in task_assignments:
            for task in task_assignments[uav.id]:
                # 处理不同格式的任务数据
                if isinstance(task, tuple):
                    target_id = task[0]
                    phi_idx = task[1]
                    start_pos = uav.current_position
                else:
                    target_id = task['target_id']
                    start_pos = task['start_pos']
                
                target = next(t for t in temp_targets if t.id == target_id)
                
                # 计算实际贡献
                contribution = np.minimum(target.remaining_resources, uav_remaining)
                actual_contribution = np.sum(contribution)
                
                if actual_contribution > 0:
                    target.remaining_resources = target.remaining_resources - contribution
                    uav_remaining = uav_remaining - contribution
                    total_contribution += actual_contribution
                
                # 计算距离
                distance = np.linalg.norm(start_pos - target.position)
                total_distance += distance
    
    # 计算实际完成率（优化版本，综合考虑多个指标）
    total_targets = len(targets)
    if total_targets > 0:
        # 1. 目标满足数量比例
        completed_targets = sum(1 for t in temp_targets if np.all(t.remaining_resources <= 0))
        target_satisfaction_rate = completed_targets / total_targets
        
        # 2. 总体资源满足率
        total_demand = sum(np.sum(t.resources) for t in targets)
        total_remaining = sum(np.sum(t.remaining_resources) for t in temp_targets)
        resource_satisfaction_rate = (total_demand - total_remaining) / total_demand if total_demand > 0 else 0
        
        # 3. 资源利用率
        total_initial_supply = sum(np.sum(uav.initial_resources) for uav in uavs)
        total_used = total_contribution
        resource_utilization_rate = total_used / total_initial_supply if total_initial_supply > 0 else 0
        
        # 综合完成率 = 目标满足率(50%) + 资源满足率(30%) + 资源利用率(20%)
        actual_completion_rate = (target_satisfaction_rate * 0.5 + 
                                resource_satisfaction_rate * 0.3 + 
                                resource_utilization_rate * 0.2)
    else:
        actual_completion_rate = 0.0
    
    # 计算资源满足率（基于实际贡献）
    total_required = sum(np.sum(t.resources) for t in targets)
    resource_satisfaction_rate = total_contribution / total_required if total_required > 0 else 0.0
    
    # 计算目标满足率（部分或完全满足）
    satisfied_targets = sum(1 for t in temp_targets if np.any(t.resources > t.remaining_resources))
    satisfied_targets_rate = satisfied_targets / total_targets if total_targets > 0 else 0.0
    
    # 计算资源效率
    resource_efficiency_score = 0.0
    if total_available > 0:
        resource_efficiency_score = (total_contribution / total_available) * 500
    
    # 计算距离成本
    distance_cost_score = -total_distance * 0.1
    
    # 计算边际效用
    marginal_utility_score = 0.0
    for target in temp_targets:
        target_initial_total = np.sum(target.resources)
        target_remaining = np.sum(target.remaining_resources)
        if target_initial_total > 0:
            completion_ratio = 1.0 - (target_remaining / target_initial_total)
            # 使用改进的边际效用计算
            marginal_utility = completion_ratio * (1.0 - completion_ratio)
            marginal_utility_score += marginal_utility * 300
    
    # 计算总奖励分数（与优化后的奖励函数保持一致）
    target_completion_score = actual_completion_rate * 500  # 与新的target_completion_reward一致
    completion_bonus = 1000 if actual_completion_rate >= 0.95 else 0  # 放宽完成标准
    
    total_reward_score = (target_completion_score + marginal_utility_score + 
                         resource_efficiency_score + distance_cost_score + completion_bonus)
    
    return {
        'completion_rate': actual_completion_rate,  # 综合完成率
        'satisfied_targets_rate': satisfied_targets_rate,  # 目标满足率
        'target_satisfaction_rate': target_satisfaction_rate,  # 完全满足的目标比例
        'resource_satisfaction_rate': resource_satisfaction_rate,  # 总体资源满足率
        'resource_utilization_rate': resource_utilization_rate,  # 资源利用率
        'total_reward_score': total_reward_score,
        'marginal_utility_score': marginal_utility_score,
        'resource_efficiency_score': resource_efficiency_score,
        'distance_cost_score': distance_cost_score,
        'target_completion_score': target_completion_score,
        'completion_bonus': completion_bonus,
        # 'actual_completion_rate': actual_completion_rate,  # 移除重复定义
        'completed_targets_count': completed_targets,  # 完全满足的目标数量
        'total_targets_count': total_targets,  # 总目标数量
        'total_contribution': total_contribution,  # 总资源贡献
        'total_demand': sum(np.sum(t.resources) for t in targets),  # 总资源需求
        'total_initial_supply': sum(np.sum(uav.initial_resources) for uav in uavs)  # 总初始资源供给
    }

def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config) -> Tuple[defaultdict, dict]:
    """(已更新) 计算经济同步速度，并返回未完成的任务以进行死锁检测。"""
    # 转换任务数据结构并补充资源消耗
    final_plan = defaultdict(list)
    uav_status = {u.id: {'pos': u.position, 'free_at': 0.0, 'heading': u.heading} for u in uavs}
    remaining_tasks = {uav_id: list(tasks) for uav_id, tasks in task_assignments.items()}
    task_step_counter = defaultdict(lambda: 1)
    
    def _plan_single_leg(args):
        uav_id, start_pos, target_pos, start_heading, end_heading, obstacles, config = args
        planner = PHCurveRRTPlanner(start_pos, target_pos, start_heading, end_heading, obstacles, config)
        return uav_id, planner.plan()
    
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
                uav_id = uav_info['uav_id']
                phi_angle = uav_info['phi_idx'] * (2 * np.pi / config.GRAPH_N_PHI)
                args = (uav_id, uav_status[uav_id]['pos'], target.position, uav_status[uav_id]['heading'], phi_angle, obstacles, config)
                _, plan_result = _plan_single_leg(args)
                if plan_result: path_planners[uav_id] = {'path_points': plan_result[0], 'distance': plan_result[1]}
            
            time_windows = []
            for uav_info in uav_infos:
                uav_id = uav_info['uav_id']
                if uav_id not in path_planners: continue
                uav = next((u for u in uavs if u.id == uav_id), None)
                if not uav: continue
                
                distance = path_planners[uav_id]['distance']
                free_at = uav_status[uav_id]['free_at']
                t_min = free_at + (distance / uav.velocity_range[1])
                t_max = free_at + (distance / uav.velocity_range[0]) if uav.velocity_range[0] > 0 else float('inf')
                t_econ = free_at + (distance / uav.economic_speed)
                time_windows.append({'uav_id': uav_id, 'phi_idx': uav_info['phi_idx'], 't_min': t_min, 't_max': t_max, 't_econ': t_econ})
            
            if not time_windows: continue
            sync_start = max(tw['t_min'] for tw in time_windows)
            sync_end = min(tw['t_max'] for tw in time_windows)
            is_feasible = sync_start <= sync_end + 1e-6
            final_sync_time = np.clip(np.median([tw['t_econ'] for tw in time_windows]), sync_start, sync_end) if is_feasible else sync_start
            group_arrival_times.append({'target_id': target_id, 'arrival_time': final_sync_time, 'uav_infos': time_windows, 'is_feasible': is_feasible, 'path_planners': path_planners})
        
        if not group_arrival_times: break
        next_event = min(group_arrival_times, key=lambda x: x['arrival_time'])
        target_pos = next(t.position for t in targets if t.id == next_event['target_id'])
        
        for uav_info in next_event['uav_infos']:
            uav_id = uav_info['uav_id']
            if uav_id not in next_event['path_planners']: continue
            
            uav, plan_data = next(u for u in uavs if u.id == uav_id), next_event['path_planners'][uav_id]
            travel_time = next_event['arrival_time'] - uav_status[uav_id]['free_at']
            speed = (plan_data['distance'] / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
            
            final_plan[uav_id].append({
                'target_id': next_event['target_id'],
                'start_pos': uav_status[uav_id]['pos'],
                'speed': np.clip(speed, uav.velocity_range[0], uav.velocity_range[1]),
                'arrival_time': next_event['arrival_time'],
                'step': task_step_counter[uav_id],
                'is_sync_feasible': next_event['is_feasible'],
                'phi_idx': uav_info['phi_idx'],
                'path_points': plan_data['path_points'],
                'distance': plan_data['distance']
            })
            
            task_step_counter[uav_id] += 1
            phi_angle = uav_info['phi_idx'] * (2 * np.pi / config.GRAPH_N_PHI)
            uav_status[uav_id].update(pos=target_pos, free_at=next_event['arrival_time'], heading=phi_angle)
            if remaining_tasks.get(uav_id): remaining_tasks[uav_id].pop(0)
    
    # 应用协同贪婪资源分配策略
    temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}
    
    events = defaultdict(list)
    for uav_id, tasks in final_plan.items():
        for task in tasks:
            event_key = (task['arrival_time'], task['target_id'])
            events[event_key].append({'uav_id': uav_id, 'task_ref': task})
    
    for event_key in sorted(events.keys()):
        arrival_time, target_id = event_key
        collaborating_tasks = events[event_key]
        target_remaining = temp_target_resources[target_id].copy()
        
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
        # 使用传入的输出目录或默认目录
        import builtins
        if hasattr(builtins, 'current_output_dir') and builtins.current_output_dir:
            img_dir = builtins.current_output_dir
            report_dir = builtins.current_output_dir
        else:
            img_dir = "output"
            report_dir = "output"
        
        os.makedirs(img_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
        base_filename = f"{clean_scenario_name}_{timestamp}"
        img_filepath = os.path.join(img_dir, f"{base_filename}.png")
        plt.savefig(img_filepath, dpi=300)
        print(f"结果图已保存至: {img_filepath}")
        
        if save_report:
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

def run_scenario(config, base_uavs, base_targets, obstacles, scenario_name, 
                network_type="SimpleNetwork", save_visualization=True, show_visualization=True, 
                save_report=False, force_retrain=False, incremental_training=False, output_base_dir=None):
    """
    运行场景 - 核心执行器 (优化版本，统一目录结构，支持TensorBoard)
    
    Args:
        config: 配置对象
        base_uavs: UAV列表
        base_targets: 目标列表
        obstacles: 障碍物列表
        scenario_name: 场景名称
        network_type: 网络类型 ("SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual")
        save_visualization: 是否保存可视化
        show_visualization: 是否显示可视化
        save_report: 是否保存报告
        force_retrain: 是否强制重新训练
        incremental_training: 是否增量训练
        output_base_dir: 基础输出目录，如果为None则使用默认
        
    Returns:
        final_plan: 最终计划
        training_time: 训练时间
        training_history: 训练历史
        evaluation_metrics: 评估指标
    """
    import time
    import os
    import pickle
    import json
    import glob
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    network_type = config.NETWORK_TYPE

    # 优化的目录结构 - 根据网络类型决定模型存储位置
    if network_type == "ZeroShotGNN":
        # ZeroShotGNN具有泛化能力，模型存储在全局目录
        model_dir = "output"
        output_dir = f"output/{scenario_name}_{network_type}"
    else:
        # 其他网络结构与场景相关，存储在场景目录
        output_dir = f"output/{scenario_name}_{network_type}"
        model_dir = output_dir
    
    # 创建统一的输出目录结构（不再创建子文件夹）
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"运行场景: {scenario_name} ({len(base_uavs)}UAV, {len(base_targets)}目标, {len(obstacles)}障碍)")
    print(f"输出目录: {output_dir}")
    
    # 创建图
    graph = DirectedGraph(base_uavs, base_targets, config.GRAPH_N_PHI, obstacles, config)
    
    # === 双模式观测支持 ===
    # 根据网络类型决定观测模式
    obs_mode = "graph" if network_type == "ZeroShotGNN" else "flat"
    print(f"观测模式: {obs_mode} (网络类型: {network_type})")
    
    # 创建求解器 - 动态计算输入维度，支持TensorBoard
    test_env = UAVTaskEnv(base_uavs, base_targets, graph, obstacles, config, obs_mode=obs_mode)
    test_state = test_env.reset()
    
    if obs_mode == "flat":
        i_dim = len(test_state)
    else:  # graph mode
        # 对于图模式，使用占位维度
        i_dim = 64  # 占位值，实际由网络内部处理
    
    h_dim = [256, 128, 64]  # 隐藏层维度列表
    o_dim = test_env.n_actions
    
    # TensorBoard目录
    tensorboard_dir = os.path.join(output_dir, "tensorboard", timestamp)
    
    solver = GraphRLSolver(
        base_uavs, base_targets, graph, obstacles, 
        i_dim, h_dim, o_dim, config, 
        network_type=network_type,
        tensorboard_dir=tensorboard_dir,
        obs_mode=obs_mode
    )
    
    # 模型文件路径策略
    if network_type == "ZeroShotGNN":
        # ZeroShotGNN使用全局最优模型
        model_path = os.path.join(model_dir, f"best_{network_type}_model.pth")
    else:
        # 其他网络使用场景特定的最优模型
        model_path = os.path.join(model_dir, f"best_{network_type}_{scenario_name}_model.pth")
    
    # 训练或加载模型
    training_time = 0
    training_history = None
    
    # 检查是否强制重训或配置为训练模式
    force_retrain = force_retrain or config.FORCE_RETRAIN
    
    if force_retrain or config.is_training_mode():
        # 检查是否需要重新训练
        should_train = force_retrain or not os.path.exists(model_path)
        
        if not should_train:
            # 尝试加载现有模型
            print(f"尝试加载已有模型: {model_path}")
            load_result = solver.load_model(model_path)
            if load_result is None:
                print("模型加载失败，将重新训练")
                should_train = True
            else:
                print("✓ 成功加载现有模型，跳过训练")
        
        if should_train:
            print(f"开始训练 {network_type} 网络，训练轮数: {config.EPISODES}")
            training_time = solver.train(
                episodes=config.EPISODES,
                patience=config.PATIENCE,
                log_interval=config.LOG_INTERVAL,
                model_save_path=model_path
            )
            
            # 保存训练历史
            training_history = {
                'episode_rewards': getattr(solver, 'episode_rewards', []),
                'episode_losses': getattr(solver, 'episode_losses', []),
                'epsilon_values': getattr(solver, 'epsilon_values', []),
                'completion_rates': getattr(solver, 'completion_rates', []),
                'episode_steps': getattr(solver, 'episode_steps', []),
                'memory_usage': getattr(solver, 'memory_usage', [])
            }
            
            # 保存训练历史到文件
            history_path = os.path.join(output_dir, f"training_history_{timestamp}.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(training_history, f)
            print(f"训练历史已保存至: {history_path}")
            
            # 保存训练曲线图
            plot_training_curves(training_history, output_dir, timestamp)
            
            # 只保存最佳模型（训练过程中已保存）
            print(f"最佳模型已在训练过程中保存")
    else:
        # 推理模式，只加载模型
        print(f"推理模式：加载模型 {model_path}")
        if os.path.exists(model_path):
            load_result = solver.load_model(model_path)
            if load_result is None:
                raise FileNotFoundError(f"推理模式下模型加载失败: {model_path}")
            print("✓ 模型加载成功")
        else:
            raise FileNotFoundError(f"推理模式下找不到模型文件: {model_path}")
    
    # 生成任务分配方案
    print("生成任务分配方案...")
    plan_start_time = time.time()
    
    # 获取任务分配
    task_assignments = solver.get_task_assignments(temperature=0.1)
    
    # 校准资源分配
    calibrated_assignments = calibrate_resource_assignments(task_assignments, base_uavs, base_targets)
    
    # 计算经济同步速度和路径
    final_plan, deadlocked_tasks = calculate_economic_sync_speeds(
        calibrated_assignments, base_uavs, base_targets, graph, obstacles, config
    )
    
    plan_generation_time = time.time() - plan_start_time
    
    # 评估方案
    from evaluate import evaluate_plan
    evaluation_metrics = evaluate_plan(final_plan, base_uavs, base_targets, deadlocked_tasks)
    
    # 保存评估结果
    evaluation_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(evaluation_metrics), f, ensure_ascii=False, indent=2)
    print(f"评估结果已保存至: {evaluation_path}")
    
    # 可视化结果
    if save_visualization or show_visualization:
        # 临时设置全局变量以传递输出目录
        import builtins
        builtins.current_output_dir = output_dir
        
        visualize_task_assignments(
            final_plan, base_uavs, base_targets, obstacles, config, scenario_name,
            training_time, plan_generation_time, 
            save_plot=save_visualization, 
            show_plot=show_visualization,
            save_report=save_report,
            deadlocked_tasks=deadlocked_tasks,
            evaluation_metrics=evaluation_metrics
        )
        
        # 清理全局变量
        if hasattr(builtins, 'current_output_dir'):
            delattr(builtins, 'current_output_dir')
    
    print(f"场景 {scenario_name} 执行完成")
    print(f"训练时间: {training_time:.2f}s, 方案生成时间: {plan_generation_time:.2f}s")
    print(f"完成率: {evaluation_metrics.get('completion_rate', 0):.3f}")
    
    return final_plan, training_time, training_history, evaluation_metrics

# =============================================================================
# section 6: 主程序入口
# =============================================================================
def main():
    """主函数，用于单独运行和调试一个默认场景。"""
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    config = Config()
    
    # 检查是否启用批量测试
    if hasattr(config, 'BATCH_TEST_SCENARIOS') and config.BATCH_TEST_SCENARIOS:
        print("检测到批量测试模式，启动批量测试...")
        # 这里可以添加批量测试逻辑
        return
    
    # 检查是否启用网络结构对比模式
    if hasattr(config, 'NETWORK_COMPARISON_MODE') and config.NETWORK_COMPARISON_MODE:
        print("检测到网络结构对比模式...")
        config.EPISODES = 1500
        complex_uavs, complex_targets, complex_obstacles = get_strategic_trap_scenario(config.OBSTACLE_TOLERANCE)
        
        # 测试不同的网络架构
        network_types = ['DeepFCN', 'DeepFCNResidual', 'ZeroShotGNN']
        
        for network_type in network_types:
            print(f"\n{'='*60}")
            print(f"测试网络架构: {network_type}")
            print(f"{'='*60}")
            
            try:
                run_scenario(config, complex_uavs, complex_targets, complex_obstacles, 
                            f"网络架构测试-{network_type}", show_visualization=False, save_report=True,
                            force_retrain=config.FORCE_RETRAIN, incremental_training=False, network_type=network_type)
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
    
    # 选择场景 - 可以修改这里来测试不同场景
    scenario_functions = {
        'small': lambda: get_small_scenario(config.OBSTACLE_TOLERANCE),
        'balanced': lambda: get_balanced_scenario(config.OBSTACLE_TOLERANCE), 
        'complex': lambda: get_complex_scenario(),
        'experimental': lambda: get_new_experimental_scenario(config.OBSTACLE_TOLERANCE),
        'complex_v4': lambda: get_complex_scenario_v4(30),
        'strategic_trap': lambda: get_strategic_trap_scenario(config.OBSTACLE_TOLERANCE)
    }
    
    # 默认运行场景列表 - 可以修改这里选择要运行的场景
    scenarios_to_run = ['experimental']  # 可以改为 ['small', 'balanced', 'strategic_trap'] 或多个场景
    
    # 使用配置中指定的网络类型experimental
    network_type = config.NETWORK_TYPE
    print(f"使用配置的网络类型: {network_type}")
    
    for scenario_name in scenarios_to_run:
        if scenario_name in scenario_functions:
            print(f"\n{'='*80}")
            print(f"运行场景: {scenario_name}")
            print(f"{'='*80}")
            
            try:
                uavs, targets, obstacles = scenario_functions[scenario_name]()
                
                run_scenario(config, uavs, targets, obstacles, 
                            f"{scenario_name}场景", show_visualization=False, save_report=True,
                            force_retrain=config.FORCE_RETRAIN, incremental_training=False, network_type=network_type)
                            
            except Exception as e:
                print(f"场景 {scenario_name} 运行失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"未知场景: {scenario_name}")
    
    print(f"\n{'='*80}")
    print("程序执行完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()