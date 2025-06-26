import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, FontManager
from scipy.spatial.distance import euclidean
import random
from typing import List, Dict, Tuple, Set
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import os
import time
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib
import matplotlib.font_manager as fm
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载OpenMP库

# 设置中文字体，增强错误处理
def set_chinese_font(font_path='SourceHanSansSC-Regular.otf'):
    """
    通过直接指定字体文件路径来设置matplotlib字体。
    这种方法绕过了系统字体缓存，更加可靠。
    """
    if not os.path.exists(font_path):
        print(f"警告：字体文件 '{font_path}' 未找到。")
        print("请下载“思源黑体”并将其 .otf 或 .ttf 文件放置在脚本同级目录下。")
        print("中文将无法正常显示。")
        return False

    try:
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"字体设置成功，已加载：{font_prop.get_name()}")
        return True
    except Exception as e:
        print(f"错误: 通过路径加载字体 '{font_path}' 时失败: {e}")
        return False


# ##################################################################
# 核心修改部分：增强版的可视化函数
# ##################################################################
def visualize_task_assignments(task_assignments, uavs, targets, show=True):
    """
    功能增强版的可视化函数，清晰展示任务分配的详细信息。
    """
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(18, 12))  # 使用 subplot 并增大画布

    # ------------------- 1. 绘制基础的无人机和目标点 -------------------
    # 绘制目标点，并标注其资源需求
    target_positions = np.array([t.position for t in targets])
    ax.scatter(target_positions[:, 0], target_positions[:, 1], c='red', marker='x', s=150, label='目标', zorder=5)
    for t in targets:
        # 将资源向量转换为字符串
        res_str = np.array2string(t.resources, formatter={'float_kind': lambda x: "%.0f" % x})
        ax.annotate(f"目标 {t.id}\n需求: {res_str}", (t.position[0], t.position[1]),
                    fontsize=10, xytext=(10, -20), textcoords='offset points', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.2))

    # 绘制无人机起点，并标注其初始资源
    uav_positions = np.array([u.position for u in uavs])
    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], c='blue', marker='s', s=150, label='无人机起点', zorder=5)
    for uav in uavs:
        res_str = np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: "%.0f" % x})
        ax.annotate(f"无人机 {uav.id}\n初始资源: {res_str}", (uav.position[0], uav.position[1]),
                    fontsize=10, xytext=(10, -20), textcoords='offset points', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.2))

    # ------------------- 2. 预计算与路径绘制 -------------------
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    # 预计算每个目标的协作者数量
    target_collaborators = defaultdict(int)
    for uav_id, tasks in task_assignments.items():
        unique_targets = {task[0] for task in tasks}
        for target_id in unique_targets:
            target_collaborators[target_id] += 1

    # 存储用于生成报告的详细信息
    report_details = defaultdict(list)

    # 绘制每个无人机的任务路径
    for uav_id, tasks in task_assignments.items():
        uav = next(u for u in uavs if u.id == uav_id)
        uav_color = colors[uav_id % len(colors)]

        current_pos = uav.position.copy()

        for i, (target_id, heading) in enumerate(tasks):
            target = next(t for t in targets if t.id == target_id)

            # 绘制路径线段
            ax.plot([current_pos[0], target.position[0]],
                    [current_pos[1], target.position[1]],
                    c=uav_color, linestyle='-', linewidth=2, alpha=0.8, marker='o', markersize=4,
                    markerfacecolor='white')

            # 在路径中点添加任务顺序标签
            mid_point = (current_pos + target.position) / 2
            ax.text(mid_point[0], mid_point[1] + 50, str(i + 1),
                    c='white', backgroundcolor=uav_color, ha='center', va='center',
                    fontsize=9, fontweight='bold', bbox=dict(boxstyle='circle,pad=0.2', fc=uav_color, ec='none'))

            collaborators_count = target_collaborators.get(target_id, 1)
            resource_cost = np.ceil(target.resources / collaborators_count).astype(np.int32)

            report_details[uav_id].append({
                "step": i + 1,
                "target_id": target_id,
                "cost": resource_cost
            })

            current_pos = target.position.copy()

    # ------------------- 3. 创建并显示详细报告 -------------------
    report_text = "---------- 任务执行报告 ----------\n\n"
    for uav in uavs:
        report_text += f"■ 无人机 {uav.id} (初始资源: {np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: '%.0f' % x})})\n"

        temp_resources = uav.initial_resources.copy()
        details = report_details.get(uav.id, [])

        if not details:
            report_text += "  - 未分配任何任务\n"
        else:
            for detail in details:
                cost_str = np.array2string(detail['cost'], formatter={'float_kind': lambda x: '%.0f' % x})

                # 创建一个副本用于显示，避免修改原始数据
                remaining_resources_after_step = temp_resources - detail['cost']
                res_str = np.array2string(remaining_resources_after_step,
                                          formatter={'float_kind': lambda x: '%.0f' % x})

                report_text += f"  {detail['step']}. 执行目标 {detail['target_id']}:\n"
                report_text += f"     消耗: {cost_str} -> 剩余: {res_str}\n"

                # 更新追踪的资源量
                temp_resources = remaining_resources_after_step

        report_text += "\n"

    fig.text(0.78, 0.95, report_text,
             transform=ax.transAxes,  # 相对于坐标轴定位
             ha="left", va="top", fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.9))

    # ------------------- 4. 最终美化和显示 -------------------
    ax.set_title("GCN-DL多无人机任务分配结果", fontsize=20, pad=20)
    ax.set_xlabel("X坐标 (m)", fontsize=14)
    ax.set_ylabel("Y坐标 (m)", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    plt.subplots_adjust(right=0.75, top=0.9)

    if show:
        plt.show()

    return fig


class UAV:
    """无人机类，存储无人机的属性和状态"""

    def __init__(self, id: int, position: np.ndarray, heading: float,
                 resources: np.ndarray, max_distance: float, velocity: float = 50):
        self.id = id
        self.position = np.array(position)
        self.heading = heading
        self.resources = np.array(resources)
        self.initial_resources = np.array(resources)
        self.max_distance = max_distance
        self.velocity = velocity
        self.task_sequence = []
        self.current_distance = 0
        self.current_position = np.array(position)

    def reset(self):
        """重置无人机状态"""
        self.resources = self.initial_resources.copy()
        self.current_distance = 0
        self.current_position = self.position.copy()
        self.task_sequence = []


class Target:
    """目标类，存储目标的属性"""

    def __init__(self, id: int, position: np.ndarray, resources: np.ndarray, value: float):
        self.id = id
        self.position = np.array(position)
        self.resources = np.array(resources)
        self.value = value
        self.allocated_uavs = []
        self.arrival_times = []
        self.is_assigned = False


class DirectedGraph:
    """有向图模型，表示任务分配和路径规划"""

    def __init__(self, uavs: List[UAV], targets: List[Target], n_phi: int = 6, cache_path=None):
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi
        self.phi_set = [2 * np.pi * i / n_phi for i in range(n_phi)]
        self.cache_path = cache_path
        self.uav_ids = {uav.id for uav in uavs}

        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self.vertices = self._create_vertices()
            self.vertex_to_idx = self._create_vertex_index()
            self.position_cache = {}
            self.edges = self._create_edges()
            self.adjacency_matrix = self._create_adjacency_matrix()
            if cache_path:
                self._save_to_cache(cache_path)

    def _create_vertices(self) -> Dict:
        vertices = {'UAVs': {}, 'Targets': {}}
        for uav in self.uavs:
            vertices['UAVs'][uav.id] = [(-uav.id, phi) for phi in [uav.heading]]
        for target in self.targets:
            vertices['Targets'][target.id] = [(target.id, phi) for phi in self.phi_set]
        return vertices

    def _create_vertex_index(self) -> Dict:
        vertex_to_idx = {}
        idx = 0
        for uav_id, uav_vertices in self.vertices['UAVs'].items():
            for vertex in uav_vertices:
                vertex_to_idx[vertex] = idx
                idx += 1
        for target_id, target_vertices in self.vertices['Targets'].items():
            for vertex in target_vertices:
                vertex_to_idx[vertex] = idx
                idx += 1
        return vertex_to_idx

    def _create_edges(self) -> List[Tuple]:
        edges = []
        for uav_id, uav_vertices in self.vertices['UAVs'].items():
            for uav_vertex in uav_vertices:
                for target_id, target_vertices in self.vertices['Targets'].items():
                    for target_vertex in target_vertices:
                        edges.append((uav_vertex, target_vertex))
        for target1_id, target1_vertices in self.vertices['Targets'].items():
            for target1_vertex in target1_vertices:
                for target2_id, target2_vertices in self.vertices['Targets'].items():
                    if target1_id != target2_id:
                        for target2_vertex in target2_vertices:
                            edges.append((target1_vertex, target2_vertex))
        return edges

    def _create_adjacency_matrix(self) -> np.ndarray:
        n_vertices = len(self.vertex_to_idx)
        adj_matrix = np.full((n_vertices, n_vertices), np.inf)
        np.fill_diagonal(adj_matrix, 0)
        print("开始计算邻接矩阵...")
        for start_vertex, end_vertex in self.edges:
            try:
                start_idx = self.vertex_to_idx[start_vertex]
                end_idx = self.vertex_to_idx[end_vertex]
                adj_matrix[start_idx, end_idx] = self._calculate_path_length(start_vertex, end_vertex)
            except KeyError as e:
                print(f"警告: 顶点 {e} 未在索引中找到。")
                continue
        print("邻接矩阵计算完成。")
        return adj_matrix

    def _get_position(self, vertex: Tuple) -> np.ndarray:
        if vertex in self.position_cache:
            return self.position_cache[vertex]
        vertex_id = vertex[0]
        if vertex_id < 0:
            uav_id = -vertex_id
            uav = next(uav for uav in self.uavs if uav.id == uav_id)
            pos = uav.position
        else:
            target = next(target for target in self.targets if target.id == vertex_id)
            pos = target.position
        self.position_cache[vertex] = pos
        return pos

    def _calculate_path_length(self, vertex1: Tuple, vertex2: Tuple) -> float:
        start_pos = self._get_position(vertex1)
        end_pos = self._get_position(vertex2)
        start_heading = vertex1[1]
        end_heading = vertex2[1]
        distance = np.sqrt(np.sum((start_pos - end_pos) ** 2))
        heading_diff = abs(start_heading - end_heading)
        heading_factor = 1.0 + 0.2 * min(heading_diff, 2 * np.pi - heading_diff)
        return distance * heading_factor

    def _save_to_cache(self, path):
        data = {
            'vertices': self.vertices, 'vertex_to_idx': self.vertex_to_idx,
            'edges': self.edges, 'adjacency_matrix': self.adjacency_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"图数据已缓存到 {path}")

    def _load_from_cache(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.vertices = data['vertices']
            self.vertex_to_idx = data['vertex_to_idx']
            self.edges = data['edges']
            self.adjacency_matrix = data['adjacency_matrix']
            self.position_cache = {}
            print(f"已从缓存加载图数据: {path}")
        except Exception as e:
            print(f"加载缓存失败: {e}，重新计算图数据")
            self.__init__(self.uavs, self.targets, self.n_phi, None)


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class UAVTaskEnv:
    def __init__(self, uavs, targets, graph, load_balance_penalty=0.1):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.load_balance_penalty = load_balance_penalty
        self.reset()

    def reset(self):
        for uav in self.uavs:
            uav.reset()
        for target in self.targets:
            target.allocated_uavs = []
            target.arrival_times = []
            target.is_assigned = False
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        state = []
        for target in self.targets:
            state.extend(target.position)
            state.extend(target.resources)
            state.append(1 if target.is_assigned else 0)
        for uav in self.uavs:
            state.extend(uav.current_position)
            state.extend(uav.resources)
            state.append(uav.heading)
            state.append(uav.current_distance)
            state.append(uav.max_distance)
        state.append(len(self.targets))
        state.append(len(self.uavs))
        return np.array(state, dtype=np.float32)

    def step(self, action):
        target_id, uav_id, phi_idx = action
        target = next(t for t in self.targets if t.id == target_id)
        uav = next(u for u in self.uavs if u.id == uav_id)

        collaborators = [a[0] for a in target.allocated_uavs]
        total_collaborators = len(collaborators) + (1 if uav_id not in collaborators else 0)
        if total_collaborators == 0: total_collaborators = 1
        resource_cost = target.resources / total_collaborators

        if np.any(uav.resources < resource_cost):
            return self.state, -10, False, {}

        uav.resources -= np.ceil(resource_cost).astype(np.int32)
        if uav_id not in collaborators:
            target.allocated_uavs.append((uav_id, phi_idx))

        if not uav.task_sequence:
            start_vertex = (-uav.id, uav.heading)
        else:
            last_target_id, last_phi_idx = uav.task_sequence[-1]
            start_vertex = (last_target_id, self.graph.phi_set[last_phi_idx])

        end_heading = self.graph.phi_set[phi_idx]
        end_vertex = (target.id, end_heading)

        try:
            start_idx = self.graph.vertex_to_idx[start_vertex]
            end_idx = self.graph.vertex_to_idx[end_vertex]
            path_length = self.graph.adjacency_matrix[start_idx, end_idx]
            if np.isnan(path_length) or np.isinf(path_length):
                path_length = 100
        except (KeyError, IndexError):
            start_pos = self.graph._get_position(start_vertex)
            end_pos = self.graph._get_position(end_vertex)
            path_length = euclidean(start_pos, end_pos)

        uav.task_sequence.append((target_id, phi_idx))
        uav.current_distance += path_length
        target.is_assigned = True

        step_bonus = 20
        base_reward = -path_length + step_bonus
        task_counts = [len(u.task_sequence) for u in self.uavs]
        imbalance_penalty = np.var(task_counts) * self.load_balance_penalty
        done = all(t.is_assigned for t in self.targets)
        completion_bonus = 100 if done else 0
        reward = base_reward - imbalance_penalty + completion_bonus
        reward = max(-1000, min(reward, 1000))

        self.state = self._get_state()
        return self.state, reward, done, {}


class GraphRLSolver:
    def __init__(self, uavs, targets, graph, input_dim, hidden_dim, output_dim,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.05, batch_size=64,
                 memory_size=10000, load_balance_penalty=0.1):
        self.graph = graph
        self.env = UAVTaskEnv(uavs, targets, graph, load_balance_penalty)
        self.model = GNN(input_dim, hidden_dim, output_dim)
        self.target_model = GNN(input_dim, hidden_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = 10
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.train_history = {
            'episode_rewards': [], 'episode_steps': [],
            'mean_q_values': [], 'losses': []
        }

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for _, _, r, _, _ in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([d for _, _, _, _, d in minibatch])).to(self.device)
        actions = np.array([self._action_to_index(a) for _, a, _, _, _ in minibatch])
        actions_tensor = torch.LongTensor(actions).to(self.device)
        q_values = self.model(states)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in minibatch])).to(self.device)
        next_q_values = self.target_model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        selected_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(selected_q_values, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def _get_valid_action_mask(self, state) -> torch.Tensor:
        n_targets = len(self.env.targets)
        n_uavs = len(self.env.uavs)
        n_phi = self.graph.n_phi
        mask = torch.zeros(n_targets * n_uavs * n_phi, dtype=torch.bool)

        available_targets = [t for t in self.env.targets if not t.is_assigned]

        for target in available_targets:
            for uav in self.env.uavs:
                collaborators = [a[0] for a in target.allocated_uavs]
                total_collaborators = len(collaborators) + (1 if uav.id not in collaborators else 0)
                if total_collaborators == 0: total_collaborators = 1
                resource_cost = target.resources / total_collaborators
                if np.all(uav.resources >= resource_cost):
                    for phi_idx in range(n_phi):
                        action_idx = self._action_to_index((target.id, uav.id, phi_idx))
                        if 0 <= action_idx < len(mask):
                            mask[action_idx] = True
        return mask.to(self.device)

    def _action_to_index(self, action):
        target_id, uav_id, phi_idx = action
        return (target_id - 1) * (len(self.env.uavs) * self.graph.n_phi) + (uav_id - 1) * self.graph.n_phi + phi_idx

    def save_model(self, path='./saved_model.pth'):
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon, 'step_count': self.step_count
        }
        torch.save(model_state, path)
        print(f"模型已保存到 {path}")

    def load_model(self, path='./saved_model.pth', only_weights=False):
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not only_weights:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.step_count = checkpoint['step_count']
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.to(self.device)
            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False

    def train(self, episodes, use_cache=True, cache_path='./saved_model.pth',
              early_stopping_patience=10, save_best_only=True, log_interval=10, enable_plotting=True):
        start_time = time.time()
        best_reward = -float('inf')
        best_model_weights = None
        early_stop_counter = 0
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor=0.5, verbose=True)

        if use_cache and self.load_model(cache_path):
            print("继续之前的训练...")

        for episode in tqdm(range(episodes), desc=f"Training ({episodes} episodes)"):
            episode_start_time = time.time()
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_losses = []
            q_values = []

            while not done:
                action = None
                if random.random() < self.epsilon:
                    available_targets = [t for t in self.env.targets if not t.is_assigned]
                    if not available_targets:
                        done = True
                        continue
                    valid_actions = []
                    for target in available_targets:
                        for uav in self.env.uavs:
                            collaborators = [a[0] for a in target.allocated_uavs]
                            total_collaborators = len(collaborators) + (1 if uav.id not in collaborators else 0)
                            if total_collaborators == 0: total_collaborators = 1
                            resource_cost = target.resources / total_collaborators
                            if np.all(uav.resources >= resource_cost):
                                for phi_idx in range(self.graph.n_phi):
                                    valid_actions.append((target.id, uav.id, phi_idx))
                    if valid_actions:
                        action = random.choice(valid_actions)
                    else:
                        done = True
                        continue
                else:
                    with torch.no_grad():
                        valid_action_mask = self._get_valid_action_mask(state)
                        if not valid_action_mask.any():
                            done = True
                            continue
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        qs = self.model(state_tensor).squeeze(0)
                        qs[~valid_action_mask] = -float('inf')
                        q_values.append(qs.max().item())
                        action_idx = qs.argmax().item()
                        n_uavs = len(self.env.uavs)
                        n_phi = self.graph.n_phi
                        target_idx = action_idx // (n_uavs * n_phi)
                        uav_idx = (action_idx % (n_uavs * n_phi)) // n_phi
                        phi_idx = action_idx % n_phi
                        action = (self.env.targets[target_idx].id, self.env.uavs[uav_idx].id, phi_idx)

                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                if len(self.memory) > self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_losses.append(loss)

            self.train_history['episode_rewards'].append(total_reward)
            self.train_history['episode_steps'].append(steps)
            self.train_history['mean_q_values'].append(np.mean(q_values) if q_values else 0)
            self.train_history['losses'].append(np.mean(episode_losses) if episode_losses else 0)

            if (episode + 1) % log_interval == 0:
                tqdm.write(
                    f"Episode {episode + 1}/{episodes} | Reward: {total_reward:.2f} | "
                    f"Steps: {steps} | Epsilon: {self.epsilon:.3f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                    f"Loss: {np.mean(episode_losses) if episode_losses else 0:.4f} | "
                    f"Time: {time.time() - episode_start_time:.2f}s"
                )

            scheduler.step(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                best_model_weights = self.model.state_dict().copy()
                early_stop_counter = 0
                if save_best_only:
                    self.save_model(cache_path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    print(f"早停触发：在 {early_stopping_patience} 轮内奖励未提升。")
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            if save_best_only:
                print("已加载训练过程中的最佳模型。")
                self.save_model(cache_path)

        total_time = time.time() - start_time
        print(f"训练完成，总耗时: {total_time:.2f}s")
        if enable_plotting:
            print("正在生成训练历史图...")
            self._plot_training_history()
        return self.train_history

    def _plot_training_history(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.train_history['episode_rewards'])
        plt.title('累计奖励')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['episode_steps'])
        plt.title('每轮步数')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['mean_q_values'])
        plt.title('平均Q值')
        plt.xlabel('Episode')
        plt.ylabel('Q-value')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(self.train_history['losses'])
        plt.title('损失函数')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def get_task_assignments(self):
        task_assignments = {uav.id: [] for uav in self.env.uavs}
        for uav in self.env.uavs:
            for target_id, phi_idx in uav.task_sequence:
                task_assignments[uav.id].append((target_id, self.graph.phi_set[phi_idx]))
        return task_assignments


def main():
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([2, 1]), max_distance=6000),
        UAV(id=2, position=np.array([1500, 0]), heading=np.pi / 6, resources=np.array([0, 1]), max_distance=6000),
        UAV(id=3, position=np.array([3000, 0]), heading=3 * np.pi / 4, resources=np.array([3, 2]), max_distance=6000),
        UAV(id=4, position=np.array([2000, 2000]), heading=np.pi / 6, resources=np.array([2, 1]), max_distance=6000)
    ]
    targets = [
        Target(id=1, position=np.array([1500, 1500]), resources=np.array([4, 2]), value=100),
        Target(id=2, position=np.array([2000, 1000]), resources=np.array([3, 3]), value=90)
    ]

    graph = DirectedGraph(uavs, targets, cache_path=None)

    best_params = {
        'learning_rate': 0.0005, 'load_balance_penalty': 1.0,
        'epsilon_decay': 0.999, 'epsilon_min': 0.05
    }

    temp_env = UAVTaskEnv(uavs, targets, graph, load_balance_penalty=best_params['load_balance_penalty'])
    input_dim = len(temp_env.reset())
    output_dim = len(targets) * len(uavs) * graph.n_phi

    solver = GraphRLSolver(
        uavs=uavs, targets=targets, graph=graph,
        input_dim=input_dim, hidden_dim=128, output_dim=output_dim,
        **best_params
    )

    solver.train(episodes=500, use_cache=False, early_stopping_patience=50, enable_plotting=True)

    task_assignments = solver.get_task_assignments()
    print("\n任务分配结果:")
    for uav_id, tasks in task_assignments.items():
        print(f"无人机{uav_id}:")
        for target_id, heading in tasks:
            print(f"  - 目标{target_id}, 航向角: {np.degrees(heading):.2f}度")

    visualize_task_assignments(task_assignments, solver.env.uavs, solver.env.targets)
    print("-----over------")


if __name__ == "__main__":
    main()
