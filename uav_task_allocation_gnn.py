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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载OpenMP库
# 图计算优化：  22:42
# 添加了图数据缓存机制，避免重复计算
# 使用多线程并行计算邻接矩阵
# 优化了 PH 曲线计算，使用简化但高效的模型
# 强化学习训练优化：
# 添加了 GPU 支持，加速神经网络计算
# 优化了状态表示和批处理操作
# 改进了经验回放机制，提高样本利用效率
# 添加了训练进度监控和时间统计
# 代码结构优化：
# 简化了一些复杂的计算逻辑
# 增加了提前终止条件，避免无效计算
# 改进了 GNN 网络结构，提高学习能力
# 内存管理优化：
# 使用更高效的数据结构和算法

# 设置中文字体，增强错误处理
def set_chinese_font(preferred_fonts=None):
    """设置matplotlib支持中文显示的字体"""
    if preferred_fonts is None:
        preferred_fonts = ["SimHei", "WenQuanYi Zen Hei", "Heiti TC", "Microsoft YaHei"]

    try:
        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]

        for font in preferred_fonts:
            if font in available_fonts:
                plt.rcParams["font.family"] = font
                return True
    except Exception:
        pass

    try:
        default_font = findfont(FontProperties(family=['sans-serif']))
        plt.rcParams["font.family"] = default_font
        print(f"警告: 未找到中文字体，将使用默认字体: {default_font}")
        return False
    except Exception:
        print("错误: 字体设置失败，中文可能无法正常显示")
        return False


def visualize_task_assignments(task_assignments, uavs, targets, show=True):
    """可视化多无人机任务分配结果"""
    font_set = set_chinese_font()

    plt.figure(figsize=(10, 8))

    # 绘制目标点
    target_positions = np.array([t.position for t in targets])
    plt.scatter(target_positions[:, 0], target_positions[:, 1], c='red', marker='o', s=100, label='目标')

    # 为每个目标添加标签
    for target in targets:
        plt.annotate(f"目标{target.id}",
                     (target.position[0], target.position[1]),
                     fontsize=12,
                     xytext=(5, 5),
                     textcoords='offset points')

    # 绘制无人机起点
    uav_positions = np.array([u.position for u in uavs])
    plt.scatter(uav_positions[:, 0], uav_positions[:, 1], c='blue', marker='s', s=100, label='无人机起点')

    # 为每个无人机添加标签
    for uav in uavs:
        plt.annotate(f"无人机{uav.id}",
                     (uav.position[0], uav.position[1]),
                     fontsize=12,
                     xytext=(5, 5),
                     textcoords='offset points')

    # 定义颜色列表
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 绘制每个无人机的任务路径
    for uav_id, tasks in task_assignments.items():
        uav = next(u for u in uavs if u.id == uav_id)
        uav_color = colors[uav_id % len(colors)]
        uav_start = uav.position

        # 绘制从起点到第一个目标的路径
        if tasks:
            first_target_id = tasks[0][0]
            first_target = next(t for t in targets if t.id == first_target_id)
            plt.plot([uav_start[0], first_target.position[0]],
                     [uav_start[1], first_target.position[1]],
                     c=uav_color, linestyle='-', linewidth=1.5, alpha=0.7)

        # 绘制目标之间的路径
        for i in range(len(tasks) - 1):
            current_target_id = tasks[i][0]
            next_target_id = tasks[i + 1][0]
            current_target = next(t for t in targets if t.id == current_target_id)
            next_target = next(t for t in targets if t.id == next_target_id)

            plt.plot([current_target.position[0], next_target.position[0]],
                     [current_target.position[1], next_target.position[1]],
                     c=uav_color, linestyle='-', linewidth=1.5, alpha=0.7)

        # 绘制无人机航向角
        for target_id, heading in tasks:
            target = next(t for t in targets if t.id == target_id)
            heading_rad = heading
            arrow_length = 2.0
            plt.arrow(target.position[0], target.position[1],
                      arrow_length * np.cos(heading_rad),
                      arrow_length * np.sin(heading_rad),
                      head_width=0.5, head_length=0.7, fc=uav_color, ec=uav_color, alpha=0.8)

    # 添加图例和标题
    plt.legend(fontsize=12)
    plt.title("GCN-DL多无人机任务分配结果", fontsize=16)
    plt.xlabel("X坐标", fontsize=14)
    plt.ylabel("Y坐标", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    if show:
        plt.show()

    return plt.gcf()


class UAV:
    """无人机类，存储无人机的属性和状态"""

    def __init__(self, id: int, position: np.ndarray, heading: float,
                 resources: np.ndarray, max_distance: float, velocity: float = 50):
        self.id = id
        self.position = np.array(position)  # 初始位置 [x, y]
        self.heading = heading  # 初始航向角（弧度）
        self.resources = np.array(resources)  # 携带资源 [类型1, 类型2, ...]
        self.initial_resources = np.array(resources)  # 初始资源
        self.max_distance = max_distance  # 最大飞行距离
        self.velocity = velocity  # 飞行速度
        self.task_sequence = []  # 任务序列 [(目标ID, 航向角), ...]
        self.current_distance = 0  # 已飞行距离
        self.current_position = np.array(position)  # 当前位置

    def reset(self):
        """重置无人机状态"""
        self.resources = self.initial_resources.copy()
        self.current_distance = 0
        self.current_position = self.position.copy()
        self.task_sequence = []

    def report(self):
        strinfo = "UAV Infomation is : \n"
        strinfo += "id:" + str(self.id) + "--初始航向角（弧度）:" + str(self.heading)
        print(strinfo)


class Target:
    """目标类，存储目标的属性"""

    def __init__(self, id: int, position: np.ndarray, resources: np.ndarray, value: float):
        self.id = id
        self.position = np.array(position)  # 位置 [x, y]
        self.resources = np.array(resources)  # 所需资源 [类型1, 类型2, ...]
        self.value = value  # 目标初始价值
        self.allocated_uavs = []  # 分配的无人机列表 [(无人机ID, 航向角), ...]
        self.arrival_times = []  # 无人机到达时间列表
        self.is_assigned = False  # 标记目标是否已分配

class DirectedGraph:
    """有向图模型，表示任务分配和路径规划"""

    def __init__(self, uavs: List[UAV], targets: List[Target], n_phi: int = 6, cache_path=None):
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi  # 航向角离散化数量
        self.phi_set = [2 * np.pi * i / n_phi for i in range(n_phi)]  # 离散化航向角集合
        self.cache_path = cache_path

        # 尝试从缓存加载
        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self.vertices = self._create_vertices()  # 图顶点
            self.vertex_to_idx = self._create_vertex_index()  # 顶点到索引的映射
            self.position_cache = {}  # 缓存顶点位置
            self.edges = self._create_edges()  # 图边
            self.adjacency_matrix = self._create_adjacency_matrix()  # 邻接矩阵

            # 保存到缓存
            if cache_path:
                self._save_to_cache(cache_path)

    def _create_vertices(self) -> Dict:
        """创建图的顶点"""
        vertices = {'UAVs': {}, 'Targets': {}}
        # 创建无人机顶点
        for uav in self.uavs:
            vertices['UAVs'][uav.id] = [(uav.id, phi) for phi in [uav.heading]]  # 初始航向角

        # 创建目标顶点
        for target in self.targets:
            vertices['Targets'][target.id] = [(target.id, phi) for phi in self.phi_set]

        return vertices

    def _create_vertex_index(self) -> Dict:
        """创建顶点到索引的映射，加速查找"""
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
        """创建图的边"""
        edges = []
        # 从无人机顶点到目标顶点的边
        for uav_id, uav_vertices in self.vertices['UAVs'].items():
            for uav_vertex in uav_vertices:
                for target_id, target_vertices in self.vertices['Targets'].items():
                    for target_vertex in target_vertices:
                        edges.append((uav_vertex, target_vertex))

        # 从目标顶点到目标顶点的边（表示无人机完成一个目标后到另一个目标）
        for target1_id, target1_vertices in self.vertices['Targets'].items():
            for target1_vertex in target1_vertices:
                for target2_id, target2_vertices in self.vertices['Targets'].items():
                    if target1_id != target2_id:
                        for target2_vertex in target2_vertices:
                            edges.append((target1_vertex, target2_vertex))

        return edges

    def _create_adjacency_matrix(self) -> np.ndarray:
        """创建邻接矩阵，存储边的权重（路径长度）"""
        n_vertices = self._count_vertices()
        adj_matrix = np.inf * np.ones((n_vertices, n_vertices))
        # 确保对角线元素为0（自身到自身的距离）
        np.fill_diagonal(adj_matrix, 0)

        # 使用多线程并行计算路径长度
        def compute_edge_weight(i, j, edge_i, edge_j):
            # 跳过对角线元素，已在上面设置为0
            if i == j:
                return
            if (edge_i, edge_j) in self.edges:
                adj_matrix[i, j] = self._calculate_path_length(edge_i, edge_j)

        vertices = self._flatten_vertices()
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, edge_i in enumerate(vertices):
                for j, edge_j in enumerate(vertices):
                    futures.append(executor.submit(compute_edge_weight, i, j, edge_i, edge_j))

            # 等待所有任务完成
            for future in futures:
                future.result()

        return adj_matrix

    def _flatten_vertices(self) -> List[Tuple]:
        """展平所有顶点"""
        vertices = []
        for uav_id, uav_vertices in self.vertices['UAVs'].items():
            vertices.extend(uav_vertices)
        for target_id, target_vertices in self.vertices['Targets'].items():
            vertices.extend(target_vertices)
        return vertices

    def _count_vertices(self) -> int:
        """计算顶点总数"""
        count = 0
        for uav_id, uav_vertices in self.vertices['UAVs'].items():
            count += len(uav_vertices)
        for target_id, target_vertices in self.vertices['Targets'].items():
            count += len(target_vertices)
        return count

    def _get_position(self, vertex: Tuple) -> np.ndarray:
        """获取顶点对应的位置，使用缓存机制避免重复计算"""
        if vertex in self.position_cache:
            return self.position_cache[vertex]

        if vertex[0] < 0:  # 无人机顶点
            uav_id = -vertex[0]
            uav = next(uav for uav in self.uavs if uav.id == uav_id)
            pos = uav.current_position
        else:
            # 目标顶点，获取目标位置
            target_id = vertex[0]
            target = next(target for target in self.targets if target.id == target_id)
            pos = target.position

        self.position_cache[vertex] = pos
        return pos

    def _calculate_path_length(self, vertex1: Tuple, vertex2: Tuple) -> float:
        """计算两点之间的PH曲线长度"""
        start_pos = self._get_position(vertex1)
        end_pos = self._get_position(vertex2)
        start_heading = vertex1[1]
        end_heading = vertex2[1]

        # 使用更精确的PH曲线计算
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # 考虑航向角的影响，添加修正因子
        heading_diff = abs(start_heading - end_heading)
        heading_factor = 1.0 + 0.2 * min(heading_diff, 2 * np.pi - heading_diff)

        return distance * heading_factor

    def _save_to_cache(self, path):
        """保存图数据到缓存文件"""
        data = {
            'vertices': self.vertices,
            'vertex_to_idx': self.vertex_to_idx,
            'edges': self.edges,
            'adjacency_matrix': self.adjacency_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"图数据已缓存到 {path}")

    def _load_from_cache(self, path):
        """从缓存文件加载图数据"""
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


class PHCurve:
    """PH曲线路径规划 - 高效实现"""

    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, start_heading: float, end_heading: float):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_heading = start_heading
        self.end_heading = end_heading
        self.length = self._calculate_length()

    def _calculate_length(self) -> float:
        """计算PH曲线长度（高效版本）"""
        dx = self.end_pos[0] - self.start_pos[0]
        dy = self.end_pos[1] - self.start_pos[1]
        straight_distance = np.sqrt(dx ** 2 + dy ** 2)

        # 航向角差异
        heading_diff = abs(self.start_heading - self.end_heading)
        heading_diff = min(heading_diff, 2 * np.pi - heading_diff)  # 取最小角度差

        # 使用简化模型：直线距离 + 转向惩罚
        turn_penalty = straight_distance * 0.2 * (heading_diff / np.pi)

        return straight_distance + turn_penalty

    def get_length(self) -> float:
        """获取路径长度"""
        return self.length


# 定义图神经网络（GNN） - 优化版本
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 添加额外层增加表达能力
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# 定义强化学习环境
class UAVTaskEnv:
    def __init__(self, uavs, targets, graph):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.reset()
        self.vertex_to_idx = self.graph._create_vertex_index()
        # 预计算目标ID到索引的映射
        self.target_id_to_idx = {target.id: i for i, target in enumerate(targets)}

    def reset(self):
        for uav in self.uavs:
            uav.reset()
        for target in self.targets:
            target.allocated_uavs = []
            target.arrival_times = []
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        # 优化：使用numpy数组预分配和直接赋值提高效率
        state = np.zeros(len(self.uavs) * (2 + 1 + len(self.uavs[0].resources)) +
                         len(self.targets) * (2 + len(self.targets[0].resources) + 1))

        offset = 0
        for uav in self.uavs:
            state[offset:offset + 2] = uav.position
            offset += 2
            state[offset] = uav.heading
            offset += 1
            state[offset:offset + len(uav.resources)] = uav.resources
            offset += len(uav.resources)

        for target in self.targets:
            state[offset:offset + 2] = target.position
            offset += 2
            state[offset:offset + len(target.resources)] = target.resources
            offset += len(target.resources)
            state[offset] = target.value
            offset += 1

        return state

    def step(self, action):
        # 解析动作
        target_id, uav_id, phi_idx = action
        target = next(t for t in self.targets if t.id == target_id)
        uav = next(u for u in self.uavs if u.id == uav_id)

        # 计算资源成本
        assigned_count = sum(1 for a in target.allocated_uavs if a[0] == uav_id)
        resource_cost = target.resources / (assigned_count + 1)

        # 检查资源约束
        if np.any(uav.resources < resource_cost):
            print(f"资源不足: UAV{uav_id}无法分配到目标{target_id}, 剩余资源: {uav.resources}, 需要: {resource_cost}")
            return self.state, -10, False, {}

        # 更新无人机和目标状态
        uav.resources -= np.array(resource_cost, dtype=np.int32)

        # 计算路径长度
        if not uav.task_sequence:
            start_vertex = (uav.id, uav.heading)
        else:
            last_target_id, last_phi_idx = uav.task_sequence[-1]
            start_vertex = (last_target_id, self.graph.phi_set[last_phi_idx])

        end_heading = self.graph.phi_set[phi_idx]
        end_vertex = (target.id, end_heading)

        # 使用邻接矩阵查找路径长度，添加异常处理
        try:
            start_idx = self.graph.vertex_to_idx[start_vertex]
            end_idx = self.graph.vertex_to_idx[end_vertex]
            path_length = self.graph.adjacency_matrix[start_idx, end_idx]

            # 特殊处理：同一顶点的路径长度应为0
            if start_vertex == end_vertex:
                path_length = 0
                if self.graph.adjacency_matrix[start_idx, end_idx] != 0:
                    print(f"警告: 邻接矩阵中顶点{start_vertex}到自身的距离不为0，已修正")
            else:
                path_length = self.graph.adjacency_matrix[start_idx, end_idx]

            # 检查路径长度有效性
            if np.isnan(path_length) or np.isinf(path_length):
                print(f"无效路径长度: {path_length}, 顶点: {start_vertex} → {end_vertex}")
                path_length = 100  # 设置默认值
        except (KeyError, IndexError) as e:
            print(f"顶点索引错误: {e}, 回退到欧氏距离计算")
            start_pos = self.graph._get_position(start_vertex)
            end_pos = self.graph._get_position(end_vertex)
            path_length = euclidean(start_pos, end_pos)

        # 更新无人机状态
        uav.task_sequence.append((target_id, phi_idx))
        target.allocated_uavs.append((uav_id, phi_idx))
        uav.current_distance += path_length

        # 标记目标为已分配
        target.is_assigned = True

        # 计算奖励（添加边界检查）
        base_reward = -path_length
        completion_bonus = 100 if all(t.is_assigned for t in self.targets) else 0
        reward = base_reward + completion_bonus

        # 限制奖励范围，防止数值不稳定
        reward = max(-1000, min(reward, 1000))

        done = all(t.is_assigned for t in self.targets)
        self.state = self._get_state()

        # 打印调试信息
        if done:
            print(f"任务完成! 总奖励: {reward}, 总路径长度: {uav.current_distance}")

        return self.state, reward, done, {}


    # 定义基于图强化学习的求解算法 - 优化版本
class GraphRLSolver:
    def __init__(self, uavs, targets, graph, input_dim, hidden_dim, output_dim,
                 learning_rate=0.001, gamma=0.99, epsilon=0.1, batch_size=64, memory_size=10000):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.env = UAVTaskEnv(uavs, targets, graph)
        self.model = GNN(input_dim, hidden_dim, output_dim)
        self.target_model = GNN(input_dim, hidden_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update_freq = 10
        self.step_count = 0

        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if np.random.rand() <= self.epsilon:
            target_id = random.choice([t.id for t in self.targets])
            uav_id = random.choice([u.id for u in self.uavs])
            phi_idx = random.randint(0, self.graph.n_phi - 1)
            return target_id, uav_id, phi_idx

        # 使用GPU加速推理
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action_index = torch.argmax(q_values).item()

        # 解析动作索引为实际动作
        target_id = action_index // (len(self.uavs) * self.graph.n_phi) + 1
        uav_id = (action_index % (len(self.uavs) * self.graph.n_phi)) // self.graph.n_phi + 1
        phi_idx = action_index % self.graph.n_phi
        return target_id, uav_id, phi_idx

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # 优化：使用随机采样的numpy数组提高效率
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]

        # 准备批次数据并移至GPU
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([r for _, _, r, _, _ in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([d for _, _, _, _, d in minibatch])).to(self.device)

        # 优化：预分配动作索引数组
        actions = np.array([self._action_to_index(a) for _, a, _, _, _ in minibatch])
        actions_tensor = torch.LongTensor(actions).to(self.device)

        # 计算目标Q值
        q_values = self.model(states)
        next_states = torch.FloatTensor(np.array([ns for _, _, _, ns, _ in minibatch])).to(self.device)
        next_q_values = self.target_model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]

        # 使用索引选择Q值进行更新
        selected_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 优化模型
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(selected_q_values, targets)
        loss.backward()
        self.optimizer.step()

        # 更新步数计数器
        self.step_count += 1
        # 定期更新目标网络
        if self.step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def _action_to_index(self, action):
        target_id, uav_id, phi_idx = action
        target_index = target_id - 1
        uav_index = uav_id - 1
        return target_index * (len(self.uavs) * self.graph.n_phi) + uav_index * self.graph.n_phi + phi_idx

    def save_model(self, path='./saved_model.pth'):
        """保存当前模型状态到指定路径"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        torch.save(model_state, path)
        print(f"模型已保存到 {path}")

    def load_model(self, path='./saved_model.pth', only_weights=False):
        """从指定路径加载模型状态"""
        if not os.path.exists(path):
            print(f"错误：模型文件 {path} 不存在")
            return False

        try:
            # 加载模型到指定设备
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not only_weights:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.step_count = checkpoint['step_count']

            # 将目标模型也移到相同设备
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.to(self.device)

            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False

    def train(self, episodes, use_cache=True, cache_path='./saved_model.pth',
              early_stopping_patience=10, save_best_only=True, log_interval=10):
        """训练模型，增加早停机制和学习率调度"""
        # 加载缓存模型
        if use_cache and self.load_model(cache_path):
            print(f"使用缓存模型 {cache_path}，跳过训练")
            return

        print(f"开始训练，共 {episodes} 轮")
        start_time = time.time()

        # 记录训练历史
        self.train_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'mean_q_values': [],
            'losses': []
        }

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 早停机制
        best_reward = -float('inf')
        early_stop_counter = 0
        best_model_weights = None

        for episode in range(episodes):
            episode_start_time = time.time()
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_losses = []
            q_values = []

            while not done:
                # ε-贪婪策略选择动作
                if random.random() < self.epsilon:
                    # 随机动作（探索）
                    available_targets = [t for t in self.targets if not t.is_assigned]

                    if not available_targets:
                        # 没有可用目标，可能所有目标都已分配
                        print("警告: 没有可用的未分配目标，探索阶段跳过")
                        # 或者可以选择结束当前episode
                        return self.model(state_tensor).argmax().item()  # 执行最优动作

                    target_id = random.choice([t.id for t in available_targets])
                    uav_id = random.choice([u.id for u in self.uavs])
                    phi_idx = random.randint(0, self.graph.n_phi - 1)
                    action = (target_id, uav_id, phi_idx)
                else:
                    # 基于Q值的最优动作（利用）
                    # 基于Q值的最优动作（利用）
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        qs = self.model(state_tensor)
                        q_values.append(qs.max().item())

                        # 从已有对象获取属性值
                        n_uavs = len(self.uavs)
                        n_phi = self.graph.n_phi

                        # 解析动作索引为具体的(目标ID, 无人机ID, 航向角)
                        action_idx = qs.argmax().item()
                        target_idx = action_idx // (n_uavs * n_phi)
                        uav_idx = (action_idx % (n_uavs * n_phi)) // n_phi
                        phi_idx = action_idx % n_phi
                        action = (target_idx + 1, uav_idx + 1, phi_idx)  # 注意+1的偏移处理

                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

                # 经验回放训练
                if len(self.memory) > self.batch_size:
                    loss = self.replay()
                    episode_losses.append(loss)

                    # 梯度裁剪（优化稳定性）
                    for param in self.model.parameters():
                        param.grad.data.clamp_(-1, 1)

                    self.optimizer.step()

            # 记录本轮训练数据
            self.train_history['episode_rewards'].append(total_reward)
            self.train_history['episode_steps'].append(steps)
            self.train_history['mean_q_values'].append(np.mean(q_values) if q_values else 0)
            self.train_history['losses'].append(np.mean(episode_losses) if episode_losses else 0)

            # 打印训练进度
            if (episode + 1) % log_interval == 0:
                elapsed = time.time() - episode_start_time
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Reward: {total_reward:.2f} | "
                      f"Steps: {steps} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Loss: {self.train_history['losses'][-1]:.4f} | "
                      f"Time: {elapsed:.2f}s")

            # 更新学习率
            scheduler.step(total_reward)

            # 早停检查
            if total_reward > best_reward:
                best_reward = total_reward
                best_model_weights = self.model.state_dict().copy()
                early_stop_counter = 0
                # 保存最佳模型
                if save_best_only:
                    self.save_model(cache_path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    print(f"早停触发：{early_stopping_patience}轮内奖励未提升")
                    break

            # 定期保存模型（除了最佳模型外）
            if not save_best_only and (episode + 1) % 10 == 0:
                self.save_model(f"{cache_path}.ep{episode + 1}")

            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 加载最佳模型权重
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            self.save_model(cache_path)

        total_time = time.time() - start_time
        print(f"训练完成，总耗时: {total_time:.2f}s")

        # 绘制训练曲线
        self._plot_training_history()

        return self.train_history

    def _plot_training_history(self):
        """绘制训练历史曲线"""
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
        task_assignments = {uav.id: [] for uav in self.uavs}
        for uav in self.uavs:
            for target_id, phi_idx in uav.task_sequence:
                task_assignments[uav.id].append((target_id, self.graph.phi_set[phi_idx]))
        return task_assignments


# 主函数
def main():
    # 定义无人机和目标
    uavs = [
        UAV(id=1, position=[0, 0], heading=0, resources=[100], max_distance=100),
        UAV(id=2, position=[10, 0], heading=0, resources=[100], max_distance=100)
    ]
    targets = [
        Target(id=1, position=[5, 5], resources=[20], value=10),
        Target(id=2, position=[8, 3], resources=[30], value=15),
        Target(id=3, position=[12, 7], resources=[25], value=12),
        Target(id=4, position=[3, 9], resources=[15], value=8),
        Target(id=5, position=[7, 12], resources=[22], value=11)
    ]

    # 创建图并使用缓存
    graph = DirectedGraph(uavs, targets, cache_path='./graph_cache.pkl')

    # 计算输入和输出维度
    input_dim = len(uavs) * (2 + 1 + len(uavs[0].resources)) + len(targets) * (2 + len(targets[0].resources) + 1)
    output_dim = len(targets) * len(uavs) * graph.n_phi

    # 创建求解器并训练
    solver = GraphRLSolver(uavs, targets, graph, input_dim, 128, output_dim)  # 增加隐藏层大小

    # 训练模型，是否使用缓存
    solver.train(episodes=100, use_cache=False)

    # 获取任务分配结果
    task_assignments = solver.get_task_assignments()
    print("\n任务分配结果:")
    for uav_id, tasks in task_assignments.items():
        print(f"无人机{uav_id}:")
        for target_id, heading in tasks:
            print(f"  - 目标{target_id}, 航向角: {np.degrees(heading):.2f}度")

    # 可视化任务分配结果
    visualize_task_assignments(task_assignments, uavs, targets)
    print("-----over------")


if __name__ == "__main__":
    main()