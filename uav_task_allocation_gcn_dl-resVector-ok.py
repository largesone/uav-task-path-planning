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
# 多无人机协同完成多目标打击任务，适应每个无人机资源小于目标资源需求的情况
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载OpenMP库


# 设置中文字体，增强错误处理
def set_chinese_font(preferred_fonts=None):
    """设置matplotlib支持中文显示的字体"""
    if preferred_fonts is None:
        preferred_fonts = ['Source Han Sans SC', 'Lucida Handwriting', 'SimHei', 'Microsoft YaHei', 'Microsoft YaHei',
                           'HYChaoCuHeiJ']

    try:
        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]
        for font in preferred_fonts:
            if font in available_fonts:
                plt.rcParams["font.family"] = font
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                return True
    except Exception:
        pass

    try:
        default_font = findfont(FontProperties(family=['sans-serif']))
        plt.rcParams["font.family"] = default_font
        plt.rcParams['axes.unicode_minus'] = False
        print(f"警告: 未找到首选字体，将使用默认字体: {default_font}")
        return False
    except Exception:
        print("错误: 字体设置失败，中文可能无法正常显示")
        return False


def visualize_task_assignments(task_assignments, uavs, targets, show=True):
    """
    功能增强版的可视化函数，清晰展示任务分配的详细信息。
    """
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 10))

    # 绘制目标点
    target_positions = np.array([t.position for t in targets])
    ax.scatter(target_positions[:, 0], target_positions[:, 1], c='red', marker='x', s=150, label='目标', zorder=5)
    for t in targets:
        res_str = np.array2string(t.resources, formatter={'float_kind': lambda x: "%.0f" % x})
        ax.annotate(f"目标 {t.id}\n需求: {res_str}", (t.position[0], t.position[1]),
                    fontsize=9, xytext=(8, -18), textcoords='offset points', ha='left')

    # 绘制无人机起点
    uav_positions = np.array([u.position for u in uavs])
    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], c='blue', marker='s', s=150, label='无人机起点', zorder=5)
    for uav in uavs:
        res_str = np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: "%.0f" % x})
        ax.annotate(f"无人机 {uav.id}\n初始资源: {res_str}", (uav.position[0], uav.position[1]),
                    fontsize=9, xytext=(8, -18), textcoords='offset points', ha='left')

    colors = plt.cm.jet(np.linspace(0, 1, len(uavs)))

    # 预计算每个目标的协作者
    target_collaborators = defaultdict(set)
    for uav_id, tasks in task_assignments.items():
        for target_id, _ in tasks:
            target_collaborators[target_id].add(uav_id)

    report_details = defaultdict(list)

    # 绘制每个无人机的任务路径
    for uav_id, tasks in task_assignments.items():
        uav = next(u for u in uavs if u.id == uav_id)
        uav_color = colors[uav_id - 1]
        current_pos = uav.position.copy()

        for i, (target_id, heading) in enumerate(tasks):
            target = next(t for t in targets if t.id == target_id)

            # 绘制路径
            ax.plot([current_pos[0], target.position[0]], [current_pos[1], target.position[1]],
                    c=uav_color, linestyle='-', linewidth=1.5, alpha=0.8)

            # 标注任务顺序
            mid_point = (current_pos + target.position) / 2
            ax.text(mid_point[0], mid_point[1], str(i + 1), c='white', backgroundcolor=uav_color,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.2', fc=uav_color, ec='none'))

            # 计算资源消耗（近似值，用于报告）
            collaborators_count = len(target_collaborators.get(target_id, {uav_id}))
            resource_cost_for_report = np.ceil(target.resources / collaborators_count).astype(np.int32)

            report_details[uav_id].append({"step": i + 1, "target_id": target_id, "cost": resource_cost_for_report})
            current_pos = target.position.copy()

    # 创建报告文本
    report_text = "---------- 任务执行报告 ----------\n\n"
    for uav in uavs:
        report_text += f"■ 无人机 {uav.id} (初始资源: {np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: '%.0f' % x})})\n"
        temp_resources = uav.initial_resources.copy()
        details = report_details.get(uav.id, [])
        if not details:
            report_text += "  - 未分配任何任务\n"
        else:
            # 注意：此处的资源消耗报告是一个近似值，实际消耗取决于动态的贡献过程
            for detail in details:
                cost_str = np.array2string(detail['cost'], formatter={'float_kind': lambda x: '%.0f' % x})
                temp_resources -= detail['cost']
                temp_resources[temp_resources < 0] = 0
                res_str = np.array2string(temp_resources, formatter={'float_kind': lambda x: '%.0f' % x})
                report_text += f"  {detail['step']}. 执行目标 {detail['target_id']}:\n"
                report_text += f"     消耗 (近似): {cost_str} -> 剩余: {res_str}\n"
        report_text += "\n"

    # 显示报告
    fig.text(0.76, 0.92, report_text, transform=plt.gcf().transFigure, ha="left", va="top", fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.8))

    # 美化图表
    ax.set_title("GCN-DL多无人机任务分配结果", fontsize=16)
    ax.set_xlabel("X坐标", fontsize=14)
    ax.set_ylabel("Y坐标", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    plt.subplots_adjust(right=0.75)
    if show:
        plt.show()
    return fig


class UAV:
    """无人机类，存储无人机的属性和状态"""

    def __init__(self, id: int, position: np.ndarray, heading: float, resources: np.ndarray, max_distance: float):
        self.id, self.position, self.heading = id, np.array(position), heading
        self.resources, self.initial_resources = np.array(resources), np.array(resources)
        self.max_distance = max_distance
        self.task_sequence, self.current_distance, self.current_position = [], 0, np.array(position)

    def reset(self):
        self.resources, self.current_distance = self.initial_resources.copy(), 0
        self.current_position, self.task_sequence = self.position.copy(), []


class Target:
    """目标类，存储目标的属性和状态"""

    def __init__(self, id: int, position: np.ndarray, resources: np.ndarray, value: float):
        self.id, self.position, self.resources, self.value = id, np.array(position), np.array(resources), value
        self.allocated_uavs, self.remaining_resources = [], np.array(resources)

    def reset(self):
        self.allocated_uavs, self.remaining_resources = [], self.resources.copy()


class DirectedGraph:
    """有向图模型，表示任务分配和路径规划"""

    def __init__(self, uavs: List[UAV], targets: List[Target], n_phi: int = 6, cache_path=None):
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi
        self.phi_set = [2 * np.pi * i / n_phi for i in range(n_phi)]
        self.cache_path = cache_path
        self.uav_ids = {uav.id for uav in uavs}
        self.position_cache = {}

        # 【AttributeError 最终修复】: 确保在加载缓存时，所有必要的实例变量都被正确初始化。
        if cache_path and os.path.exists(cache_path) and self._load_from_cache(cache_path):
            # 如果从缓存加载成功，则初始化完成
            pass
        else:
            # 如果缓存不存在或加载失败，则重新计算所有属性
            print("正在重新计算图属性...")
            # 强制对齐无人机航向角
            for uav in self.uavs:
                uav.heading = self._snap_to_phi_set(uav.heading)

            # 初始化核心属性
            self.vertices = self._create_vertices()
            self.vertex_to_idx = self._create_vertex_index()
            self.edges = self._create_edges()
            self.adjacency_matrix = self._create_adjacency_matrix()

            # 如果指定了路径，则保存新生成的图到缓存
            if cache_path:
                self._save_to_cache(cache_path)

    def _snap_to_phi_set(self, heading: float) -> float:
        """将给定的航向角对齐到离散集合phi_set中最近的一个值。"""
        diffs = np.abs(np.array(self.phi_set) - heading)
        wrapped_diffs = np.minimum(diffs, 2 * np.pi - diffs)
        closest_idx = np.argmin(wrapped_diffs)
        return self.phi_set[closest_idx]

    def _create_vertices(self) -> Dict:
        """创建图的顶点"""
        vertices = {'UAVs': {}, 'Targets': {}}
        for uav in self.uavs:
            vertices['UAVs'][uav.id] = [(-uav.id, None)]
        for target in self.targets:
            vertices['Targets'][target.id] = [(target.id, phi) for phi in self.phi_set]
        return vertices

    def _create_vertex_index(self) -> Dict:
        vertex_to_idx, idx = {}, 0
        # 确保 self.vertices 已经被初始化
        if not hasattr(self, 'vertices'):
            self.vertices = self._create_vertices()
        for vertices_list in (self.vertices['UAVs'].values(), self.vertices['Targets'].values()):
            for vertices in vertices_list:
                for vertex in vertices:
                    vertex_to_idx[vertex], idx = idx, idx + 1
        return vertex_to_idx

    def _create_edges(self) -> List[Tuple]:
        edges = []
        if not hasattr(self, 'vertices'):
            self.vertices = self._create_vertices()
        for uav_vertex in (v for vs in self.vertices['UAVs'].values() for v in vs):
            for target_vertex in (v for vs in self.vertices['Targets'].values() for v in vs):
                edges.append((uav_vertex, target_vertex))
        for t1_id, t1_vertices in self.vertices['Targets'].items():
            for t1_vertex in t1_vertices:
                for t2_id, t2_vertices in self.vertices['Targets'].items():
                    if t1_id != t2_id:
                        for t2_vertex in t2_vertices:
                            edges.append((t1_vertex, t2_vertex))
        return edges

    def _create_adjacency_matrix(self) -> np.ndarray:
        n = len(self.vertex_to_idx)
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0)
        print("开始计算邻接矩阵...")
        for start_v, end_v in self.edges:
            try:
                adj[self.vertex_to_idx[start_v], self.vertex_to_idx[end_v]] = self._calculate_path_length(start_v,
                                                                                                          end_v)
            except KeyError as e:
                print(f"警告: 顶点 {e} 未在索引中找到。")
        print("邻接矩阵计算完成。")
        return adj

    def _get_position(self, vertex: Tuple) -> np.ndarray:
        if vertex in self.position_cache:
            return self.position_cache[vertex]
        v_id, pos = vertex[0], None
        if v_id < 0:
            pos = next(u.position for u in self.uavs if u.id == -v_id)
        else:
            pos = next(t.position for t in self.targets if t.id == v_id)
        self.position_cache[vertex] = pos
        return pos

    def _calculate_path_length(self, v1: Tuple, v2: Tuple) -> float:
        """计算两点之间的PH曲线长度"""
        p1, p2 = self._get_position(v1), self._get_position(v2)
        h1, h2 = v1[1], v2[1]

        if h1 is None:
            uav_id = -v1[0]
            uav = next(u for u in self.uavs if u.id == uav_id)
            h1 = uav.heading

        dist = np.linalg.norm(p2 - p1)
        if h1 is not None and h2 is not None:
            h_diff = abs(h1 - h2)
            h_factor = 1.0 + 0.2 * min(h_diff, 2 * np.pi - h_diff)
            return dist * h_factor
        return dist

    def _save_to_cache(self, path):
        """保存需要缓存的属性到文件"""
        data_to_save = {
            'vertices': self.vertices,
            'vertex_to_idx': self.vertex_to_idx,
            'edges': self.edges,
            'adjacency_matrix': self.adjacency_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"图数据已缓存到 {path}")

    def _load_from_cache(self, path) -> bool:
        """从缓存加载属性，如果成功则返回True"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # 将加载的数据赋值给实例属性
            self.vertices = data['vertices']
            self.vertex_to_idx = data['vertex_to_idx']
            self.edges = data['edges']
            self.adjacency_matrix = data['adjacency_matrix']
            print(f"已从缓存加载图数据: {path}")
            return True
        except (Exception, KeyError) as e:
            print(f"加载缓存失败或缓存文件已过期: {e}")
            return False

class GNN(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim):
        super(GNN, self).__init__()
        self.l = nn.Sequential(nn.Linear(i_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU(),
                               nn.Linear(h_dim, o_dim))

    def forward(self, x): return self.l(x)


class UAVTaskEnv:
    """强化学习环境"""

    def __init__(self, uavs, targets, graph, load_balance_penalty=0.1, alliance_bonus=10.0):
        self.uavs, self.targets, self.graph = uavs, targets, graph
        self.load_balance_penalty, self.alliance_bonus = load_balance_penalty, alliance_bonus
        self.reset()

    def reset(self):
        for uav in self.uavs: uav.reset()
        for target in self.targets: target.reset()
        return self._get_state()

    def _get_state(self):
        state = []
        for t in self.targets: state.extend([*t.position, *t.remaining_resources, *t.resources])
        for u in self.uavs: state.extend([*u.current_position, *u.resources, u.heading, u.current_distance])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        target_id, uav_id, phi_idx = action
        target = next((t for t in self.targets if t.id == target_id), None)
        uav = next((u for u in self.uavs if u.id == uav_id), None)

        if target is None or uav is None: return self.state, -100, True, {}

        actual_contribution = np.minimum(target.remaining_resources, uav.resources)

        if np.all(actual_contribution <= 0):
            return self.state, -20, False, {}

        collaborators = {a[0] for a in target.allocated_uavs}
        is_joining_alliance = len(collaborators) > 0 and uav_id not in collaborators

        uav.resources -= actual_contribution
        target.remaining_resources -= actual_contribution

        if uav_id not in collaborators: target.allocated_uavs.append((uav_id, phi_idx))

        # 【KeyError 最终修复】: 查找路径时，使用与顶点创建时一致的键(None)。
        start_v = (-uav.id, None) if not uav.task_sequence else (uav.task_sequence[-1][0],
                                                                 self.graph.phi_set[uav.task_sequence[-1][1]])
        end_v = (target.id, self.graph.phi_set[phi_idx])
        path_len = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], self.graph.vertex_to_idx[end_v]]

        uav.task_sequence.append((target_id, phi_idx));
        uav.current_distance += path_len

        reward = -path_len
        if np.all(target.remaining_resources <= 0): reward += 50
        if is_joining_alliance: reward += self.alliance_bonus

        imbalance_penalty = np.var([len(u.task_sequence) for u in self.uavs]) * self.load_balance_penalty

        done = all(np.all(t.remaining_resources <= 0) for t in self.targets)
        completion_bonus = 500 if done else 0

        final_reward = reward - imbalance_penalty + completion_bonus
        return self._get_state(), final_reward, done, {}
class GraphRLSolver:
    """基于图强化学习的求解器"""

    def __init__(self, uavs, targets, graph, i_dim, h_dim, o_dim, **params):
        self.graph, self.env = graph, UAVTaskEnv(uavs, targets, graph, params.get('load_balance_penalty', 0.1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.model = GNN(i_dim, h_dim, o_dim).to(self.device)
        self.target_model = GNN(i_dim, h_dim, o_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict());
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=params.get('learning_rate', 0.001))
        self.gamma = params.get('gamma', 0.99)
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1.0, params.get('epsilon_decay', 0.999), params.get(
            'epsilon_min', 0.05)
        self.batch_size, self.memory = params.get('batch_size', 64), deque(maxlen=params.get('memory_size', 10000))
        self.target_update_freq, self.step_count = 10, 0
        self.train_history = defaultdict(list)

    def _action_to_index(self, action: Tuple) -> int:
        t_id, u_id, p_idx = action
        t_idx = t_id - 1
        u_idx = u_id - 1
        return t_idx * (len(self.env.uavs) * self.graph.n_phi) + u_idx * self.graph.n_phi + p_idx

    def _index_to_action(self, index: int) -> Tuple:
        n_u, n_p = len(self.env.uavs), self.graph.n_phi
        t_idx = index // (n_u * n_p)
        u_idx = (index % (n_u * n_p)) // n_p
        p_idx = index % n_p
        return (self.env.targets[t_idx].id, self.env.uavs[u_idx].id, p_idx)

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self):
        if len(self.memory) < self.batch_size: return None

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = map(torch.tensor, zip(*[(s, self._action_to_index(a), r, ns, d) for s, a, r, ns, d in batch]))
        s, a, r, ns, d = s.float().to(self.device), a.long().to(self.device), r.float().to(self.device), ns.float().to(
            self.device), d.float().to(self.device)

        q_vals = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_vals = self.target_model(ns).max(1)[0]
        targets = r + self.gamma * next_q_vals * (1 - d)

        loss = nn.MSELoss()(q_vals, targets)
        self.optimizer.zero_grad();
        loss.backward();
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0);
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0: self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def _get_valid_action_mask(self) -> torch.Tensor:
        n_t, n_u, n_p = len(self.env.targets), len(self.env.uavs), self.graph.n_phi
        mask = torch.zeros(n_t * n_u * n_p, dtype=torch.bool, device=self.device)

        for t_idx, t in enumerate(self.env.targets):
            if np.all(t.remaining_resources <= 0): continue

            collaborators = {a[0] for a in t.allocated_uavs}

            for u_idx, u in enumerate(self.env.uavs):
                if u.id in collaborators: continue

                # 【核心逻辑2】: 只要无人机能做出任何有效贡献，动作就有效
                can_contribute = np.any((u.resources > 0) & (t.remaining_resources > 0))

                if can_contribute:
                    start_idx = t_idx * (n_u * n_p) + u_idx * n_p
                    mask[start_idx: start_idx + n_p] = True
        return mask

    def train(self, episodes, **kwargs):
        patience = kwargs.get('early_stopping_patience', 30)
        log_interval = kwargs.get('log_interval', 10)

        start_time, best_reward, early_stop_counter = time.time(), -float('inf'), 0
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=10, factor=0.5)

        for ep in tqdm(range(episodes), desc=f"训练 ({episodes} episodes)"):
            state, done, total_reward, steps, loss = self.env.reset(), False, 0, 0, None

            max_steps = len(self.env.targets) * len(self.env.uavs) * 3  # 增加最大步数
            while not done and steps < max_steps:
                action_idx = -1
                valid_mask = self._get_valid_action_mask()
                if not valid_mask.any(): break

                if random.random() < self.epsilon:
                    action_idx = random.choice(torch.where(valid_mask)[0]).item()
                else:
                    with torch.no_grad():
                        qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                        qs[~valid_mask] = -float('inf')
                        action_idx = qs.argmax().item()

                action = self._index_to_action(action_idx)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state, total_reward, steps = next_state, total_reward + reward, steps + 1
                loss = self.replay()

            self.train_history['episode_rewards'].append(total_reward)
            self.train_history['episode_steps'].append(steps)
            if loss is not None: self.train_history['losses'].append(loss)

            if (ep + 1) % log_interval == 0:
                tqdm.write(
                    f"轮次 {ep + 1}/{episodes} | 奖励: {total_reward:.2f} | 步数: {steps} | Epsilon: {self.epsilon:.3f} | 损失: {loss or 0:.4f}")

            scheduler.step(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if total_reward > best_reward:
                best_reward, early_stop_counter = total_reward, 0
                self.save_model(kwargs.get('cache_path', './saved_model.pth'))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"早停触发：在 {patience} 轮内奖励未提升。")
                    break

        print(f"训练完成，总耗时: {time.time() - start_time:.2f}s")
        if kwargs.get('enable_plotting', True): self._plot_training_history()

    def evaluate(self, episodes=10):
        print("\n--- 开始评估最优模型 ---")
        self.model.eval()
        all_rewards, all_steps = [], []

        for _ in tqdm(range(episodes), desc="评估中"):
            state, done, ep_reward, steps = self.env.reset(), False, 0, 0
            max_steps = len(self.env.targets) * len(self.env.uavs) * 3
            while not done and steps < max_steps:
                with torch.no_grad():
                    valid_mask = self._get_valid_action_mask()
                    if not valid_mask.any(): break
                    qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                    qs[~valid_mask] = -float('inf')
                    action_idx = qs.argmax().item()
                action = self._index_to_action(action_idx)
                state, reward, done, _ = self.env.step(action)
                ep_reward, steps = ep_reward + reward, steps + 1
            all_rewards.append(ep_reward)
            all_steps.append(steps)

        self.model.train()
        avg_reward, avg_steps = np.mean(all_rewards), np.mean(all_steps)
        print(f"--- 评估完成 ---")
        print(f"平均奖励: {avg_reward:.2f} | 平均步数: {avg_steps:.2f}\n")
        return avg_reward, avg_steps

    def _plot_training_history(self):
        print("正在生成训练历史图...")
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].plot(self.train_history['episode_rewards']);
        axs[0].set_title('每轮奖励');
        axs[0].grid(True)
        axs[1].plot(self.train_history['episode_steps']);
        axs[1].set_title('每轮步数');
        axs[1].grid(True)
        axs[2].plot(self.train_history['losses']);
        axs[2].set_title('损失函数');
        axs[2].grid(True)
        plt.tight_layout();
        plt.savefig('training_history.png');
        plt.close()

    def get_task_assignments(self):
        print("正在为最终可视化生成任务分配...")
        self.model.eval()
        state, done, steps = self.env.reset(), False, 0
        max_steps = len(self.env.targets) * len(self.env.uavs) * 3
        while not done and steps < max_steps:
            with torch.no_grad():
                valid_mask = self._get_valid_action_mask()
                if not valid_mask.any(): break
                qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                qs[~valid_mask] = -float('inf')
                action_idx = qs.argmax().item()
            action = self._index_to_action(action_idx)
            state, _, done, _ = self.env.step(action)
            steps += 1
        self.model.train()

        assignments = {u.id: u.task_sequence for u in self.env.uavs}
        for uav_id, tasks in assignments.items():
            assignments[uav_id] = [(t_id, self.graph.phi_set[p_idx]) for t_id, p_idx in tasks]
        return assignments

    def save_model(self, path):
        torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict()}, path)

    def load_model(self, path):
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model']);
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer.load_state_dict(ckpt['optim'])
            print(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}"); return False


def main():
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([2, 1]), max_distance=6000),
        UAV(id=2, position=np.array([1500, 0]), heading=0.0, resources=np.array([0, 1]), max_distance=6000),
        UAV(id=3, position=np.array([3000, 0]), heading=3 * np.pi / 4, resources=np.array([3, 2]), max_distance=6000),
        UAV(id=4, position=np.array([2000, 2000]), heading=np.pi / 6, resources=np.array([2, 1]), max_distance=6000)
    ]
    targets = [
        Target(id=1, position=np.array([1500, 1500]), resources=np.array([4, 2]), value=100),
        Target(id=2, position=np.array([2000, 1000]), resources=np.array([3, 3]), value=90)
    ]

    graph = DirectedGraph(uavs, targets, cache_path='./graph_cache.pkl')

    params = {
        'learning_rate': 0.0005, 'load_balance_penalty': 1.0, 'epsilon_decay': 0.9995,
        'epsilon_min': 0.1, 'gamma': 0.98, 'batch_size': 128, 'memory_size': 20000
    }

    temp_env = UAVTaskEnv(uavs, targets, graph)
    i_dim, o_dim = len(temp_env.reset()), len(targets) * len(uavs) * graph.n_phi

    solver = GraphRLSolver(uavs, targets, graph, i_dim, 256, o_dim, **params)

    # 训练模型
    solver.train(episodes=1000, early_stopping_patience=50)

    # 评估最优模型
    solver.load_model('./saved_model.pth')
    solver.evaluate()

    # 获取并可视化最终结果
    task_assignments = solver.get_task_assignments()
    print("\n---------- 最终任务分配方案 ----------")
    for uav_id, tasks in task_assignments.items():
        if tasks:
            print(f"无人机 {uav_id}:")
            for t_id, heading in tasks: print(f"  - 目标 {t_id}, 航向角: {np.degrees(heading):.2f}°")
        else:
            print(f"无人机 {uav_id}: 未分配任务")

    visualize_task_assignments(task_assignments, uavs, targets)
    print("----- 运行结束 -----")


if __name__ == "__main__":
    main()
