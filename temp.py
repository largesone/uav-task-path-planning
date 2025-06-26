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

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# “规划-验证”系统**。它能：
## 生成资源最优的规划。
# 对该规划进行精确的、物理可行的时间同步验证。
# 清晰地报告规划中存在的所有冲突点。
# 虽然它不能自动解决冲突，但它为您提供了做出最终决策所需的所有关键信息
# 在满足所有物理约束（速度上下限）的前提下，寻找一个能让整个无人机编队的总飞行速度与各自的经济速度偏差最小的同步方案
def set_chinese_font(preferred_fonts=None):
    """
    设置matplotlib支持中文显示的字体，增强了错误处理能力。
    """
    if preferred_fonts is None:
        preferred_fonts = ['Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

    try:
        # 优先使用用户指定的字体
        for font in preferred_fonts:
            if findfont(FontProperties(family=font)):
                plt.rcParams["font.family"] = font
                plt.rcParams['axes.unicode_minus'] = False
                return True
    except Exception:
        pass

    try:
        # 如果首选字体不可用，则回退到系统默认的无衬线字体
        default_font = findfont(FontProperties(family=['sans-serif']))
        if default_font:
            plt.rcParams["font.family"] = default_font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"警告: 未找到任何首选字体，将使用系统默认字体: {os.path.basename(default_font)}")
            return True
    except Exception as e:
        print(f"错误: 字体设置失败，中文可能无法正常显示。错误信息: {e}")

    return False


def visualize_task_assignments(final_plan, uavs, targets, show=True):
    """
    【最终版】可视化函数，增加对“同步可行性”的标注。
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
        ax.annotate(f"无人机 {uav.id}\n速度: {uav.velocity_range[0]}-{uav.velocity_range[1]} m/s",
                    (uav.position[0], uav.position[1]),
                    fontsize=9, xytext=(8, -18), textcoords='offset points', ha='left')

    colors = plt.cm.jet(np.linspace(0, 1, len(uavs)))

    for uav_id, tasks in final_plan.items():
        uav_color = colors[uav_id - 1]
        for i, task_info in enumerate(tasks):
            target = next(t for t in targets if t.id == task_info['target_id'])
            start_pos = task_info['start_pos']

            # 如果同步不可行，使用虚线表示，增加视觉区分度
            linestyle = '-' if task_info['is_sync_feasible'] else '--'
            linewidth = 1.5 if task_info['is_sync_feasible'] else 2.0

            ax.plot([start_pos[0], target.position[0]], [start_pos[1], target.position[1]],
                    c=uav_color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
            mid_point = (start_pos + target.position) / 2
            ax.text(mid_point[0], mid_point[1], str(task_info['step']), c='white', backgroundcolor=uav_color,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.2', fc=uav_color, ec='none'))

    report_text = "---------- 任务执行报告 ----------\n\n"
    for uav in uavs:
        report_text += f"■ 无人机 {uav.id} (经济/最低/最高速度: {uav.economic_speed}/{uav.velocity_range[0]}/{uav.velocity_range[1]} m/s)\n"
        details = sorted(final_plan.get(uav.id, []), key=lambda x: x['step'])
        if not details:
            report_text += "  - 未分配任何任务\n"
        else:
            for detail in details:
                sync_status = "" if detail['is_sync_feasible'] else " (警告: 无法满足同步)"
                report_text += f"  {detail['step']}. 飞向目标 {detail['target_id']}{sync_status}:\n"
                report_text += f"     飞行速度: {detail['speed']:.2f} m/s\n"
                report_text += f"     预计到达时间点: {detail['arrival_time']:.2f} s\n"
        report_text += "\n"

    fig.text(0.76, 0.92, report_text, transform=plt.gcf().transFigure, ha="left", va="top", fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.8))

    ax.set_title("GCN-DL多无人机任务分配结果 (含经济速度与同步约束验证)", fontsize=16)
    ax.set_xlabel("X坐标 (m)", fontsize=14)
    ax.set_ylabel("Y坐标 (m)", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    plt.subplots_adjust(right=0.75)
    if show: plt.show()
    return fig


class UAV:
    """无人机类，存储无人机的属性和状态"""

    def __init__(self, id: int, position: np.ndarray, heading: float, resources: np.ndarray, max_distance: float,
                 velocity_range: Tuple[float, float], economic_speed: float):
        self.id = id
        self.position = np.array(position)
        self.heading = heading
        self.resources = np.array(resources)
        self.initial_resources = np.array(resources)
        self.max_distance = max_distance
        self.velocity_range = velocity_range
        # 新增: 经济飞行速度
        assert velocity_range[0] <= economic_speed <= velocity_range[1], "经济速度必须在速度范围内"
        self.economic_speed = economic_speed

        self.task_sequence = []
        self.current_distance = 0
        self.current_position = np.array(position)

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

        if cache_path and os.path.exists(cache_path) and self._load_from_cache(cache_path):
            # 如果从缓存加载成功，则初始化完成
            pass
        else:
            # 如果缓存不存在或加载失败，则重新计算所有属性
            print("正在重新计算图属性...")
            for uav in self.uavs:
                uav.heading = self._snap_to_phi_set(uav.heading)

            # ---【已修正】---
            # 将并行的元组赋值拆分为串行赋值，确保正确的初始化顺序

            # 1. 先创建 vertices
            self.vertices = self._create_vertices()
            # 2. 再创建依赖 vertices 的 vertex_to_idx
            self.vertex_to_idx = self._create_vertex_index()
            # 3. 然后创建 edges
            self.edges = self._create_edges()
            # 4. 最后创建依赖 edges 的 adjacency_matrix
            self.adjacency_matrix = self._create_adjacency_matrix()

            # 如果指定了路径，则保存新生成的图到缓存
            if cache_path:
                self._save_to_cache(cache_path)

    def _snap_to_phi_set(self, heading: float) -> float:
        diffs = np.abs(np.array(self.phi_set) - heading)
        wrapped_diffs = np.minimum(diffs, 2 * np.pi - diffs)
        return self.phi_set[np.argmin(wrapped_diffs)]

    def _create_vertices(self) -> Dict:
        vertices = {'UAVs': {}, 'Targets': {}}
        for uav in self.uavs:
            vertices['UAVs'][uav.id] = [(-uav.id, None)]
        for target in self.targets:
            vertices['Targets'][target.id] = [(target.id, phi) for phi in self.phi_set]
        return vertices

    def _create_vertex_index(self) -> Dict:
        vertex_to_idx, idx = {}, 0
        # 虽然内部有hasattr检查，但从源头修正初始化顺序是最佳实践
        if not hasattr(self, 'vertices'):
            self.vertices = self._create_vertices()
        for v_list in (self.vertices['UAVs'].values(), self.vertices['Targets'].values()):
            for vs in v_list:
                for v in vs:
                    vertex_to_idx[v], idx = idx, idx + 1
        return vertex_to_idx

    def _create_edges(self) -> List[Tuple]:
        edges = []
        if not hasattr(self, 'vertices'):
            self.vertices = self._create_vertices()
        uav_vs = (v for vs in self.vertices['UAVs'].values() for v in vs)
        target_vs = [v for vs in self.vertices['Targets'].values() for v in vs]
        for uav_v in uav_vs:
            for target_v in target_vs:
                edges.append((uav_v, target_v))
        for t1_v in target_vs:
            for t2_v in target_vs:
                if t1_v[0] != t2_v[0]:
                    edges.append((t1_v, t2_v))
        return edges

    def _create_adjacency_matrix(self) -> np.ndarray:
        n = len(self.vertex_to_idx)
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0)
        print("开始计算邻接矩阵...")
        # 此处 self.edges 现在可以安全地被访问
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
        p1, p2 = self._get_position(v1), self._get_position(v2)
        h1, h2 = v1[1], v2[1]
        if h1 is None:
            h1 = next(u.heading for u in self.uavs if u.id == -v1[0])
        dist = np.linalg.norm(p2 - p1)
        if h1 is not None and h2 is not None:
            h_diff = abs(h1 - h2)
            h_factor = 1.0 + 0.2 * min(h_diff, 2 * np.pi - h_diff)
            return dist * h_factor
        return dist

    def _save_to_cache(self, path):
        data = {
            'vertices': self.vertices,
            'vertex_to_idx': self.vertex_to_idx,
            'edges': self.edges,
            'adjacency_matrix': self.adjacency_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"图数据已缓存到 {path}")

    def _load_from_cache(self, path) -> bool:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.vertices = data['vertices']
            self.vertex_to_idx = data['vertex_to_idx']
            self.edges = data['edges']
            self.adjacency_matrix = data['adjacency_matrix']
            print(f"已从缓存加载图数据: {path}")
            return True
        except (Exception, KeyError) as e:
            print(f"加载缓存失败或缓存文件已过期: {e}")
            return False
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
        if target is None or uav is None: return self._get_state(), -100, True, {}
        actual_contribution = np.minimum(target.remaining_resources, uav.resources)
        if np.all(actual_contribution <= 0): return self._get_state(), -20, False, {}
        collaborators = {a[0] for a in target.allocated_uavs}
        is_joining_alliance = len(collaborators) > 0 and uav_id not in collaborators
        uav.resources -= actual_contribution;
        target.remaining_resources -= actual_contribution
        if uav_id not in collaborators: target.allocated_uavs.append((uav_id, phi_idx))
        start_v = (-uav.id, None) if not uav.task_sequence else (uav.task_sequence[-1][0],
                                                                 self.graph.phi_set[uav.task_sequence[-1][1]])
        end_v = (target.id, self.graph.phi_set[phi_idx])
        path_len = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], self.graph.vertex_to_idx[end_v]]
        travel_time = path_len / uav.velocity_range[1]
        reward = -travel_time
        uav.task_sequence.append((target_id, phi_idx));
        uav.current_distance += path_len;
        uav.current_position = target.position
        if np.all(target.remaining_resources <= 0): reward += 50
        if is_joining_alliance: reward += self.alliance_bonus
        imbalance_penalty = np.var([len(u.task_sequence) for u in self.uavs]) * self.load_balance_penalty
        done = all(np.all(t.remaining_resources <= 0) for t in self.targets)
        completion_bonus = 500 if done else 0
        final_reward = reward - imbalance_penalty + completion_bonus
        return self._get_state(), final_reward, done, {}


class GNN(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim):
        super(GNN, self).__init__()
        self.l = nn.Sequential(nn.Linear(i_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU(),
                               nn.Linear(h_dim, o_dim))

    def forward(self, x): return self.l(x)


class GraphRLSolver:
    """基于图强化学习的求解器"""

    def __init__(self, uavs, targets, graph, i_dim, h_dim, o_dim, **params):
        self.graph, self.env = graph, UAVTaskEnv(uavs, targets, graph, params.get('load_balance_penalty', 0.1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        print(f"使用设备: {self.device}")
        self.model = GNN(i_dim, h_dim, o_dim).to(self.device);
        self.target_model = GNN(i_dim, h_dim, o_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict());
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.get('learning_rate', 0.001))
        self.gamma, self.epsilon, self.epsilon_decay, self.epsilon_min = params.get('gamma', 0.99), 1.0, params.get(
            'epsilon_decay', 0.999), params.get('epsilon_min', 0.05)
        self.batch_size, self.memory = params.get('batch_size', 64), deque(maxlen=params.get('memory_size', 10000))
        self.target_update_freq, self.step_count, self.train_history = 10, 0, defaultdict(list)

    def _action_to_index(self, a: Tuple) -> int:
        t_id, u_id, p_idx = a;
        t_idx, u_idx = t_id - 1, u_id - 1
        return t_idx * (len(self.env.uavs) * self.graph.n_phi) + u_idx * self.graph.n_phi + p_idx

    def _index_to_action(self, i: int) -> Tuple:
        n_u, n_p = len(self.env.uavs), self.graph.n_phi
        t_idx, u_idx, p_idx = i // (n_u * n_p), (i % (n_u * n_p)) // n_p, i % n_p
        return (self.env.targets[t_idx].id, self.env.uavs[u_idx].id, p_idx)

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self):
        if len(self.memory) < self.batch_size: return None
        s, a, r, ns, d = map(torch.tensor, zip(*[(s, self._action_to_index(a), r, ns, d) for s, a, r, ns, d in
                                                 random.sample(self.memory, self.batch_size)]))
        s, a, r, ns, d = s.float().to(self.device), a.long().to(self.device), r.float().to(self.device), ns.float().to(
            self.device), d.float().to(self.device)
        q_vals = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        targets = r + self.gamma * self.target_model(ns).max(1)[0] * (1 - d)
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
                if np.any((u.resources > 0) & (t.remaining_resources > 0)):
                    start_idx = t_idx * (n_u * n_p) + u_idx * n_p;
                    mask[start_idx: start_idx + n_p] = True
        return mask

    def train(self, episodes, **kwargs):
        p, log_int = kwargs.get('patience', 30), kwargs.get('log_interval', 10)
        start_t, best_r, early_stop_c = time.time(), -float('inf'), 0
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=10, factor=0.5)
        for ep in tqdm(range(episodes), desc=f"训练 ({episodes} episodes)"):
            state, done, total_r, steps, loss = self.env.reset(), False, 0, 0, None
            max_steps = len(self.env.targets) * len(self.env.uavs) * 3
            while not done and steps < max_steps:
                action_idx = -1;
                valid_mask = self._get_valid_action_mask()
                if not valid_mask.any(): break
                if random.random() < self.epsilon:
                    action_idx = random.choice(torch.where(valid_mask)[0]).item()
                else:
                    with torch.no_grad():
                        qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                        qs[~valid_mask] = -float('inf');
                        action_idx = qs.argmax().item()
                action = self._index_to_action(action_idx)
                next_s, r, done, _ = self.env.step(action);
                self.remember(state, action, r, next_s, done)
                state, total_r, steps = next_s, total_r + r, steps + 1;
                loss = self.replay()
            self.train_history['episode_rewards'].append(total_r);
            self.train_history['episode_steps'].append(steps)
            if loss is not None: self.train_history['losses'].append(loss)
            if (ep + 1) % log_int == 0: tqdm.write(
                f"轮次 {ep + 1}/{episodes} | 奖励: {total_r:.2f} | 步数: {steps} | Epsilon: {self.epsilon:.3f} | 损失: {loss or 0:.4f}")
            scheduler.step(total_r);
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if total_r > best_r:
                best_r, early_stop_c = total_r, 0; self.save_model(kwargs.get('cache_path', './saved_model.pth'))
            else:
                early_stop_c += 1
                if early_stop_c >= p: print(f"早停触发：在 {p} 轮内奖励未提升。"); break
        print(f"训练完成，总耗时: {time.time() - start_t:.2f}s")
        if kwargs.get('plot', True): self._plot_training_history()

    def evaluate(self, episodes=10):
        print("\n--- 开始评估最优模型 ---");
        self.model.eval();
        all_rewards, all_steps = [], []
        for _ in tqdm(range(episodes), desc="评估中"):
            state, done, ep_r, steps = self.env.reset(), False, 0, 0
            max_steps = len(self.env.targets) * len(self.env.uavs) * 3
            while not done and steps < max_steps:
                with torch.no_grad():
                    valid_mask = self._get_valid_action_mask()
                    if not valid_mask.any(): break
                    qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                    qs[~valid_mask] = -float('inf');
                    action_idx = qs.argmax().item()
                action = self._index_to_action(action_idx)
                state, r, done, _ = self.env.step(action);
                ep_r, steps = ep_r + r, steps + 1
            all_rewards.append(ep_r);
            all_steps.append(steps)
        self.model.train();
        avg_r, avg_s = np.mean(all_rewards), np.mean(all_steps)
        print(f"--- 评估完成 ---\n平均奖励: {avg_r:.2f} | 平均步数: {avg_s:.2f}\n");
        return avg_r, avg_s

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
        print("正在为最终可视化生成任务分配...");
        self.model.eval();
        state, done, steps = self.env.reset(), False, 0
        max_steps = len(self.env.targets) * len(self.env.uavs) * 3
        while not done and steps < max_steps:
            with torch.no_grad():
                valid_mask = self._get_valid_action_mask()
                if not valid_mask.any(): break
                qs = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze(0)
                qs[~valid_mask] = -float('inf');
                action_idx = qs.argmax().item()
            action = self._index_to_action(action_idx);
            state, _, done, _ = self.env.step(action);
            steps += 1
        self.model.train()
        assignments = {u.id: u.task_sequence for u in self.env.uavs}
        for uav_id, tasks in assignments.items(): assignments[uav_id] = [(t_id, p_idx) for t_id, p_idx in tasks]
        return assignments

    def save_model(self, path):
        torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict()}, path)

    def load_model(self, path):
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location=self.device);
            self.model.load_state_dict(ckpt['model'])
            self.target_model.load_state_dict(self.model.state_dict());
            self.optimizer.load_state_dict(ckpt['optim'])
            print(f"模型已从 {path} 加载");
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}"); return False


def calculate_economic_sync_speeds(task_assignments, uavs, targets, graph):
    """
    【v4.1 修正版】后处理函数，以经济速度为基准，在可行时间窗口内计算最优同步速度。
    """
    print("\n正在计算经济同步到达速度 (v4.1 修正逻辑)...")
    final_plan = defaultdict(list)

    # 初始化无人机状态：当前位置，何时空闲
    uav_status = {u.id: {'pos': u.position, 'free_at': 0.0} for u in uavs}

    # ---【已修正】---
    # 创建每个无人机待办任务的副本
    # 将 `u.id` 修正为正确的循环变量 `uav_id`
    remaining_tasks = {uav_id: list(tasks) for uav_id, tasks in task_assignments.items()}

    task_step_counter = defaultdict(lambda: 1)

    # 循环直到所有任务完成
    while any(remaining_tasks.values()):
        # 1. 找出所有无人机的下一个任务，并按目标分组
        next_target_groups = defaultdict(list)
        for uav_id, tasks in remaining_tasks.items():
            if tasks:
                next_target_groups[tasks[0][0]].append(uav_id)

        if not next_target_groups: break  # 没有更多可执行的任务了

        # 2. 对每个目标组，计算其最优同步到达时间
        group_arrival_times = []
        for target_id, uav_ids in next_target_groups.items():
            target = next(t for t in targets if t.id == target_id)

            # a. 计算每个无人机的时间窗口 [T_min, T_max] 和经济时间 T_econ
            time_windows = []
            for uav_id in uav_ids:
                uav = next(u for u in uavs if u.id == uav_id)
                start_pos = uav_status[uav_id]['pos']
                free_at = uav_status[uav_id]['free_at']
                distance = np.linalg.norm(target.position - start_pos)

                t_min = free_at + (distance / uav.velocity_range[1])
                t_max = free_at + (distance / uav.velocity_range[0])
                t_econ = free_at + (distance / uav.economic_speed)
                time_windows.append({'uav_id': uav_id, 't_min': t_min, 't_max': t_max, 't_econ': t_econ})

            # b. 寻找所有时间窗口的交集
            sync_start = max(tw['t_min'] for tw in time_windows)
            sync_end = min(tw['t_max'] for tw in time_windows)

            # c. 检查同步可行性
            if sync_start > sync_end + 1e-6:  # 交集不存在
                is_feasible = False
                # 即使不可行，也必须选择一个时间点继续模拟，这里选择最早能完成的时间
                final_sync_time = sync_start
            else:
                is_feasible = True
                # d. 在交集内寻找最优同步时间点（以经济时间的中位数为基准）
                econ_times = [tw['t_econ'] for tw in time_windows]
                preferred_sync_time = np.median(econ_times)
                final_sync_time = np.clip(preferred_sync_time, sync_start, sync_end)

            group_arrival_times.append({
                'target_id': target_id,
                'arrival_time': final_sync_time,
                'uav_ids': uav_ids,
                'is_feasible': is_feasible
            })

        # 3. 找到所有组中最早能完成的那个组
        next_event = min(group_arrival_times, key=lambda x: x['arrival_time'])

        # 4. 处理这个最早的事件
        target_to_process_id = next_event['target_id']
        uavs_to_process = next_event['uav_ids']
        final_arrival_time = next_event['arrival_time']
        is_group_sync_feasible = next_event['is_feasible']
        target_to_process = next(t for t in targets if t.id == target_to_process_id)

        for uav_id in uavs_to_process:
            uav = next(u for u in uavs if u.id == uav_id)

            # 计算并存储该段任务的详细信息
            start_pos = uav_status[uav_id]['pos']
            distance = np.linalg.norm(target_to_process.position - start_pos)
            travel_time = final_arrival_time - uav_status[uav_id]['free_at']

            required_speed = (distance / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
            optimized_speed = np.clip(required_speed, uav.velocity_range[0], uav.velocity_range[1])

            final_plan[uav_id].append({
                'target_id': target_to_process_id,
                'start_pos': start_pos,
                'speed': optimized_speed,
                'arrival_time': final_arrival_time,
                'step': task_step_counter[uav_id],
                'is_sync_feasible': is_group_sync_feasible
            })
            task_step_counter[uav_id] += 1

            # 更新无人机状态
            uav_status[uav_id]['pos'] = target_to_process.position
            uav_status[uav_id]['free_at'] = final_arrival_time

            # 从待办清单中移除已处理的任务
            remaining_tasks[uav_id].pop(0)

    return final_plan

def main():
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([80, 50]), max_distance=6000,
            velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([60, 90]), max_distance=6000,
            velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([100, 70]), max_distance=6000,
            velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([70, 80]), max_distance=6000,
            velocity_range=(50, 160), economic_speed=110)
    ]
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80)
    ]

    graph = DirectedGraph(uavs, targets, cache_path='./graph_cache_v2.pkl')

    params = {
        'learning_rate': 0.0005, 'load_balance_penalty': 1.0, 'epsilon_decay': 0.9995,
        'epsilon_min': 0.1, 'gamma': 0.98, 'batch_size': 128, 'memory_size': 20000
    }

    temp_env = UAVTaskEnv(uavs, targets, graph)
    i_dim, o_dim = len(temp_env.reset()), len(targets) * len(uavs) * graph.n_phi

    solver = GraphRLSolver(uavs, targets, graph, i_dim, 256, o_dim, **params)
    solver.train(episodes=100, patience=20, log_interval=5, plot=False)
    solver.load_model('./saved_model.pth')
    solver.evaluate(episodes=5)

    task_assignments = solver.get_task_assignments()

    final_plan = calculate_economic_sync_speeds(task_assignments, uavs, targets, graph)

    print("\n---------- 最终任务分配方案 (含经济速度优化与同步验证) ----------")
    for uav_id in sorted(final_plan.keys()):
        tasks = sorted(final_plan[uav_id], key=lambda x: x['step'])
        if tasks:
            uav = next(u for u in uavs if u.id == uav_id)
            print(f"无人机 {uav.id} (经济速度: {uav.economic_speed} m/s):")
            for task in tasks:
                sync_status = "" if task['is_sync_feasible'] else " (警告: 无法满足同步)"
                print(
                    f"  - 第 {task['step']} 步: 飞向目标 {task['target_id']}{sync_status}, 飞行速度: {task['speed']:.2f} m/s, 到达时间点: {task['arrival_time']:.2f}s")
        else:
            print(f"无人机 {uav_id}: 未分配任务")

    visualize_task_assignments(final_plan, uavs, targets)
    print("----- 运行结束 -----")


if __name__ == "__main__":
    main()