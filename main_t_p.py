# -*- coding: utf-8 -*-
import pro
from swarms_planning import UAV
from swarms_planning import DirectedGraph
from swarms_planning import Target
from swarms_planning import GeneticAlgorithm
from swarms_planning import PHCurve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from typing import List, Dict, Tuple, Set
import matplotlib.font_manager as fm

# self.position = np.array(position)  # 初始位置 [x, y]
# self.heading = heading  # 初始航向角（弧度）
# self.resources = np.array(resources)  # 携带资源 [类型1, 类型2, ...]
# self.initial_resources = np.array(resources)  # 初始资源
# self.max_distance = max_distance  # 最大飞行距离
# self.velocity = velocity  # 飞行速度
# self.task_sequence = []  # 任务序列 [(目标ID, 航向角), ...]
# self.current_distance = 0  # 已飞行距离
# self.current_position = np.array(position)  # 当前位置
# 求解多无人机协同任务分配问题
def solve_multi_uav_task_allocation():
    # 1. 初始化无人机和目标
    uavs = [
        UAV(1, np.array([0, 0]), 0, np.array([100, 50, 20]), 1000),
        UAV(2, np.array([50, 0]), np.pi / 4, np.array([80, 70, 30]), 1200),
        UAV(3, np.array([0, 50]), np.pi / 2, np.array([60, 90, 10]), 800)
    ]

    targets = [
        Target(1, np.array([200, 200]), np.array([50, 20, 10]), 100),
        Target(2, np.array([300, 100]), np.array([30, 40, 5]), 80),
        Target(3, np.array([150, 300]), np.array([70, 10, 15]), 120)
    ]

    # 2. 创建有向图模型
    graph = DirectedGraph(uavs, targets, n_phi=6)

    # 3. 可视化有向图（可选）
  #  visualize_directed_graph(graph)

    # 4. 创建遗传算法实例并求解
    ga = GeneticAlgorithm(
        uavs=uavs,
        targets=targets,
        graph=graph,
        population_size=100,
        max_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.1
    )

    # 5. 运行遗传算法
    best_fitness_history = []
    for generation in range(ga.max_generations):
        # 选择
        parents = ga.selection()

        # 交叉
        offspring = ga.crossover(parents)

        # 变异
        offspring = ga.mutation(offspring)

        # 替换当前种群
        ga.population = offspring

        # 记录最佳适应度
        current_best_fitness = max([ga.evaluate_fitness(chrom) for chrom in ga.population])
        best_fitness_history.append(current_best_fitness)

        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {current_best_fitness:.6f}")

    # 6. 获取最优解
    best_chromosome = ga.best_chromosome
    best_fitness = ga.best_fitness
    print(f"\nOptimization completed!")
    print(f"Best fitness: {best_fitness:.6f}")

    # 7. 解析最优解并输出结果
    task_assignments = parse_chromosome(best_chromosome, uavs, targets)
    visualize_task_assignments(uavs, targets, task_assignments,graph)

    # 8. 可视化适应度历史
    visualize_fitness_history(best_fitness_history)

    return task_assignments


def parse_chromosome(chromosome: np.ndarray, uavs: List[UAV], targets: List[Target]) -> Dict:
    """解析染色体，获取任务分配结果"""
    task_assignments = {uav.id: [] for uav in uavs}

    for i in range(chromosome.shape[1]):
        target_id = chromosome[0, i]
        uav_id = -chromosome[1, i] if chromosome[1, i] != 0 else None
        phi_idx = chromosome[2, i]

        if uav_id and uav_id in task_assignments:
            # 获取目标和航向角
            target = next(t for t in targets if t.id == target_id)
            heading = DirectedGraph(uavs, targets).phi_set[phi_idx]

            # 添加到任务分配结果
            task_assignments[uav_id].append((target_id, heading))

    # 按目标ID排序
    for uav_id in task_assignments:
        task_assignments[uav_id].sort(key=lambda x: x[0])

    return task_assignments


def visualize_task_assignments(uavs: List[UAV], targets: List[Target],
                               task_assignments: Dict, graph=None):
    """可视化任务分配结果"""
    # 获取所有可用字体
    #fonts = fm.findSystemFonts()
    # 筛选包含中文字符的字体
    #chinese_fonts = [f for f in fonts if 'hei' in f.lower() or 'song' in f.lower() or 'microsoftyahei' in f.lower()]

    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 使用ASCII减号替代Unicode减号
    plt.figure(figsize=(12, 10))
    # 绘制目标点（用编号标记）
    target_x = [target.position[0] for target in targets]
    target_y = [target.position[1] for target in targets]
    plt.scatter(target_x, target_y, c='red', s=120, marker='o', label='目标')

    # 为每个目标添加编号标签
    for i, target in enumerate(targets):
        plt.annotate(f'T{i}',  # 目标编号
                     xy=(target.position[0], target.position[1]),
                     xytext=(8, 8),  # 偏移量
                     textcoords='offset points',
                     fontsize=12,
                     color='red',
                     fontweight='bold')

        # 绘制无人机初始位置并显示初始航向角
    uav_x = [uav.position[0] for uav in uavs]
    uav_y = [uav.position[1] for uav in uavs]
    plt.scatter(uav_x, uav_y, c='blue', s=150, marker='^', label='无人机')

    for i, uav in enumerate(uavs):
        # 显示无人机编号
        plt.annotate(f'U{i}',
                     xy=(uav.position[0], uav.position[1]),
                     xytext=(8, 8),
                     textcoords='offset points',
                     fontsize=12,
                     color='blue',
                     fontweight='bold')

        # 显示起始位置
        plt.annotate(f'起始位置: ({uav.position[0]:.1f}, {uav.position[1]:.1f})',
                     xy=(uav.position[0], uav.position[1]),
                     xytext=(10, -20),
                     textcoords='offset points',
                     fontsize=9,
                     color='blue',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

        # 显示初始航向角（从heading属性获取，单位为角度）
        if hasattr(uav, 'heading'):
            plt.annotate(f'初始航向角: {uav.heading:.1f}°',
                         xy=(uav.position[0], uav.position[1]),
                         xytext=(10, -40),  # 调整位置避免重叠
                         textcoords='offset points',
                         fontsize=9,
                         color='blue',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

            # 绘制表示初始航向的箭头
            arrow_length = 1.0  # 箭头长度
            angle_rad = np.radians(uav.heading)
            dx = arrow_length * np.cos(angle_rad)
            dy = arrow_length * np.sin(angle_rad)
            plt.arrow(uav.position[0], uav.position[1], dx, dy,
                      head_width=0.3, head_length=0.4, fc='blue', ec='blue', alpha=0.7)

        # 绘制无人机路径和任务点航向角
        colors = plt.cm.rainbow(np.linspace(0, 1, len(uavs)))
        for uav_id, tasks in task_assignments.items():
            uav = next(u for u in uavs if u.id == uav_id)
            color = colors[uav_id - 1]

            # 从无人机初始位置开始
            path_x = [uav.position[0]]
            path_y = [uav.position[1]]

            # 添加任务目标位置
            for target_id, phi_idx in tasks:
                target = next(t for t in targets if t.id == target_id)
                path_x.append(target.position[0])
                path_y.append(target.position[1])

            # 绘制路径
            plt.plot(path_x, path_y, '-', color=color, linewidth=2.0,
                     label=f'UAV {uav_id}' if uav_id == 1 else "")

            # 在路径上标记任务顺序和航向角
            for j, (x, y) in enumerate(zip(path_x[1:], path_y[1:])):
                target_id, phi_idx = tasks[j]

                # 显示任务顺序和目标编号
                plt.annotate(f'{j+1}\n(T{target_id})',
                             xy=(x, y),
                             xytext=(-10, 10),
                             textcoords='offset points',
                             fontsize=9,
                             color='white',
                             fontweight='bold',
                             bbox=dict(boxstyle="circle,pad=0.2", fc=color, ec="none", alpha=0.8))

                # 显示任务点航向角信息（从phi_idx计算，单位为角度）
                angle_degrees = phi_idx * (360 / graph.n_phi) if graph else phi_idx
                plt.annotate(f'航向角: {angle_degrees:.1f}°',
                             xy=(x, y),
                             xytext=(15, -15),
                             textcoords='offset points',
                             fontsize=9,
                             color='black',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

                # 绘制表示航向的箭头
                arrow_length = 1.0
                angle_rad = np.radians(angle_degrees)
                dx = arrow_length * np.cos(angle_rad)
                dy = arrow_length * np.sin(angle_rad)
                plt.arrow(x, y, dx, dy,
                          head_width=0.3, head_length=0.4, fc=color, ec=color, alpha=0.7)

    # 添加图例和标题
    plt.legend(loc='upper right')
    plt.title('多无人机协同任务分配结果', fontsize=16)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_fitness_history(fitness_history: List[float]):
    """可视化适应度历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.title('遗传算法优化过程', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def visualize_directed_graph(graph: DirectedGraph, show_labels=True, color_nodes_by_type=True,
                             edge_width_by_distance=True, show_edge_labels=True):
    """
    可视化有向图

    参数:
    - show_labels: 是否显示节点和边的标签
    - color_nodes_by_type: 是否按类型为节点着色
    - edge_width_by_distance: 是否根据距离调整边的宽度
    - show_edge_labels: 是否显示边的标签(距离)
    """
    plt.figure(figsize=(12, 10))

    # 绘制节点
    node_positions = {}
    node_colors = {}
    node_sizes = {}

    # 收集无人机节点信息
    for uav in graph.uavs:
        node_id = f"U{uav.id}"
        node_positions[node_id] = uav.position
        node_colors[node_id] = 'blue' if color_nodes_by_type else 'gray'
        node_sizes[node_id] = 150 + uav.resources.sum() * 2  # 资源越多，节点越大

    # 收集目标节点信息
    for target in graph.targets:
        node_id = f"T{target.id}"
        node_positions[node_id] = target.position
        node_colors[node_id] = 'red' if color_nodes_by_type else 'gray'
        node_sizes[node_id] = 150 + target.value * 3  # 价值越高，节点越大

    # 绘制节点
    for node_id, pos in node_positions.items():
        plt.scatter(pos[0], pos[1], c=node_colors[node_id], s=node_sizes[node_id],
                    alpha=0.7, edgecolors='black', linewidths=1.5)

        # 添加节点标签
        if show_labels:
            plt.text(pos[0], pos[1], node_id, fontsize=12,
                     ha='center', va='center', color='white', fontweight='bold')

    # 绘制边
    max_distance = 0
    edge_distances = {}

    # 计算最大距离，用于边宽的归一化
    for uav_id, uav_vertices in graph.vertices['UAVs'].items():
        for uav_vertex in uav_vertices:
            for target_id, target_vertices in graph.vertices['Targets'].items():
                for target_vertex in target_vertices:
                    # 计算边的权重（路径长度）
                    vertices = graph._flatten_vertices()
                    start_idx = vertices.index(uav_vertex)
                    end_idx = vertices.index(target_vertex)
                    distance = graph.adjacency_matrix[start_idx, end_idx]

                    if distance < np.inf:  # 忽略不可达的边
                        edge_distances[(uav_id, target_id)] = distance
                        max_distance = max(max_distance, distance)

    # 绘制边
    for (uav_id, target_id), distance in edge_distances.items():
        start_pos = node_positions[f"U{uav_id}"]
        end_pos = node_positions[f"T{target_id}"]

        # 根据距离调整边的宽度
        if edge_width_by_distance:
            edge_width = 1.0 + 3.0 * (distance / max_distance)
        else:
            edge_width = 1.5

        # 绘制箭头
        plt.arrow(start_pos[0], start_pos[1],
                  end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
                  head_width=5, head_length=7, fc='gray', ec='gray',
                  alpha=0.6, width=edge_width * 0.05)

        # 添加边标签
        if show_edge_labels:
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            plt.text(mid_x, mid_y, f"{distance:.1f}", fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=10, label='uav'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='target'),
        Line2D([0], [1], color='gray', lw=2, label='path')
    ]

    plt.legend(handles=legend_elements, loc='upper right')

    # 设置标题和坐标轴
    plt.title('uavs task allocation directed graph', fontsize=16)
    plt.xlabel('X axis', fontsize=12)
    plt.ylabel('Y axis', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')  # 确保X和Y轴比例相同
    plt.tight_layout()
    plt.show()

# 运行求解函数
if __name__ == "__main__":
    task_assignments = solve_multi_uav_task_allocation()

    # 输出任务分配详情
    print("\n任务分配详情:")
    for uav_id, tasks in task_assignments.items():
        print(f"无人机 {uav_id}:")
        for i, (target_id, heading) in enumerate(tasks):
            print(f"  任务 {i+1}: 攻击目标 {target_id}, 到达航向角 {heading:.2f} 弧度")