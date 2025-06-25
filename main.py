# -*- coding: utf-8 -*-
import pro
#import swarms_planning
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from swarms_planning import UAV
from swarms_planning import DirectedGraph
from swarms_planning import Target

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


def configparas():
    uav1 = UAV(id=1,
                               position=np.array([0, 0]),  # 初始位置
                               heading=0,  # 初始航向角（弧度）
                               resources=np.array([100, 50, 20]),  # 携带资源 [弹药, 燃料, 侦察设备]
                               max_distance=1000  # 最大飞行距离
                               )
    uav1.report()

    uav2 = UAV(
        id=2,
        position=np.array([50, 0]),
        heading=np.pi / 4,
        resources=np.array([80, 70, 30]),
        max_distance=1200
    )

    uav3 = UAV(
        id=3,
        position=np.array([0, 50]),
        heading=np.pi / 2,
        resources=np.array([60, 90, 10]),
        max_distance=800
    )

    # 创建目标实例
    target1 = Target(
        id=1,
        position=np.array([200, 200]),  # 目标位置
        resources=np.array([50, 20, 10]),  # 所需资源 [攻击, 压制, 侦察]
        value=100  # 目标价值
    )

    target2 = Target(
        id=2,
        position=np.array([300, 100]),
        resources=np.array([30, 40, 5]),
        value=80
    )

    target3 = Target(
        id=3,
        position=np.array([150, 300]),
        resources=np.array([70, 10, 15]),
        value=120
    )

    # 收集所有无人机和目标
    uavs = [uav1, uav2, uav3]
    targets = [target1, target2, target3]

    # 创建有向图模型
    graph = DirectedGraph(uavs, targets)
    visualize_directed_graph(graph)  #可视化有向图





if __name__ == "__main__":

    pro1msg = pro.HelloClass( "测试类")
    #pro1msg.msg = "测试类1"
    pro1msg.outputmsg()

    configparas()



    print("---end----")



