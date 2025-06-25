# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import random
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
import sys

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


class DirectedGraph:
    """有向图模型，表示任务分配和路径规划"""

    def __init__(self, uavs: List[UAV], targets: List[Target], n_phi: int = 6):
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi  # 航向角离散化数量
        self.phi_set = [2 * np.pi * i / n_phi for i in range(n_phi)]  # 离散化航向角集合
        self.vertices = self._create_vertices()  # 图顶点
        self.edges = self._create_edges()  # 图边
        self.adjacency_matrix = self._create_adjacency_matrix()  # 邻接矩阵

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

        # 为每条边计算权重（PH曲线长度）
        for i, edge_i in enumerate(self._flatten_vertices()):
            for j, edge_j in enumerate(self._flatten_vertices()):
                if (edge_i, edge_j) in self.edges:
                    adj_matrix[i, j] = self._calculate_path_length(edge_i, edge_j)

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

    def _calculate_path_length(self, vertex1: Tuple, vertex2: Tuple) -> float:
        """计算两点之间的PH曲线长度"""
        if vertex1[0] < 0:  # 无人机顶点
            uav_id = -vertex1[0]
            uav = next(uav for uav in self.uavs if uav.id == uav_id)
            start_pos = uav.current_position
        else:
            # 目标顶点，获取目标位置
            target_id = vertex1[0]
            target = next(target for target in self.targets if target.id == target_id)
            start_pos = target.position

        if vertex2[0] < 0:  # 无人机顶点，这里可能是结束状态，暂时用初始位置
            end_pos = self.uavs[-vertex2[0]].position
        else:
            target_id = vertex2[0]
            target = next(target for target in self.targets if target.id == target_id)
            end_pos = target.position

        # 简化计算，使用欧氏距离作为路径长度
        return euclidean(start_pos, end_pos)

class PHCurve:
    """PH曲线路径规划"""

    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, start_heading: float, end_heading: float):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_heading = start_heading
        self.end_heading = end_heading

    def calculate_path(self, num_points: int = 50) -> np.ndarray:
        """计算PH曲线路径点"""
        # 简化实现，使用线性插值加航向角调整
        path = np.zeros((num_points, 2))
        for i in range(num_points):
            t = i / (num_points - 1)
            # 线性插值位置
            path[i] = self.start_pos + t * (self.end_pos - self.start_pos)
            # 平滑调整航向角
            current_heading = self.start_heading + t * (self.end_heading - self.start_heading)
            # 这里可以添加更复杂的PH曲线计算
        return path

    def get_length(self) -> float:
        """获取路径长度"""
        return euclidean(self.start_pos, self.end_pos)


class GeneticAlgorithm:
    """改进的遗传算法"""

    def __init__(self, uavs: List[UAV], targets: List[Target], graph: DirectedGraph,
                 population_size: int = 100, max_generations: int = 200,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        self.best_chromosome = None
        self.best_fitness = 0

    def _initialize_population(self) -> List[np.ndarray]:
    # 初始化种群，确保生成的染色体满足基本约束"""
        population = []

        for _ in range(self.population_size):
            # 为每个染色体初始化基因
            n_genes = len(self.targets) * 3  # 每个目标最多分配3架无人机
            chromosome = np.zeros((3, n_genes), dtype=int)

            # 跟踪每架无人机的剩余资源
            uav_resources = {uav.id: uav.initial_resources.copy() for uav in self.uavs}

            # 为每个目标分配无人机
            for target_idx, target in enumerate(self.targets):
                # 尝试为当前目标分配1-3架无人机
                n_assignments = random.randint(1, 3)

                for assignment in range(n_assignments):
                    gene_idx = target_idx * 3 + assignment

                    # 设置目标ID
                    chromosome[0, gene_idx] = target.id

                    # 选择能够提供足够资源的无人机
                    feasible_uavs = []
                    for uav in self.uavs:
                        # 计算该无人机需要提供的资源量
                        required_resources = target.resources / n_assignments
                        # 根据uav_resources的数据类型调整计算
                        if uav_resources[uav.id].dtype == np.int32:
                            # 整数类型：四舍五入资源消耗
                            required_resources = np.round(target.resources / n_assignments).astype(np.int32)
                        else:
                            # 浮点数类型
                            required_resources = target.resources / n_assignments

                        # 检查剩余资源是否足够
                        if np.all(uav_resources[uav.id] >= required_resources):
                            feasible_uavs.append(uav.id)

                    # 如果有可行的无人机，随机选择一个
                    if feasible_uavs:
                        uav_id = random.choice(feasible_uavs)
                        chromosome[1, gene_idx] = -uav_id  # 负号表示分配的无人机ID

                        # 更新剩余资源
                        #uav_resources[uav_id] -= np.array( target.resources / n_assignments, np.int32)
                        # 更新资源（保持浮点数精度）
                        uav_resources[uav_id] -= required_resources
                        # 随机分配航向角索引
                        chromosome[2, gene_idx] = random.randint(0, self.graph.n_phi - 1)

            population.append(chromosome)

        return population

    #改进初始种群生成（增加可行解概率）    策略基于约束的初始化
    def _initialize_population_1(self) -> List[np.ndarray]:
        population = []
        n_genes = sum(3 for _ in self.targets)

        for _ in range(self.population_size):
            chromosome = np.zeros((3, n_genes), dtype=int)
            target_indices = []
            for target in self.targets:
                target_indices.extend([target.id] * 3)
            chromosome[0] = target_indices

            # 关键改进：优先分配资源充足的无人机
            uav_resources_remaining = {uav.id: uav.initial_resources.copy() for uav in self.uavs}
            for i in range(n_genes):
                target_id = chromosome[0, i]
                target = next(t for t in self.targets if t.id == target_id)

                # 筛选资源充足的无人机
                feasible_uavs = [uav.id for uav in self.uavs
                                 if np.all(uav_resources_remaining[uav.id] >= target.resources / 3)]  # 假设每目标最多3架无人机
                if feasible_uavs:
                    uav_id = random.choice(feasible_uavs)
                    chromosome[1, i] = -uav_id
                    uav_resources_remaining[uav_id] -= np.array(target.resources / 3,dtype = np.int32)
                    chromosome[2, i] = random.randint(0, self.graph.n_phi - 1)
                else:
                    chromosome[1, i] = 0  # 无可行无人机，暂不分配

            population.append(chromosome)
        return population

    # def evaluate_fitness(self, chromosome: np.ndarray) -> float:
    #     """评估染色体适应度"""
    #     total_distance = 0
    #     max_distance = 0
    #     uav_resources = {uav.id: uav.initial_resources.copy() for uav in self.uavs}
    #     target_allocation = {target.id: np.zeros(len(target.resources)) for target in self.targets}
    #
    #     # 解析染色体，计算每架无人机的任务序列和路径长度
    #     for i in range(chromosome.shape[1]):
    #         target_id = chromosome[0, i]
    #         uav_id = -chromosome[1, i] if chromosome[1, i] != 0 else None
    #         phi_idx = chromosome[2, i]
    #
    #         if uav_id and uav_id in uav_resources:
    #             # 找到对应的目标和无人机
    #             target = next(t for t in self.targets if t.id == target_id)
    #             uav = next(u for u in self.uavs if u.id == uav_id)
    #
    #             # 计算路径长度
    #             start_vertex = (uav.id, uav.heading) if not uav.task_sequence else uav.task_sequence[-1][1]
    #             end_heading = self.graph.phi_set[phi_idx]
    #             end_vertex = (target.id, end_heading)
    #
    #             # 在邻接矩阵中找到路径长度
    #             vertices = self.graph._flatten_vertices()
    #             start_idx = vertices.index(start_vertex)
    #             end_idx = vertices.index(end_vertex)
    #             path_length = self.graph.adjacency_matrix[start_idx, end_idx]
    #
    #             uav.current_distance += path_length
    #             total_distance += path_length
    #             max_distance = max(max_distance, uav.current_distance)
    #
    #             # 检查资源约束
    #             resource_cost = self._calculate_resource_cost(uav, target)
    #             if np.any(uav_resources[uav.id] < resource_cost):
    #                 return 0  # 资源不足，适应度为0
    #
    #             uav_resources[uav.id] -= np.array(resource_cost, dtype=np.int32)
    #             target_allocation[target.id] += np.array(resource_cost, dtype=np.int32)
    #
    #     # 检查目标资源需求是否满足
    #     for target_id, required in target_allocation.items():
    #         target = next(t for t in self.targets if t.id == target_id)
    #         if np.any(required < target.resources):
    #             return 0  # 目标资源不足，适应度为0
    #
    #     # 检查燃油约束
    #     for uav in self.uavs:
    #         if uav.current_distance > uav.max_distance:
    #             return 0
    #
    #     # 计算适应度（目标函数的倒数）
    #     alpha = 0.5  # 平衡因子
    #     objective = total_distance + alpha * max_distance
    #     fitness = 1.0 / (objective + 1)  # 避免除以零
    #
    #     if fitness > self.best_fitness:
    #         self.best_fitness = fitness
    #         self.best_chromosome = chromosome.copy()
    #
    #     return fitness
    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """评估染色体适应度，修复适应度始终为0的问题"""
        # 初始化参数
        total_distance = 0
        max_distance = 0
        uav_resources = {uav.id: uav.initial_resources.copy() for uav in self.uavs}
        target_allocation = {target.id: np.zeros(len(target.resources)) for target in self.targets}
        uav_positions = {uav.id: uav.position.copy() for uav in self.uavs}
        uav_distances = {uav.id: 0 for uav in self.uavs}

        # 解析染色体，计算每架无人机的任务序列和路径长度
        for i in range(chromosome.shape[1]):
            target_id = chromosome[0, i]
            uav_id = -chromosome[1, i] if chromosome[1, i] != 0 else None
            phi_idx = chromosome[2, i]

            if uav_id and uav_id in uav_resources:
                # 找到对应的目标和无人机
                target = next((t for t in self.targets if t.id == target_id), None)
                if not target:
                    continue

                uav = next(u for u in self.uavs if u.id == uav_id)

                # 计算从当前位置到目标的距离
                start_pos = uav_positions[uav_id]
                distance = np.linalg.norm(target.position - start_pos)

                # 更新无人机位置和总飞行距离
                uav_positions[uav_id] = target.position.copy()
                uav_distances[uav_id] += distance

                # 检查资源约束（使用软约束，而非直接拒绝）
                resource_cost = target.resources / len([g for g in range(chromosome.shape[1])
                                                        if chromosome[0, g] == target_id and chromosome[1, g] != 0])

                # 累加资源分配
                uav_resources[uav_id] -= np.array(resource_cost, np.int32)
                target_allocation[target_id] += np.array(resource_cost, np.int32)

        # 计算约束违反程度
        resource_violation = 0
        for uav_id, resources in uav_resources.items():
            resource_violation += sum(max(0, -r) for r in resources)  # 资源不足部分

        range_violation = 0
        for uav_id, dist in uav_distances.items():
            if dist > self.uavs[uav_id - 1].max_distance:
                range_violation += dist - self.uavs[uav_id - 1].max_distance  # 超出航程部分

        target_violation = 0
        for target_id, allocated in target_allocation.items():
            target = next(t for t in self.targets if t.id == target_id)
            target_violation += sum(max(0, required - allocated[i])
                                    for i, required in enumerate(target.resources))  # 未满足的资源需求

        # 计算总距离和最大距离
        total_distance = sum(uav_distances.values())
        max_distance = max(uav_distances.values())

        # 使用柔性适应度计算（避免直接返回0）
        alpha = 0.5  # 平衡因子
        beta = 10  # 约束惩罚因子
        objective = total_distance + alpha * max_distance
        penalty = beta * (resource_violation + range_violation + target_violation)

        # 确保适应度始终为正数，约束违反会降低适应度但不会使其为0
        fitness = 1.0 / (1 + objective + penalty)

        # 记录最佳染色体（即使适应度很低）
        if self.best_chromosome is None or fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_chromosome = chromosome.copy()

        return fitness

#松弛约束或调整适应度函数（柔性惩罚）
    def evaluate_fitness_soft(self, chromosome: np.ndarray) -> float:
        """评估染色体适应度，修复适应度始终为0的问题"""
        # 初始化参数
        total_distance = 0
        max_distance = 0
        uav_resources = {uav.id: uav.initial_resources.copy() for uav in self.uavs}
        target_allocation = {target.id: np.zeros(len(target.resources)) for target in self.targets}
        uav_positions = {uav.id: uav.position.copy() for uav in self.uavs}
        uav_distances = {uav.id: 0 for uav in self.uavs}

        # 解析染色体，计算每架无人机的任务序列和路径长度
        for i in range(chromosome.shape[1]):
            target_id = chromosome[0, i]
            uav_id = -chromosome[1, i] if chromosome[1, i] != 0 else None
            phi_idx = chromosome[2, i]

            if uav_id and uav_id in uav_resources:
                # 找到对应的目标和无人机
                target = next((t for t in self.targets if t.id == target_id), None)
                if not target:
                    continue

                uav = next(u for u in self.uavs if u.id == uav_id)

                # 计算从当前位置到目标的距离
                start_pos = uav_positions[uav_id]
                distance = np.linalg.norm(target.position - start_pos)

                # 更新无人机位置和总飞行距离
                uav_positions[uav_id] = target.position.copy()
                uav_distances[uav_id] += distance

                # 检查资源约束（使用软约束，而非直接拒绝）
                resource_cost = target.resources / len([g for g in range(chromosome.shape[1])
                                                        if chromosome[0, g] == target_id and chromosome[1, g] != 0])

                # 累加资源分配
                uav_resources[uav_id] -= np.array(resource_cost,np.int32)
                target_allocation[target_id] += np.array(resource_cost,np.int32)

        # 计算约束违反程度
        resource_violation = 0
        for uav_id, resources in uav_resources.items():
            resource_violation += sum(max(0, -r) for r in resources)  # 资源不足部分

        range_violation = 0
        for uav_id, dist in uav_distances.items():
            if dist > self.uavs[uav_id - 1].max_distance:
                range_violation += dist - self.uavs[uav_id - 1].max_distance  # 超出航程部分

        target_violation = 0
        for target_id, allocated in target_allocation.items():
            target = next(t for t in self.targets if t.id == target_id)
            target_violation += sum(max(0, required - allocated[i])
                                    for i, required in enumerate(target.resources))  # 未满足的资源需求

        # 计算总距离和最大距离
        total_distance = sum(uav_distances.values())
        max_distance = max(uav_distances.values())

        # 使用柔性适应度计算（避免直接返回0）
        alpha = 0.5  # 平衡因子
        beta = 10  # 约束惩罚因子
        objective = total_distance + alpha * max_distance
        penalty = beta * (resource_violation + range_violation + target_violation)

        # 确保适应度始终为正数，约束违反会降低适应度但不会使其为0
        fitness = 1.0 / (1 + objective + penalty)

        # 记录最佳染色体（即使适应度很低）
        if self.best_chromosome is None or fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_chromosome = chromosome.copy()

        return fitness

    def _calculate_resource_cost(self, uav: UAV, target: Target) -> np.ndarray:
        """计算无人机攻击目标的资源消耗"""
        # 简化的资源消耗计算，实际应根据任务类型和目标需求计算
        return target.resources / len([u for u in self.uavs if u.id == uav.id])

    def selection(self) -> List[np.ndarray]:
        """选择操作"""
        fitnesses = [self.evaluate_fitness(chrom) for chrom in self.population]
        total_fitness = sum(fitnesses)
        # 处理total_fitness为0的情况
        if total_fitness == 0:
            #print("警告：所有染色体的适应度都为0！重新初始化种群...")
            #self.population = self._initialize_population()
            #fitnesses = [self.evaluate_fitness(chrom) for chrom in self.population]
            self.population = self._initialize_population_1()
            fitnesses = [self.evaluate_fitness_soft(chrom) for chrom in self.population]  # 修改适应度计算函数

            total_fitness = sum(fitnesses)
            if total_fitness == 0:
                #print("警告：所有染色体的适应度都为0！重新初始化种群后仍然为0，改变种群初始化策略为：基于约束的初始化")
                self.population = self._initialize_population_1()
                fitnesses = [self.evaluate_fitness_soft(chrom) for chrom in self.population]#修改适应度计算函数
                total_fitness = sum(fitnesses)
                if total_fitness == 0:
                    print("警告：所有染色体的适应度都为0！第1次更新种群初始化策略、松弛适应度计算函数后仍然为0。\n---分析原因---")

                    # 计算无人机总资源和目标总需求
                    uav_total_resources = np.sum([uav.resources for uav in self.uavs], axis=0)
                    target_total_demand = np.sum([target.resources for target in self.targets], axis=0)
                    print(f"无人机总资源: {uav_total_resources}")
                    print(f"目标总需求: {target_total_demand}")
                    if np.any(uav_total_resources < target_total_demand):
                        print("警告：无人机资源总量不足，需调整目标需求或增加无人机！")
                        sys.exit(1)  # 错误退出
                    else:
                        print("检查1：无人机资源总量大于目标需求，通过！")

                    # 检查航程可行性
                    for uav in self.uavs:
                        for target in self.targets:
                            distance = euclidean(uav.position, target.position)
                            if distance > uav.max_distance:
                                print(f"警告：UAV{uav.id}到目标T{target.id}的距离({distance:.1f})超过最大航程({uav.max_distance})")
                                sys.exit(1)  # 错误退出
                            else:
                                print(
                                    "检查2：航程可行性，通过！" + f"警告：UAV{uav.id}到目标T{target.id}的距离({distance:.1f})超过最大航程({uav.max_distance})")

                    sys.exit(1)



        selection_probs = [f / total_fitness for f in fitnesses]

        # 轮盘赌选择
        parents = []
        for _ in range(self.population_size):
            idx = np.random.choice(len(self.population), p=selection_probs)
            parents.append(self.population[idx].copy())

        return parents

    def crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """交叉操作"""
        offspring = []
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size and random.random() < self.crossover_rate:
                # 单点交叉
                parent1, parent2 = parents[i], parents[i + 1]
                n_genes = parent1.shape[1]
                crossover_point = random.randint(1, n_genes - 1)

                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent2)

                child1[:, :crossover_point] = parent1[:, :crossover_point]
                child1[:, crossover_point:] = parent2[:, crossover_point:]

                child2[:, :crossover_point] = parent2[:, :crossover_point]
                child2[:, crossover_point:] = parent1[:, crossover_point:]

                # 修复交叉后的染色体，确保资源约束
                child1 = self._repair_chromosome(child1)
                child2 = self._repair_chromosome(child2)

                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i].copy())

        return offspring

    def mutation(self, offspring: List[np.ndarray]) -> List[np.ndarray]:
        """变异操作，修复None值导致的错误"""
        mutated_offspring = []

        for i, chrom in enumerate(offspring):
            # 关键修复：检查染色体是否为None
            if chrom is None:
                print(f"警告：第 {i} 个染色体为None，重新初始化")
                chrom = self._initialize_chromosome()  # 创建新的随机染色体

            if random.random() < self.mutation_rate:
                n_genes = chrom.shape[1]  # 此时chrom一定不是None

                # 随机选择一个基因进行变异
                gene_idx = random.randint(0, n_genes - 1)

                # 变异航向角（第三行）
                chrom[2, gene_idx] = random.randint(0, self.graph.n_phi - 1)

                # 有一定概率变异无人机分配（第二行）
                if random.random() < 0.5:
                    feasible_uavs = [uav.id for uav in self.uavs]
                    if feasible_uavs:
                        chrom[1, gene_idx] = -random.choice(feasible_uavs)

                # 修复变异后的染色体，确保满足约束
                chrom = self._repair_chromosome(chrom)

            mutated_offspring.append(chrom)

        return mutated_offspring

    def _initialize_chromosome(self) -> np.ndarray:
        """创建满足基本资源约束的初始染色体"""
        n_genes = len(self.targets) * 3
        chromosome = np.zeros((3, n_genes), dtype=int)

        # 跟踪每架无人机的剩余资源
        uav_resources = {uav.id: uav.initial_resources.copy() for uav in self.uavs}

        for target_idx, target in enumerate(self.targets):
            # 为每个目标分配至少1架无人机
            assigned_uavs = []

            # 计算需要的最小无人机数量（基于资源需求）
            min_uavs_needed = max(1, int(
                np.ceil(np.max(target.resources / np.min([uav.resources for uav in self.uavs], axis=0)))))
            n_assignments = min(min_uavs_needed, 3)  # 最多3架无人机

            for assignment in range(n_assignments):
                gene_idx = target_idx * 3 + assignment
                chromosome[0, gene_idx] = target.id

                # 筛选有足够资源的无人机
                feasible_uavs = [
                    uav.id for uav in self.uavs
                    if uav.id not in assigned_uavs and np.all(uav_resources[uav.id] >= target.resources / n_assignments)
                ]

                if feasible_uavs:
                    uav_id = random.choice(feasible_uavs)
                    chromosome[1, gene_idx] = -uav_id
                    chromosome[2, gene_idx] = random.randint(0, self.graph.n_phi - 1)

                    # 更新资源使用
                    uav_resources[uav_id] -= target.resources / n_assignments
                    assigned_uavs.append(uav_id)

        return chromosome

    def _repair_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """修复染色体，确保满足资源约束条件"""
        n_genes = chromosome.shape[1] if chromosome is not None else 0
        if n_genes == 0:
            return self._initialize_chromosome()  # 创建新染色体

        # 跟踪每架无人机的剩余资源
        uav_resources = {uav.id: uav.initial_resources.copy() for uav in self.uavs}

        # 记录每个目标已分配的无人机数量
        target_assignments = {target.id: 0 for target in self.targets}

        # 第一遍扫描：移除不可行的任务分配
        for i in range(n_genes):
            target_id = chromosome[0, i]
            uav_id = -chromosome[1, i] if chromosome[1, i] != 0 else None

            if uav_id and uav_id in uav_resources:
                target = next((t for t in self.targets if t.id == target_id), None)
                if not target:
                    chromosome[1, i] = 0  # 无效目标，取消分配
                    continue

                # 计算该无人机需要提供的资源
                assigned_uavs = [g for g in range(n_genes)
                                 if chromosome[0, g] == target_id and chromosome[1, g] != 0]
                n_assigned = len(assigned_uavs)

                if n_assigned > 0:
                    resource_cost = target.resources / n_assigned
                else:
                    resource_cost = target.resources  # 只有当前无人机

                # 检查资源是否足够
                if np.any(uav_resources[uav_id] < resource_cost):
                    chromosome[1, i] = 0  # 资源不足，取消分配
                else:
                    # 更新剩余资源（仅用于检查后续任务）
                    uav_resources[uav_id] -= np.array(resource_cost,np.int32)
                    target_assignments[target_id] += 1

        # 第二遍扫描：确保每个目标至少有一个无人机
        for target_id in target_assignments:
            if target_assignments[target_id] == 0:
                # 目标未分配任何无人机，尝试分配一个
                target = next(t for t in self.targets if t.id == target_id)
                feasible_uavs = [
                    uav.id for uav in self.uavs
                    if np.all(uav_resources[uav.id] >= target.resources)
                ]

                if feasible_uavs:
                    # 随机选择一个可行的无人机
                    uav_id = random.choice(feasible_uavs)

                    # 找到第一个未分配的基因位置
                    for i in range(n_genes):
                        if chromosome[0, i] == target_id and chromosome[1, i] == 0:
                            chromosome[1, i] = -uav_id
                            chromosome[2, i] = random.randint(0, self.graph.n_phi - 1)
                            break

        return chromosome