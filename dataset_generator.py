# -*- coding: utf-8 -*-
# 文件名: dataset_generator.py
# 描述: 为验证GATNetwork等高级网络结构的性能，生成一套全面的、层次化的实验数据集。

import os
import pickle
import numpy as np
import random
from tqdm import tqdm
from entities import UAV, Target
from path_planning import CircularObstacle, PolygonalObstacle


class DatasetGenerator:
    """
    为无人机任务规划算法生成一套全面的、可复现的验证数据集。
    每个场景都被设计用来测试算法在特定方面的能力。
    """

    def __init__(self, base_dir: str = "scenarios", map_size: tuple = (8000, 6000), tolerance: float = 50.0):
        """
        初始化数据集生成器。

        Args:
            base_dir: 保存生成的数据集的根目录。
            map_size: 场景的地图尺寸 (宽度, 高度)。
            tolerance: 障碍物的安全容忍距离。
        """
        self.base_dir = base_dir
        self.map_width, self.map_height = map_size
        self.tolerance = tolerance
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"数据集将保存至: {self.base_dir}")


    # --- 核心辅助生成函数 ---

    def _create_uavs(self, num_uavs: int, resource_type: str = 'balanced') -> list[UAV]:
        """
        创建一个无人机列表。

        Args:
            num_uavs: 要创建的无人机数量。
            resource_type: 资源类型 ('balanced', 'type_a_heavy', 'type_b_heavy')。

        Returns:
            一个UAV对象的列表。
        """
        uavs = []
        for i in range(1, num_uavs + 1):
            if resource_type == 'type_a_heavy':
                resources = np.array([random.randint(120, 150), random.randint(40, 60)])
            elif resource_type == 'type_b_heavy':
                resources = np.array([random.randint(40, 60), random.randint(120, 150)])
            else: # balanced
                resources = np.array([random.randint(80, 120), random.randint(80, 120)])
            
            uavs.append(UAV(
                id=i,
                position=np.array([random.uniform(0, self.map_width), random.uniform(0, self.map_height)]),
                heading=random.uniform(0, 2 * np.pi),
                resources=resources,
                max_distance=20000,
                velocity_range=(50, 150),
                economic_speed=100
            ))
        return uavs

    def _create_targets(self, num_targets: int, demand_level: str = 'medium', value_range: tuple = (80, 150)) -> list[Target]:
        """
        创建一个目标列表。

        Args:
            num_targets: 要创建的目标数量。
            demand_level: 资源需求水平 ('low', 'medium', 'high')。
            value_range: 目标的价值范围。

        Returns:
            一个Target对象的列表。
        """
        targets = []
        for i in range(1, num_targets + 1):
            if demand_level == 'low':
                demand = np.array([random.randint(40, 60), random.randint(40, 60)])
            elif demand_level == 'high':
                demand = np.array([random.randint(150, 200), random.randint(150, 200)])
            else: # medium
                demand = np.array([random.randint(90, 130), random.randint(90, 130)])

            targets.append(Target(
                id=i,
                position=np.array([
                    random.uniform(self.map_width * 0.1, self.map_width * 0.9),
                    random.uniform(self.map_height * 0.1, self.map_height * 0.9)
                ]),
                resources=demand,
                value=random.randint(value_range[0], value_range[1])
            ))
        return targets
    
    def _save_scenario(self, category: str, name: str, uavs: list, targets: list, obstacles: list):
        """将一个场景保存到文件。"""
        category_dir = os.path.join(self.base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        scenario_data = {
            'uavs': uavs,
            'targets': targets,
            'obstacles': obstacles,
            'scenario_name': name
        }
        
        filepath = os.path.join(category_dir, f'{name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(scenario_data, f)

    # --- A. 基础泛化能力测试集 ---

    def generate_A1_permutation_invariance(self):
        """
        Purpose: 验证模型是否对输入实体的顺序不敏感（排列不变性）。
                 一个好的模型，对于实体顺序打乱的同一个场景，应该输出完全相同或极其相似的决策。
        Characteristics: 生成两个完全相同但实体列表顺序不同的场景。
        Expected Behavior: 模型在 A1_a 和 A1_b 上的最终评估指标（如完成率、总奖励）应几乎没有差异。
        """
        uavs = self._create_uavs(5)
        targets = self._create_targets(4)
        obstacles = [] # 简化场景，聚焦于排列不变性
        
        # 保存原始顺序
        self._save_scenario("A_Generalization", "A1_a_OriginalOrder", uavs, targets, obstacles)
        
        # 打乱顺序后保存
        random.shuffle(uavs)
        random.shuffle(targets)
        self._save_scenario("A_Generalization", "A1_b_ShuffledOrder", uavs, targets, obstacles)

    def generate_A2_count_generalization(self):
        """
        Purpose: 测试模型从“M-to-N”零样本迁移的基本能力。
        Characteristics: 生成一个用于训练的小规模场景，和一个实体数量不同的中等规模场景用于评估。
        Expected Behavior: 在A2_Train上训练好的模型，应能直接在A2_Test上运行并给出一个合理的（不一定是完美）规划方案。
        """
        # 训练场景
        train_uavs = self._create_uavs(6)
        train_targets = self._create_targets(5)
        self._save_scenario("A_Generalization", "A2_Train_6UAV_5TGT", train_uavs, train_targets, [])
        
        # 测试场景
        test_uavs = self._create_uavs(10)
        test_targets = self._create_targets(8)
        self._save_scenario("A_Generalization", "A2_Test_10UAV_8TGT", test_uavs, test_targets, [])

    # --- B. 规模扩展性测试集 ---

    def generate_B1_uav_scaling(self):
        """
        Purpose: 测试在目标数量固定时，模型随无人机数量增加的性能变化。
        Characteristics: 4个场景，目标数量固定为10，无人机数量从5, 10, 15, 20递增。
        Expected Behavior: 随着无人机数量增加，任务完成率和总奖励应稳定提升并最终饱和。
                         同时应观察模型的决策时间（推理耗时）如何变化。
        """
        num_targets = 10
        targets = self._create_targets(num_targets)
        
        for num_uavs in [5, 10, 15, 20]:
            uavs = self._create_uavs(num_uavs)
            name = f"B1_UAV_Scaling_{num_uavs:02d}UAV_{num_targets:02d}TGT"
            self._save_scenario("B_Scalability", name, uavs, targets, [])

    def generate_B2_target_scaling(self):
        """
        Purpose: 测试在无人机数量固定时，模型随目标数量增加的性能变化。
        Characteristics: 4个场景，无人机数量固定为12，目标数量从5, 10, 15, 20递增。
        Expected Behavior: 随着目标数量增加，模型的规划难度增大，可能会出现性能瓶颈。
                         这是一个很好的压力测试，用于观察算法的决策上限。
        """
        num_uavs = 12
        uavs = self._create_uavs(num_uavs)
        
        for num_targets in [5, 10, 15, 20]:
            targets = self._create_targets(num_targets)
            name = f"B2_Target_Scaling_{num_uavs:02d}UAV_{num_targets:02d}TGT"
            self._save_scenario("B_Scalability", name, targets, uavs, []) # 修正参数顺序

    # --- C. 资源约束敏感性测试集 ---

    def generate_C1_resource_scarcity(self):
        """
        Purpose: 模拟资源极度紧缺，考验模型的“精打细算”和优先级排序能力。
        Characteristics: 无人机总资源供给远小于目标总需求。
        Expected Behavior: 一个好的模型应该优先完成价值最高的目标，并最大化已分配资源的利用率，
                         而不是平均地、零散地攻击所有目标导致没有一个能完成。
        """
        uavs = self._create_uavs(8)
        # 创造高需求目标
        targets = self._create_targets(10, demand_level='high', value_range=(50, 200))
        self._save_scenario("C_Resource_Constraints", "C1_ResourceScarcity", uavs, targets, [])

    def generate_C2_resource_abundance(self):
        """
        Purpose: 模拟资源极度丰富，考验模型是否会因为资源过多而产生不必要的浪费。
        Characteristics: 无人机总资源供给远大于目标总需求。
        Expected Behavior: 模型应能以最高效的方式（最短总路径、最少无人机出动）完成所有任务，
                         而不是派遣所有无人机进行“饱和式攻击”。
        """
        uavs = self._create_uavs(15)
        # 创造低需求目标
        targets = self._create_targets(8, demand_level='low')
        self._save_scenario("C_Resource_Constraints", "C2_ResourceAbundance", uavs, targets, [])
        
    # --- D. 空间拓扑鲁棒性测试集 ---
    
    def generate_D1_clustered_targets(self):
        """
        Purpose: 考验模型处理局部拥堵和高效协同的能力。
        Characteristics: 目标集中在地图一小块区域，无人机分散在四周。
        Expected Behavior: 模型应能避免让过多无人机挤入狭小区域导致冲突或低效，
                         并能有效地为近处和远处的无人机分配任务以实现快速协同。
        """
        uavs = self._create_uavs(10)
        targets = self._create_targets(8)
        # 将所有目标移动到地图中心的一个小区域
        for target in targets:
            target.position = np.array([
                random.uniform(self.map_width * 0.4, self.map_width * 0.6),
                random.uniform(self.map_height * 0.4, self.map_height * 0.6)
            ])
        self._save_scenario("D_Spatial_Topology", "D1_ClusteredTargets", uavs, targets, [])

    def generate_D2_dispersed_targets(self):
        """
        Purpose: 考验模型的“分兵”和广域覆盖能力。
        Characteristics: 无人机集中在地图一角，目标广泛分散在整个地图。
        Expected Behavior: 模型应能做出合理的“分兵”决策，为每架无人机规划出不交叉、高效的
                         远距离任务路径，以最小化总航程和完成时间。
        """
        targets = self._create_targets(10)
        uavs = self._create_uavs(8)
        # 将所有无人机移动到地图左下角
        for uav in uavs:
            uav.position = np.array([
                random.uniform(0, self.map_width * 0.2),
                random.uniform(0, self.map_height * 0.2)
            ])
        self._save_scenario("D_Spatial_Topology", "D2_DispersedTargets", uavs, targets, [])
        
    # --- E. 战略决策能力测试集 ---

    def generate_E1_strategic_trap(self):
        """
        Purpose: 考验模型的长期规划与权衡能力，避免“短视”的贪婪决策。
        Characteristics: 存在一个价值极高但被障碍物包围、位置偏远的“陷阱”目标，
                         同时存在一簇价值中等但易于攻击的“常规”目标。
        Expected Behavior: 一个优秀的模型应能计算出攻击“陷阱”的长期成本（高昂的飞行惩罚、
                         无法兼顾其他目标的机会成本），并可能选择放弃它，以换取
                         高效完成其他目标所带来的更高总体回报。
        """
        uavs = self._create_uavs(8)
        # 中等价值的目标集群
        targets = self._create_targets(6, value_range=(80, 120))
        # 高价值的“陷阱”目标
        trap_target = Target(id=7, position=np.array([self.map_width * 0.9, self.map_height * 0.9]),
                             resources=np.array([150, 150]), value=300)
        targets.append(trap_target)
        
        # 障碍物包围陷阱
        obstacles = [
            CircularObstacle(center=trap_target.position, radius=600, tolerance=self.tolerance)
        ]
        self._save_scenario("E_Strategic_Decision", "E1_StrategicTrap", uavs, targets, obstacles)
        
    def generate_E2_dynamic_priorities(self):
        """
        Purpose: 考验模型对任务优先级的理解，是否能区分高价值和低价值目标。
        Characteristics: 场景中混合了少量价值极高（必须完成）和大量价值极低（可以放弃）的目标。
        Expected Behavior: 模型应集中优势兵力，优先确保高价值目标的完成，即使这意味着
                         要放弃一些唾手可得的低价值目标。
        """
        uavs = self._create_uavs(10)
        # 3个高价值目标
        high_value_targets = self._create_targets(3, value_range=(200, 250))
        # 12个低价值“干扰”目标
        low_value_targets = self._create_targets(12, value_range=(10, 30))
        
        # 重新分配ID
        all_targets = high_value_targets + low_value_targets
        for i, target in enumerate(all_targets):
            target.id = i + 1
            
        self._save_scenario("E_Strategic_Decision", "E2_DynamicPriorities", uavs, all_targets, [])

    def generate_all(self):
        """生成所有定义的数据集。"""
        print("开始生成所有实验数据集...")
        
        # 使用tqdm显示进度条
        generation_tasks = {
            "A1: Permutation Invariance": self.generate_A1_permutation_invariance,
            "A2: Count Generalization": self.generate_A2_count_generalization,
            "B1: UAV Scaling": self.generate_B1_uav_scaling,
            "B2: Target Scaling": self.generate_B2_target_scaling,
            "C1: Resource Scarcity": self.generate_C1_resource_scarcity,
            "C2: Resource Abundance": self.generate_C2_resource_abundance,
            "D1: Clustered Targets": self.generate_D1_clustered_targets,
            "D2: Dispersed Targets": self.generate_D2_dispersed_targets,
            "E1: Strategic Trap": self.generate_E1_strategic_trap,
            "E2: Dynamic Priorities": self.generate_E2_dynamic_priorities,
        }
        
        with tqdm(total=len(generation_tasks), desc="生成数据集中") as pbar:
            for name, func in generation_tasks.items():
                pbar.set_description(f"正在生成: {name}")
                func()
                pbar.update(1)
                
        print("\n所有实验数据集已成功生成！")


if __name__ == '__main__':
    # 创建生成器实例
    generator = DatasetGenerator()
    
    # 执行生成
    generator.generate_all()