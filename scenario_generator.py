# -*- coding: utf-8 -*-
# 文件名: scenario_generator.py
# 描述: 动态场景生成器 - 支持不同复杂度的场景生成

import numpy as np
import random
from typing import List, Tuple, Dict, Any
from entities import UAV, Target
from path_planning import CircularObstacle

class DynamicScenarioGenerator:
    """动态场景生成器"""
    
    def __init__(self, map_size: Tuple[int, int] = (1000, 1000)):
        self.map_size = map_size
        self.complexity_levels = {
            'simple': {
                'uav_range': (2, 4),
                'target_range': (2, 5),
                'obstacle_range': (1, 3),
                'resource_variance': 0.2,
                'spatial_clustering': 0.3
            },
            'medium': {
                'uav_range': (4, 7),
                'target_range': (5, 10),
                'obstacle_range': (3, 6),
                'resource_variance': 0.4,
                'spatial_clustering': 0.5
            },
            'complex': {
                'uav_range': (6, 10),
                'target_range': (8, 15),
                'obstacle_range': (5, 10),
                'resource_variance': 0.6,
                'spatial_clustering': 0.7
            }
        }
    
    def generate_scenario(self, complexity: str = 'medium', 
                         seed: int = None) -> Tuple[List[UAV], List[Target], List]:
        """
        生成指定复杂度的场景
        
        Args:
            complexity: 复杂度级别 ('simple', 'medium', 'complex')
            seed: 随机种子
        
        Returns:
            (uavs, targets, obstacles)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        config = self.complexity_levels.get(complexity, self.complexity_levels['medium'])
        
        # 生成数量
        num_uavs = np.random.randint(config['uav_range'][0], config['uav_range'][1] + 1)
        num_targets = np.random.randint(config['target_range'][0], config['target_range'][1] + 1)
        num_obstacles = np.random.randint(config['obstacle_range'][0], config['obstacle_range'][1] + 1)
        
        # 生成实体
        uavs = self._generate_uavs(num_uavs, config)
        targets = self._generate_targets(num_targets, config)
        obstacles = self._generate_obstacles(num_obstacles, config)
        
        # 确保场景有效性
        self._validate_scenario(uavs, targets, obstacles)
        
        print(f"生成{complexity}场景: {num_uavs}个UAV, {num_targets}个目标, {num_obstacles}个障碍物")
        
        return uavs, targets, obstacles
    
    def _generate_uavs(self, num_uavs: int, config: Dict) -> List[UAV]:
        """生成UAV"""
        uavs = []
        
        # 根据聚类程度决定位置分布
        if config['spatial_clustering'] > 0.5:
            # 高聚类：生成几个聚类中心
            num_clusters = max(1, num_uavs // 3)
            cluster_centers = [
                np.random.uniform([100, 100], [self.map_size[0]-100, self.map_size[1]-100])
                for _ in range(num_clusters)
            ]
        
        for i in range(num_uavs):
            # 位置生成
            if config['spatial_clustering'] > 0.5 and cluster_centers:
                # 聚类分布
                center = random.choice(cluster_centers)
                position = center + np.random.normal(0, 100, 2)
                position = np.clip(position, [50, 50], [self.map_size[0]-50, self.map_size[1]-50])
            else:
                # 均匀分布
                position = np.random.uniform([50, 50], [self.map_size[0]-50, self.map_size[1]-50])
            
            # 资源生成
            base_resources = np.array([50.0, 60.0])
            variance = config['resource_variance']
            resources = base_resources * (1 + np.random.uniform(-variance, variance, 2))
            resources = np.maximum(resources, [10.0, 10.0])
            
            # 速度范围
            velocity_range = [
                np.random.uniform(8, 15),
                np.random.uniform(25, 45)
            ]
            
            # 朝向
            heading = np.random.uniform(0, 2 * np.pi)
            
            # 计算最大距离和经济速度
            max_distance = np.random.uniform(800, 1200)
            economic_speed = (velocity_range[0] + velocity_range[1]) / 2
            
            uav = UAV(i + 1, position, heading, resources, max_distance, velocity_range, economic_speed)
            uavs.append(uav)
        
        return uavs
    
    def _generate_targets(self, num_targets: int, config: Dict) -> List[Target]:
        """生成目标"""
        targets = []
        
        # 根据聚类程度决定位置分布
        if config['spatial_clustering'] > 0.6:
            # 高聚类：生成任务区域
            num_task_areas = max(1, num_targets // 4)
            task_centers = [
                np.random.uniform([150, 150], [self.map_size[0]-150, self.map_size[1]-150])
                for _ in range(num_task_areas)
            ]
        
        for i in range(num_targets):
            # 位置生成
            if config['spatial_clustering'] > 0.6 and task_centers:
                # 任务区域分布
                center = random.choice(task_centers)
                position = center + np.random.normal(0, 80, 2)
                position = np.clip(position, [50, 50], [self.map_size[0]-50, self.map_size[1]-50])
            else:
                # 均匀分布
                position = np.random.uniform([50, 50], [self.map_size[0]-50, self.map_size[1]-50])
            
            # 资源需求生成
            base_demand = np.array([40.0, 50.0])
            variance = config['resource_variance']
            resources = base_demand * (1 + np.random.uniform(-variance, variance, 2))
            resources = np.maximum(resources, [5.0, 5.0])
            
            # 计算目标价值
            value = np.sum(resources) * np.random.uniform(0.8, 1.2)
            target = Target(i + 1, position, resources, value)
            targets.append(target)
        
        return targets
    
    def _generate_obstacles(self, num_obstacles: int, config: Dict) -> List:
        """生成障碍物"""
        obstacles = []
        
        for i in range(num_obstacles):
            # 位置：避免边界
            center = np.random.uniform([100, 100], [self.map_size[0]-100, self.map_size[1]-100])
            
            # 半径：根据复杂度调整
            if config == self.complexity_levels['simple']:
                radius = np.random.uniform(20, 40)
            elif config == self.complexity_levels['medium']:
                radius = np.random.uniform(30, 60)
            else:  # complex
                radius = np.random.uniform(40, 80)
            
            obstacle = CircularObstacle(center, radius, tolerance=50.0)
            obstacles.append(obstacle)
        
        return obstacles
    
    def _validate_scenario(self, uavs: List[UAV], targets: List[Target], obstacles: List):
        """验证场景有效性"""
        # 检查资源平衡
        total_supply = sum(np.sum(uav.resources) for uav in uavs)
        total_demand = sum(np.sum(target.resources) for target in targets)
        
        # 如果供给严重不足，调整目标需求
        if total_supply < total_demand * 0.5:
            scale_factor = (total_supply * 0.8) / total_demand
            for target in targets:
                target.resources = target.resources * scale_factor
                target.remaining_resources = target.resources.copy()
        
        # 如果供给过多，增加一些目标需求
        elif total_supply > total_demand * 2.0:
            scale_factor = min(1.5, (total_supply * 0.6) / total_demand)
            for target in targets:
                target.resources = target.resources * scale_factor
                target.remaining_resources = target.resources.copy()
        
        # 检查实体间的最小距离，避免重叠
        self._ensure_minimum_distances(uavs, targets, obstacles)
    
    def _ensure_minimum_distances(self, uavs: List[UAV], targets: List[Target], obstacles: List):
        """确保实体间的最小距离"""
        min_distance = 50.0
        
        # 调整UAV位置
        for i, uav in enumerate(uavs):
            for j, other_uav in enumerate(uavs[i+1:], i+1):
                distance = np.linalg.norm(uav.position - other_uav.position)
                if distance < min_distance:
                    # 移动第二个UAV
                    direction = (other_uav.position - uav.position) / (distance + 1e-6)
                    other_uav.position = uav.position + direction * min_distance
                    # 确保在地图范围内
                    other_uav.position = np.clip(other_uav.position, [50, 50], 
                                                [self.map_size[0]-50, self.map_size[1]-50])
        
        # 调整目标位置
        for i, target in enumerate(targets):
            for j, other_target in enumerate(targets[i+1:], i+1):
                distance = np.linalg.norm(target.position - other_target.position)
                if distance < min_distance:
                    # 移动第二个目标
                    direction = (other_target.position - target.position) / (distance + 1e-6)
                    other_target.position = target.position + direction * min_distance
                    # 确保在地图范围内
                    other_target.position = np.clip(other_target.position, [50, 50], 
                                                   [self.map_size[0]-50, self.map_size[1]-50])
    
    def generate_curriculum_scenarios(self, num_scenarios: int = 10) -> List[Tuple[str, List[UAV], List[Target], List]]:
        """
        生成课程学习场景序列
        
        Args:
            num_scenarios: 生成的场景数量
        
        Returns:
            场景列表，每个元素为 (complexity, uavs, targets, obstacles)
        """
        scenarios = []
        
        # 课程学习策略：从简单到复杂
        complexity_sequence = ['simple'] * 3 + ['medium'] * 4 + ['complex'] * 3
        
        for i in range(num_scenarios):
            complexity = complexity_sequence[i % len(complexity_sequence)]
            uavs, targets, obstacles = self.generate_scenario(complexity, seed=i)
            scenarios.append((complexity, uavs, targets, obstacles))
        
        return scenarios
    
    def generate_transfer_test_scenarios(self) -> Dict[str, Tuple[List[UAV], List[Target], List]]:
        """
        生成迁移测试场景
        
        Returns:
            测试场景字典
        """
        test_scenarios = {}
        
        # 生成各种复杂度的测试场景
        for complexity in ['simple', 'medium', 'complex']:
            for i in range(3):  # 每种复杂度生成3个场景
                scenario_name = f"{complexity}_test_{i+1}"
                uavs, targets, obstacles = self.generate_scenario(complexity, seed=100+i)
                test_scenarios[scenario_name] = (uavs, targets, obstacles)
        
        # 生成极端场景
        # 1. 资源稀缺场景
        uavs_scarce, targets_scarce, obstacles_scarce = self._generate_resource_scarce_scenario()
        test_scenarios['resource_scarce'] = (uavs_scarce, targets_scarce, obstacles_scarce)
        
        # 2. 高密度场景
        uavs_dense, targets_dense, obstacles_dense = self._generate_high_density_scenario()
        test_scenarios['high_density'] = (uavs_dense, targets_dense, obstacles_dense)
        
        # 3. 不平衡场景
        uavs_unbalanced, targets_unbalanced, obstacles_unbalanced = self._generate_unbalanced_scenario()
        test_scenarios['unbalanced'] = (uavs_unbalanced, targets_unbalanced, obstacles_unbalanced)
        
        return test_scenarios
    
    def _generate_resource_scarce_scenario(self) -> Tuple[List[UAV], List[Target], List]:
        """生成资源稀缺场景"""
        # 少量UAV，大量目标
        num_uavs = 3
        num_targets = 8
        num_obstacles = 4
        
        uavs = []
        for i in range(num_uavs):
            position = np.random.uniform([100, 100], [900, 900])
            # 资源较少
            resources = np.random.uniform([20, 25], [35, 40])
            velocity_range = [10, 30]
            heading = np.random.uniform(0, 2 * np.pi)
            max_distance = np.random.uniform(800, 1200)
            economic_speed = (velocity_range[0] + velocity_range[1]) / 2
            uav = UAV(i + 1, position, heading, resources, max_distance, velocity_range, economic_speed)
            uavs.append(uav)
        
        targets = []
        for i in range(num_targets):
            position = np.random.uniform([100, 100], [900, 900])
            # 需求较高
            resources = np.random.uniform([25, 30], [40, 50])
            value = np.sum(resources) * np.random.uniform(0.8, 1.2)
            target = Target(i + 1, position, resources, value)
            targets.append(target)
        
        obstacles = []
        for i in range(num_obstacles):
            center = np.random.uniform([150, 150], [850, 850])
            radius = np.random.uniform(30, 50)
            obstacle = CircularObstacle(center, radius, tolerance=50.0)
            obstacles.append(obstacle)
        
        return uavs, targets, obstacles
    
    def _generate_high_density_scenario(self) -> Tuple[List[UAV], List[Target], List]:
        """生成高密度场景"""
        # 大量实体在较小区域内
        num_uavs = 8
        num_targets = 12
        num_obstacles = 8
        
        # 限制在较小的区域内
        area_size = 600
        offset = (self.map_size[0] - area_size) // 2
        
        uavs = []
        for i in range(num_uavs):
            position = np.random.uniform([offset, offset], [offset + area_size, offset + area_size])
            resources = np.random.uniform([40, 50], [70, 80])
            velocity_range = [12, 35]
            heading = np.random.uniform(0, 2 * np.pi)
            max_distance = np.random.uniform(800, 1200)
            economic_speed = (velocity_range[0] + velocity_range[1]) / 2
            uav = UAV(i + 1, position, heading, resources, max_distance, velocity_range, economic_speed)
            uavs.append(uav)
        
        targets = []
        for i in range(num_targets):
            position = np.random.uniform([offset, offset], [offset + area_size, offset + area_size])
            resources = np.random.uniform([30, 40], [50, 60])
            value = np.sum(resources) * np.random.uniform(0.8, 1.2)
            target = Target(i + 1, position, resources, value)
            targets.append(target)
        
        obstacles = []
        for i in range(num_obstacles):
            center = np.random.uniform([offset + 50, offset + 50], 
                                     [offset + area_size - 50, offset + area_size - 50])
            radius = np.random.uniform(20, 35)
            obstacle = CircularObstacle(center, radius, tolerance=50.0)
            obstacles.append(obstacle)
        
        # 确保最小距离
        self._ensure_minimum_distances(uavs, targets, obstacles)
        
        return uavs, targets, obstacles
    
    def _generate_unbalanced_scenario(self) -> Tuple[List[UAV], List[Target], List]:
        """生成不平衡场景"""
        # 资源类型严重不平衡
        num_uavs = 6
        num_targets = 8
        num_obstacles = 5
        
        uavs = []
        for i in range(num_uavs):
            position = np.random.uniform([100, 100], [900, 900])
            
            # 创建资源不平衡：一半UAV资源类型1多，一半资源类型2多
            if i < num_uavs // 2:
                resources = np.array([np.random.uniform(60, 80), np.random.uniform(10, 20)])
            else:
                resources = np.array([np.random.uniform(10, 20), np.random.uniform(60, 80)])
            
            velocity_range = [10, 35]
            heading = np.random.uniform(0, 2 * np.pi)
            max_distance = np.random.uniform(800, 1200)
            economic_speed = (velocity_range[0] + velocity_range[1]) / 2
            uav = UAV(i + 1, position, heading, resources, max_distance, velocity_range, economic_speed)
            uavs.append(uav)
        
        targets = []
        for i in range(num_targets):
            position = np.random.uniform([100, 100], [900, 900])
            
            # 目标需求也不平衡
            if i < num_targets // 2:
                resources = np.array([np.random.uniform(40, 60), np.random.uniform(5, 15)])
            else:
                resources = np.array([np.random.uniform(5, 15), np.random.uniform(40, 60)])
            
            value = np.sum(resources) * np.random.uniform(0.8, 1.2)
            target = Target(i + 1, position, resources, value)
            targets.append(target)
        
        obstacles = []
        for i in range(num_obstacles):
            center = np.random.uniform([150, 150], [850, 850])
            radius = np.random.uniform(35, 55)
            obstacle = CircularObstacle(center, radius, tolerance=50.0)
            obstacles.append(obstacle)
        
        return uavs, targets, obstacles
    
    def get_scenario_statistics(self, uavs: List[UAV], targets: List[Target], obstacles: List) -> Dict[str, Any]:
        """获取场景统计信息"""
        # 基础统计
        stats = {
            'num_uavs': len(uavs),
            'num_targets': len(targets),
            'num_obstacles': len(obstacles),
            'map_size': self.map_size
        }
        
        # 资源统计
        if uavs and targets:
            total_supply = sum(np.sum(uav.resources) for uav in uavs)
            total_demand = sum(np.sum(target.resources) for target in targets)
            
            stats.update({
                'total_supply': total_supply,
                'total_demand': total_demand,
                'supply_demand_ratio': total_supply / total_demand if total_demand > 0 else float('inf'),
                'avg_uav_resources': total_supply / len(uavs),
                'avg_target_demand': total_demand / len(targets)
            })
            
            # 资源类型平衡
            uav_resources = np.array([uav.resources for uav in uavs])
            target_resources = np.array([target.resources for target in targets])
            
            supply_balance = np.std(np.sum(uav_resources, axis=0)) / np.mean(np.sum(uav_resources, axis=0))
            demand_balance = np.std(np.sum(target_resources, axis=0)) / np.mean(np.sum(target_resources, axis=0))
            
            stats.update({
                'supply_balance': supply_balance,
                'demand_balance': demand_balance
            })
        
        # 空间分布统计
        if uavs:
            uav_positions = np.array([uav.position for uav in uavs])
            uav_distances = []
            for i in range(len(uavs)):
                for j in range(i+1, len(uavs)):
                    distance = np.linalg.norm(uav_positions[i] - uav_positions[j])
                    uav_distances.append(distance)
            
            stats['avg_uav_distance'] = np.mean(uav_distances) if uav_distances else 0
            stats['min_uav_distance'] = np.min(uav_distances) if uav_distances else 0
        
        if targets:
            target_positions = np.array([target.position for target in targets])
            target_distances = []
            for i in range(len(targets)):
                for j in range(i+1, len(targets)):
                    distance = np.linalg.norm(target_positions[i] - target_positions[j])
                    target_distances.append(distance)
            
            stats['avg_target_distance'] = np.mean(target_distances) if target_distances else 0
            stats['min_target_distance'] = np.min(target_distances) if target_distances else 0
        
        # 复杂度评估
        complexity_score = self._calculate_complexity_score(uavs, targets, obstacles)
        stats['complexity_score'] = complexity_score
        
        return stats
    
    def _calculate_complexity_score(self, uavs: List[UAV], targets: List[Target], obstacles: List) -> float:
        """计算场景复杂度分数"""
        score = 0.0
        
        # 实体数量复杂度
        entity_complexity = (len(uavs) + len(targets)) / 20.0  # 归一化到[0,1]
        score += entity_complexity * 0.3
        
        # 资源平衡复杂度
        if uavs and targets:
            total_supply = sum(np.sum(uav.resources) for uav in uavs)
            total_demand = sum(np.sum(target.resources) for target in targets)
            ratio = total_supply / total_demand if total_demand > 0 else 1.0
            
            # 比例越接近1，复杂度越高（需要精确分配）
            balance_complexity = 1.0 - abs(ratio - 1.0) / max(ratio, 1.0)
            score += balance_complexity * 0.3
        
        # 空间分布复杂度
        if len(obstacles) > 0:
            obstacle_complexity = min(1.0, len(obstacles) / 10.0)
            score += obstacle_complexity * 0.2
        
        # 密度复杂度
        if uavs and targets:
            all_positions = [uav.position for uav in uavs] + [target.position for target in targets]
            positions = np.array(all_positions)
            
            # 计算平均最近邻距离
            min_distances = []
            for i, pos in enumerate(positions):
                distances = [np.linalg.norm(pos - other_pos) for j, other_pos in enumerate(positions) if i != j]
                if distances:
                    min_distances.append(min(distances))
            
            if min_distances:
                avg_min_distance = np.mean(min_distances)
                # 距离越小，密度越高，复杂度越高
                density_complexity = max(0, 1.0 - avg_min_distance / 200.0)
                score += density_complexity * 0.2
        
        return min(1.0, score)

# 测试代码
if __name__ == "__main__":
    print("测试动态场景生成器...")
    
    generator = DynamicScenarioGenerator()
    
    # 测试不同复杂度的场景生成
    for complexity in ['simple', 'medium', 'complex']:
        print(f"\n生成{complexity}场景:")
        uavs, targets, obstacles = generator.generate_scenario(complexity, seed=42)
        
        stats = generator.get_scenario_statistics(uavs, targets, obstacles)
        print(f"  统计信息: {stats}")
    
    # 测试课程学习场景
    print(f"\n生成课程学习场景:")
    curriculum_scenarios = generator.generate_curriculum_scenarios(5)
    for i, (complexity, uavs, targets, obstacles) in enumerate(curriculum_scenarios):
        print(f"  场景{i+1}: {complexity} - {len(uavs)}UAV, {len(targets)}目标")
    
    # 测试迁移测试场景
    print(f"\n生成迁移测试场景:")
    transfer_scenarios = generator.generate_transfer_test_scenarios()
    for name, (uavs, targets, obstacles) in transfer_scenarios.items():
        print(f"  {name}: {len(uavs)}UAV, {len(targets)}目标, {len(obstacles)}障碍物")
    
    print("动态场景生成器测试完成！")
