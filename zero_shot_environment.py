# -*- coding: utf-8 -*-
# 文件名: zero_shot_environment.py
# 描述: 零样本迁移环境适配器

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from environment import UAVTaskEnv

class ZeroShotEnvironmentAdapter:
    """
    零样本环境适配器
    
    功能：
    1. 将传统的扁平状态转换为结构化的图状态
    2. 支持可变数量的UAV和目标
    3. 提供标准化的特征提取
    4. 处理动作空间的动态映射
    """
    
    def __init__(self, base_env: UAVTaskEnv):
        self.base_env = base_env
        self.uavs = base_env.uavs
        self.targets = base_env.targets
        self.graph = base_env.graph
        self.obstacles = base_env.obstacles
        self.config = base_env.config
        
        # 动作空间映射
        self.num_uavs = len(self.uavs)
        self.num_targets = len(self.targets)
        self.num_directions = 4  # 简化为4个方向
        self.n_actions = self.num_uavs * self.num_targets * self.num_directions
        
        print(f"[ZeroShotEnvironmentAdapter] 初始化完成")
        print(f"  - UAV数量: {self.num_uavs}")
        print(f"  - 目标数量: {self.num_targets}")
        print(f"  - 动作空间大小: {self.n_actions}")
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境并返回结构化状态"""
        flat_state = self.base_env.reset()
        return self._convert_to_graph_state()
    
    def step(self, action_idx: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """执行动作并返回结构化状态"""
        # 将动作索引转换为具体动作
        uav_idx, target_idx, direction_idx = self._decode_action(action_idx)
        
        # 转换为原始环境的动作格式
        original_action_idx = self._convert_to_original_action(uav_idx, target_idx, direction_idx)
        
        # 执行动作
        flat_next_state, reward, done, truncated, info = self.base_env.step(original_action_idx)
        
        # 转换状态
        graph_state = self._convert_to_graph_state()
        
        return graph_state, reward, done, truncated, info
    
    def _convert_to_graph_state(self) -> Dict[str, torch.Tensor]:
        """将当前环境状态转换为图结构状态"""
        # 提取UAV特征
        uav_features = []
        for uav in self.uavs:
            features = [
                uav.current_position[0] / 1000.0,  # 归一化位置
                uav.current_position[1] / 1000.0,
                uav.resources[0] / 100.0,  # 归一化资源
                uav.resources[1] / 100.0,
                uav.velocity_range[0] / 100.0,  # 归一化速度
                uav.velocity_range[1] / 100.0,
                uav.heading / (2 * np.pi),  # 归一化朝向
                float(len([t for t in self.targets if np.linalg.norm(uav.current_position - t.position) < 200])) / len(self.targets)  # 附近目标比例
            ]
            uav_features.append(features)
        
        # 提取目标特征
        target_features = []
        for target in self.targets:
            features = [
                target.position[0] / 1000.0,  # 归一化位置
                target.position[1] / 1000.0,
                target.remaining_resources[0] / 100.0,  # 归一化剩余资源需求
                target.remaining_resources[1] / 100.0
            ]
            target_features.append(features)
        
        # 转换为张量
        uav_tensor = torch.FloatTensor(uav_features).unsqueeze(0)  # [1, num_uavs, 8]
        target_tensor = torch.FloatTensor(target_features).unsqueeze(0)  # [1, num_targets, 4]
        
        return {
            'uav_features': uav_tensor,
            'target_features': target_tensor
        }
    
    def _decode_action(self, action_idx: int) -> Tuple[int, int, int]:
        """解码动作索引"""
        # 确保动作索引在有效范围内
        action_idx = action_idx % self.n_actions
        
        uav_idx = action_idx // (self.num_targets * self.num_directions)
        remaining = action_idx % (self.num_targets * self.num_directions)
        target_idx = remaining // self.num_directions
        direction_idx = remaining % self.num_directions
        
        return uav_idx, target_idx, direction_idx
    
    def _convert_to_original_action(self, uav_idx: int, target_idx: int, direction_idx: int) -> int:
        """转换为原始环境的动作索引"""
        # 简化映射：使用方向索引作为phi索引
        phi_idx = direction_idx
        
        # 计算原始动作索引
        n_u = len(self.base_env.uavs)
        n_p = self.base_env.graph.n_phi
        
        original_action_idx = target_idx * (n_u * n_p) + uav_idx * n_p + phi_idx
        
        # 确保在有效范围内
        return original_action_idx % self.base_env.n_actions
    
    def get_action_mask(self) -> torch.Tensor:
        """获取有效动作掩码"""
        mask = torch.ones(self.n_actions, dtype=torch.bool)
        
        # 简化版本：所有动作都有效
        # 在实际应用中，可以根据UAV资源、目标状态等添加约束
        
        return mask
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'num_uavs': self.num_uavs,
            'num_targets': self.num_targets,
            'num_actions': self.n_actions,
            'uav_positions': [uav.current_position.tolist() for uav in self.uavs],
            'target_positions': [target.position.tolist() for target in self.targets],
            'uav_resources': [uav.resources.tolist() for uav in self.uavs],
            'target_remaining_resources': [target.remaining_resources.tolist() for target in self.targets]
        }

class ZeroShotTrainingEnvironment:
    """零样本训练环境 - 支持多规模场景训练"""
    
    def __init__(self, scenario_configs: List[Dict]):
        """
        Args:
            scenario_configs: 场景配置列表，每个配置包含UAV和目标的数量范围
        """
        self.scenario_configs = scenario_configs
        self.current_env = None
        self.current_adapter = None
        
        print(f"[ZeroShotTrainingEnvironment] 初始化完成")
        print(f"  - 支持场景数量: {len(scenario_configs)}")
        for i, config in enumerate(scenario_configs):
            print(f"    场景{i+1}: UAV({config['uav_range']}) 目标({config['target_range']})")
    
    def sample_scenario(self) -> ZeroShotEnvironmentAdapter:
        """随机采样一个场景配置"""
        config = np.random.choice(self.scenario_configs)
        
        # 随机生成UAV和目标数量
        num_uavs = np.random.randint(config['uav_range'][0], config['uav_range'][1] + 1)
        num_targets = np.random.randint(config['target_range'][0], config['target_range'][1] + 1)
        
        # 创建场景（这里需要实际的场景生成函数）
        uavs, targets, obstacles = self._generate_random_scenario(num_uavs, num_targets)
        
        # 创建环境
        from config import Config
        config_obj = Config()
        
        from environment import DirectedGraph
        graph = DirectedGraph(uavs, targets, config_obj.GRAPH_N_PHI, obstacles, config_obj)
        
        base_env = UAVTaskEnv(uavs, targets, graph, obstacles, config_obj, obs_mode="flat")
        adapter = ZeroShotEnvironmentAdapter(base_env)
        
        self.current_env = base_env
        self.current_adapter = adapter
        
        return adapter
    
    def _generate_random_scenario(self, num_uavs: int, num_targets: int):
        """生成随机场景"""
        from entities import UAV, Target
        from obstacles import CircularObstacle
        
        # 生成随机UAV
        uavs = []
        for i in range(num_uavs):
            position = np.random.uniform([100, 100], [900, 900])
            resources = np.random.uniform([20, 20], [80, 80])
            uav = UAV(i+1, position, resources, velocity_range=[10, 50], heading=0)
            uavs.append(uav)
        
        # 生成随机目标
        targets = []
        for i in range(num_targets):
            position = np.random.uniform([100, 100], [900, 900])
            resources = np.random.uniform([10, 10], [60, 60])
            target = Target(i+1, position, resources)
            targets.append(target)
        
        # 生成随机障碍物
        obstacles = []
        num_obstacles = np.random.randint(2, 6)
        for i in range(num_obstacles):
            center = np.random.uniform([200, 200], [800, 800])
            radius = np.random.uniform(30, 80)
            obstacle = CircularObstacle(center, radius)
            obstacles.append(obstacle)
        
        return uavs, targets, obstacles

# 测试代码
if __name__ == "__main__":
    print("测试零样本环境适配器...")
    
    # 创建测试场景
    from scenarios import get_small_scenario
    from config import Config
    from environment import DirectedGraph, UAVTaskEnv
    
    # 设置环境变量
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    try:
        uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
        config = Config()
        graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
        base_env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
        
        # 创建适配器
        adapter = ZeroShotEnvironmentAdapter(base_env)
        
        # 测试重置
        state = adapter.reset()
        print(f"初始状态:")
        print(f"  UAV特征形状: {state['uav_features'].shape}")
        print(f"  目标特征形状: {state['target_features'].shape}")
        
        # 测试步进
        action_idx = 0
        next_state, reward, done, truncated, info = adapter.step(action_idx)
        print(f"执行动作后:")
        print(f"  奖励: {reward}")
        print(f"  完成: {done}")
        print(f"  状态形状: UAV{next_state['uav_features'].shape}, 目标{next_state['target_features'].shape}")
        
        # 测试状态信息
        state_info = adapter.get_state_info()
        print(f"状态信息: {state_info}")
        
        print("零样本环境适配器测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()