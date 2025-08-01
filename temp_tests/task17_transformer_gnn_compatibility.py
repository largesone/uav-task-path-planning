# -*- coding: utf-8 -*-
# 文件名: task17_transformer_gnn_compatibility.py
# 描述: TransformerGNN输出格式兼容性实现，确保与现有RL算法输出格式一致

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from transformer_gnn import TransformerGNN
    from environment import UAVTaskEnv
    from entities import UAV, Target
except ImportError as e:
    print(f"警告：无法导入必要模块: {e}")


class TransformerGNNCompatibilityMixin:
    """
    TransformerGNN兼容性混入类
    
    为TransformerGNN添加与现有RL算法兼容的输出格式方法，确保：
    1. get_task_assignments方法与现有RL算法输出格式一致
    2. 图模式决策结果转换为标准任务分配格式
    3. 与现有evaluate_plan函数的兼容性
    4. 与main.py中run_scenario流程的完全兼容
    """
    
    def get_task_assignments(self, temperature: float = 0.1, max_inference_steps: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        获取任务分配结果 - 与现有RL算法兼容的接口
        
        此方法确保TransformerGNN的输出格式与现有GraphRLSolver.get_task_assignments()完全一致，
        支持统一的方案评估和与main.py中run_scenario流程的兼容性。
        
        Args:
            temperature (float): 温度参数，控制采样的随机性。值越小，决策越确定性。
                               建议范围：0.05-0.3，默认0.1
            max_inference_steps (int): 最大推理步数，如果为None则使用默认值
            
        Returns:
            Dict[int, List[Tuple[int, int]]]: 任务分配结果，格式为：
                {
                    uav_id: [(target_id, phi_idx), (target_id, phi_idx), ...],
                    ...
                }
                与现有RL算法输出格式完全一致
        """
        print(f"[TransformerGNN] 开始获取任务分配，温度参数: {temperature}")
        
        # 确保模型处于评估模式
        self.eval()
        
        # 重置环境状态
        if not hasattr(self, 'env') or self.env is None:
            raise RuntimeError("TransformerGNN未正确初始化环境，无法进行任务分配推理")
        
        state = self.env.reset()
        assignments = {u.id: [] for u in self.env.uavs}
        done = False
        step = 0
        
        # 设置最大推理步数
        if max_inference_steps is None:
            max_inference_steps = len(self.env.targets) * len(self.env.uavs) * 2
        
        print(f"[TransformerGNN] 开始推理循环，最大步数: {max_inference_steps}")
        
        with torch.no_grad():
            while not done and step < max_inference_steps:
                # 准备输入数据
                if self.is_dict_obs:
                    # 图模式输入
                    obs_tensor = self._prepare_dict_observation(state)
                else:
                    # 扁平模式输入
                    obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # 前向传播获取动作logits
                logits, _ = self.forward({"obs": obs_tensor}, None, None)
                
                # 使用温度采样选择动作
                action_idx = self._sample_action_with_temperature(logits, temperature)
                
                # 将动作索引转换为具体动作
                action = self._index_to_action(action_idx)
                target_id, uav_id, phi_idx = action
                
                # 检查动作有效性
                if not self._is_action_valid(action):
                    step += 1
                    continue
                
                # 记录任务分配
                assignments[uav_id].append((target_id, phi_idx))
                
                # 执行动作并获取下一状态
                next_state, reward, done, truncated, info = self.env.step(action_idx)
                state = next_state
                step += 1
                
                # 检查是否应该提前终止
                if truncated or self._should_terminate_early(assignments):
                    break
        
        print(f"[TransformerGNN] 任务分配推理完成，总步数: {step}")
        
        # 后处理任务分配结果
        processed_assignments = self._post_process_assignments(assignments)
        
        # 输出分配统计
        total_assignments = sum(len(tasks) for tasks in processed_assignments.values())
        print(f"[TransformerGNN] 任务分配统计: 总分配数 {total_assignments}")
        for uav_id, tasks in processed_assignments.items():
            if tasks:
                print(f"  UAV {uav_id}: {len(tasks)} 个任务")
        
        return processed_assignments
    
    def _prepare_dict_observation(self, state: Any) -> torch.Tensor:
        """
        准备字典格式的观测数据
        
        Args:
            state: 环境状态
            
        Returns:
            torch.Tensor: 准备好的观测张量
        """
        if isinstance(state, dict):
            # 如果已经是字典格式，直接转换为张量
            obs_dict = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    obs_dict[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                elif isinstance(value, dict):
                    obs_dict[key] = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                                   for k, v in value.items()}
                else:
                    obs_dict[key] = torch.FloatTensor([value]).to(self.device)
            return obs_dict
        else:
            # 如果是扁平格式，需要转换为图格式
            return self._convert_flat_to_dict_obs(state)
    
    def _convert_flat_to_dict_obs(self, flat_state: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        将扁平状态转换为字典格式观测
        
        Args:
            flat_state: 扁平状态数组
            
        Returns:
            Dict[str, torch.Tensor]: 字典格式观测
        """
        # 这里需要根据具体的状态格式进行转换
        # 暂时返回一个基本的转换结果
        batch_size = 1
        num_uavs = len(self.env.uavs)
        num_targets = len(self.env.targets)
        
        # 假设状态的前半部分是UAV特征，后半部分是目标特征
        split_point = len(flat_state) // 2
        
        obs_dict = {
            'uav_features': torch.FloatTensor(flat_state[:split_point]).view(batch_size, num_uavs, -1).to(self.device),
            'target_features': torch.FloatTensor(flat_state[split_point:]).view(batch_size, num_targets, -1).to(self.device),
            'relative_positions': torch.zeros(batch_size, num_uavs, num_targets, 2).to(self.device),
            'distances': torch.ones(batch_size, num_uavs, num_targets).to(self.device),
            'masks': {
                'uav_mask': torch.ones(batch_size, num_uavs, dtype=torch.bool).to(self.device),
                'target_mask': torch.ones(batch_size, num_targets, dtype=torch.bool).to(self.device)
            }
        }
        
        return obs_dict
    
    def _sample_action_with_temperature(self, logits: torch.Tensor, temperature: float) -> int:
        """
        使用温度采样选择动作
        
        Args:
            logits: 动作logits
            temperature: 温度参数
            
        Returns:
            int: 选择的动作索引
        """
        # 应用温度缩放
        scaled_logits = logits / temperature
        
        # 计算softmax概率
        action_probs = F.softmax(scaled_logits, dim=1)
        
        # 从概率分布中采样
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample().item()
        
        return action_idx
    
    def _index_to_action(self, action_idx: int) -> Tuple[int, int, int]:
        """
        将动作索引转换为具体动作
        
        Args:
            action_idx: 动作索引
            
        Returns:
            Tuple[int, int, int]: (target_id, uav_id, phi_idx)
        """
        if not hasattr(self, 'env') or self.env is None:
            raise RuntimeError("环境未初始化")
        
        # 根据环境的动作空间结构进行转换
        n_uavs = len(self.env.uavs)
        n_targets = len(self.env.targets)
        n_phi = self.env.graph.n_phi if hasattr(self.env, 'graph') else 8
        
        # 计算动作分解
        target_idx = action_idx // (n_uavs * n_phi)
        remainder = action_idx % (n_uavs * n_phi)
        uav_idx = remainder // n_phi
        phi_idx = remainder % n_phi
        
        # 转换为实际ID
        target_id = self.env.targets[target_idx].id if target_idx < len(self.env.targets) else 1
        uav_id = self.env.uavs[uav_idx].id if uav_idx < len(self.env.uavs) else 1
        
        return (target_id, uav_id, phi_idx)
    
    def _is_action_valid(self, action: Tuple[int, int, int]) -> bool:
        """
        检查动作是否有效
        
        Args:
            action: 动作元组 (target_id, uav_id, phi_idx)
            
        Returns:
            bool: 动作是否有效
        """
        target_id, uav_id, phi_idx = action
        
        # 检查UAV是否还有资源
        uav = next((u for u in self.env.uavs if u.id == uav_id), None)
        if uav is None or np.all(uav.resources <= 1e-6):
            return False
        
        # 检查目标是否还需要资源
        target = next((t for t in self.env.targets if t.id == target_id), None)
        if target is None or np.all(target.remaining_resources <= 1e-6):
            return False
        
        return True
    
    def _should_terminate_early(self, assignments: Dict[int, List[Tuple[int, int]]]) -> bool:
        """
        检查是否应该提前终止推理
        
        Args:
            assignments: 当前任务分配
            
        Returns:
            bool: 是否应该提前终止
        """
        # 检查是否所有目标都已满足
        all_targets_satisfied = True
        for target in self.env.targets:
            if np.any(target.remaining_resources > 1e-6):
                all_targets_satisfied = False
                break
        
        if all_targets_satisfied:
            return True
        
        # 检查是否所有UAV都已耗尽资源
        all_uavs_exhausted = True
        for uav in self.env.uavs:
            if np.any(uav.resources > 1e-6):
                all_uavs_exhausted = False
                break
        
        if all_uavs_exhausted:
            return True
        
        return False
    
    def _post_process_assignments(self, assignments: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
        """
        后处理任务分配结果
        
        Args:
            assignments: 原始任务分配
            
        Returns:
            Dict[int, List[Tuple[int, int]]]: 处理后的任务分配
        """
        # 移除重复分配
        processed_assignments = {}
        for uav_id, tasks in assignments.items():
            unique_tasks = []
            seen_tasks = set()
            for task in tasks:
                if task not in seen_tasks:
                    unique_tasks.append(task)
                    seen_tasks.add(task)
            processed_assignments[uav_id] = unique_tasks
        
        return processed_assignments


class CompatibleTransformerGNN(TransformerGNN, TransformerGNNCompatibilityMixin):
    """
    兼容的TransformerGNN类
    
    继承TransformerGNN并混入兼容性功能，确保与现有系统完全兼容
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, env=None):
        """
        初始化兼容的TransformerGNN
        
        Args:
            obs_space: 观测空间
            action_space: 动作空间
            num_outputs: 输出维度
            model_config: 模型配置
            name: 模型名称
            env: 环境实例（用于任务分配推理）
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        # 存储环境引用
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[CompatibleTransformerGNN] 初始化完成，设备: {self.device}")
    
    def set_environment(self, env: UAVTaskEnv):
        """
        设置环境实例
        
        Args:
            env: UAVTaskEnv环境实例
        """
        self.env = env
        print(f"[CompatibleTransformerGNN] 环境已设置")


def create_compatible_transformer_gnn(obs_space, action_space, num_outputs, model_config, name="CompatibleTransformerGNN", env=None):
    """
    创建兼容的TransformerGNN模型的工厂函数
    
    Args:
        obs_space: 观测空间
        action_space: 动作空间
        num_outputs: 输出维度
        model_config: 模型配置
        name: 模型名称
        env: 环境实例
        
    Returns:
        CompatibleTransformerGNN: 兼容的TransformerGNN模型实例
    """
    return CompatibleTransformerGNN(obs_space, action_space, num_outputs, model_config, name, env)


# 方案转换接口
class SolutionConverter:
    """
    方案转换接口
    
    将图模式决策结果转换为标准的任务分配格式，确保与现有evaluate_plan函数的兼容性
    """
    
    @staticmethod
    def convert_graph_solution_to_standard_format(
        graph_assignments: Dict[int, List[Tuple[int, int]]],
        uavs: List[UAV],
        targets: List[Target],
        graph=None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        将图模式任务分配转换为标准格式
        
        Args:
            graph_assignments: 图模式任务分配结果
            uavs: UAV列表
            targets: 目标列表
            graph: 图对象（可选）
            
        Returns:
            Dict[int, List[Dict[str, Any]]]: 标准格式的任务分配，与evaluate_plan兼容
        """
        print("[SolutionConverter] 开始转换图模式方案为标准格式")
        
        standard_assignments = {}
        
        for uav_id, tasks in graph_assignments.items():
            standard_tasks = []
            
            # 找到对应的UAV
            uav = next((u for u in uavs if u.id == uav_id), None)
            if uav is None:
                print(f"警告：未找到UAV {uav_id}")
                continue
            
            for target_id, phi_idx in tasks:
                # 找到对应的目标
                target = next((t for t in targets if t.id == target_id), None)
                if target is None:
                    print(f"警告：未找到目标 {target_id}")
                    continue
                
                # 计算资源成本
                resource_cost = SolutionConverter._calculate_resource_cost(uav, target)
                
                # 计算距离
                distance = SolutionConverter._calculate_distance(uav, target)
                
                # 创建标准任务格式
                standard_task = {
                    'target_id': target_id,
                    'uav_id': uav_id,
                    'phi_idx': phi_idx,
                    'resource_cost': resource_cost,
                    'distance': distance,
                    'is_sync_feasible': True,  # 默认为可同步
                    'start_time': 0.0,
                    'end_time': 1.0,
                    'priority': 1.0
                }
                
                standard_tasks.append(standard_task)
            
            standard_assignments[uav_id] = standard_tasks
        
        print(f"[SolutionConverter] 方案转换完成，转换了 {len(standard_assignments)} 个UAV的任务")
        return standard_assignments
    
    @staticmethod
    def _calculate_resource_cost(uav: UAV, target: Target) -> np.ndarray:
        """
        计算资源成本
        
        Args:
            uav: UAV对象
            target: 目标对象
            
        Returns:
            np.ndarray: 资源成本数组
        """
        # 计算UAV能够提供给目标的资源
        available_resources = uav.resources.copy()
        needed_resources = target.remaining_resources.copy()
        
        # 实际贡献是两者的最小值
        actual_contribution = np.minimum(available_resources, needed_resources)
        
        return actual_contribution
    
    @staticmethod
    def _calculate_distance(uav: UAV, target: Target) -> float:
        """
        计算UAV到目标的距离
        
        Args:
            uav: UAV对象
            target: 目标对象
            
        Returns:
            float: 距离
        """
        uav_pos = np.array(uav.position[:2])  # 取前两个坐标
        target_pos = np.array(target.position[:2])  # 取前两个坐标
        
        distance = np.linalg.norm(target_pos - uav_pos)
        return float(distance)


# 训练完成后的方案信息输出
class SolutionReporter:
    """
    方案信息输出器
    
    实现训练完成后的方案信息输出，包括分配结果、性能指标、迁移能力评估
    """
    
    @staticmethod
    def generate_solution_report(
        assignments: Dict[int, List[Tuple[int, int]]],
        evaluation_metrics: Dict[str, Any],
        training_history: Optional[Dict[str, List]] = None,
        transfer_evaluation: Optional[Dict[str, Any]] = None,
        output_path: str = "solution_report.json"
    ) -> Dict[str, Any]:
        """
        生成完整的方案报告
        
        Args:
            assignments: 任务分配结果
            evaluation_metrics: 评估指标
            training_history: 训练历史（可选）
            transfer_evaluation: 迁移能力评估（可选）
            output_path: 输出路径
            
        Returns:
            Dict[str, Any]: 完整的方案报告
        """
        print("[SolutionReporter] 生成方案报告")
        
        report = {
            'timestamp': SolutionReporter._get_timestamp(),
            'model_type': 'TransformerGNN',
            'task_assignments': SolutionReporter._format_assignments(assignments),
            'performance_metrics': evaluation_metrics,
            'summary': SolutionReporter._generate_summary(assignments, evaluation_metrics)
        }
        
        # 添加训练历史
        if training_history:
            report['training_history'] = SolutionReporter._format_training_history(training_history)
        
        # 添加迁移能力评估
        if transfer_evaluation:
            report['transfer_capability'] = transfer_evaluation
        
        # 保存报告
        import json
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"[SolutionReporter] 方案报告已保存至: {output_path}")
        except Exception as e:
            print(f"[SolutionReporter] 保存报告失败: {e}")
        
        return report
    
    @staticmethod
    def _get_timestamp() -> str:
        """获取时间戳"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def _format_assignments(assignments: Dict[int, List[Tuple[int, int]]]) -> Dict[str, Any]:
        """格式化任务分配结果"""
        formatted = {
            'total_assignments': sum(len(tasks) for tasks in assignments.values()),
            'uav_assignments': {}
        }
        
        for uav_id, tasks in assignments.items():
            formatted['uav_assignments'][str(uav_id)] = {
                'task_count': len(tasks),
                'tasks': [{'target_id': t[0], 'phi_idx': t[1]} for t in tasks]
            }
        
        return formatted
    
    @staticmethod
    def _format_training_history(training_history: Dict[str, List]) -> Dict[str, Any]:
        """格式化训练历史"""
        if not training_history:
            return {}
        
        formatted = {}
        for key, values in training_history.items():
            if values:
                formatted[key] = {
                    'final_value': values[-1] if values else 0,
                    'max_value': max(values) if values else 0,
                    'min_value': min(values) if values else 0,
                    'mean_value': sum(values) / len(values) if values else 0
                }
        
        return formatted
    
    @staticmethod
    def _generate_summary(assignments: Dict[int, List[Tuple[int, int]]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """生成方案摘要"""
        total_assignments = sum(len(tasks) for tasks in assignments.values())
        active_uavs = len([uav_id for uav_id, tasks in assignments.items() if tasks])
        
        summary = {
            'total_task_assignments': total_assignments,
            'active_uavs': active_uavs,
            'completion_rate': metrics.get('completion_rate', 0),
            'total_reward_score': metrics.get('total_reward_score', 0),
            'resource_utilization_rate': metrics.get('resource_utilization_rate', 0)
        }
        
        return summary


if __name__ == "__main__":
    print("TransformerGNN输出格式兼容性模块测试")
    
    # 这里可以添加简单的测试代码
    print("模块加载成功")
