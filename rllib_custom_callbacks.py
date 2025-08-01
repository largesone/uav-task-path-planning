# -*- coding: utf-8 -*-
"""
Ray RLlib自定义回调函数
实现尺度不变指标计算和TensorBoard集成
"""

import numpy as np
from typing import Dict, Any, Optional
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import logging

logger = logging.getLogger(__name__)


class ScaleInvariantMetricsCallbacks(DefaultCallbacks):
    """
    尺度不变指标计算的自定义回调函数
    
    实现以下指标:
    1. Per-Agent Reward: total_reward / N_active
    2. Normalized Completion Score: satisfied_targets_rate * (1 - average_congestion_metric)
    3. Efficiency Metric: total_completed_targets / total_flight_distance
    """
    
    def __init__(self):
        super().__init__()
        self.episode_metrics_history = []
        self.stage_metrics_accumulator = {}
        
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[PolicyID, Policy], episode: EpisodeV2,
                        env_index: Optional[int] = None, **kwargs) -> None:
        """
        回合开始时的回调
        初始化指标计算所需的状态
        """
        # 获取环境实例
        env = base_env.get_sub_environments()[env_index or 0]
        
        # 初始化回合级别的指标累积器
        episode.custom_metrics.update({
            "scale_invariant_metrics": {
                "per_agent_reward": 0.0,
                "normalized_completion_score": 0.0,
                "efficiency_metric": 0.0,
                "n_active_uavs": 0,
                "total_flight_distance": 0.0,
                "completed_targets": 0,
                "satisfied_targets_rate": 0.0,
                "average_congestion_metric": 0.0,
                "step_count": 0
            }
        })
        
        # 记录初始状态
        if hasattr(env, 'uavs') and hasattr(env, 'targets'):
            episode.custom_metrics["scale_invariant_metrics"]["n_active_uavs"] = len([
                uav for uav in env.uavs if np.any(uav.resources > 0)
            ])
            episode.custom_metrics["scale_invariant_metrics"]["total_targets"] = len(env.targets)
            
            logger.debug(f"回合开始 - 活跃UAV数量: {episode.custom_metrics['scale_invariant_metrics']['n_active_uavs']}, "
                        f"目标数量: {episode.custom_metrics['scale_invariant_metrics']['total_targets']}")
    
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy], episode: EpisodeV2,
                       env_index: Optional[int] = None, **kwargs) -> None:
        """
        每步执行后的回调
        累积计算尺度不变指标
        """
        # 获取环境实例
        env = base_env.get_sub_environments()[env_index or 0]
        
        if not hasattr(env, 'uavs') or not hasattr(env, 'targets'):
            return
        
        # 更新步数计数
        episode.custom_metrics["scale_invariant_metrics"]["step_count"] += 1
        
        # 计算当前活跃UAV数量
        n_active_uavs = len([uav for uav in env.uavs if np.any(uav.resources > 0)])
        episode.custom_metrics["scale_invariant_metrics"]["n_active_uavs"] = n_active_uavs
        
        # 累积飞行距离
        total_flight_distance = 0.0
        for uav in env.uavs:
            if hasattr(uav, 'task_sequence') and len(uav.task_sequence) > 0:
                # 计算UAV的总飞行距离
                current_pos = np.array(uav.current_position)
                initial_pos = np.array(uav.position)
                total_flight_distance += np.linalg.norm(current_pos - initial_pos)
        
        episode.custom_metrics["scale_invariant_metrics"]["total_flight_distance"] = total_flight_distance
        
        # 计算完成的目标数量
        completed_targets = sum(1 for target in env.targets 
                               if np.all(target.remaining_resources <= 0))
        episode.custom_metrics["scale_invariant_metrics"]["completed_targets"] = completed_targets
        
        # 计算目标满足率
        total_targets = len(env.targets)
        satisfied_targets_rate = completed_targets / total_targets if total_targets > 0 else 0.0
        episode.custom_metrics["scale_invariant_metrics"]["satisfied_targets_rate"] = satisfied_targets_rate
        
        # 计算平均拥堵指标
        average_congestion_metric = self._calculate_congestion_metric(env)
        episode.custom_metrics["scale_invariant_metrics"]["average_congestion_metric"] = average_congestion_metric
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                      policies: Dict[PolicyID, Policy], episode: Episode,
                      env_index: Optional[int] = None, **kwargs) -> None:
        """
        回合结束时的回调
        计算最终的尺度不变指标并记录到TensorBoard
        """
        metrics = episode.custom_metrics["scale_invariant_metrics"]
        
        # 1. 计算Per-Agent Reward
        total_reward = episode.total_reward
        n_active_uavs = max(metrics["n_active_uavs"], 1)  # 避免除零
        per_agent_reward = total_reward / n_active_uavs
        
        # 2. 计算Normalized Completion Score
        satisfied_targets_rate = metrics["satisfied_targets_rate"]
        average_congestion_metric = metrics["average_congestion_metric"]
        normalized_completion_score = satisfied_targets_rate * (1 - average_congestion_metric)
        
        # 3. 计算Efficiency Metric
        completed_targets = metrics["completed_targets"]
        total_flight_distance = max(metrics["total_flight_distance"], 1e-6)  # 避免除零
        efficiency_metric = completed_targets / total_flight_distance
        
        # 更新最终指标
        episode.custom_metrics["per_agent_reward"] = per_agent_reward
        episode.custom_metrics["normalized_completion_score"] = normalized_completion_score
        episode.custom_metrics["efficiency_metric"] = efficiency_metric
        
        # 记录详细的调试信息
        episode.custom_metrics.update({
            "debug_info": {
                "total_reward": total_reward,
                "n_active_uavs": n_active_uavs,
                "satisfied_targets_rate": satisfied_targets_rate,
                "average_congestion_metric": average_congestion_metric,
                "completed_targets": completed_targets,
                "total_flight_distance": total_flight_distance,
                "episode_length": episode.length
            }
        })
        
        # 保存到历史记录
        episode_summary = {
            "episode_id": episode.episode_id,
            "per_agent_reward": per_agent_reward,
            "normalized_completion_score": normalized_completion_score,
            "efficiency_metric": efficiency_metric,
            "episode_length": episode.length,
            "total_reward": total_reward,
            "n_active_uavs": n_active_uavs
        }
        self.episode_metrics_history.append(episode_summary)
        
        logger.info(f"回合结束 - Per-Agent Reward: {per_agent_reward:.4f}, "
                   f"Normalized Completion Score: {normalized_completion_score:.4f}, "
                   f"Efficiency Metric: {efficiency_metric:.6f}")
    
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """
        训练结果回调
        将尺度不变指标添加到训练结果中，用于TensorBoard记录
        """
        if not self.episode_metrics_history:
            return
        
        # 计算最近N个回合的平均指标
        recent_episodes = self.episode_metrics_history[-10:]  # 最近10个回合
        
        if recent_episodes:
            avg_per_agent_reward = np.mean([ep["per_agent_reward"] for ep in recent_episodes])
            avg_normalized_completion_score = np.mean([ep["normalized_completion_score"] for ep in recent_episodes])
            avg_efficiency_metric = np.mean([ep["efficiency_metric"] for ep in recent_episodes])
            
            # 添加到训练结果中
            result["custom_metrics"].update({
                "scale_invariant_per_agent_reward_mean": avg_per_agent_reward,
                "scale_invariant_normalized_completion_score_mean": avg_normalized_completion_score,
                "scale_invariant_efficiency_metric_mean": avg_efficiency_metric,
                
                # 添加标准差信息
                "scale_invariant_per_agent_reward_std": np.std([ep["per_agent_reward"] for ep in recent_episodes]),
                "scale_invariant_normalized_completion_score_std": np.std([ep["normalized_completion_score"] for ep in recent_episodes]),
                "scale_invariant_efficiency_metric_std": np.std([ep["efficiency_metric"] for ep in recent_episodes]),
                
                # 添加趋势信息
                "scale_invariant_metrics_trend": self._calculate_metrics_trend(recent_episodes)
            })
            
            logger.debug(f"训练结果更新 - 平均Per-Agent Reward: {avg_per_agent_reward:.4f}, "
                        f"平均Normalized Completion Score: {avg_normalized_completion_score:.4f}")
    
    def _calculate_congestion_metric(self, env) -> float:
        """
        计算平均拥堵指标
        
        Args:
            env: 环境实例
            
        Returns:
            float: 平均拥堵指标 [0, 1]，0表示无拥堵，1表示严重拥堵
        """
        if not hasattr(env, 'targets'):
            return 0.0
        
        congestion_scores = []
        
        for target in env.targets:
            if hasattr(target, 'allocated_uavs'):
                # 计算分配到该目标的UAV数量
                allocated_count = len(target.allocated_uavs)
                
                # 计算理想分配数量（基于目标资源需求）
                if hasattr(target, 'resources'):
                    total_demand = np.sum(target.resources)
                    # 假设每个UAV平均能提供的资源
                    avg_uav_capacity = 50.0  # 可以根据实际情况调整
                    ideal_allocation = max(1, int(np.ceil(total_demand / avg_uav_capacity)))
                    
                    # 计算拥堵程度：实际分配数 / 理想分配数
                    if ideal_allocation > 0:
                        congestion_ratio = allocated_count / ideal_allocation
                        # 将拥堵比例映射到[0, 1]范围，超过2倍理想分配视为完全拥堵
                        congestion_score = min(1.0, max(0.0, (congestion_ratio - 1.0) / 1.0))
                        congestion_scores.append(congestion_score)
        
        return np.mean(congestion_scores) if congestion_scores else 0.0
    
    def _calculate_metrics_trend(self, recent_episodes: list) -> float:
        """
        计算指标趋势
        
        Args:
            recent_episodes: 最近的回合列表
            
        Returns:
            float: 趋势值，正数表示上升趋势，负数表示下降趋势
        """
        if len(recent_episodes) < 3:
            return 0.0
        
        # 使用normalized_completion_score作为主要趋势指标
        scores = [ep["normalized_completion_score"] for ep in recent_episodes]
        
        # 计算简单的线性趋势
        x = np.arange(len(scores))
        trend = np.polyfit(x, scores, 1)[0]  # 线性回归的斜率
        
        return float(trend)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Returns:
            Dict: 指标摘要信息
        """
        if not self.episode_metrics_history:
            return {}
        
        all_episodes = self.episode_metrics_history
        
        return {
            "total_episodes": len(all_episodes),
            "avg_per_agent_reward": np.mean([ep["per_agent_reward"] for ep in all_episodes]),
            "avg_normalized_completion_score": np.mean([ep["normalized_completion_score"] for ep in all_episodes]),
            "avg_efficiency_metric": np.mean([ep["efficiency_metric"] for ep in all_episodes]),
            "best_per_agent_reward": max([ep["per_agent_reward"] for ep in all_episodes]),
            "best_normalized_completion_score": max([ep["normalized_completion_score"] for ep in all_episodes]),
            "best_efficiency_metric": max([ep["efficiency_metric"] for ep in all_episodes])
        }


class CurriculumLearningCallbacks(ScaleInvariantMetricsCallbacks):
    """
    课程学习专用回调函数
    继承尺度不变指标计算，并添加课程学习相关功能
    """
    
    def __init__(self):
        super().__init__()
        self.current_stage = 0
        self.stage_episode_count = 0
        self.stage_performance_history = {}
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                      policies: Dict[PolicyID, Policy], episode: Episode,
                      env_index: Optional[int] = None, **kwargs) -> None:
        """
        扩展回合结束回调，添加课程学习相关指标
        """
        # 调用父类方法计算尺度不变指标
        super().on_episode_end(worker=worker, base_env=base_env, 
                              policies=policies, episode=episode, 
                              env_index=env_index, **kwargs)
        
        # 添加课程学习相关指标
        self.stage_episode_count += 1
        
        # 记录当前阶段信息
        episode.custom_metrics.update({
            "curriculum_stage": self.current_stage,
            "stage_episode_count": self.stage_episode_count,
            "stage_progress": self._calculate_stage_progress()
        })
        
        # 更新阶段性能历史
        if self.current_stage not in self.stage_performance_history:
            self.stage_performance_history[self.current_stage] = []
        
        stage_metrics = {
            "per_agent_reward": episode.custom_metrics["per_agent_reward"],
            "normalized_completion_score": episode.custom_metrics["normalized_completion_score"],
            "efficiency_metric": episode.custom_metrics["efficiency_metric"],
            "episode_count": self.stage_episode_count
        }
        
        self.stage_performance_history[self.current_stage].append(stage_metrics)
        
        logger.info(f"课程学习 - 阶段: {self.current_stage}, "
                   f"阶段回合数: {self.stage_episode_count}, "
                   f"阶段进度: {self._calculate_stage_progress():.2f}")
    
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """
        扩展训练结果回调，添加课程学习指标
        """
        # 调用父类方法
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        
        # 添加课程学习相关指标
        result["custom_metrics"].update({
            "curriculum_current_stage": self.current_stage,
            "curriculum_stage_episode_count": self.stage_episode_count,
            "curriculum_stage_progress": self._calculate_stage_progress(),
            "curriculum_total_stages_completed": len([s for s in self.stage_performance_history.keys() if self._is_stage_completed(s)])
        })
        
        # 计算各阶段的平均性能
        for stage_id, stage_history in self.stage_performance_history.items():
            if stage_history:
                recent_stage_episodes = stage_history[-5:]  # 最近5个回合
                avg_stage_performance = np.mean([ep["normalized_completion_score"] for ep in recent_stage_episodes])
                result["custom_metrics"][f"curriculum_stage_{stage_id}_avg_performance"] = avg_stage_performance
    
    def advance_to_next_stage(self):
        """
        推进到下一个课程学习阶段
        """
        self.current_stage += 1
        self.stage_episode_count = 0
        
        logger.info(f"课程学习阶段推进: 进入阶段 {self.current_stage}")
    
    def rollback_to_previous_stage(self):
        """
        回退到上一个课程学习阶段
        """
        if self.current_stage > 0:
            self.current_stage -= 1
            self.stage_episode_count = 0
            
            logger.warning(f"课程学习阶段回退: 回到阶段 {self.current_stage}")
    
    def _calculate_stage_progress(self) -> float:
        """
        计算当前阶段的进度
        
        Returns:
            float: 阶段进度 [0, 1]
        """
        # 简单的基于回合数的进度计算
        max_episodes_per_stage = 100  # 可以根据实际情况调整
        progress = min(1.0, self.stage_episode_count / max_episodes_per_stage)
        return progress
    
    def _is_stage_completed(self, stage_id: int) -> bool:
        """
        判断指定阶段是否已完成
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            bool: 是否已完成
        """
        if stage_id not in self.stage_performance_history:
            return False
        
        stage_history = self.stage_performance_history[stage_id]
        
        # 简单的完成判断：最近10个回合的平均性能超过阈值
        if len(stage_history) >= 10:
            recent_performance = np.mean([ep["normalized_completion_score"] for ep in stage_history[-10:]])
            return recent_performance >= 0.8  # 80%的完成阈值
        
        return False
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """
        获取课程学习摘要
        
        Returns:
            Dict: 课程学习摘要信息
        """
        summary = {
            "current_stage": self.current_stage,
            "stage_episode_count": self.stage_episode_count,
            "total_stages_attempted": len(self.stage_performance_history),
            "completed_stages": [s for s in self.stage_performance_history.keys() if self._is_stage_completed(s)],
            "stage_performance_summary": {}
        }
        
        # 添加各阶段性能摘要
        for stage_id, stage_history in self.stage_performance_history.items():
            if stage_history:
                summary["stage_performance_summary"][stage_id] = {
                    "episodes": len(stage_history),
                    "avg_performance": np.mean([ep["normalized_completion_score"] for ep in stage_history]),
                    "best_performance": max([ep["normalized_completion_score"] for ep in stage_history]),
                    "completed": self._is_stage_completed(stage_id)
                }
        
        return summary

