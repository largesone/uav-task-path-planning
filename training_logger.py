"""
训练数据保存与TensorBoard集成模块
实现课程学习各阶段的训练指标记录、模型检查点保存和可视化支持
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
try:
    from ray.rllib.evaluation.episode import Episode
except ImportError:
    from ray.rllib.evaluation.episode_v2 import EpisodeV2 as Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class CurriculumTensorBoardLogger:
    """
    课程学习专用的TensorBoard日志记录器
    支持尺度不变指标记录和课程学习进度可视化
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "curriculum_training"):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化TensorBoard writer
        tensorboard_dir = self.log_dir / "tensorboard" / experiment_name
        self.writer = SummaryWriter(str(tensorboard_dir))
        
        # 训练历史记录
        self.training_history = {
            "stages": [],
            "metrics": [],
            "stage_transitions": [],
            "rollback_events": []
        }
        
        # 当前阶段信息
        self.current_stage = 0
        self.stage_start_step = 0
        
        print(f"TensorBoard日志初始化完成，日志目录: {tensorboard_dir}")
    
    def log_stage_transition(self, from_stage: int, to_stage: int, step: int, 
                           reason: str = "performance_threshold"):
        """
        记录课程学习阶段切换事件
        
        Args:
            from_stage: 源阶段
            to_stage: 目标阶段
            step: 当前训练步数
            reason: 切换原因
        """
        transition_info = {
            "timestamp": datetime.now().isoformat(),
            "from_stage": from_stage,
            "to_stage": to_stage,
            "step": step,
            "reason": reason
        }
        
        self.training_history["stage_transitions"].append(transition_info)
        
        # 记录到TensorBoard
        self.writer.add_scalar("Curriculum/Current_Stage", to_stage, step)
        self.writer.add_text("Curriculum/Stage_Transition", 
                           f"阶段 {from_stage} -> {to_stage} (原因: {reason})", step)
        
        # 更新当前阶段信息
        self.current_stage = to_stage
        self.stage_start_step = step
        
        print(f"阶段切换记录: {from_stage} -> {to_stage} (步数: {step}, 原因: {reason})")
    
    def log_rollback_event(self, stage: int, step: int, performance_drop: float, 
                          threshold: float):
        """
        记录课程学习回退事件
        
        Args:
            stage: 回退的阶段
            step: 当前训练步数
            performance_drop: 性能下降幅度
            threshold: 回退阈值
        """
        rollback_info = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "step": step,
            "performance_drop": performance_drop,
            "threshold": threshold
        }
        
        self.training_history["rollback_events"].append(rollback_info)
        
        # 记录到TensorBoard
        self.writer.add_scalar("Curriculum/Rollback_Events", 1, step)
        self.writer.add_scalar("Curriculum/Performance_Drop", performance_drop, step)
        self.writer.add_text("Curriculum/Rollback_Reason", 
                           f"阶段 {stage} 性能下降 {performance_drop:.3f} > 阈值 {threshold:.3f}", 
                           step)
        
        print(f"回退事件记录: 阶段 {stage}, 性能下降 {performance_drop:.3f}")
    
    def log_scale_invariant_metrics(self, metrics: Dict[str, float], step: int, 
                                   stage: int, n_uavs: int, n_targets: int):
        """
        记录尺度不变的评价指标
        
        Args:
            metrics: 指标字典，包含per_agent_reward, normalized_completion_score, efficiency_metric
            step: 训练步数
            stage: 当前阶段
            n_uavs: 无人机数量
            n_targets: 目标数量
        """
        # 记录尺度不变指标
        if "per_agent_reward" in metrics:
            self.writer.add_scalar("Metrics/Per_Agent_Reward", 
                                 metrics["per_agent_reward"], step)
        
        if "normalized_completion_score" in metrics:
            self.writer.add_scalar("Metrics/Normalized_Completion_Score", 
                                 metrics["normalized_completion_score"], step)
        
        if "efficiency_metric" in metrics:
            self.writer.add_scalar("Metrics/Efficiency_Metric", 
                                 metrics["efficiency_metric"], step)
        
        # 记录场景规模信息
        self.writer.add_scalar("Scenario/N_UAVs", n_uavs, step)
        self.writer.add_scalar("Scenario/N_Targets", n_targets, step)
        self.writer.add_scalar("Scenario/Scale_Factor", n_uavs * n_targets, step)
        
        # 记录阶段信息
        self.writer.add_scalar("Curriculum/Current_Stage", stage, step)
        self.writer.add_scalar("Curriculum/Steps_In_Stage", step - self.stage_start_step, step)
        
        # 保存到训练历史
        metric_record = {
            "step": step,
            "stage": stage,
            "n_uavs": n_uavs,
            "n_targets": n_targets,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.training_history["metrics"].append(metric_record)
        
        print(f"指标记录 (步数: {step}, 阶段: {stage}): {metrics}")
    
    def log_training_progress(self, episode_reward: float, episode_length: int, 
                            step: int, stage: int):
        """
        记录基础训练进度指标
        
        Args:
            episode_reward: 回合奖励
            episode_length: 回合长度
            step: 训练步数
            stage: 当前阶段
        """
        self.writer.add_scalar("Training/Episode_Reward", episode_reward, step)
        self.writer.add_scalar("Training/Episode_Length", episode_length, step)
        self.writer.add_scalar("Training/Learning_Rate", 
                             self._get_current_lr(), step)
        
        # 记录阶段进度
        stage_progress = (step - self.stage_start_step) / max(1, step)
        self.writer.add_scalar("Curriculum/Stage_Progress", stage_progress, step)
    
    def _get_current_lr(self) -> float:
        """获取当前学习率（占位符实现）"""
        return 0.001  # 实际实现中应该从训练器获取
    
    def save_training_history(self):
        """保存训练历史到文件"""
        history_file = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        print(f"训练历史已保存到: {history_file}")
    
    def close(self):
        """关闭日志记录器"""
        self.save_training_history()
        self.writer.close()
        print("TensorBoard日志记录器已关闭")


class ModelCheckpointManager:
    """
    模型检查点管理器
    负责保存和加载训练过程中的模型检查点
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保存检查点数量
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        
        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点历史记录
        self.checkpoint_history = []
        
        print(f"模型检查点管理器初始化完成，目录: {checkpoint_dir}")
    
    def save_checkpoint(self, model_state: Dict[str, Any], optimizer_state: Dict[str, Any],
                       metrics: Dict[str, float], step: int, stage: int, 
                       is_best: bool = False) -> str:
        """
        保存模型检查点
        
        Args:
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            metrics: 当前性能指标
            step: 训练步数
            stage: 当前阶段
            is_best: 是否为最佳模型
            
        Returns:
            检查点文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            checkpoint_name = f"best_stage_{stage}_step_{step}_{timestamp}.pt"
        else:
            checkpoint_name = f"checkpoint_stage_{stage}_step_{step}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 准备检查点数据
        checkpoint_data = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
            "step": step,
            "stage": stage,
            "timestamp": timestamp,
            "is_best": is_best
        }
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        
        # 更新历史记录
        checkpoint_info = {
            "path": str(checkpoint_path),
            "step": step,
            "stage": stage,
            "timestamp": timestamp,
            "is_best": is_best,
            "metrics": metrics.copy()
        }
        self.checkpoint_history.append(checkpoint_info)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        print(f"检查点已保存: {checkpoint_name}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点数据字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        print(f"检查点已加载: {checkpoint_path}")
        
        return checkpoint_data
    
    def get_best_checkpoint(self, stage: Optional[int] = None) -> Optional[str]:
        """
        获取最佳检查点路径
        
        Args:
            stage: 指定阶段，None表示全局最佳
            
        Returns:
            最佳检查点路径，如果不存在则返回None
        """
        best_checkpoints = [cp for cp in self.checkpoint_history if cp["is_best"]]
        
        if stage is not None:
            best_checkpoints = [cp for cp in best_checkpoints if cp["stage"] == stage]
        
        if not best_checkpoints:
            return None
        
        # 按性能指标排序（假设使用normalized_completion_score）
        best_checkpoints.sort(
            key=lambda x: x["metrics"].get("normalized_completion_score", 0), 
            reverse=True
        )
        
        return best_checkpoints[0]["path"]
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        # 保留最佳检查点和最近的检查点
        non_best_checkpoints = [cp for cp in self.checkpoint_history if not cp["is_best"]]
        
        if len(non_best_checkpoints) > self.max_checkpoints:
            # 按时间排序，删除最旧的检查点
            non_best_checkpoints.sort(key=lambda x: x["timestamp"])
            
            to_remove = non_best_checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                try:
                    os.remove(checkpoint["path"])
                    self.checkpoint_history.remove(checkpoint)
                    print(f"已删除旧检查点: {checkpoint['path']}")
                except OSError as e:
                    print(f"删除检查点失败: {e}")
    
    def save_checkpoint_history(self):
        """保存检查点历史记录"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint_history, f, indent=2, ensure_ascii=False)
        
        print(f"检查点历史已保存到: {history_file}")


class CurriculumTrainingCallbacks(DefaultCallbacks):
    """
    课程学习训练回调函数
    集成TensorBoard日志记录和模型检查点保存
    """
    
    def __init__(self):
        super().__init__()
        self.logger = None
        self.checkpoint_manager = None
        self.best_metrics = {}
        
    def on_algorithm_init(self, *, algorithm, **kwargs):
        """算法初始化时的回调"""
        # 初始化日志记录器和检查点管理器
        log_dir = algorithm.config.get("log_dir", "./logs")
        experiment_name = algorithm.config.get("experiment_name", "curriculum_training")
        
        self.logger = CurriculumTensorBoardLogger(log_dir, experiment_name)
        
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        self.checkpoint_manager = ModelCheckpointManager(checkpoint_dir)
        
        print("课程学习训练回调函数初始化完成")
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                      policies: Dict[PolicyID, Policy], episode: Episode, **kwargs):
        """回合结束时的回调"""
        if self.logger is None:
            return
        
        # 获取回合信息
        episode_reward = episode.total_reward
        episode_length = episode.length
        
        # 从环境获取当前场景信息
        env = base_env.get_sub_environments()[0]
        current_stage = getattr(env, 'current_stage', 0)
        n_uavs = getattr(env, 'n_uavs', 0)
        n_targets = getattr(env, 'n_targets', 0)
        
        # 计算尺度不变指标
        metrics = self._calculate_scale_invariant_metrics(episode, env)
        
        # 记录到TensorBoard
        step = worker.global_vars.get("timestep", 0)
        self.logger.log_scale_invariant_metrics(metrics, step, current_stage, n_uavs, n_targets)
        self.logger.log_training_progress(episode_reward, episode_length, step, current_stage)
    
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """训练结果回调"""
        if self.checkpoint_manager is None:
            return
        
        # 获取训练指标
        step = result.get("timesteps_total", 0)
        stage = result.get("custom_metrics", {}).get("current_stage", 0)
        
        # 计算当前性能指标
        current_metrics = {
            "episode_reward_mean": result.get("episode_reward_mean", 0),
            "per_agent_reward": result.get("custom_metrics", {}).get("per_agent_reward", 0),
            "normalized_completion_score": result.get("custom_metrics", {}).get("normalized_completion_score", 0),
            "efficiency_metric": result.get("custom_metrics", {}).get("efficiency_metric", 0)
        }
        
        # 判断是否为最佳模型
        is_best = self._is_best_model(current_metrics, stage)
        
        # 保存检查点
        if step % 1000 == 0 or is_best:  # 每1000步或最佳模型时保存
            model_state = algorithm.get_policy().get_state()
            optimizer_state = {}  # 实际实现中应该获取优化器状态
            
            self.checkpoint_manager.save_checkpoint(
                model_state, optimizer_state, current_metrics, 
                step, stage, is_best
            )
    
    def _calculate_scale_invariant_metrics(self, episode: Episode, env) -> Dict[str, float]:
        """计算尺度不变指标"""
        # 获取环境信息
        n_active_uavs = getattr(env, 'n_active_uavs', 1)
        total_targets = getattr(env, 'n_targets', 1)
        completed_targets = getattr(env, 'completed_targets', 0)
        total_flight_distance = getattr(env, 'total_flight_distance', 1)
        congestion_metric = getattr(env, 'average_congestion', 0)
        
        # 计算尺度不变指标
        per_agent_reward = episode.total_reward / max(1, n_active_uavs)
        
        completion_rate = completed_targets / max(1, total_targets)
        normalized_completion_score = completion_rate * (1 - congestion_metric)
        
        efficiency_metric = completed_targets / max(1, total_flight_distance)
        
        return {
            "per_agent_reward": per_agent_reward,
            "normalized_completion_score": normalized_completion_score,
            "efficiency_metric": efficiency_metric
        }
    
    def _is_best_model(self, current_metrics: Dict[str, float], stage: int) -> bool:
        """判断是否为最佳模型"""
        stage_key = f"stage_{stage}"
        
        if stage_key not in self.best_metrics:
            self.best_metrics[stage_key] = current_metrics.copy()
            return True
        
        # 使用normalized_completion_score作为主要判断标准
        current_score = current_metrics.get("normalized_completion_score", 0)
        best_score = self.best_metrics[stage_key].get("normalized_completion_score", 0)
        
        if current_score > best_score:
            self.best_metrics[stage_key] = current_metrics.copy()
            return True
        
        return False
    
    def on_algorithm_exit(self, *, algorithm, **kwargs):
        """算法退出时的回调"""
        if self.logger:
            self.logger.close()
        
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint_history()
        
        print("课程学习训练回调函数已清理")


def create_training_config_with_logging(base_config: Dict[str, Any], 
                                      log_dir: str = "./logs",
                                      experiment_name: str = "curriculum_training") -> Dict[str, Any]:
    """
    创建包含日志记录配置的训练配置
    
    Args:
        base_config: 基础训练配置
        log_dir: 日志目录
        experiment_name: 实验名称
        
    Returns:
        增强的训练配置
    """
    config = base_config.copy()
    
    # 添加日志配置
    config.update({
        "log_dir": log_dir,
        "experiment_name": experiment_name,
        "callbacks": CurriculumTrainingCallbacks,
        
        # TensorBoard配置
        "logger_config": {
            "type": "ray.tune.logger.TBXLogger",
            "config": {
                "logdir": os.path.join(log_dir, "tensorboard")
            }
        },
        
        # 自定义指标配置
        "custom_metrics_smoothing_window": 100,
        "metrics_episode_collection_timeout_s": 60,
        
        # 检查点配置
        "checkpoint_freq": 10,
        "checkpoint_at_end": True,
        "keep_checkpoints_num": 5,
    })
    
    return config


if __name__ == "__main__":
    # 测试代码
    print("训练数据保存与TensorBoard集成模块测试")
    
    # 测试TensorBoard日志记录器
    logger = CurriculumTensorBoardLogger("./test_logs", "test_experiment")
    
    # 模拟训练过程
    for step in range(100):
        stage = step // 30
        
        # 模拟指标
        metrics = {
            "per_agent_reward": np.random.normal(10, 2),
            "normalized_completion_score": np.random.uniform(0.5, 1.0),
            "efficiency_metric": np.random.uniform(0.1, 0.5)
        }
        
        logger.log_scale_invariant_metrics(metrics, step, stage, 5, 3)
        
        # 模拟阶段切换
        if step in [30, 60, 90]:
            logger.log_stage_transition(stage-1, stage, step)
    
    logger.close()
    
    # 测试检查点管理器
    checkpoint_manager = ModelCheckpointManager("./test_checkpoints")
    
    # 模拟保存检查点
    for i in range(3):
        model_state = {"layer1.weight": torch.randn(10, 5)}
        optimizer_state = {"lr": 0.001}
        metrics = {"score": np.random.uniform(0.5, 1.0)}
        
        checkpoint_manager.save_checkpoint(
            model_state, optimizer_state, metrics, 
            i*100, i, is_best=(i==2)
        )
    
    print("测试完成")
