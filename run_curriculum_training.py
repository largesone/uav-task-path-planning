# -*- coding: utf-8 -*-
"""
文件名: run_curriculum_training.py
描述: 课程学习训练协调器 - 最高级别训练协调器
作者: AI Assistant
日期: 2024

核心功能:
1. 管理多阶段训练流程，从简单场景逐步过渡到复杂场景
2. 实现训练阶段配置管理，支持不同复杂度场景的渐进式训练
3. 集成Ray RLlib的训练接口，确保与现有训练流程兼容
4. 提供回退机制防止灾难性发散
5. 实现混合经验回放策略
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Ray RLlib imports
try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    from ray.rllib.algorithms.dqn import DQN, DQNConfig
    from ray.rllib.models import ModelCatalog
    from ray.rllib.env.env_context import EnvContext
    from ray.rllib.utils.framework import try_import_torch
    RLLIB_AVAILABLE = True
except ImportError:
    print("警告: Ray RLlib未安装，将使用兼容模式")
    RLLIB_AVAILABLE = False

# 本地模块导入
from config import Config
from environment import UAVTaskEnv, DirectedGraph
from entities import UAV, Target
from scenarios import (
    get_small_scenario, get_balanced_scenario, 
    get_complex_scenario, get_new_experimental_scenario
)
from transformer_gnn import TransformerGNN

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CurriculumStage:
    """课程学习阶段配置"""
    name: str
    description: str
    scenario_func: callable
    scenario_params: Dict[str, Any]
    max_episodes: int
    success_threshold: float = 0.8
    min_episodes: int = 100
    performance_window: int = 50
    
    # 训练参数
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    
    # 回退机制参数
    fallback_threshold: float = 0.6
    fallback_patience: int = 3

@dataclass
class CurriculumConfig:
    """课程学习配置"""
    stages: List[CurriculumStage] = field(default_factory=list)
    mixed_replay_ratio: float = 0.3  # 旧阶段经验比例
    evaluation_frequency: int = 10   # 评估频率（每N个episode）
    checkpoint_frequency: int = 100  # 检查点保存频率
    output_dir: str = "curriculum_output"
    
    # Ray RLlib配置
    num_workers: int = 4
    num_envs_per_worker: int = 1
    framework: str = "torch"
    
    def __post_init__(self):
        """初始化默认课程阶段"""
        if not self.stages:
            self.stages = self._create_default_curriculum()
    
    def _create_default_curriculum(self) -> List[CurriculumStage]:
        """创建默认课程学习阶段"""
        return [
            CurriculumStage(
                name="stage_1_simple",
                description="简单场景：2无人机，3目标，无障碍物",
                scenario_func=get_small_scenario,
                scenario_params={"obstacle_tolerance": 30.0},
                max_episodes=500,
                success_threshold=0.85,
                learning_rate=5e-4,
                batch_size=32
            ),
            CurriculumStage(
                name="stage_2_balanced",
                description="平衡场景：4无人机，6目标，少量障碍物",
                scenario_func=get_balanced_scenario,
                scenario_params={"obstacle_tolerance": 50.0},
                max_episodes=800,
                success_threshold=0.80,
                learning_rate=3e-4,
                batch_size=64
            ),
            CurriculumStage(
                name="stage_3_complex",
                description="复杂场景：6无人机，10目标，多障碍物",
                scenario_func=get_complex_scenario,
                scenario_params={"obstacle_tolerance": 80.0},
                max_episodes=1200,
                success_threshold=0.75,
                learning_rate=2e-4,
                batch_size=128
            ),
            CurriculumStage(
                name="stage_4_experimental",
                description="实验场景：8无人机，15目标，复杂障碍物",
                scenario_func=get_new_experimental_scenario,
                scenario_params={"obstacle_tolerance": 100.0},
                max_episodes=1500,
                success_threshold=0.70,
                learning_rate=1e-4,
                batch_size=128
            )
        ]

class CurriculumTrainer:
    """课程学习训练器"""
    
    def __init__(self, config: CurriculumConfig, base_config: Config):
        """
        初始化课程学习训练器
        
        Args:
            config: 课程学习配置
            base_config: 基础配置对象
        """
        self.curriculum_config = config
        self.base_config = base_config
        self.current_stage_idx = 0
        self.training_history = {}
        self.performance_history = {}
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化Ray（如果可用）
        if RLLIB_AVAILABLE:
            self._init_ray()
        
        logger.info(f"课程学习训练器初始化完成，共{len(config.stages)}个阶段")
    
    def _init_ray(self):
        """初始化Ray"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray初始化完成")
        
        # 注册自定义模型
        ModelCatalog.register_custom_model("transformer_gnn", TransformerGNN)
        logger.info("TransformerGNN模型已注册到Ray ModelCatalog")
    
    def train(self) -> Dict[str, Any]:
        """
        执行完整的课程学习训练
        
        Returns:
            Dict: 训练结果摘要
        """
        logger.info("开始课程学习训练")
        start_time = time.time()
        
        training_summary = {
            "stages_completed": 0,
            "total_episodes": 0,
            "total_training_time": 0,
            "final_performance": {},
            "stage_results": []
        }
        
        try:
            for stage_idx, stage in enumerate(self.curriculum_config.stages):
                self.current_stage_idx = stage_idx
                logger.info(f"开始阶段 {stage_idx + 1}/{len(self.curriculum_config.stages)}: {stage.name}")
                
                # 训练当前阶段
                stage_result = self._train_stage(stage, stage_idx)
                training_summary["stage_results"].append(stage_result)
                training_summary["total_episodes"] += stage_result["episodes_trained"]
                
                # 检查是否成功完成阶段
                if stage_result["success"]:
                    training_summary["stages_completed"] += 1
                    logger.info(f"阶段 {stage.name} 训练成功完成")
                else:
                    logger.warning(f"阶段 {stage.name} 训练未达到成功标准，但继续下一阶段")
                
                # 保存检查点
                self.checkpoint_manager.save_stage_checkpoint(stage_idx, stage_result)
        
        except Exception as e:
            logger.error(f"课程学习训练过程中发生错误: {e}")
            raise
        
        finally:
            training_summary["total_training_time"] = time.time() - start_time
            self._save_training_summary(training_summary)
            logger.info(f"课程学习训练完成，总耗时: {training_summary['total_training_time']:.2f}秒")
        
        return training_summary
    
    def _train_stage(self, stage: CurriculumStage, stage_idx: int) -> Dict[str, Any]:
        """
        训练单个课程阶段
        
        Args:
            stage: 课程阶段配置
            stage_idx: 阶段索引
            
        Returns:
            Dict: 阶段训练结果
        """
        logger.info(f"训练阶段: {stage.description}")
        
        # 创建环境
        env_config = self._create_env_config(stage)
        
        # 由于Ray RLlib API兼容性问题，暂时使用回退训练方法
        logger.info("由于Ray RLlib API兼容性问题，使用回退训练方法")
        return self._train_stage_fallback(stage, stage_idx, env_config)
    
    def _create_env_config(self, stage: CurriculumStage) -> Dict[str, Any]:
        """创建环境配置"""
        # 生成场景数据
        uavs, targets, obstacles = stage.scenario_func(**stage.scenario_params)
        
        return {
            "uavs": uavs,
            "targets": targets,
            "obstacles": obstacles,
            "config": self.base_config,
            "obs_mode": "graph"  # 使用图模式观测
        }
    
    def _train_stage_with_rllib(self, stage: CurriculumStage, stage_idx: int, env_config: Dict) -> Dict[str, Any]:
        """使用Ray RLlib训练阶段"""
        
        # 创建算法配置
        if hasattr(self.base_config, 'ALGORITHM') and self.base_config.ALGORITHM == 'PPO':
            config = PPOConfig()
        else:
            config = DQNConfig()
        
        # 配置算法参数 - 使用旧API栈以兼容custom_model
        config = config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        ).training(
            lr=stage.learning_rate,
            train_batch_size=stage.batch_size,
            gamma=stage.gamma,
            model={
                "custom_model": "transformer_gnn",
                "custom_model_config": {
                    "obs_mode": "graph",
                    "hidden_dim": 256,
                    "num_heads": 8,
                    "num_layers": 4
                }
            },
            replay_buffer_config={
                'type': 'MultiAgentPrioritizedReplayBuffer',
                'prioritized_replay_alpha': 0.6,
                'prioritized_replay_beta': 0.4,
                'prioritized_replay_eps': 1e-6,
            }
        ).environment(
            env=UAVTaskEnv,
            env_config=env_config
        ).env_runners(
            num_env_runners=self.curriculum_config.num_workers,
            num_envs_per_env_runner=self.curriculum_config.num_envs_per_worker
        ).framework(self.curriculum_config.framework)
        
        # 创建算法实例
        if hasattr(self.base_config, 'ALGORITHM') and self.base_config.ALGORITHM == 'PPO':
            algo = config.build(env=UAVTaskEnv)
        else:
            algo = config.build(env=UAVTaskEnv)
        
        # 训练循环
        stage_result = {
            "stage_name": stage.name,
            "episodes_trained": 0,
            "success": False,
            "final_performance": 0.0,
            "performance_history": [],
            "fallback_count": 0
        }
        
        consecutive_poor_performance = 0
        best_performance = 0.0
        
        try:
            for episode in range(stage.max_episodes):
                # 训练一个批次
                result = algo.train()
                
                # 记录性能
                episode_reward_mean = result.get("episode_reward_mean", 0)
                stage_result["performance_history"].append(episode_reward_mean)
                stage_result["episodes_trained"] = episode + 1
                
                # 评估性能
                if episode % self.curriculum_config.evaluation_frequency == 0:
                    performance = self._evaluate_performance(algo, env_config)
                    
                    logger.info(f"阶段 {stage.name}, Episode {episode}: 性能 = {performance:.4f}")
                    
                    # 更新最佳性能
                    if performance > best_performance:
                        best_performance = performance
                        consecutive_poor_performance = 0
                    else:
                        consecutive_poor_performance += 1
                    
                    # 检查回退条件
                    if (stage_idx > 0 and 
                        consecutive_poor_performance >= stage.fallback_patience and
                        performance < stage.fallback_threshold * best_performance):
                        
                        logger.warning(f"性能下降，触发回退机制")
                        stage_result["fallback_count"] += 1
                        
                        # 实施回退策略
                        self._implement_fallback_strategy(algo, stage_idx)
                        consecutive_poor_performance = 0
                    
                    # 检查成功条件
                    if (episode >= stage.min_episodes and 
                        performance >= stage.success_threshold):
                        stage_result["success"] = True
                        logger.info(f"阶段 {stage.name} 达到成功标准")
                        break
                
                # 保存检查点
                if episode % self.curriculum_config.checkpoint_frequency == 0:
                    checkpoint_path = f"{self.curriculum_config.output_dir}/stage_{stage_idx}_episode_{episode}"
                    algo.save(checkpoint_path)
        
        finally:
            stage_result["final_performance"] = best_performance
            algo.stop()
        
        return stage_result
    
    def _train_stage_fallback(self, stage: CurriculumStage, stage_idx: int, env_config: Dict) -> Dict[str, Any]:
        """回退训练方法（当Ray RLlib不可用时）"""
        logger.warning("使用回退训练方法（Ray RLlib不可用）")
        
        # 这里可以集成现有的训练逻辑
        from main import run_scenario
        
        # 创建场景数据
        uavs = env_config["uavs"]
        targets = env_config["targets"]
        obstacles = env_config["obstacles"]
        config = env_config["config"]
        
        # 更新配置参数
        config.update_training_params(
            episodes=stage.max_episodes,
            learning_rate=stage.learning_rate,
            batch_size=stage.batch_size
        )
        
        # 运行训练
        final_plan, training_time, training_history, evaluation_metrics = run_scenario(
            config, uavs, targets, obstacles, 
            scenario_name=f"curriculum_{stage.name}",
            network_type="TransformerGNN",
            save_visualization=True,
            show_visualization=False,
            output_base_dir=self.curriculum_config.output_dir
        )
        
        # 构建结果
        stage_result = {
            "stage_name": stage.name,
            "episodes_trained": len(training_history.get("episode_rewards", [])),
            "success": evaluation_metrics.get("completion_rate", 0) >= stage.success_threshold,
            "final_performance": evaluation_metrics.get("completion_rate", 0),
            "performance_history": training_history.get("episode_rewards", []),
            "fallback_count": 0,
            "training_time": training_time,
            "evaluation_metrics": evaluation_metrics
        }
        
        return stage_result
    
    def _evaluate_performance(self, algo, env_config: Dict) -> float:
        """评估当前模型性能"""
        # 创建评估环境
        eval_env = UAVTaskEnv(**env_config)
        
        total_reward = 0
        num_episodes = 5  # 评估轮数
        
        for _ in range(num_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = algo.compute_single_action(obs)
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if truncated:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def _implement_fallback_strategy(self, algo, stage_idx: int):
        """实施回退策略"""
        if stage_idx == 0:
            logger.info("已在第一阶段，无法回退，降低学习率")
            # 降低学习率
            current_lr = algo.get_policy().config.get("lr", 3e-4)
            new_lr = current_lr * 0.5
            algo.get_policy().config["lr"] = new_lr
            logger.info(f"学习率从 {current_lr} 降低到 {new_lr}")
        else:
            logger.info(f"回退到阶段 {stage_idx - 1}")
            # 加载上一阶段的检查点
            prev_checkpoint = self.checkpoint_manager.get_latest_checkpoint(stage_idx - 1)
            if prev_checkpoint:
                algo.restore(prev_checkpoint)
                logger.info(f"已恢复到检查点: {prev_checkpoint}")
            
            # 实施混合经验回放
            self._implement_mixed_replay(algo, stage_idx)
    
    def _implement_mixed_replay(self, algo, stage_idx: int):
        """实施混合经验回放策略"""
        logger.info("实施混合经验回放策略")
        
        # 这里需要根据具体的算法实现混合经验回放
        # 70%当前阶段经验 + 30%旧阶段经验
        
        try:
            # 获取当前经验回放缓冲区
            replay_buffer = algo.local_replay_buffer
            
            # 加载旧阶段经验（如果存在）
            old_experiences_path = f"{self.curriculum_config.output_dir}/stage_{stage_idx-1}_experiences.pkl"
            if os.path.exists(old_experiences_path):
                with open(old_experiences_path, 'rb') as f:
                    old_experiences = pickle.load(f)
                
                # 混合经验（简化实现）
                old_sample_size = int(len(old_experiences) * self.curriculum_config.mixed_replay_ratio)
                if old_sample_size > 0:
                    old_sample = np.random.choice(old_experiences, old_sample_size, replace=False)
                    
                    # 将旧经验添加到当前缓冲区
                    for experience in old_sample:
                        replay_buffer.add(**experience)
                    
                    logger.info(f"已添加 {old_sample_size} 个旧阶段经验到回放缓冲区")
        
        except Exception as e:
            logger.warning(f"混合经验回放实施失败: {e}")
    
    def _save_training_summary(self, summary: Dict[str, Any]):
        """保存训练摘要"""
        summary_path = f"{self.curriculum_config.output_dir}/training_summary.json"
        
        # 转换numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        # 递归转换
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        serializable_summary = recursive_convert(summary)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练摘要已保存到: {summary_path}")

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def save_stage_checkpoint(self, stage_idx: int, stage_result: Dict[str, Any]):
        """保存阶段检查点"""
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            f"stage_{stage_idx}_checkpoint.json"
        )
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(stage_result, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"阶段 {stage_idx} 检查点已保存到: {checkpoint_path}")
    
    def get_latest_checkpoint(self, stage_idx: int) -> Optional[str]:
        """获取最新检查点路径"""
        checkpoint_pattern = f"stage_{stage_idx}_episode_*"
        checkpoint_dir = os.path.join(self.checkpoints_dir, checkpoint_pattern)
        
        # 查找匹配的检查点目录
        import glob
        checkpoints = glob.glob(checkpoint_dir)
        
        if checkpoints:
            # 返回最新的检查点
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            return latest_checkpoint
        
        return None

def create_env_wrapper(env_config: Dict[str, Any]):
    """创建环境包装器，用于Ray RLlib"""
    return UAVTaskEnv(**env_config)

def main():
    """主函数 - 课程学习训练入口"""
    print("课程学习训练协调器")
    print("=" * 50)
    
    # 创建配置
    base_config = Config()
    curriculum_config = CurriculumConfig(
        output_dir="curriculum_training_output",
        num_workers=2,  # 根据系统资源调整
        evaluation_frequency=20
    )
    
    # 创建训练器
    trainer = CurriculumTrainer(curriculum_config, base_config)
    
    try:
        # 执行训练
        training_summary = trainer.train()
        
        # 输出结果
        print("\n课程学习训练完成!")
        print(f"完成阶段数: {training_summary['stages_completed']}/{len(curriculum_config.stages)}")
        print(f"总训练轮数: {training_summary['total_episodes']}")
        print(f"总训练时间: {training_summary['total_training_time']:.2f}秒")
        
        # 输出各阶段结果
        print("\n各阶段训练结果:")
        for i, stage_result in enumerate(training_summary['stage_results']):
            status = "✓ 成功" if stage_result['success'] else "✗ 未达标"
            print(f"  阶段 {i+1}: {stage_result['stage_name']} - {status}")
            print(f"    训练轮数: {stage_result['episodes_trained']}")
            print(f"    最终性能: {stage_result['final_performance']:.4f}")
            if stage_result['fallback_count'] > 0:
                print(f"    回退次数: {stage_result['fallback_count']}")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        raise
    finally:
        # 清理资源
        if RLLIB_AVAILABLE and ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
