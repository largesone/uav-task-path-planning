"""
Ray RLlib集成适配器
将MixedExperienceReplay集成到Ray RLlib的训练流程中
支持课程学习和分布式训练
"""

from typing import Dict, Any, Optional
import logging

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AlgorithmConfigDict

from mixed_experience_replay import MixedExperienceReplay, ExperiencePoolManager

# 创建全局经验池管理器实例
experience_pool_manager = ExperiencePoolManager()


class MixedReplayDQNConfig(DQNConfig):
    """
    支持混合经验回放的DQN配置
    """
    
    def __init__(self):
        super().__init__()
        # 混合回放配置
        self.mixed_replay_config = {
            'current_stage_ratio': 0.7,
            'historical_stage_ratio': 0.3,
            'max_stages_to_keep': 3,
            'buffer_capacity': 100000
        }
    
    def replay_buffer_config(self, **kwargs) -> "MixedReplayDQNConfig":
        """
        配置混合经验回放缓冲区
        """
        self.mixed_replay_config.update(kwargs)
        return self


class MixedReplayDQN(DQN):
    """
    支持混合经验回放的DQN算法
    """
    
    @classmethod
    def get_default_config(cls) -> AlgorithmConfigDict:
        config = super().get_default_config()
        config.update({
            'mixed_replay_config': {
                'current_stage_ratio': 0.7,
                'historical_stage_ratio': 0.3,
                'max_stages_to_keep': 3,
                'buffer_capacity': 100000
            }
        })
        return config
    
    def setup(self, config: AlgorithmConfigDict) -> None:
        """
        初始化算法，设置混合经验回放缓冲区
        """
        super().setup(config)
        
        # 创建混合经验回放缓冲区
        mixed_config = config.get('mixed_replay_config', {})
        self.mixed_replay_buffer = experience_pool_manager.create_buffer(
            buffer_id=f"dqn_{id(self)}",
            **mixed_config
        )
        
        # 替换默认的replay buffer
        if hasattr(self, 'local_replay_buffer'):
            self.local_replay_buffer = self.mixed_replay_buffer
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化混合经验回放DQN算法")
    
    def set_curriculum_stage(self, stage_id: int) -> None:
        """
        设置课程学习阶段
        
        Args:
            stage_id: 阶段ID
        """
        if hasattr(self, 'mixed_replay_buffer'):
            self.mixed_replay_buffer.set_current_stage(stage_id)
            self.logger.info(f"DQN算法切换到课程阶段 {stage_id}")


class MixedReplayPPOConfig(PPOConfig):
    """
    支持混合经验回放的PPO配置（用于off-policy变体）
    """
    
    def __init__(self):
        super().__init__()
        self.mixed_replay_config = {
            'current_stage_ratio': 0.7,
            'historical_stage_ratio': 0.3,
            'max_stages_to_keep': 3,
            'buffer_capacity': 50000  # PPO通常需要较小的buffer
        }


class CurriculumLearningCallback:
    """
    课程学习回调函数
    用于在训练过程中管理阶段切换和经验回放
    """
    
    def __init__(self, curriculum_config: Dict[str, Any]):
        self.curriculum_config = curriculum_config
        self.current_stage = 0
        self.stage_episodes = 0
        self.logger = logging.getLogger(__name__)
    
    def on_episode_end(self, algorithm, episode_data: Dict[str, Any]) -> None:
        """
        每个episode结束时的回调
        
        Args:
            algorithm: 训练算法实例
            episode_data: episode数据
        """
        self.stage_episodes += 1
        
        # 检查是否需要切换阶段
        if self._should_advance_stage(episode_data):
            self._advance_stage(algorithm)
        elif self._should_rollback_stage(episode_data):
            self._rollback_stage(algorithm)
    
    def _should_advance_stage(self, episode_data: Dict[str, Any]) -> bool:
        """
        判断是否应该进入下一阶段
        
        Args:
            episode_data: episode数据
            
        Returns:
            是否应该进入下一阶段
        """
        # 实现阶段推进逻辑
        # 这里可以基于性能指标、episode数量等条件判断
        stage_config = self.curriculum_config.get('stages', {}).get(self.current_stage, {})
        max_episodes = stage_config.get('max_episodes', 1000)
        success_threshold = stage_config.get('success_threshold', 0.8)
        
        # 检查episode数量和性能阈值
        if self.stage_episodes >= max_episodes:
            avg_reward = episode_data.get('episode_reward_mean', 0)
            if avg_reward >= success_threshold:
                return True
        
        return False
    
    def _should_rollback_stage(self, episode_data: Dict[str, Any]) -> bool:
        """
        判断是否应该回退到上一阶段
        
        Args:
            episode_data: episode数据
            
        Returns:
            是否应该回退
        """
        # 实现回退逻辑
        if self.current_stage == 0:
            return False
        
        # 检查性能是否显著下降
        avg_reward = episode_data.get('episode_reward_mean', 0)
        stage_config = self.curriculum_config.get('stages', {}).get(self.current_stage, {})
        rollback_threshold = stage_config.get('rollback_threshold', 0.6)
        
        return avg_reward < rollback_threshold
    
    def _advance_stage(self, algorithm) -> None:
        """
        推进到下一阶段
        
        Args:
            algorithm: 训练算法实例
        """
        next_stage = self.current_stage + 1
        max_stages = len(self.curriculum_config.get('stages', {}))
        
        if next_stage < max_stages:
            self.current_stage = next_stage
            self.stage_episodes = 0
            
            # 更新算法的阶段设置
            if hasattr(algorithm, 'set_curriculum_stage'):
                algorithm.set_curriculum_stage(self.current_stage)
            
            # 更新全局经验池管理器
            experience_pool_manager.set_stage_for_all(self.current_stage)
            
            self.logger.info(f"课程学习推进到阶段 {self.current_stage}")
    
    def _rollback_stage(self, algorithm) -> None:
        """
        回退到上一阶段
        
        Args:
            algorithm: 训练算法实例
        """
        if self.current_stage > 0:
            prev_stage = self.current_stage - 1
            self.current_stage = prev_stage
            self.stage_episodes = 0
            
            # 更新算法的阶段设置
            if hasattr(algorithm, 'set_curriculum_stage'):
                algorithm.set_curriculum_stage(self.current_stage)
            
            # 更新全局经验池管理器
            experience_pool_manager.set_stage_for_all(self.current_stage)
            
            self.logger.warning(f"课程学习回退到阶段 {self.current_stage}")


def create_mixed_replay_config(
    algorithm_type: str = "DQN",
    mixed_replay_config: Optional[Dict[str, Any]] = None
) -> AlgorithmConfig:
    """
    创建支持混合经验回放的算法配置
    
    Args:
        algorithm_type: 算法类型 ("DQN", "PPO")
        mixed_replay_config: 混合回放配置
        
    Returns:
        配置好的算法配置对象
    """
    if mixed_replay_config is None:
        mixed_replay_config = {
            'current_stage_ratio': 0.7,
            'historical_stage_ratio': 0.3,
            'max_stages_to_keep': 3,
            'buffer_capacity': 100000
        }
    
    if algorithm_type.upper() == "DQN":
        config = MixedReplayDQNConfig()
        config.mixed_replay_config.update(mixed_replay_config)
    elif algorithm_type.upper() == "PPO":
        config = MixedReplayPPOConfig()
        config.mixed_replay_config.update(mixed_replay_config)
    else:
        raise ValueError(f"不支持的算法类型: {algorithm_type}")
    
    return config
