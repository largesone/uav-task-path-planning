"""
混合经验回放机制实现

该模块实现了支持课程学习的混合经验回放机制，用于防止灾难性遗忘。
在课程学习第二阶段及以后，采样批次包含70%当前阶段经验和30%旧阶段经验。

核心特性：
1. 基于Ray RLlib的自定义Replay Buffer API
2. 支持多阶段经验池管理
3. 混合采样策略（70%当前 + 30%历史）
4. 防止灾难性遗忘机制
"""

import numpy as np
import random
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import logging

try:
    from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.utils.typing import SampleBatchType
except ImportError:
    # 如果Ray RLlib不可用，提供基础接口
    class ReplayBuffer:
        pass
    class SampleBatch:
        pass
    SampleBatchType = Any

logger = logging.getLogger(__name__)


class MixedExperienceReplay:
    """
    混合经验回放机制
    
    该类实现了支持课程学习的经验回放机制，能够在不同训练阶段间
    维护经验池，并实现混合采样策略以防止灾难性遗忘。
    
    主要功能：
    - 多阶段经验池管理
    - 混合采样策略（当前阶段70% + 历史阶段30%）
    - 经验池容量管理和老化策略
    - 与Ray RLlib的无缝集成
    """
    
    def __init__(
        self,
        capacity_per_stage: int = 100000,
        current_stage_ratio: float = 0.7,
        historical_stage_ratio: float = 0.3,
        max_stages_to_keep: int = 3,
        min_historical_samples: int = 1000,
        capacity: Optional[int] = None  # 兼容性参数
    ):
        """
        初始化混合经验回放机制
        
        Args:
            capacity_per_stage: 每个阶段的经验池容量
            current_stage_ratio: 当前阶段经验的采样比例（默认0.7）
            historical_stage_ratio: 历史阶段经验的采样比例（默认0.3）
            max_stages_to_keep: 最多保留的历史阶段数量
            min_historical_samples: 历史阶段的最小样本数量
            capacity: 兼容性参数，如果提供则覆盖capacity_per_stage
        """
        # 处理兼容性参数
        if capacity is not None:
            capacity_per_stage = capacity
            
        self.capacity_per_stage = capacity_per_stage
        self.current_stage_ratio = current_stage_ratio
        self.historical_stage_ratio = historical_stage_ratio
        self.max_stages_to_keep = max_stages_to_keep
        self.min_historical_samples = min_historical_samples
        
        # 多阶段经验池存储
        # 格式: {stage_id: deque of experiences}
        self.stage_buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=capacity_per_stage)
        )
        
        # 当前训练阶段
        self.current_stage = 0
        
        # 统计信息
        self.stats = {
            'total_samples_added': 0,
            'samples_per_stage': defaultdict(int),
            'mixed_batches_generated': 0,
            'current_stage_samples_used': 0,
            'historical_samples_used': 0
        }
        
        logger.info(f"初始化混合经验回放机制 - 容量: {capacity_per_stage}, "
                   f"当前阶段比例: {current_stage_ratio}, 历史阶段比例: {historical_stage_ratio}")
    
    @property
    def capacity(self) -> int:
        """获取每阶段容量（兼容性属性）"""
        return self.capacity_per_stage
    
    @property
    def current_stage_id(self) -> int:
        """获取当前阶段ID（兼容性属性）"""
        return self.current_stage
    
    def add(self, experience: Dict[str, Any], stage_id: Optional[int] = None) -> None:
        """添加经验的兼容性方法"""
        self.add_experience(experience, stage_id)
    
    def sample(self, batch_size: int):
        """采样的兼容性方法，返回单个SampleBatch"""
        experiences = self.sample_mixed_batch(batch_size)
        
        if not experiences:
            # 如果没有经验，返回空的SampleBatch
            try:
                from ray.rllib.policy.sample_batch import SampleBatch
                return SampleBatch({})
            except ImportError:
                return {}
        
        # 将经验列表转换为单个SampleBatch
        try:
            from ray.rllib.policy.sample_batch import SampleBatch
            
            # 合并所有经验到一个批次中
            batch_dict = {}
            for key in experiences[0].keys():
                if key not in ['stage_id', 'timestamp']:  # 排除内部字段
                    batch_dict[key] = [exp[key] for exp in experiences]
            
            return SampleBatch(batch_dict)
        except ImportError:
            # 如果Ray RLlib不可用，返回字典格式
            return experiences
    
    def __len__(self) -> int:
        """获取当前阶段的经验数量"""
        return len(self.stage_buffers.get(self.current_stage, []))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息的兼容性方法"""
        return self.get_statistics()
    
    def set_current_stage(self, stage_id: int) -> None:
        """
        设置当前训练阶段
        
        Args:
            stage_id: 当前训练阶段ID
        """
        if stage_id != self.current_stage:
            logger.info(f"切换训练阶段: {self.current_stage} -> {stage_id}")
            self.current_stage = stage_id
            
            # 清理过旧的阶段数据
            self._cleanup_old_stages()
    
    def add_experience(self, experience: Dict[str, Any], stage_id: Optional[int] = None) -> None:
        """
        添加经验到指定阶段的经验池
        
        Args:
            experience: 经验数据字典
            stage_id: 阶段ID，如果为None则使用当前阶段
        """
        if stage_id is None:
            stage_id = self.current_stage
        
        # 添加阶段标识到经验中
        experience_with_stage = experience.copy()
        experience_with_stage['stage_id'] = stage_id
        experience_with_stage['timestamp'] = np.random.random()  # 简单的时间戳
        
        # 存储到对应阶段的缓冲区
        self.stage_buffers[stage_id].append(experience_with_stage)
        
        # 更新统计信息
        self.stats['total_samples_added'] += 1
        self.stats['samples_per_stage'][stage_id] += 1
        
        if len(self.stage_buffers[stage_id]) % 1000 == 0:
            logger.debug(f"阶段 {stage_id} 经验池大小: {len(self.stage_buffers[stage_id])}")
    
    def sample_mixed_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        采样混合批次数据
        
        在第一阶段只从当前阶段采样，从第二阶段开始实施混合采样策略。
        
        Args:
            batch_size: 批次大小
            
        Returns:
            混合采样的经验批次
        """
        if self.current_stage == 0 or not self._has_sufficient_historical_data():
            # 第一阶段或历史数据不足时，只从当前阶段采样
            return self._sample_from_current_stage(batch_size)
        else:
            # 第二阶段及以后，实施混合采样策略
            return self._sample_mixed_strategy(batch_size)
    
    def _sample_mixed_strategy(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        实施混合采样策略：70%当前阶段 + 30%历史阶段
        
        Args:
            batch_size: 批次大小
            
        Returns:
            混合采样的经验批次
        """
        current_samples_count = int(batch_size * self.current_stage_ratio)
        historical_samples_count = batch_size - current_samples_count
        
        # 从当前阶段采样
        current_samples = self._sample_from_current_stage(current_samples_count)
        
        # 从历史阶段采样
        historical_samples = self._sample_from_historical_stages(historical_samples_count)
        
        # 合并并随机打乱
        mixed_batch = current_samples + historical_samples
        random.shuffle(mixed_batch)
        
        # 更新统计信息
        self.stats['mixed_batches_generated'] += 1
        self.stats['current_stage_samples_used'] += len(current_samples)
        self.stats['historical_samples_used'] += len(historical_samples)
        
        logger.debug(f"混合采样批次 - 当前阶段: {len(current_samples)}, "
                    f"历史阶段: {len(historical_samples)}, 总计: {len(mixed_batch)}")
        
        return mixed_batch
    
    def _sample_from_current_stage(self, sample_count: int) -> List[Dict[str, Any]]:
        """
        从当前阶段采样
        
        Args:
            sample_count: 采样数量
            
        Returns:
            当前阶段的经验样本
        """
        current_buffer = self.stage_buffers[self.current_stage]
        
        if len(current_buffer) == 0:
            logger.warning(f"当前阶段 {self.current_stage} 经验池为空")
            return []
        
        # 随机采样
        available_count = min(sample_count, len(current_buffer))
        indices = random.sample(range(len(current_buffer)), available_count)
        
        return [current_buffer[i] for i in indices]
    
    def _sample_from_historical_stages(self, sample_count: int) -> List[Dict[str, Any]]:
        """
        从历史阶段采样
        
        Args:
            sample_count: 采样数量
            
        Returns:
            历史阶段的经验样本
        """
        historical_stages = [stage_id for stage_id in self.stage_buffers.keys() 
                           if stage_id < self.current_stage]
        
        if not historical_stages:
            logger.warning("没有可用的历史阶段数据")
            return []
        
        # 计算每个历史阶段的采样权重（基于数据量）
        stage_weights = {}
        total_historical_samples = 0
        
        for stage_id in historical_stages:
            stage_size = len(self.stage_buffers[stage_id])
            stage_weights[stage_id] = stage_size
            total_historical_samples += stage_size
        
        if total_historical_samples == 0:
            return []
        
        # 按权重分配采样数量
        samples = []
        remaining_count = sample_count
        
        for stage_id in historical_stages[:-1]:  # 除了最后一个阶段
            stage_ratio = stage_weights[stage_id] / total_historical_samples
            stage_sample_count = int(remaining_count * stage_ratio)
            
            if stage_sample_count > 0:
                stage_samples = self._sample_from_stage(stage_id, stage_sample_count)
                samples.extend(stage_samples)
                remaining_count -= len(stage_samples)
        
        # 最后一个阶段采样剩余数量
        if remaining_count > 0 and historical_stages:
            last_stage = historical_stages[-1]
            last_samples = self._sample_from_stage(last_stage, remaining_count)
            samples.extend(last_samples)
        
        return samples
    
    def _sample_from_stage(self, stage_id: int, sample_count: int) -> List[Dict[str, Any]]:
        """
        从指定阶段采样
        
        Args:
            stage_id: 阶段ID
            sample_count: 采样数量
            
        Returns:
            指定阶段的经验样本
        """
        stage_buffer = self.stage_buffers[stage_id]
        
        if len(stage_buffer) == 0:
            return []
        
        available_count = min(sample_count, len(stage_buffer))
        indices = random.sample(range(len(stage_buffer)), available_count)
        
        return [stage_buffer[i] for i in indices]
    
    def _has_sufficient_historical_data(self) -> bool:
        """
        检查是否有足够的历史数据进行混合采样
        
        Returns:
            是否有足够的历史数据
        """
        historical_stages = [stage_id for stage_id in self.stage_buffers.keys() 
                           if stage_id < self.current_stage]
        
        total_historical_samples = sum(
            len(self.stage_buffers[stage_id]) for stage_id in historical_stages
        )
        
        return total_historical_samples >= self.min_historical_samples
    
    def _cleanup_old_stages(self) -> None:
        """
        清理过旧的阶段数据，保持内存使用合理
        """
        if len(self.stage_buffers) <= self.max_stages_to_keep:
            return
            
        # 保留当前阶段和最近的几个阶段
        stages_to_keep = sorted([
            stage_id for stage_id in self.stage_buffers.keys()
            if stage_id >= self.current_stage - self.max_stages_to_keep + 1
        ])
        
        # 如果保留的阶段数仍然超过限制，只保留最新的几个
        if len(stages_to_keep) > self.max_stages_to_keep:
            stages_to_keep = stages_to_keep[-self.max_stages_to_keep:]
        
        # 删除过旧的阶段
        stages_to_remove = [
            stage_id for stage_id in self.stage_buffers.keys()
            if stage_id not in stages_to_keep
        ]
        
        for stage_id in stages_to_remove:
            removed_count = len(self.stage_buffers[stage_id])
            del self.stage_buffers[stage_id]
            logger.info(f"清理阶段 {stage_id} 的 {removed_count} 个经验样本")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取经验回放的统计信息
        
        Returns:
            统计信息字典
        """
        stage_sizes = {
            stage_id: len(buffer) 
            for stage_id, buffer in self.stage_buffers.items()
        }
        
        return {
            'current_stage': self.current_stage,
            'stage_buffer_sizes': stage_sizes,
            'total_samples_added': self.stats['total_samples_added'],
            'samples_per_stage': dict(self.stats['samples_per_stage']),
            'mixed_batches_generated': self.stats['mixed_batches_generated'],
            'current_stage_samples_used': self.stats['current_stage_samples_used'],
            'historical_samples_used': self.stats['historical_samples_used'],
            'has_sufficient_historical_data': self._has_sufficient_historical_data()
        }
    
    def clear_stage(self, stage_id: int) -> None:
        """
        清空指定阶段的经验池
        
        Args:
            stage_id: 要清空的阶段ID
        """
        if stage_id in self.stage_buffers:
            cleared_count = len(self.stage_buffers[stage_id])
            self.stage_buffers[stage_id].clear()
            logger.info(f"清空阶段 {stage_id} 的 {cleared_count} 个经验样本")
    
    def get_stage_size(self, stage_id: int) -> int:
        """
        获取指定阶段的经验池大小
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            经验池大小
        """
        return len(self.stage_buffers.get(stage_id, []))


class RLlibMixedReplayBuffer(ReplayBuffer):
    """
    Ray RLlib兼容的混合经验回放缓冲区
    
    该类继承自Ray RLlib的ReplayBuffer，提供与RLlib训练框架的无缝集成。
    内部使用MixedExperienceReplay实现混合采样逻辑。
    """
    
    def __init__(self, capacity: int = 100000, **kwargs):
        """
        初始化RLlib兼容的混合回放缓冲区
        
        Args:
            capacity: 缓冲区总容量
            **kwargs: 其他参数传递给MixedExperienceReplay
        """
        super().__init__(capacity)
        
        # 初始化混合经验回放机制
        self.mixed_replay = MixedExperienceReplay(
            capacity_per_stage=capacity // 4,  # 每阶段容量为总容量的1/4
            **kwargs
        )
        
        logger.info(f"初始化RLlib混合回放缓冲区，容量: {capacity}")
    
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """
        添加样本批次到回放缓冲区
        
        Args:
            batch: RLlib样本批次
            **kwargs: 其他参数
        """
        # 将RLlib的SampleBatch转换为字典格式
        if hasattr(batch, 'data'):
            # 处理SampleBatch对象
            for i in range(len(batch)):
                experience = {key: batch[key][i] for key in batch.keys()}
                self.mixed_replay.add_experience(experience)
        else:
            # 处理字典格式
            self.mixed_replay.add_experience(batch)
    
    def sample(self, batch_size: int, **kwargs) -> SampleBatchType:
        """
        从回放缓冲区采样批次
        
        Args:
            batch_size: 批次大小
            **kwargs: 其他参数
            
        Returns:
            RLlib格式的样本批次
        """
        # 使用混合采样策略获取经验
        experiences = self.mixed_replay.sample_mixed_batch(batch_size)
        
        if not experiences:
            # 如果没有经验，返回空批次
            return SampleBatch({})
        
        # 将经验列表转换为RLlib的SampleBatch格式
        batch_dict = {}
        for key in experiences[0].keys():
            if key not in ['stage_id', 'timestamp']:  # 排除内部字段
                batch_dict[key] = np.array([exp[key] for exp in experiences])
        
        return SampleBatch(batch_dict)
    
    def set_current_stage(self, stage_id: int) -> None:
        """
        设置当前训练阶段
        
        Args:
            stage_id: 当前训练阶段ID
        """
        self.mixed_replay.set_current_stage(stage_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取回放缓冲区统计信息
        
        Returns:
            统计信息字典
        """
        return self.mixed_replay.get_statistics()


class ExperiencePoolManager:
    """
    经验池管理器
    
    该类负责管理多个训练阶段的经验池，提供统一的接口来
    添加、检索和管理不同阶段的经验数据。
    
    主要功能：
    - 多阶段经验池的创建和管理
    - 经验池容量控制和内存管理
    - 阶段间经验数据的迁移和清理
    - 统计信息收集和监控
    """
    
    def __init__(
        self,
        default_capacity: int = 50000,
        max_stages: int = 10,
        auto_cleanup: bool = True
    ):
        """
        初始化经验池管理器
        
        Args:
            default_capacity: 默认经验池容量
            max_stages: 最大支持的训练阶段数
            auto_cleanup: 是否自动清理旧阶段数据
        """
        self.default_capacity = default_capacity
        self.max_stages = max_stages
        self.auto_cleanup = auto_cleanup
        
        # 经验池存储：{stage_id: MixedExperienceReplay}
        self.pools: Dict[int, MixedExperienceReplay] = {}
        
        # 全局统计信息
        self.global_stats = {
            'total_pools_created': 0,
            'total_experiences_managed': 0,
            'active_stages': set(),
            'cleanup_operations': 0
        }
        
        logger.info(f"初始化经验池管理器 - 默认容量: {default_capacity}, "
                   f"最大阶段数: {max_stages}")
    
    @property
    def replay_buffers(self) -> Dict[str, MixedExperienceReplay]:
        """获取所有回放缓冲区的兼容性属性"""
        # 将stage_id映射回buffer_id的简化版本
        return {f"stage_{stage_id}": pool for stage_id, pool in self.pools.items()}
    
    def create_pool(
        self, 
        stage_id: int, 
        capacity: Optional[int] = None,
        **kwargs
    ) -> MixedExperienceReplay:
        """
        为指定阶段创建经验池
        
        Args:
            stage_id: 训练阶段ID
            capacity: 经验池容量，如果为None则使用默认容量
            **kwargs: 传递给MixedExperienceReplay的其他参数
            
        Returns:
            创建的经验池实例
        """
        if stage_id in self.pools:
            logger.warning(f"阶段 {stage_id} 的经验池已存在，返回现有实例")
            return self.pools[stage_id]
        
        if capacity is None:
            capacity = self.default_capacity
        
        # 创建新的经验池
        pool = MixedExperienceReplay(
            capacity_per_stage=capacity,
            **kwargs
        )
        pool.set_current_stage(stage_id)
        
        self.pools[stage_id] = pool
        self.global_stats['total_pools_created'] += 1
        self.global_stats['active_stages'].add(stage_id)
        
        logger.info(f"为阶段 {stage_id} 创建经验池，容量: {capacity}")
        
        # 自动清理检查
        if self.auto_cleanup and len(self.pools) > self.max_stages:
            self._auto_cleanup()
        
        return pool
    
    def get_pool(self, stage_id: int) -> Optional[MixedExperienceReplay]:
        """
        获取指定阶段的经验池
        
        Args:
            stage_id: 训练阶段ID
            
        Returns:
            经验池实例，如果不存在则返回None
        """
        return self.pools.get(stage_id)
    
    def get_or_create_pool(
        self, 
        stage_id: int, 
        **kwargs
    ) -> MixedExperienceReplay:
        """
        获取或创建指定阶段的经验池
        
        Args:
            stage_id: 训练阶段ID
            **kwargs: 创建经验池时的参数
            
        Returns:
            经验池实例
        """
        pool = self.get_pool(stage_id)
        if pool is None:
            pool = self.create_pool(stage_id, **kwargs)
        return pool
    
    def add_experience_to_stage(
        self, 
        stage_id: int, 
        experience: Dict[str, Any],
        auto_create: bool = True
    ) -> bool:
        """
        向指定阶段添加经验
        
        Args:
            stage_id: 训练阶段ID
            experience: 经验数据
            auto_create: 如果经验池不存在是否自动创建
            
        Returns:
            是否成功添加经验
        """
        pool = self.get_pool(stage_id)
        
        if pool is None:
            if auto_create:
                pool = self.create_pool(stage_id)
            else:
                logger.error(f"阶段 {stage_id} 的经验池不存在且未启用自动创建")
                return False
        
        pool.add_experience(experience, stage_id)
        self.global_stats['total_experiences_managed'] += 1
        
        return True
    
    def sample_from_stage(
        self, 
        stage_id: int, 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """
        从指定阶段采样经验批次
        
        Args:
            stage_id: 训练阶段ID
            batch_size: 批次大小
            
        Returns:
            采样的经验批次
        """
        pool = self.get_pool(stage_id)
        
        if pool is None:
            logger.error(f"阶段 {stage_id} 的经验池不存在")
            return []
        
        return pool.sample_mixed_batch(batch_size)
    
    def transfer_experiences(
        self, 
        from_stage: int, 
        to_stage: int, 
        ratio: float = 0.1
    ) -> int:
        """
        在阶段间转移经验数据
        
        Args:
            from_stage: 源阶段ID
            to_stage: 目标阶段ID
            ratio: 转移比例（0.0-1.0）
            
        Returns:
            转移的经验数量
        """
        source_pool = self.get_pool(from_stage)
        target_pool = self.get_or_create_pool(to_stage)
        
        if source_pool is None:
            logger.error(f"源阶段 {from_stage} 的经验池不存在")
            return 0
        
        # 获取源阶段的经验数据
        source_buffer = source_pool.stage_buffers.get(from_stage, deque())
        transfer_count = int(len(source_buffer) * ratio)
        
        if transfer_count == 0:
            return 0
        
        # 随机选择要转移的经验
        indices = random.sample(range(len(source_buffer)), transfer_count)
        transferred_experiences = [source_buffer[i] for i in indices]
        
        # 添加到目标阶段
        for experience in transferred_experiences:
            target_pool.add_experience(experience, to_stage)
        
        logger.info(f"从阶段 {from_stage} 向阶段 {to_stage} 转移了 {transfer_count} 个经验")
        
        return transfer_count
    
    def clear_stage(self, stage_id: int) -> bool:
        """
        清空指定阶段的经验池
        
        Args:
            stage_id: 训练阶段ID
            
        Returns:
            是否成功清空
        """
        pool = self.get_pool(stage_id)
        
        if pool is None:
            logger.warning(f"阶段 {stage_id} 的经验池不存在")
            return False
        
        pool.clear_stage(stage_id)
        return True
    
    def remove_stage(self, stage_id: int) -> bool:
        """
        完全移除指定阶段的经验池
        
        Args:
            stage_id: 训练阶段ID
            
        Returns:
            是否成功移除
        """
        if stage_id not in self.pools:
            logger.warning(f"阶段 {stage_id} 的经验池不存在")
            return False
        
        # 获取统计信息用于日志
        pool_stats = self.pools[stage_id].get_statistics()
        
        # 移除经验池
        del self.pools[stage_id]
        self.global_stats['active_stages'].discard(stage_id)
        
        logger.info(f"移除阶段 {stage_id} 的经验池，"
                   f"包含 {pool_stats.get('total_samples_added', 0)} 个经验")
        
        return True
    
    def create_buffer(
        self, 
        buffer_id: str, 
        capacity: Optional[int] = None,
        **kwargs
    ) -> MixedExperienceReplay:
        """
        创建命名的经验缓冲区（兼容性方法）
        
        Args:
            buffer_id: 缓冲区标识符
            capacity: 缓冲区容量
            **kwargs: 其他参数
            
        Returns:
            创建的经验回放实例
        """
        if capacity is None:
            capacity = self.default_capacity
        
        # 使用buffer_id作为stage_id的哈希值
        stage_id = hash(buffer_id) % 1000
        
        return self.create_pool(stage_id, capacity, **kwargs)
    
    def get_buffer(self, buffer_id: str) -> Optional[MixedExperienceReplay]:
        """
        获取命名的经验缓冲区（兼容性方法）
        
        Args:
            buffer_id: 缓冲区标识符
            
        Returns:
            经验回放实例
        """
        stage_id = hash(buffer_id) % 1000
        return self.get_pool(stage_id)
    
    def set_stage_for_all(self, stage_id: int) -> None:
        """
        为所有经验池设置当前阶段
        
        Args:
            stage_id: 新的阶段ID
        """
        for pool in self.pools.values():
            pool.set_current_stage(stage_id)
        
        logger.info(f"为所有 {len(self.pools)} 个经验池设置阶段为 {stage_id}")
    
    def _auto_cleanup(self) -> None:
        """
        自动清理最旧的经验池以控制内存使用
        """
        if len(self.pools) <= self.max_stages:
            return
        
        # 按阶段ID排序，移除最旧的阶段
        sorted_stages = sorted(self.pools.keys())
        stages_to_remove = sorted_stages[:len(self.pools) - self.max_stages]
        
        for stage_id in stages_to_remove:
            self.remove_stage(stage_id)
        
        self.global_stats['cleanup_operations'] += 1
        logger.info(f"自动清理完成，移除了 {len(stages_to_remove)} 个旧阶段")
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        获取全局统计信息
        
        Returns:
            全局统计信息字典
        """
        # 收集所有经验池的统计信息
        pool_stats = {}
        total_experiences = 0
        total_mixed_batches = 0
        
        for stage_id, pool in self.pools.items():
            stats = pool.get_statistics()
            pool_stats[stage_id] = stats
            total_experiences += stats.get('total_samples_added', 0)
            total_mixed_batches += stats.get('mixed_batches_generated', 0)
        
        return {
            'global_stats': dict(self.global_stats),
            'active_stages': list(self.global_stats['active_stages']),
            'total_active_pools': len(self.pools),
            'pool_statistics': pool_stats,
            'aggregated_stats': {
                'total_experiences_across_pools': total_experiences,
                'total_mixed_batches_generated': total_mixed_batches
            }
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计信息的兼容性方法"""
        return self.get_global_statistics()
    
    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """
        估算内存使用情况
        
        Returns:
            内存使用估算信息
        """
        memory_info = {}
        total_samples = 0
        
        for stage_id, pool in self.pools.items():
            stage_samples = sum(
                len(buffer) for buffer in pool.stage_buffers.values()
            )
            memory_info[f'stage_{stage_id}_samples'] = stage_samples
            total_samples += stage_samples
        
        # 简单的内存估算（假设每个样本平均1KB）
        estimated_memory_mb = total_samples * 1024 / (1024 * 1024)
        
        return {
            'total_samples': total_samples,
            'estimated_memory_mb': estimated_memory_mb,
            'samples_per_stage': memory_info,
            'average_samples_per_pool': total_samples / len(self.pools) if self.pools else 0
        }


# 创建全局经验池管理器实例
experience_pool_manager = ExperiencePoolManager()
