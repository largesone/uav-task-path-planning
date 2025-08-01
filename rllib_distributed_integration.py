# -*- coding: utf-8 -*-
# 文件名: rllib_distributed_integration.py
# 描述: Ray RLlib分布式训练集成，实现数据一致性和稳定性保障

import torch
import logging
from typing import Dict, Any, Optional, List
import time

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.utils.annotations import override

from distributed_training_utils import distributed_processor, prepare_sample_batch_for_sharing


class DistributedRolloutWorker(RolloutWorker):
    """
    支持分布式图数据处理的RolloutWorker
    
    主要功能：
    1. 在数据发送前对图数据张量进行内存共享预处理
    2. 处理稀疏张量的跨进程传输兼容性
    3. 实现异常处理和重试机制
    """
    
    def __init__(self, *args, **kwargs):
        """初始化分布式RolloutWorker"""
        super().__init__(*args, **kwargs)
        
        self.logger = logging.getLogger(__name__)
        self.distributed_config = kwargs.get('distributed_config', {})
        self.enable_graph_data_sharing = self.distributed_config.get('enable_graph_data_sharing', True)
        
        # 统计信息
        self.batch_count = 0
        self.last_stats_log = 0
        self.stats_log_interval = self.distributed_config.get('stats_log_interval', 100)
        
        self.logger.info(f"分布式RolloutWorker初始化完成，图数据共享: {self.enable_graph_data_sharing}")
    
    @override(RolloutWorker)
    def sample(self) -> SampleBatchType:
        """
        采样数据并准备分布式传输
        
        Returns:
            准备好分布式传输的样本批次
        """
        # 调用父类的采样方法
        sample_batch = super().sample()
        
        # 如果启用了图数据共享，对样本批次进行预处理
        if self.enable_graph_data_sharing:
            try:
                sample_batch = self._prepare_batch_for_distributed_training(sample_batch)
                self.batch_count += 1
                
                # 定期记录统计信息
                if self.batch_count - self.last_stats_log >= self.stats_log_interval:
                    self._log_processing_stats()
                    self.last_stats_log = self.batch_count
                    
            except Exception as e:
                self.logger.error(f"样本批次分布式预处理失败: {e}")
                # 返回原始批次作为备用方案
                pass
        
        return sample_batch
    
    def _prepare_batch_for_distributed_training(self, sample_batch: SampleBatchType) -> SampleBatchType:
        """
        为分布式训练准备样本批次
        
        Args:
            sample_batch: 原始样本批次
            
        Returns:
            准备好分布式传输的样本批次
        """
        if isinstance(sample_batch, SampleBatch):
            return prepare_sample_batch_for_sharing(sample_batch)
        elif isinstance(sample_batch, dict):
            # 处理多智能体样本批次
            processed_batch = {}
            for policy_id, batch in sample_batch.items():
                if isinstance(batch, SampleBatch):
                    processed_batch[policy_id] = prepare_sample_batch_for_sharing(batch)
                else:
                    processed_batch[policy_id] = batch
            return processed_batch
        else:
            return sample_batch
    
    def _log_processing_stats(self):
        """记录处理统计信息"""
        stats = distributed_processor.get_stats()
        self.logger.info(f"分布式数据处理统计 - 已处理批次: {stats['processed_batches']}, "
                        f"失败批次: {stats['failed_batches']}, "
                        f"重试次数: {stats['retry_count']}, "
                        f"成功率: {stats['success_rate']:.3f}, "
                        f"平均处理时间: {stats['avg_processing_time']:.4f}秒")


class DistributedLearner:
    """
    支持分布式训练的Learner组件
    
    主要功能：
    1. 配置数据加载器使用pin_memory=True
    2. 处理GPU数据传输优化
    3. 实现数据一致性检查
    """
    
    def __init__(self, learner_config: Dict[str, Any]):
        """
        初始化分布式Learner
        
        Args:
            learner_config: Learner配置
        """
        self.config = learner_config
        self.logger = logging.getLogger(__name__)
        
        # 数据加载器配置
        self.data_loader_config = distributed_processor.configure_data_loader_for_learner(
            self.config.get('data_loader_config', {})
        )
        
        # 数据一致性检查
        self.enable_consistency_check = self.config.get('enable_data_consistency_check', True)
        
        self.logger.info(f"分布式Learner初始化完成")
        self.logger.info(f"数据加载器配置: {self.data_loader_config}")
    
    def create_data_loader(self, dataset, **kwargs) -> torch.utils.data.DataLoader:
        """
        创建优化的数据加载器
        
        Args:
            dataset: 数据集
            **kwargs: 额外的数据加载器参数
            
        Returns:
            优化的数据加载器
        """
        # 合并配置
        loader_config = self.data_loader_config.copy()
        loader_config.update(kwargs)
        
        # 创建数据加载器
        data_loader = torch.utils.data.DataLoader(dataset, **loader_config)
        
        self.logger.debug(f"创建数据加载器，配置: {loader_config}")
        
        return data_loader
    
    def validate_batch_consistency(self, batch: Dict[str, Any]) -> bool:
        """
        验证批次数据一致性
        
        Args:
            batch: 数据批次
            
        Returns:
            数据是否一致
        """
        if not self.enable_consistency_check:
            return True
        
        try:
            # 检查张量形状一致性
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    # 检查张量是否有效
                    if torch.isnan(value).any():
                        self.logger.warning(f"检测到NaN值在张量 {key}")
                        return False
                    
                    if torch.isinf(value).any():
                        self.logger.warning(f"检测到无穷值在张量 {key}")
                        return False
                
                elif isinstance(value, dict):
                    # 递归检查嵌套字典
                    if not self.validate_batch_consistency(value):
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据一致性检查失败: {e}")
            return False


class DistributedPPOConfig(PPOConfig):
    """
    支持分布式图数据处理的PPO配置
    """
    
    def __init__(self):
        super().__init__()
        
        # 分布式训练配置
        self.distributed_config = {
            'enable_graph_data_sharing': True,
            'rollout_worker_class': DistributedRolloutWorker,
            'learner_config': {
                'data_loader_config': {
                    'pin_memory': True,
                    'num_workers': 2,
                    'persistent_workers': True,
                    'prefetch_factor': 2
                },
                'enable_data_consistency_check': True
            },
            'stats_log_interval': 100
        }
    
    def distributed_training(self, **kwargs) -> "DistributedPPOConfig":
        """
        配置分布式训练参数
        
        Args:
            **kwargs: 分布式训练配置参数
            
        Returns:
            配置对象
        """
        self.distributed_config.update(kwargs)
        return self
    
    def rollouts(self, **kwargs) -> "DistributedPPOConfig":
        """
        重写rollouts配置，使用分布式RolloutWorker
        """
        config = super().rollouts(**kwargs)
        
        # 设置分布式RolloutWorker
        if 'rollout_worker_class' not in kwargs:
            config.rollout_worker_class = DistributedRolloutWorker
        
        return config


class DistributedPPO(PPO):
    """
    支持分布式图数据处理的PPO算法
    """
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        config = super().get_default_config()
        
        # 添加分布式配置
        config.update({
            'distributed_config': {
                'enable_graph_data_sharing': True,
                'stats_log_interval': 100,
                'learner_config': {
                    'data_loader_config': {
                        'pin_memory': True,
                        'num_workers': 2,
                        'persistent_workers': True,
                        'prefetch_factor': 2
                    },
                    'enable_data_consistency_check': True
                }
            }
        })
        
        return config
    
    def setup(self, config: Dict[str, Any]) -> None:
        """
        初始化分布式PPO算法
        
        Args:
            config: 算法配置
        """
        super().setup(config)
        
        # 初始化分布式Learner
        distributed_config = config.get('distributed_config', {})
        learner_config = distributed_config.get('learner_config', {})
        
        self.distributed_learner = DistributedLearner(learner_config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("分布式PPO算法初始化完成")
    
    def training_step(self) -> Dict[str, Any]:
        """
        重写训练步骤，添加分布式数据处理
        
        Returns:
            训练结果
        """
        try:
            # 调用父类的训练步骤
            result = super().training_step()
            
            # 添加分布式处理统计信息
            distributed_stats = distributed_processor.get_stats()
            result.update({
                'distributed_stats': distributed_stats
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"分布式训练步骤失败: {e}")
            # 重置分布式处理器状态
            distributed_processor.reset_stats()
            raise


def create_distributed_training_algorithm(
    algorithm_type: str = "PPO",
    config_overrides: Optional[Dict[str, Any]] = None
) -> AlgorithmConfig:
    """
    创建支持分布式图数据处理的训练算法
    
    Args:
        algorithm_type: 算法类型
        config_overrides: 配置覆盖
        
    Returns:
        配置好的算法配置
    """
    if algorithm_type.upper() == "PPO":
        config = DistributedPPOConfig()
    else:
        raise ValueError(f"暂不支持的算法类型: {algorithm_type}")
    
    # 应用配置覆盖
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # 添加到distributed_config中
                if not hasattr(config, 'distributed_config'):
                    config.distributed_config = {}
                config.distributed_config[key] = value
    
    return config
