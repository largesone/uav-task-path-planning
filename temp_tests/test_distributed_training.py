# -*- coding: utf-8 -*-
# 文件名: temp_tests/test_distributed_training.py
# 描述: 分布式训练数据一致性测试

import torch
import numpy as np
import pytest
import logging
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed_training_utils import (
    DistributedDataProcessor, 
    distributed_processor,
    prepare_sample_batch_for_sharing,
    create_distributed_training_config
)
from rllib_distributed_integration import (
    DistributedRolloutWorker,
    DistributedLearner,
    DistributedPPOConfig,
    create_distributed_training_algorithm
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDistributedDataProcessor:
    """测试分布式数据处理器"""
    
    def setup_method(self):
        """测试前设置"""
        self.processor = DistributedDataProcessor(max_retries=2, retry_delay=0.01)
    
    def test_prepare_graph_data_basic(self):
        """测试基本图数据内存共享准备"""
        # 创建测试数据
        obs_dict = {
            'uav_features': torch.randn(2, 3, 10),
            'target_features': torch.randn(2, 5, 8),
            'relative_positions': torch.randn(2, 3, 5, 2),
            'distances': torch.randn(2, 3, 5),
            'masks': {
                'uav_mask': torch.ones(2, 3, dtype=torch.bool),
                'target_mask': torch.ones(2, 5, dtype=torch.bool)
            }
        }
        
        # 处理数据
        processed_dict = self.processor.prepare_graph_data_for_sharing(obs_dict)
        
        # 验证结果
        assert 'uav_features' in processed_dict
        assert 'target_features' in processed_dict
        assert 'relative_positions' in processed_dict
        assert 'distances' in processed_dict
        assert 'masks' in processed_dict
        
        # 验证张量已移到CPU
        for key in ['uav_features', 'target_features', 'relative_positions', 'distances']:
            tensor = processed_dict[key]
            assert tensor.device.type == 'cpu'
            assert tensor.is_shared()  # 验证内存共享已启用
        
        # 验证嵌套字典处理
        for key in ['uav_mask', 'target_mask']:
            tensor = processed_dict['masks'][key]
            assert tensor.device.type == 'cpu'
            assert tensor.is_shared()
        
        logger.info("基本图数据内存共享准备测试通过")
    
    def test_sparse_tensor_handling(self):
        """测试稀疏张量处理"""
        # 创建稀疏张量
        indices = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        values = torch.FloatTensor([3, 4, 5])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 3))
        
        # 处理稀疏张量
        processed_tensor = self.processor._handle_sparse_tensor(sparse_tensor)
        
        # 验证结果
        assert processed_tensor is not None
        assert not processed_tensor.is_sparse or processed_tensor.layout == torch.sparse_coo
        
        logger.info("稀疏张量处理测试通过")
    
    def test_data_loader_config_optimization(self):
        """测试数据加载器配置优化"""
        original_config = {
            'batch_size': 32,
            'shuffle': True
        }
        
        optimized_config = self.processor.configure_data_loader_for_learner(original_config)
        
        # 验证优化配置
        assert optimized_config['pin_memory'] is True
        assert 'num_workers' in optimized_config
        assert 'persistent_workers' in optimized_config
        assert 'prefetch_factor' in optimized_config
        assert optimized_config['batch_size'] == 32  # 原始配置保持不变
        assert optimized_config['shuffle'] is True
        
        logger.info("数据加载器配置优化测试通过")
    
    def test_retry_mechanism(self):
        """测试重试机制"""
        call_count = 0
        
        @self.processor.retry_on_failure(max_retries=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"模拟失败 {call_count}")
            return "成功"
        
        # 测试重试成功
        result = failing_function()
        assert result == "成功"
        assert call_count == 3
        
        # 测试重试失败
        call_count = 0
        
        @self.processor.retry_on_failure(max_retries=1)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"总是失败 {call_count}")
        
        with pytest.raises(ValueError):
            always_failing_function()
        
        assert call_count == 2  # 1次原始调用 + 1次重试
        
        logger.info("重试机制测试通过")
    
    def test_stats_tracking(self):
        """测试统计信息跟踪"""
        # 重置统计
        self.processor.reset_stats()
        
        # 模拟一些处理
        obs_dict = {'test_tensor': torch.randn(2, 3)}
        self.processor.prepare_graph_data_for_sharing(obs_dict)
        
        # 获取统计信息
        stats = self.processor.get_stats()
        
        assert stats['processed_batches'] == 1
        assert stats['failed_batches'] == 0
        assert stats['avg_processing_time'] > 0
        assert stats['success_rate'] == 1.0
        
        logger.info("统计信息跟踪测试通过")


class TestDistributedLearner:
    """测试分布式Learner"""
    
    def setup_method(self):
        """测试前设置"""
        learner_config = {
            'data_loader_config': {
                'batch_size': 32
            },
            'enable_data_consistency_check': True
        }
        self.learner = DistributedLearner(learner_config)
    
    def test_data_loader_creation(self):
        """测试数据加载器创建"""
        # 创建模拟数据集
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        )
        
        # 创建数据加载器
        data_loader = self.learner.create_data_loader(dataset, batch_size=16)
        
        # 验证配置
        assert data_loader.pin_memory is True
        assert data_loader.batch_size == 16  # 覆盖的配置
        
        logger.info("数据加载器创建测试通过")
    
    def test_batch_consistency_validation(self):
        """测试批次一致性验证"""
        # 测试正常批次
        normal_batch = {
            'obs': torch.randn(4, 10),
            'actions': torch.randint(0, 5, (4,)),
            'rewards': torch.randn(4)
        }
        
        assert self.learner.validate_batch_consistency(normal_batch) is True
        
        # 测试包含NaN的批次
        nan_batch = {
            'obs': torch.tensor([[1.0, float('nan')], [2.0, 3.0]]),
            'actions': torch.tensor([0, 1])
        }
        
        assert self.learner.validate_batch_consistency(nan_batch) is False
        
        # 测试包含无穷值的批次
        inf_batch = {
            'obs': torch.tensor([[1.0, float('inf')], [2.0, 3.0]]),
            'actions': torch.tensor([0, 1])
        }
        
        assert self.learner.validate_batch_consistency(inf_batch) is False
        
        logger.info("批次一致性验证测试通过")


class TestDistributedIntegration:
    """测试分布式集成"""
    
    def test_distributed_config_creation(self):
        """测试分布式配置创建"""
        config = create_distributed_training_config()
        
        # 验证配置结构
        assert 'rollout_worker_config' in config
        assert 'learner_config' in config
        assert 'error_handling_config' in config
        assert 'monitoring_config' in config
        
        # 验证具体配置
        assert config['rollout_worker_config']['enable_graph_data_sharing'] is True
        assert config['learner_config']['data_loader_config']['pin_memory'] is True
        assert config['error_handling_config']['max_retries'] == 3
        
        logger.info("分布式配置创建测试通过")
    
    def test_distributed_ppo_config(self):
        """测试分布式PPO配置"""
        config = DistributedPPOConfig()
        
        # 验证分布式配置
        assert hasattr(config, 'distributed_config')
        assert config.distributed_config['enable_graph_data_sharing'] is True
        
        # 测试配置方法
        config.distributed_training(stats_log_interval=50)
        assert config.distributed_config['stats_log_interval'] == 50
        
        logger.info("分布式PPO配置测试通过")
    
    def test_algorithm_creation(self):
        """测试算法创建"""
        config = create_distributed_training_algorithm(
            algorithm_type="PPO",
            config_overrides={'stats_log_interval': 200}
        )
        
        assert isinstance(config, DistributedPPOConfig)
        assert config.distributed_config['stats_log_interval'] == 200
        
        logger.info("算法创建测试通过")


def test_end_to_end_data_flow():
    """端到端数据流测试"""
    logger.info("开始端到端数据流测试")
    
    # 1. 创建模拟图数据
    obs_dict = {
        'uav_features': torch.randn(2, 4, 12),
        'target_features': torch.randn(2, 6, 10),
        'relative_positions': torch.randn(2, 4, 6, 2),
        'distances': torch.randn(2, 4, 6),
        'masks': {
            'uav_mask': torch.ones(2, 4, dtype=torch.bool),
            'target_mask': torch.ones(2, 6, dtype=torch.bool)
        }
    }
    
    # 2. 数据预处理（RolloutWorker阶段）
    processed_obs = distributed_processor.prepare_graph_data_for_sharing(obs_dict)
    
    # 验证预处理结果
    assert all(tensor.device.type == 'cpu' for key, tensor in processed_obs.items() 
              if isinstance(tensor, torch.Tensor))
    
    # 3. 数据加载器配置（Learner阶段）
    learner_config = {
        'data_loader_config': {'batch_size': 8},
        'enable_data_consistency_check': True
    }
    learner = DistributedLearner(learner_config)
    
    # 4. 数据一致性验证
    assert learner.validate_batch_consistency(processed_obs) is True
    
    # 5. 获取处理统计
    stats = distributed_processor.get_stats()
    assert stats['processed_batches'] > 0
    
    logger.info("端到端数据流测试通过")
    logger.info(f"处理统计: {stats}")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
