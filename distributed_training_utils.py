# -*- coding: utf-8 -*-
# 文件名: distributed_training_utils.py
# 描述: 分布式训练数据一致性工具，解决GNN稀疏张量跨进程传输问题

import torch
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Union, List
from functools import wraps
import traceback

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType


class DistributedDataProcessor:
    """
    分布式数据处理器
    
    负责处理图数据在分布式训练中的数据一致性问题：
    1. 张量内存共享预处理
    2. 稀疏张量兼容性处理
    3. 数据传输异常处理和重试
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        """
        初始化分布式数据处理器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            'processed_batches': 0,
            'failed_batches': 0,
            'retry_count': 0,
            'total_processing_time': 0.0
        }
    
    def prepare_graph_data_for_sharing(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        为图数据字典准备内存共享
        
        在RolloutWorker中调用，对图数据字典的张量调用.cpu().share_memory_()
        确保数据可以在进程间安全传输
        
        Args:
            obs_dict: 观测字典，包含图结构数据
            
        Returns:
            处理后的观测字典，张量已准备好内存共享
        """
        start_time = time.time()
        
        try:
            processed_dict = {}
            
            for key, value in obs_dict.items():
                if isinstance(value, torch.Tensor):
                    # 将张量移到CPU并启用内存共享
                    processed_tensor = value.detach().cpu()
                    
                    # 处理稀疏张量的特殊情况
                    if processed_tensor.is_sparse:
                        processed_tensor = self._handle_sparse_tensor(processed_tensor)
                    
                    # 启用内存共享
                    processed_tensor.share_memory_()
                    processed_dict[key] = processed_tensor
                    
                    self.logger.debug(f"张量 {key} 已准备内存共享，形状: {processed_tensor.shape}")
                    
                elif isinstance(value, dict):
                    # 递归处理嵌套字典
                    processed_dict[key] = self.prepare_graph_data_for_sharing(value)
                    
                elif isinstance(value, (list, tuple)):
                    # 处理张量列表
                    processed_list = []
                    for item in value:
                        if isinstance(item, torch.Tensor):
                            processed_tensor = item.detach().cpu()
                            if processed_tensor.is_sparse:
                                processed_tensor = self._handle_sparse_tensor(processed_tensor)
                            processed_tensor.share_memory_()
                            processed_list.append(processed_tensor)
                        else:
                            processed_list.append(item)
                    
                    processed_dict[key] = type(value)(processed_list)
                    
                else:
                    # 非张量数据直接复制
                    processed_dict[key] = value
            
            processing_time = time.time() - start_time
            self.stats['processed_batches'] += 1
            self.stats['total_processing_time'] += processing_time
            
            self.logger.debug(f"图数据内存共享准备完成，耗时: {processing_time:.4f}秒")
            
            return processed_dict
            
        except Exception as e:
            self.stats['failed_batches'] += 1
            self.logger.error(f"图数据内存共享准备失败: {e}")
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            
            # 返回原始数据作为备用方案
            return obs_dict
    
    def _handle_sparse_tensor(self, sparse_tensor: torch.Tensor) -> torch.Tensor:
        """
        处理稀疏张量的跨进程传输兼容性
        
        Args:
            sparse_tensor: 稀疏张量
            
        Returns:
            处理后的张量（可能转换为密集张量）
        """
        try:
            # 检查稀疏张量的大小和稀疏度
            total_elements = sparse_tensor.numel()
            sparse_elements = sparse_tensor._nnz()
            sparsity_ratio = sparse_elements / total_elements if total_elements > 0 else 0
            
            self.logger.debug(f"稀疏张量统计 - 总元素: {total_elements}, 非零元素: {sparse_elements}, 稀疏度: {sparsity_ratio:.4f}")
            
            # 如果稀疏度较低（密集度高），转换为密集张量
            if sparsity_ratio > 0.1:  # 如果非零元素超过10%，转换为密集张量
                dense_tensor = sparse_tensor.to_dense()
                self.logger.debug(f"稀疏张量转换为密集张量，形状: {dense_tensor.shape}")
                return dense_tensor
            else:
                # 保持稀疏格式，但确保兼容性
                # 将稀疏张量转换为COO格式，这是最兼容的格式
                if sparse_tensor.layout != torch.sparse_coo:
                    coo_tensor = sparse_tensor.coalesce()
                    self.logger.debug(f"稀疏张量转换为COO格式")
                    return coo_tensor
                else:
                    return sparse_tensor.coalesce()
                    
        except Exception as e:
            self.logger.warning(f"稀疏张量处理失败，转换为密集张量: {e}")
            # 备用方案：强制转换为密集张量
            return sparse_tensor.to_dense()
    
    def configure_data_loader_for_learner(self, data_loader_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为Learner配置数据加载器
        
        配置数据加载器使用pin_memory=True，优化GPU数据传输
        
        Args:
            data_loader_config: 数据加载器配置
            
        Returns:
            优化后的数据加载器配置
        """
        optimized_config = data_loader_config.copy()
        
        # 启用内存固定，加速CPU到GPU的数据传输
        optimized_config['pin_memory'] = True
        
        # 设置合适的工作进程数量
        if 'num_workers' not in optimized_config:
            optimized_config['num_workers'] = min(4, torch.get_num_threads())
        
        # 启用持久化工作进程（如果支持）
        if 'persistent_workers' not in optimized_config and optimized_config['num_workers'] > 0:
            optimized_config['persistent_workers'] = True
        
        # 设置预取因子
        if 'prefetch_factor' not in optimized_config and optimized_config['num_workers'] > 0:
            optimized_config['prefetch_factor'] = 2
        
        self.logger.info(f"数据加载器配置已优化: {optimized_config}")
        
        return optimized_config
    
    def retry_on_failure(self, max_retries: Optional[int] = None):
        """
        异常处理和重试机制装饰器
        
        Args:
            max_retries: 最大重试次数，None表示使用默认值
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries if max_retries is not None else self.max_retries
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        if attempt > 0:
                            self.logger.info(f"函数 {func.__name__} 在第 {attempt + 1} 次尝试后成功")
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        self.stats['retry_count'] += 1
                        
                        if attempt < retries:
                            self.logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                            self.logger.warning(f"将在 {self.retry_delay} 秒后重试...")
                            time.sleep(self.retry_delay)
                        else:
                            self.logger.error(f"函数 {func.__name__} 在 {retries + 1} 次尝试后仍然失败")
                            self.logger.error(f"最终错误: {e}")
                            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
                
                # 所有重试都失败，抛出最后一个异常
                raise last_exception
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        if stats['processed_batches'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['processed_batches']
            stats['success_rate'] = (stats['processed_batches'] - stats['failed_batches']) / stats['processed_batches']
        else:
            stats['avg_processing_time'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'processed_batches': 0,
            'failed_batches': 0,
            'retry_count': 0,
            'total_processing_time': 0.0
        }


# 全局分布式数据处理器实例
distributed_processor = DistributedDataProcessor()


def prepare_sample_batch_for_sharing(sample_batch: SampleBatch) -> SampleBatch:
    """
    为SampleBatch准备内存共享
    
    Args:
        sample_batch: 原始样本批次
        
    Returns:
        准备好内存共享的样本批次
    """
    @distributed_processor.retry_on_failure(max_retries=2)
    def _prepare_batch():
        processed_data = {}
        
        for key, value in sample_batch.data.items():
            if isinstance(value, torch.Tensor):
                processed_data[key] = distributed_processor.prepare_graph_data_for_sharing({key: value})[key]
            elif isinstance(value, dict):
                processed_data[key] = distributed_processor.prepare_graph_data_for_sharing(value)
            else:
                processed_data[key] = value
        
        return SampleBatch(processed_data)
    
    return _prepare_batch()


def create_distributed_training_config() -> Dict[str, Any]:
    """
    创建分布式训练配置
    
    Returns:
        分布式训练配置字典
    """
    return {
        # RolloutWorker配置
        'rollout_worker_config': {
            'enable_graph_data_sharing': True,
            'data_processor': distributed_processor,
            'batch_preparation_timeout': 30.0
        },
        
        # Learner配置
        'learner_config': {
            'data_loader_config': {
                'pin_memory': True,
                'num_workers': min(4, torch.get_num_threads()),
                'persistent_workers': True,
                'prefetch_factor': 2
            },
            'enable_data_consistency_check': True
        },
        
        # 异常处理配置
        'error_handling_config': {
            'max_retries': 3,
            'retry_delay': 0.1,
            'enable_fallback_mode': True
        },
        
        # 监控配置
        'monitoring_config': {
            'enable_stats_logging': True,
            'stats_log_interval': 100,  # 每100个batch记录一次统计
            'enable_performance_profiling': False
        }
    }
