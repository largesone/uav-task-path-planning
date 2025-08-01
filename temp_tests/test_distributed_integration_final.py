# -*- coding: utf-8 -*-
# 文件名: test_distributed_integration_final.py
# 描述: 分布式训练集成测试 - 验证数据一致性和稳定性

import unittest
import torch
import numpy as np
import logging
import time
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
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
        DistributedPPO,
        create_distributed_training_algorithm
    )
    
    # 模拟Ray RLlib组件
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入分布式训练模块: {e}")
    IMPORTS_AVAILABLE = False


class TestDistributedDataProcessor(unittest.TestCase):
    """测试分布式数据处理器"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("分布式训练模块不可用")
        
        self.processor = DistributedDataProcessor(max_retries=2, retry_delay=0.01)
        
        # 创建测试用的图数据
        self.test_graph_data = {
            'uav_features': torch.randn(2, 3, 9),  # [batch, n_uavs, features]
            'target_features': torch.randn(2, 4, 8),  # [batch, n_targets, features]
            'relative_positions': torch.randn(2, 3, 4, 2),  # [batch, n_uavs, n_targets, 2]
            'distances': torch.randn(2, 3, 4),  # [batch, n_uavs, n_targets]
            'masks': {
                'uav_mask': torch.ones(2, 3, dtype=torch.bool),
                'target_mask': torch.ones(2, 4, dtype=torch.bool)
            }
        }
    
    def test_prepare_graph_data_for_sharing(self):
        """测试图数据内存共享准备"""
        print("\n=== 测试图数据内存共享准备 ===")
        
        # 准备数据
        processed_data = self.processor.prepare_graph_data_for_sharing(self.test_graph_data)
        
        # 验证数据结构保持不变
        self.assertEqual(set(processed_data.keys()), set(self.test_graph_data.keys()))
        
        # 验证张量已移到CPU
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, 'cpu')
                print(f"✓ 张量 {key} 已移到CPU，形状: {value.shape}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        self.assertEqual(sub_value.device.type, 'cpu')
                        print(f"✓ 嵌套张量 {key}.{sub_key} 已移到CPU，形状: {sub_value.shape}")
        
        print("✓ 图数据内存共享准备测试通过")
    
    def test_sparse_tensor_handling(self):
        """测试稀疏张量处理"""
        print("\n=== 测试稀疏张量处理 ===")
        
        # 创建稀疏张量
        indices = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        values = torch.FloatTensor([3, 4, 5])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 3))
        
        # 处理稀疏张量
        processed_tensor = self.processor._handle_sparse_tensor(sparse_tensor)
        
        # 验证处理结果
        self.assertIsInstance(processed_tensor, torch.Tensor)
        print(f"✓ 稀疏张量处理完成，输出形状: {processed_tensor.shape}")
        print(f"✓ 输出张量类型: {'稀疏' if processed_tensor.is_sparse else '密集'}")
        
        # 测试高密度稀疏张量（应转换为密集张量）
        dense_indices = torch.LongTensor([[i, j] for i in range(10) for j in range(10)]).t()
        dense_values = torch.randn(100)
        dense_sparse = torch.sparse_coo_tensor(dense_indices, dense_values, (10, 10))
        
        processed_dense = self.processor._handle_sparse_tensor(dense_sparse)
        self.assertFalse(processed_dense.is_sparse)
        print("✓ 高密度稀疏张量正确转换为密集张量")
    
    def test_data_loader_configuration(self):
        """测试数据加载器配置"""
        print("\n=== 测试数据加载器配置 ===")
        
        # 基础配置
        base_config = {'batch_size': 32}
        
        # 优化配置
        optimized_config = self.processor.configure_data_loader_for_learner(base_config)
        
        # 验证优化配置
        self.assertTrue(optimized_config['pin_memory'])
        self.assertIn('num_workers', optimized_config)
        self.assertIn('persistent_workers', optimized_config)
        self.assertIn('prefetch_factor', optimized_config)
        
        print(f"✓ 数据加载器配置优化完成: {optimized_config}")
    
    def test_retry_mechanism(self):
        """测试重试机制"""
        print("\n=== 测试重试机制 ===")
        
        # 创建会失败的函数
        call_count = 0
        
        @self.processor.retry_on_failure(max_retries=2)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"模拟失败 {call_count}")
            return "成功"
        
        # 测试重试机制
        result = failing_function()
        
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)
        print(f"✓ 重试机制测试通过，总调用次数: {call_count}")
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        print("\n=== 测试统计信息跟踪 ===")
        
        # 重置统计
        self.processor.reset_stats()
        
        # 处理一些数据
        for i in range(5):
            self.processor.prepare_graph_data_for_sharing(self.test_graph_data)
        
        # 获取统计信息
        stats = self.processor.get_stats()
        
        self.assertEqual(stats['processed_batches'], 5)
        self.assertGreaterEqual(stats['success_rate'], 0.8)
        
        print(f"✓ 统计信息跟踪测试通过: {stats}")


class TestDistributedRolloutWorker(unittest.TestCase):
    """测试分布式RolloutWorker"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("分布式训练模块不可用")
        
        # 模拟RolloutWorker的基础功能
        self.mock_env = Mock()
        self.mock_policy = Mock()
        
        # 创建测试用的样本批次
        self.test_sample_batch = SampleBatch({
            'obs': self.create_test_observations(),
            'actions': torch.randint(0, 10, (32,)),
            'rewards': torch.randn(32),
            'dones': torch.zeros(32, dtype=torch.bool)
        })
    
    def create_test_observations(self):
        """创建测试观测数据"""
        return {
            'uav_features': torch.randn(32, 3, 9),
            'target_features': torch.randn(32, 4, 8),
            'relative_positions': torch.randn(32, 3, 4, 2),
            'distances': torch.randn(32, 3, 4),
            'masks': {
                'uav_mask': torch.ones(32, 3, dtype=torch.bool),
                'target_mask': torch.ones(32, 4, dtype=torch.bool)
            }
        }
    
    @patch('ray.rllib.evaluation.rollout_worker.RolloutWorker.__init__')
    def test_distributed_rollout_worker_init(self, mock_init):
        """测试分布式RolloutWorker初始化"""
        print("\n=== 测试分布式RolloutWorker初始化 ===")
        
        mock_init.return_value = None
        
        # 创建分布式配置
        distributed_config = {
            'enable_graph_data_sharing': True,
            'stats_log_interval': 50
        }
        
        # 初始化worker
        worker = DistributedRolloutWorker(distributed_config=distributed_config)
        
        self.assertTrue(worker.enable_graph_data_sharing)
        self.assertEqual(worker.stats_log_interval, 50)
        
        print("✓ 分布式RolloutWorker初始化测试通过")
    
    def test_sample_batch_preparation(self):
        """测试样本批次准备"""
        print("\n=== 测试样本批次准备 ===")
        
        # 准备样本批次
        processed_batch = prepare_sample_batch_for_sharing(self.test_sample_batch)
        
        # 验证批次结构
        self.assertIsInstance(processed_batch, SampleBatch)
        self.assertEqual(set(processed_batch.keys()), set(self.test_sample_batch.keys()))
        
        # 验证观测数据处理
        if isinstance(processed_batch['obs'], dict):
            for key, value in processed_batch['obs'].items():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, 'cpu')
        
        print("✓ 样本批次准备测试通过")


class TestDistributedLearner(unittest.TestCase):
    """测试分布式Learner"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("分布式训练模块不可用")
        
        self.learner_config = {
            'data_loader_config': {
                'batch_size': 32,
                'shuffle': True
            },
            'enable_data_consistency_check': True
        }
        
        self.learner = DistributedLearner(self.learner_config)
    
    def test_data_loader_creation(self):
        """测试数据加载器创建"""
        print("\n=== 测试数据加载器创建 ===")
        
        # 创建模拟数据集
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        # 创建数据加载器
        data_loader = self.learner.create_data_loader(mock_dataset, batch_size=16)
        
        # 验证配置
        self.assertTrue(data_loader.pin_memory)
        self.assertEqual(data_loader.batch_size, 16)
        
        print("✓ 数据加载器创建测试通过")
    
    def test_batch_consistency_validation(self):
        """测试批次一致性验证"""
        print("\n=== 测试批次一致性验证 ===")
        
        # 测试正常批次
        normal_batch = {
            'features': torch.randn(10, 5),
            'labels': torch.randint(0, 2, (10,))
        }
        
        self.assertTrue(self.learner.validate_batch_consistency(normal_batch))
        print("✓ 正常批次验证通过")
        
        # 测试包含NaN的批次
        nan_batch = {
            'features': torch.tensor([[1.0, float('nan'), 3.0]]),
            'labels': torch.tensor([1])
        }
        
        self.assertFalse(self.learner.validate_batch_consistency(nan_batch))
        print("✓ NaN批次检测通过")
        
        # 测试包含无穷值的批次
        inf_batch = {
            'features': torch.tensor([[1.0, float('inf'), 3.0]]),
            'labels': torch.tensor([1])
        }
        
        self.assertFalse(self.learner.validate_batch_consistency(inf_batch))
        print("✓ 无穷值批次检测通过")


class TestDistributedPPOIntegration(unittest.TestCase):
    """测试分布式PPO集成"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("分布式训练模块不可用")
    
    def test_distributed_ppo_config(self):
        """测试分布式PPO配置"""
        print("\n=== 测试分布式PPO配置 ===")
        
        # 创建配置
        config = DistributedPPOConfig()
        
        # 验证分布式配置存在
        self.assertIn('distributed_config', config.__dict__)
        self.assertTrue(config.distributed_config['enable_graph_data_sharing'])
        
        # 测试配置方法
        config.distributed_training(stats_log_interval=200)
        self.assertEqual(config.distributed_config['stats_log_interval'], 200)
        
        print("✓ 分布式PPO配置测试通过")
    
    def test_create_distributed_algorithm(self):
        """测试创建分布式算法"""
        print("\n=== 测试创建分布式算法 ===")
        
        # 创建算法配置
        config = create_distributed_training_algorithm(
            algorithm_type="PPO",
            config_overrides={'enable_graph_data_sharing': True}
        )
        
        self.assertIsInstance(config, DistributedPPOConfig)
        
        print("✓ 分布式算法创建测试通过")


class TestDistributedTrainingConfig(unittest.TestCase):
    """测试分布式训练配置"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("分布式训练模块不可用")
    
    def test_create_distributed_training_config(self):
        """测试创建分布式训练配置"""
        print("\n=== 测试创建分布式训练配置 ===")
        
        config = create_distributed_training_config()
        
        # 验证配置结构
        required_sections = [
            'rollout_worker_config',
            'learner_config', 
            'error_handling_config',
            'monitoring_config'
        ]
        
        for section in required_sections:
            self.assertIn(section, config)
            print(f"✓ 配置节 {section} 存在")
        
        # 验证关键配置项
        self.assertTrue(config['rollout_worker_config']['enable_graph_data_sharing'])
        self.assertTrue(config['learner_config']['data_loader_config']['pin_memory'])
        self.assertEqual(config['error_handling_config']['max_retries'], 3)
        
        print("✓ 分布式训练配置创建测试通过")


class TestEndToEndDistributedTraining(unittest.TestCase):
    """端到端分布式训练测试"""
    
    def setUp(self):
        """测试前准备"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("分布式训练模块不可用")
        
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_distributed_pipeline(self):
        """测试完整的分布式训练流水线"""
        print("\n=== 测试完整的分布式训练流水线 ===")
        
        try:
            # 1. 创建分布式配置
            config = create_distributed_training_config()
            print("✓ 步骤1: 分布式配置创建成功")
            
            # 2. 创建数据处理器
            processor = DistributedDataProcessor()
            print("✓ 步骤2: 数据处理器创建成功")
            
            # 3. 准备测试数据
            test_data = {
                'uav_features': torch.randn(4, 2, 9),
                'target_features': torch.randn(4, 3, 8),
                'distances': torch.randn(4, 2, 3),
                'masks': {
                    'uav_mask': torch.ones(4, 2, dtype=torch.bool),
                    'target_mask': torch.ones(4, 3, dtype=torch.bool)
                }
            }
            
            # 4. 数据预处理
            processed_data = processor.prepare_graph_data_for_sharing(test_data)
            print("✓ 步骤3: 数据预处理成功")
            
            # 5. 创建Learner
            learner = DistributedLearner(config['learner_config'])
            print("✓ 步骤4: Learner创建成功")
            
            # 6. 验证数据一致性
            consistency_check = learner.validate_batch_consistency(processed_data)
            self.assertTrue(consistency_check)
            print("✓ 步骤5: 数据一致性验证通过")
            
            # 7. 获取处理统计
            stats = processor.get_stats()
            print(f"✓ 步骤6: 处理统计获取成功 - {stats}")
            
            print("✓ 完整的分布式训练流水线测试通过")
            
        except Exception as e:
            self.fail(f"分布式训练流水线测试失败: {e}")
    
    def test_error_recovery_and_stability(self):
        """测试错误恢复和稳定性"""
        print("\n=== 测试错误恢复和稳定性 ===")
        
        processor = DistributedDataProcessor(max_retries=2, retry_delay=0.01)
        
        # 模拟网络错误
        error_count = 0
        
        @processor.retry_on_failure()
        def simulate_network_error():
            nonlocal error_count
            error_count += 1
            if error_count < 3:
                raise ConnectionError("模拟网络错误")
            return "恢复成功"
        
        # 测试错误恢复
        result = simulate_network_error()
        self.assertEqual(result, "恢复成功")
        self.assertEqual(error_count, 3)
        
        print("✓ 错误恢复和稳定性测试通过")


def run_distributed_integration_tests():
    """运行所有分布式集成测试"""
    print("开始运行分布式训练集成测试...")
    print("=" * 60)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试套件
    test_classes = [
        TestDistributedDataProcessor,
        TestDistributedRolloutWorker,
        TestDistributedLearner,
        TestDistributedPPOIntegration,
        TestDistributedTrainingConfig,
        TestEndToEndDistributedTraining
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n运行测试类: {test_class.__name__}")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        
        if result.failures:
            failed_tests.extend([f"{test_class.__name__}: {f[0]}" for f in result.failures])
        if result.errors:
            failed_tests.extend([f"{test_class.__name__}: {e[0]}" for e in result.errors])
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("分布式训练集成测试总结")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {len(failed_tests)}")
    print(f"成功率: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
    
    if failed_tests:
        print("\n失败的测试:")
        for failed_test in failed_tests:
            print(f"  - {failed_test}")
    
    print("\n分布式训练集成测试完成!")
    
    return total_tests, passed_tests, len(failed_tests)


if __name__ == "__main__":
    # 运行测试
    total, passed, failed = run_distributed_integration_tests()
    
    # 退出码
    sys.exit(0 if failed == 0 else 1)
