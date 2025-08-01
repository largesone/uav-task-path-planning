#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 简化的分布式训练测试

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_distributed_data_processor():
    """测试分布式数据处理器"""
    print("=== 测试分布式数据处理器 ===")
    
    try:
        from distributed_training_utils import DistributedDataProcessor
        
        # 创建处理器
        processor = DistributedDataProcessor()
        print("✓ 分布式数据处理器创建成功")
        
        # 创建测试数据
        test_data = {
            'uav_features': torch.randn(2, 3, 9),
            'target_features': torch.randn(2, 4, 8),
            'distances': torch.randn(2, 3, 4),
            'masks': {
                'uav_mask': torch.ones(2, 3, dtype=torch.bool),
                'target_mask': torch.ones(2, 4, dtype=torch.bool)
            }
        }
        
        # 测试数据处理
        processed_data = processor.prepare_graph_data_for_sharing(test_data)
        print("✓ 图数据内存共享准备成功")
        
        # 验证数据已移到CPU
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == 'cpu', f"张量 {key} 未移到CPU"
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        assert sub_value.device.type == 'cpu', f"嵌套张量 {key}.{sub_key} 未移到CPU"
        
        print("✓ CPU内存共享验证通过")
        
        # 测试统计功能
        stats = processor.get_stats()
        print(f"✓ 统计信息获取成功: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ 分布式数据处理器测试失败: {e}")
        return False

def test_distributed_learner():
    """测试分布式Learner"""
    print("\n=== 测试分布式Learner ===")
    
    try:
        from rllib_distributed_integration import DistributedLearner
        
        # 创建配置
        config = {
            'data_loader_config': {'batch_size': 32},
            'enable_data_consistency_check': True
        }
        
        # 创建Learner
        learner = DistributedLearner(config)
        print("✓ 分布式Learner创建成功")
        
        # 测试数据一致性检查
        test_batch = {
            'features': torch.randn(10, 5),
            'labels': torch.randint(0, 2, (10,))
        }
        
        is_consistent = learner.validate_batch_consistency(test_batch)
        assert is_consistent, "正常数据应该通过一致性检查"
        print("✓ 数据一致性检查通过")
        
        # 测试NaN检测
        nan_batch = {
            'features': torch.tensor([[1.0, float('nan'), 3.0]]),
            'labels': torch.tensor([1])
        }
        
        is_nan_detected = not learner.validate_batch_consistency(nan_batch)
        assert is_nan_detected, "应该检测到NaN值"
        print("✓ NaN值检测通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 分布式Learner测试失败: {e}")
        return False

def test_distributed_config():
    """测试分布式配置"""
    print("\n=== 测试分布式配置 ===")
    
    try:
        from distributed_training_utils import create_distributed_training_config
        
        # 创建配置
        config = create_distributed_training_config()
        print("✓ 分布式训练配置创建成功")
        
        # 验证配置结构
        required_sections = [
            'rollout_worker_config',
            'learner_config',
            'error_handling_config',
            'monitoring_config'
        ]
        
        for section in required_sections:
            assert section in config, f"缺少配置节: {section}"
        
        print("✓ 配置结构验证通过")
        
        # 验证关键配置
        assert config['learner_config']['data_loader_config']['pin_memory'] == True
        assert config['error_handling_config']['max_retries'] == 3
        
        print("✓ 关键配置验证通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 分布式配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始分布式训练集成测试...")
    print("=" * 50)
    
    tests = [
        test_distributed_data_processor,
        test_distributed_learner,
        test_distributed_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"总测试数: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    print(f"成功率: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("✓ 所有分布式训练集成测试通过!")
    else:
        print("✗ 部分测试失败，请检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)