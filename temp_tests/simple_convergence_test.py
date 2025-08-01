# -*- coding: utf-8 -*-
# 文件名: simple_convergence_test.py
# 描述: 简化的收敛性测试脚本

import os
import time
from config import Config
from main import run_scenario
from scenarios import get_small_scenario

def test_single_network():
    """测试单个网络的改进效果"""
    print("=" * 60)
    print("简化收敛性测试")
    print("=" * 60)
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = f"output/simple_convergence_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置参数
    config = Config()
    config.NETWORK_TYPE = "DeepFCNResidual"  # 使用优化的网络
    
    # 获取测试场景
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    scenario_name = "convergence_test"
    
    print(f"测试配置:")
    print(f"- 网络类型: {config.NETWORK_TYPE}")
    print(f"- 训练轮次: {config.training_config.episodes}")
    print(f"- 学习率: {config.training_config.learning_rate}")
    print(f"- 早停耐心值: {config.training_config.patience}")
    print(f"- 探索率衰减: {config.training_config.epsilon_decay}")
    print()
    
    try:
        # 运行训练和测试
        result = run_scenario(
            config=config,
            base_uavs=uavs,
            base_targets=targets,
            obstacles=obstacles,
            scenario_name=scenario_name,
            network_type=config.NETWORK_TYPE,
            output_base_dir=output_dir,
            force_retrain=True  # 强制重新训练
        )
        
        print("\n" + "=" * 60)
        print("测试结果摘要:")
        print("=" * 60)
        
        if result:
            training_time = result.get('training_time', 0)
            evaluation_metrics = result.get('evaluation_metrics', {})
            
            print(f"训练时间: {training_time:.2f}秒")
            print(f"完成率: {evaluation_metrics.get('completion_rate', 0):.3f}")
            print(f"目标满足率: {evaluation_metrics.get('satisfied_targets_rate', 0):.3f}")
            print(f"资源利用率: {evaluation_metrics.get('resource_utilization_rate', 0):.3f}")
            print(f"总奖励分数: {evaluation_metrics.get('total_reward_score', 0):.2f}")
            
            # 分析改进效果
            print("\n改进效果分析:")
            print("-" * 30)
            
            completion_rate = evaluation_metrics.get('completion_rate', 0)
            if completion_rate >= 0.9:
                print("✓ 完成率优秀 (≥90%)")
            elif completion_rate >= 0.8:
                print("⚠ 完成率良好 (80-90%)")
            else:
                print("✗ 完成率需要改进 (<80%)")
            
            resource_util = evaluation_metrics.get('resource_utilization_rate', 0)
            if resource_util >= 0.8:
                print("✓ 资源利用率高效 (≥80%)")
            elif resource_util >= 0.6:
                print("⚠ 资源利用率中等 (60-80%)")
            else:
                print("✗ 资源利用率低 (<60%)")
        
        else:
            print("✗ 测试失败，无结果返回")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n测试完成，结果保存在: {output_dir}")
    return output_dir

if __name__ == "__main__":
    test_single_network()