# -*- coding: utf-8 -*-
# 文件名: quick_batch_test.py
# 描述: 快速批处理测试脚本

import sys
sys.path.append('..')  # 添加父目录到路径

from batch_scenario_validation import BatchScenarioValidator

def quick_test():
    """快速测试批处理验证功能"""
    print("=== 快速批处理测试 ===")
    
    # 创建验证器
    validator = BatchScenarioValidator(output_dir="../output")
    
    # 测试少量场景
    test_scenarios = ['experimental', 'balanced']
    test_networks = ['DeepFCNResidual']
    
    print(f"测试场景: {test_scenarios}")
    print(f"测试网络: {test_networks}")
    
    # 运行批处理验证
    validator.run_batch_validation(
        scenario_types=test_scenarios,
        network_types=test_networks,
        episodes=50,  # 减少训练轮数用于快速测试
        force_retrain=False
    )

if __name__ == "__main__":
    quick_test() 