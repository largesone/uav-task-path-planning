# -*- coding: utf-8 -*-
"""
测试改进的推理机制

主要测试内容：
1. 验证低温softmax采样机制
2. 验证最佳模型加载功能
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

# 添加父目录到路径，确保能正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from main import run_scenario
from config import Config
from scenarios import get_new_experimental_scenario

def test_improved_inference():
    """测试改进的推理机制"""
    print("=== 测试改进的推理机制 ===")
    
    # 创建测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(20.0)
    config = Config()
    
    # 修改配置以快速训练
    config.training_config.episodes = 15
    config.training_config.patience = 3
    
    print("开始训练测试模型...")
    start_time = time.time()
    
    # 运行场景（会训练并保存最佳模型，然后加载最佳模型进行推理）
    final_plan, training_time, training_history, evaluation_metrics = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="test_inference_improvements",
        network_type="DeepFCNResidual",
        save_visualization=False,
        show_visualization=False,
        force_retrain=True,
        output_base_dir="temp_tests/output"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 测试结果 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"训练时间: {training_time:.2f}秒")
    
    if evaluation_metrics:
        print(f"评估指标:")
        for key, value in evaluation_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("测试完成！")
    return final_plan, training_time, training_history, evaluation_metrics

def main():
    """主测试函数"""
    print("开始测试改进的推理机制...")
    
    try:
        # 测试改进的推理机制
        plan, train_time, history, metrics = test_improved_inference()
        
        print("\n=== 测试完成 ===")
        print("改进的推理机制测试已成功完成")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 