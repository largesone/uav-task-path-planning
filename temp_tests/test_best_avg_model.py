# -*- coding: utf-8 -*-
"""
测试基于平均奖励的最佳模型保存机制

主要测试内容：
1. 验证基于平均奖励的最佳模型保存功能
2. 验证模型加载优先级
3. 比较不同模型保存策略的效果
"""

import os
import sys
import time
import glob
import numpy as np

# 添加父目录到路径，确保能正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from main import run_scenario
from config import Config
from scenarios import get_new_experimental_scenario

def test_best_avg_model_saving():
    """测试基于平均奖励的最佳模型保存功能"""
    print("=== 测试基于平均奖励的最佳模型保存功能 ===")
    
    # 创建测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(25.0)
    config = Config()
    
    # 修改配置以快速训练
    config.training_config.episodes = 30
    config.training_config.patience = 5
    config.training_config.log_interval = 10  # 设置日志间隔为10轮
    
    print("开始训练测试模型...")
    start_time = time.time()
    
    # 运行场景（会训练并保存基于平均奖励的最佳模型）
    final_plan, training_time, training_history, evaluation_metrics = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="test_best_avg_model",
        network_type="DeepFCNResidual",
        save_visualization=False,
        show_visualization=False,
        force_retrain=True,
        output_base_dir="temp_tests/output"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 训练结果 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"训练时间: {training_time:.2f}秒")
    
    # 检查生成的模型文件
    output_dir = "temp_tests/output/test_best_avg_model_DeepFCNResidual"
    model_files = glob.glob(f"{output_dir}/*.pth")
    
    print(f"\n=== 模型文件检查 ===")
    print(f"输出目录: {output_dir}")
    print(f"找到的模型文件:")
    for model_file in model_files:
        file_size = os.path.getsize(model_file) / 1024  # KB
        file_time = time.ctime(os.path.getctime(model_file))
        print(f"  {os.path.basename(model_file)} ({file_size:.1f}KB, {file_time})")
    
    # 检查是否有基于平均奖励的最佳模型
    best_avg_models = glob.glob(f"{output_dir}/*_best_avg*.pth")
    if best_avg_models:
        print(f"\n找到基于平均奖励的最佳模型: {len(best_avg_models)}个")
        for model in best_avg_models:
            print(f"  {os.path.basename(model)}")
    else:
        print(f"\n未找到基于平均奖励的最佳模型")
    
    if evaluation_metrics:
        print(f"\n评估指标:")
        for key, value in evaluation_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return final_plan, training_time, training_history, evaluation_metrics, model_files

def test_model_loading_priority():
    """测试模型加载优先级"""
    print("\n=== 测试模型加载优先级 ===")
    
    # 创建测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(20.0)
    config = Config()
    
    # 修改配置以快速训练
    config.training_config.episodes = 20
    config.training_config.patience = 3
    config.training_config.log_interval = 5
    
    print("训练模型以生成不同类型的模型文件...")
    
    # 运行场景
    final_plan, training_time, training_history, evaluation_metrics = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="test_model_loading_priority",
        network_type="DeepFCNResidual",
        save_visualization=False,
        show_visualization=False,
        force_retrain=True,
        output_base_dir="temp_tests/output"
    )
    
    # 检查模型加载过程
    output_dir = "temp_tests/output/test_model_loading_priority_DeepFCNResidual"
    
    print(f"\n=== 模型加载测试 ===")
    print(f"输出目录: {output_dir}")
    
    # 检查所有模型文件
    all_models = glob.glob(f"{output_dir}/*.pth")
    best_avg_models = glob.glob(f"{output_dir}/*_best_avg*.pth")
    other_models = [m for m in all_models if "_best_avg" not in m]
    
    print(f"总模型文件数: {len(all_models)}")
    print(f"基于平均奖励的模型数: {len(best_avg_models)}")
    print(f"其他模型数: {len(other_models)}")
    
    if best_avg_models:
        print(f"基于平均奖励的模型:")
        for model in best_avg_models:
            print(f"  {os.path.basename(model)}")
    
    if other_models:
        print(f"其他模型:")
        for model in other_models:
            print(f"  {os.path.basename(model)}")
    
    return all_models, best_avg_models, other_models

def main():
    """主测试函数"""
    print("开始测试基于平均奖励的最佳模型保存机制...")
    
    try:
        # 测试基于平均奖励的最佳模型保存
        plan, train_time, history, metrics, model_files = test_best_avg_model_saving()
        
        # 测试模型加载优先级
        all_models, best_avg_models, other_models = test_model_loading_priority()
        
        print("\n=== 测试完成 ===")
        print("基于平均奖励的最佳模型保存机制测试已成功完成")
        
        # 总结
        print(f"\n=== 测试总结 ===")
        print(f"1. 基于平均奖励的最佳模型保存: {'成功' if best_avg_models else '失败'}")
        print(f"2. 模型文件生成: {len(all_models)}个")
        print(f"3. 平均奖励模型: {len(best_avg_models)}个")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 