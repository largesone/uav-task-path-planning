# -*- coding: utf-8 -*-
# 文件名: test_improvements.py
# 描述: 测试改进功能的脚本

import os
import sys
import time
import pickle
import json
sys.path.append('..')  # 添加父目录到路径

from main import run_scenario, set_chinese_font
from config import Config
from scenarios import get_new_experimental_scenario

def test_model_loading():
    """测试模型加载功能"""
    print("=== 测试模型加载功能 ===")
    
    # 创建配置
    config = Config()
    config.NETWORK_TYPE = "DeepFCNResidual"
    config.training_config.episodes = 100  # 减少训练轮次用于测试
    
    # 加载测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(50.0)
    
    print("第一次运行 - 训练新模型")
    start_time = time.time()
    result1 = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="测试场景",
        force_retrain=False,
        output_base_dir="temp_test"
    )
    first_run_time = time.time() - start_time
    print(f"第一次运行耗时: {first_run_time:.2f}秒")
    
    print("\n第二次运行 - 应该加载已训练模型")
    start_time = time.time()
    result2 = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="测试场景",
        force_retrain=False,
        output_base_dir="temp_test"
    )
    second_run_time = time.time() - start_time
    print(f"第二次运行耗时: {second_run_time:.2f}秒")
    
    print(f"\n时间节省: {((first_run_time - second_run_time) / first_run_time * 100):.1f}%")
    
    return result1, result2

def test_visualization_improvements():
    """测试可视化改进"""
    print("\n=== 测试可视化改进 ===")
    
    # 创建配置
    config = Config()
    config.NETWORK_TYPE = "DeepFCNResidual"
    config.training_config.episodes = 50
    
    # 加载测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(50.0)
    
    # 运行场景并生成可视化
    result = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="可视化测试",
        save_visualization=True,
        output_base_dir="temp_test"
    )
    
    print("可视化改进测试完成")
    return result

def test_convergence_analysis():
    """测试收敛性分析改进"""
    print("\n=== 测试收敛性分析改进 ===")
    
    # 创建配置
    config = Config()
    config.NETWORK_TYPE = "DeepFCNResidual"
    config.training_config.episodes = 100
    
    # 加载测试场景
    uavs, targets, obstacles = get_new_experimental_scenario(50.0)
    
    # 运行场景并生成收敛性分析
    result = run_scenario(
        config=config,
        base_uavs=uavs,
        base_targets=targets,
        obstacles=obstacles,
        scenario_name="收敛性测试",
        save_visualization=True,
        output_base_dir="temp_test"
    )
    
    print("收敛性分析改进测试完成")
    return result

def cleanup_test_files():
    """清理测试文件"""
    import shutil
    if os.path.exists("temp_test"):
        shutil.rmtree("temp_test")
        print("已清理测试文件")

def main():
    """主测试函数"""
    print("开始测试改进功能...")
    
    try:
        # 测试模型加载功能
        test_model_loading()
        
        # 测试可视化改进
        test_visualization_improvements()
        
        # 测试收敛性分析改进
        test_convergence_analysis()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        cleanup_test_files()

if __name__ == "__main__":
    main() 