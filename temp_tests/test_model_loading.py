# -*- coding: utf-8 -*-
# 文件名: test_model_loading.py
# 描述: 测试模型加载功能

import os
import glob
import time
import sys
sys.path.append('..')  # 添加父目录到路径

from main import run_scenario
from config import Config
from scenarios import get_new_experimental_scenario

def test_model_loading():
    """测试模型加载功能"""
    print("=== 测试模型加载功能 ===")
    
    # 创建配置
    config = Config()
    config.NETWORK_TYPE = "DeepFCNResidual"
    config.training_config.episodes = 50  # 减少训练轮次用于测试
    
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
    
    if second_run_time < first_run_time * 0.5:  # 如果第二次运行时间少于第一次的一半
        print(f"✅ 模型加载功能正常！时间节省: {((first_run_time - second_run_time) / first_run_time * 100):.1f}%")
    else:
        print(f"❌ 模型加载功能可能有问题，时间节省不明显")
    
    return result1, result2

def check_model_files():
    """检查模型文件"""
    print("\n=== 检查模型文件 ===")
    
    # 检查temp_test目录下的模型文件
    if os.path.exists("temp_test"):
        for root, dirs, files in os.walk("temp_test"):
            print(f"目录: {root}")
            for file in files:
                if file.endswith('.pth'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_time = time.ctime(os.path.getctime(file_path))
                    print(f"  模型文件: {file} ({file_size} bytes, {file_time})")
                elif file.endswith('.json'):
                    print(f"  信息文件: {file}")
    
    # 检查output目录下的模型文件
    if os.path.exists("output"):
        for root, dirs, files in os.walk("output"):
            print(f"目录: {root}")
            for file in files:
                if file.endswith('.pth'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_time = time.ctime(os.path.getctime(file_path))
                    print(f"  模型文件: {file} ({file_size} bytes, {file_time})")

def cleanup_test_files():
    """清理测试文件"""
    import shutil
    if os.path.exists("temp_test"):
        shutil.rmtree("temp_test")
        print("已清理测试文件")

def main():
    """主测试函数"""
    print("开始测试模型加载功能...")
    
    try:
        # 测试模型加载功能
        test_model_loading()
        
        # 检查模型文件
        check_model_files()
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        cleanup_test_files()

if __name__ == "__main__":
    main() 