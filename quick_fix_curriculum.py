"""
快速修复课程学习训练问题的脚本
"""
import os
import shutil
import logging
from pathlib import Path

def quick_fix():
    """快速修复主要问题"""
    print("开始快速修复...")
    
    # 1. 清理输出目录
    output_dir = Path("curriculum_training_output")
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
            print("✓ 已清理输出目录")
        except:
            print("⚠ 输出目录清理部分失败，继续...")
    
    # 2. 重新创建目录
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o755)
    print("✓ 已重新创建输出目录")
    
    # 3. 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('curriculum_training_fix.log'),
            logging.StreamHandler()
        ]
    )
    print("✓ 已配置日志系统")
    
    # 4. 检查依赖
    try:
        import torch
        import numpy as np
        print("✓ 核心依赖检查通过")
    except ImportError as e:
        print(f"✗ 依赖检查失败: {e}")
        return False
    
    print("快速修复完成！现在可以运行修复版本的训练脚本。")
    return True

if __name__ == "__main__":
    quick_fix()
