#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: run_curriculum_tests_stable.py
描述: 课程学习训练协调器稳定测试运行脚本
作者: AI Assistant
日期: 2024

特点:
- 解决日志冲突问题
- 稳定的测试环境
- 简化的输出
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "temp_tests"))

def run_basic_tests():
    """运行基本功能测试"""
    print("运行基本功能测试...")
    try:
        import temp_tests.test_curriculum_final as test_module
        return test_module.run_basic_functionality_test()
    except ImportError as e:
        print(f"无法导入测试模块: {e}")
        return False
    except Exception as e:
        print(f"测试执行失败: {e}")
        return False

def run_unit_tests():
    """运行单元测试"""
    print("运行单元测试...")
    try:
        import temp_tests.test_curriculum_final as test_module
        exit_code = test_module.main()
        return exit_code == 0
    except ImportError as e:
        print(f"无法导入测试模块: {e}")
        return False
    except Exception as e:
        print(f"测试执行失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="课程学习训练协调器稳定测试运行器")
    parser.add_argument("--basic-only", action="store_true", help="仅运行基本功能测试")
    parser.add_argument("--unit-only", action="store_true", help="仅运行单元测试")
    
    args = parser.parse_args()
    
    print("课程学习训练协调器稳定测试运行器")
    print("=" * 50)
    
    success = True
    
    if args.basic_only:
        success = run_basic_tests()
    elif args.unit_only:
        success = run_unit_tests()
    else:
        # 运行所有测试
        print("运行完整测试套件...\n")
        
        # 1. 基本功能测试
        basic_success = run_basic_tests()
        print()
        
        if basic_success:
            # 2. 单元测试
            unit_success = run_unit_tests()
            success = basic_success and unit_success
        else:
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过!")
        print("课程学习训练协调器验证完成，功能正常")
        sys.exit(0)
    else:
        print("✗ 部分测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
