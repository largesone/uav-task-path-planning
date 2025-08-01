#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: run_curriculum_tests_fixed.py
描述: 课程学习训练协调器测试运行脚本（修复版）
作者: AI Assistant
日期: 2024

使用方法:
python run_curriculum_tests_fixed.py [--basic-only] [--unit-only] [--integration-only]
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
        # 使用完整修复版本的测试
        import temp_tests.test_curriculum_training_complete as test_module
        return test_module.run_basic_functionality_test()
    except ImportError as e:
        print(f"无法导入基本测试: {e}")
        # 回退到简化版本
        try:
            import temp_tests.test_curriculum_training_simple as simple_test
            return simple_test.run_basic_functionality_test()
        except ImportError as e2:
            print(f"无法导入简化测试: {e2}")
            return False
    except AttributeError as e:
        print(f"无法找到基本测试函数: {e}")
        return False

def run_unit_tests():
    """运行单元测试"""
    print("运行单元测试...")
    try:
        import unittest
        import temp_tests.test_curriculum_training_complete as test_module
        
        # 创建测试套件
        suite = unittest.TestSuite()
        test_classes = [
            test_module.TestCurriculumStage, 
            test_module.TestCurriculumConfig,
            test_module.TestCheckpointManager, 
            test_module.TestCurriculumTrainer
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"无法导入单元测试: {e}")
        return False
    except AttributeError as e:
        print(f"无法找到测试类: {e}")
        return False

def run_integration_tests():
    """运行集成测试"""
    print("运行集成测试...")
    try:
        import unittest
        import temp_tests.test_curriculum_training_complete as test_module
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_module.TestIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"无法导入集成测试: {e}")
        return False
    except AttributeError as e:
        print(f"无法找到集成测试类: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="课程学习训练协调器测试运行器（修复版）")
    parser.add_argument("--basic-only", action="store_true", help="仅运行基本功能测试")
    parser.add_argument("--unit-only", action="store_true", help="仅运行单元测试")
    parser.add_argument("--integration-only", action="store_true", help="仅运行集成测试")
    
    args = parser.parse_args()
    
    print("课程学习训练协调器测试运行器（修复版）")
    print("=" * 50)
    
    success = True
    
    if args.basic_only:
        success = run_basic_tests()
    elif args.unit_only:
        success = run_unit_tests()
    elif args.integration_only:
        success = run_integration_tests()
    else:
        # 运行所有测试
        print("运行完整测试套件...\n")
        
        # 1. 基本功能测试
        basic_success = run_basic_tests()
        print()
        
        if basic_success:
            # 2. 单元测试
            unit_success = run_unit_tests()
            print()
            
            # 3. 集成测试
            integration_success = run_integration_tests()
            
            success = basic_success and unit_success and integration_success
        else:
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过!")
        sys.exit(0)
    else:
        print("✗ 部分测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
