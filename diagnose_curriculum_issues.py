"""
诊断课程学习训练问题的脚本
"""
import os
import sys
import logging
import importlib.util
from pathlib import Path

def diagnose_environment():
    """诊断环境问题"""
    print("=== 环境诊断 ===")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查当前目录
    print(f"当前目录: {os.getcwd()}")
    
    # 检查关键文件
    key_files = ["main.py", "run_curriculum_training.py", "json_serialization_fix.py"]
    for file in key_files:
        if Path(file).exists():
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
    
    # 检查输出目录权限
    output_dir = Path("curriculum_training_output")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
        print("✓ 输出目录权限正常")
    except Exception as e:
        print(f"✗ 输出目录权限问题: {e}")
    
    # 检查依赖模块
    modules = ["torch", "numpy", "json"]
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module} 可用")
        except ImportError:
            print(f"✗ {module} 不可用")

def diagnose_main_module():
    """诊断main模块"""
    print("\n=== main模块诊断 ===")
    
    try:
        from main import run_scenario
        print("✓ run_scenario 函数可导入")
        
        # 检查函数签名
        import inspect
        sig = inspect.signature(run_scenario)
        print(f"函数签名: {sig}")
        
    except ImportError as e:
        print(f"✗ main模块导入失败: {e}")
    except Exception as e:
        print(f"✗ main模块检查失败: {e}")

def diagnose_curriculum_trainer():
    """诊断课程训练器"""
    print("\n=== 课程训练器诊断 ===")
    
    try:
        from run_curriculum_training import CurriculumTrainer, CurriculumConfig
        print("✓ 原始课程训练器可导入")
        
        # 检查构造函数
        import inspect
        sig = inspect.signature(CurriculumTrainer.__init__)
        print(f"CurriculumTrainer.__init__ 签名: {sig}")
        
    except ImportError as e:
        print(f"✗ 课程训练器导入失败: {e}")
    except Exception as e:
        print(f"✗ 课程训练器检查失败: {e}")

def main():
    """主诊断函数"""
    logging.basicConfig(level=logging.INFO)
    
    print("开始诊断课程学习训练问题...\n")
    
    diagnose_environment()
    diagnose_main_module()
    diagnose_curriculum_trainer()
    
    print("\n=== 诊断完成 ===")
    print("请根据上述诊断结果选择合适的修复脚本运行。")

if __name__ == "__main__":
    main()
