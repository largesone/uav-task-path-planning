"""
任务14综合集成测试
测试训练数据保存与TensorBoard集成的完整功能
"""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有测试模块
from test_training_logger import run_training_logger_tests
from test_curriculum_progress_visualizer import run_visualizer_tests
from test_stage_config_manager import run_stage_config_tests
from test_tensorboard_integration import run_tensorboard_integration_tests


class TestTask14Integration(unittest.TestCase):
    """任务14综合集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_all_modules_importable(self):
        """测试所有模块可以正常导入"""
        try:
            import training_logger
            import curriculum_progress_visualizer
            import stage_config_manager
            import tensorboard_integration
            print("✅ 所有模块导入成功")
        except ImportError as e:
            self.fail(f"模块导入失败: {e}")
    
    def test_dependencies_available(self):
        """测试依赖包可用性"""
        required_packages = [
            'torch',
            'numpy',
            'matplotlib',
            'seaborn',
            'pandas',
            'plotly',
            'tensorboard'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"⚠️ 缺少依赖包: {missing_packages}")
        else:
            print("✅ 所有依赖包可用")
    
    def test_file_structure_integrity(self):
        """测试文件结构完整性"""
        expected_files = [
            'training_logger.py',
            'curriculum_progress_visualizer.py',
            'stage_config_manager.py',
            'tensorboard_integration.py'
        ]
        
        project_root = Path(__file__).parent.parent
        missing_files = []
        
        for file_name in expected_files:
            file_path = project_root / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.fail(f"缺少文件: {missing_files}")
        else:
            print("✅ 文件结构完整")


def run_all_task14_tests():
    """运行任务14的所有测试"""
    print("🚀 开始任务14综合测试...")
    print("=" * 60)
    
    # 运行综合测试
    print("\n1️⃣ 运行综合集成测试...")
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestTask14Integration)
    integration_runner = unittest.TextTestRunner(verbosity=1)
    integration_result = integration_runner.run(integration_suite)
    
    # 运行各模块测试
    test_results = {}
    
    print("\n2️⃣ 运行训练日志记录器测试...")
    test_results['training_logger'] = run_training_logger_tests()
    
    print("\n3️⃣ 运行进度可视化器测试...")
    test_results['visualizer'] = run_visualizer_tests()
    
    print("\n4️⃣ 运行阶段配置管理器测试...")
    test_results['stage_config'] = run_stage_config_tests()
    
    print("\n5️⃣ 运行TensorBoard集成测试...")
    test_results['tensorboard'] = run_tensorboard_integration_tests()
    
    # 输出总体结果
    print("\n" + "=" * 60)
    print("📋 任务14测试总结报告")
    print("=" * 60)
    
    total_success = integration_result.wasSuccessful()
    for module_name, success in test_results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {module_name:25} : {status}")
        total_success = total_success and success
    
    print(f"\n🎯 总体结果: {'✅ 全部通过' if total_success else '❌ 存在失败'}")
    
    if total_success:
        print("\n🎉 恭喜！任务14的所有功能测试通过")
        print("   - TensorBoard集成功能正常")
        print("   - 训练数据保存机制工作正常")
        print("   - 尺度不变指标记录正确")
        print("   - 课程学习进度可视化完整")
        print("   - 模型检查点管理有效")
    else:
        print("\n⚠️ 部分测试失败，请检查相关模块")
    
    return total_success


if __name__ == "__main__":
    success = run_all_task14_tests()
    sys.exit(0 if success else 1)
