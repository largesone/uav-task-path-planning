"""
课程学习进度可视化器测试
测试各种可视化功能和图表生成
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum_progress_visualizer import CurriculumProgressVisualizer


class TestCurriculumProgressVisualizer(unittest.TestCase):
    """测试课程学习进度可视化器"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = CurriculumProgressVisualizer(self.test_dir, "test_experiment")
        
        # 创建测试数据
        self.test_history = {
            "metrics": [
                {
                    "step": i,
                    "stage": i // 30,
                    "n_uavs": 3 + (i // 30),
                    "n_targets": 2 + (i // 30),
                    "per_agent_reward": 10 + np.random.normal(0, 1),
                    "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
                    "efficiency_metric": 0.3 + np.random.uniform(-0.05, 0.1),
                    "timestamp": datetime.now().isoformat()
                }
                for i in range(100)
            ],
            "stage_transitions": [
                {"step": 30, "from_stage": 0, "to_stage": 1, "reason": "performance_threshold"},
                {"step": 60, "from_stage": 1, "to_stage": 2, "reason": "performance_threshold"},
                {"step": 90, "from_stage": 2, "to_stage": 3, "reason": "performance_threshold"}
            ],
            "rollback_events": [
                {"step": 45, "stage": 1, "performance_drop": 0.15, "threshold": 0.1}
            ]
        }
        
        # 保存测试历史数据
        history_file = Path(self.test_dir) / "test_experiment_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_history, f, indent=2)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.visualizer.viz_dir.exists())
        self.assertEqual(self.visualizer.experiment_name, "test_experiment")
    
    def test_load_training_history(self):
        """测试训练历史加载"""
        history = self.visualizer.load_training_history()
        
        self.assertIn("metrics", history)
        self.assertIn("stage_transitions", history)
        self.assertIn("rollback_events", history)
        
        self.assertEqual(len(history["metrics"]), 100)
        self.assertEqual(len(history["stage_transitions"]), 3)
        self.assertEqual(len(history["rollback_events"]), 1)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_scale_invariant_metrics(self, mock_close, mock_savefig):
        """测试尺度不变指标绘图"""
        history = self.visualizer.load_training_history()
        save_path = self.visualizer.plot_scale_invariant_metrics(history)
        
        # 验证保存路径
        expected_path = self.visualizer.viz_dir / "scale_invariant_metrics.png"
        self.assertEqual(save_path, str(expected_path))
        
        # 验证matplotlib函数被调用
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_curriculum_stages(self, mock_close, mock_savefig):
        """测试课程学习阶段绘图"""
        history = self.visualizer.load_training_history()
        save_path = self.visualizer.plot_curriculum_stages(history)
        
        # 验证保存路径
        expected_path = self.visualizer.viz_dir / "curriculum_stages.png"
        self.assertEqual(save_path, str(expected_path))
        
        # 验证matplotlib函数被调用
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_scale_transfer_analysis(self, mock_close, mock_savefig):
        """测试尺度迁移分析绘图"""
        history = self.visualizer.load_training_history()
        save_path = self.visualizer.plot_scale_transfer_analysis(history)
        
        # 验证保存路径
        expected_path = self.visualizer.viz_dir / "scale_transfer_analysis.png"
        self.assertEqual(save_path, str(expected_path))
        
        # 验证matplotlib函数被调用
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_interactive_dashboard(self, mock_write_html):
        """测试交互式仪表板创建"""
        history = self.visualizer.load_training_history()
        save_path = self.visualizer.create_interactive_dashboard(history)
        
        # 验证保存路径
        expected_path = self.visualizer.viz_dir / "interactive_dashboard.html"
        self.assertEqual(save_path, str(expected_path))
        
        # 验证plotly函数被调用
        mock_write_html.assert_called_once()
    
    def test_generate_training_report(self):
        """测试训练报告生成"""
        history = self.visualizer.load_training_history()
        report_path = self.visualizer.generate_training_report(history)
        
        # 验证报告文件存在
        self.assertTrue(os.path.exists(report_path))
        
        # 验证报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("课程学习训练报告", content)
        self.assertIn("test_experiment", content)
        self.assertIn("阶段切换次数", content)
        self.assertIn("回退事件次数", content)
    
    @patch('curriculum_progress_visualizer.CurriculumProgressVisualizer.plot_scale_invariant_metrics')
    @patch('curriculum_progress_visualizer.CurriculumProgressVisualizer.plot_curriculum_stages')
    @patch('curriculum_progress_visualizer.CurriculumProgressVisualizer.plot_scale_transfer_analysis')
    @patch('curriculum_progress_visualizer.CurriculumProgressVisualizer.create_interactive_dashboard')
    @patch('curriculum_progress_visualizer.CurriculumProgressVisualizer.generate_training_report')
    def test_generate_all_visualizations(self, mock_report, mock_dashboard, 
                                       mock_transfer, mock_stages, mock_metrics):
        """测试生成所有可视化"""
        # 设置模拟返回值
        mock_metrics.return_value = "metrics.png"
        mock_stages.return_value = "stages.png"
        mock_transfer.return_value = "transfer.png"
        mock_dashboard.return_value = "dashboard.html"
        mock_report.return_value = "report.md"
        
        results = self.visualizer.generate_all_visualizations()
        
        # 验证所有方法被调用
        mock_metrics.assert_called_once()
        mock_stages.assert_called_once()
        mock_transfer.assert_called_once()
        mock_dashboard.assert_called_once()
        mock_report.assert_called_once()
        
        # 验证返回结果
        expected_keys = ['metrics', 'stages', 'transfer', 'dashboard', 'report']
        for key in expected_keys:
            self.assertIn(key, results)


class TestVisualizationDataProcessing(unittest.TestCase):
    """测试可视化数据处理"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = CurriculumProgressVisualizer(self.test_dir, "test_experiment")
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_history = {
            "metrics": [],
            "stage_transitions": [],
            "rollback_events": []
        }
        
        # 保存空历史数据
        history_file = Path(self.test_dir) / "test_experiment_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(empty_history, f)
        
        # 测试各种可视化函数不会崩溃
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            metrics_path = self.visualizer.plot_scale_invariant_metrics(empty_history)
            stages_path = self.visualizer.plot_curriculum_stages(empty_history)
            transfer_path = self.visualizer.plot_scale_transfer_analysis(empty_history)
        
        # 验证路径正确返回
        self.assertTrue(metrics_path.endswith("scale_invariant_metrics.png"))
        self.assertTrue(stages_path.endswith("curriculum_stages.png"))
        self.assertTrue(transfer_path.endswith("scale_transfer_analysis.png"))
    
    def test_missing_columns_handling(self):
        """测试缺失列处理"""
        incomplete_history = {
            "metrics": [
                {"step": 1, "stage": 0},  # 缺少其他指标
                {"step": 2, "stage": 0, "per_agent_reward": 10.0}  # 部分指标
            ],
            "stage_transitions": [],
            "rollback_events": []
        }
        
        # 测试不会因为缺失列而崩溃
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            try:
                self.visualizer.plot_scale_invariant_metrics(incomplete_history)
                self.visualizer.plot_scale_transfer_analysis(incomplete_history)
            except Exception as e:
                self.fail(f"可视化函数在处理不完整数据时崩溃: {e}")


def run_visualizer_tests():
    """运行所有可视化器测试"""
    print("🎨 开始课程学习进度可视化器测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCurriculumProgressVisualizer,
        TestVisualizationDataProcessing
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print(f"\n📊 可视化器测试结果摘要:")
    print(f"   总测试数: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_visualizer_tests()
    sys.exit(0 if success else 1)
