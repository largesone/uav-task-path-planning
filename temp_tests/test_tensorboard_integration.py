"""
TensorBoard集成测试
测试自定义TensorBoard功能和高级可视化
"""

import os
import sys
import unittest
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorboard_integration import CurriculumTensorBoardWriter, TensorBoardCustomPlugin


class TestCurriculumTensorBoardWriter(unittest.TestCase):
    """测试课程学习TensorBoard写入器"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.writer = CurriculumTensorBoardWriter(self.test_dir, "test_experiment")
    
    def tearDown(self):
        """测试后清理"""
        self.writer.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.writer.tb_dir.exists())
        self.assertEqual(self.writer.experiment_name, "test_experiment")
        self.assertIsNotNone(self.writer.writer)
        self.assertIn("Scale_Invariant_Metrics", self.writer.scalar_groups)
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_hparams')
    def test_log_hparams(self, mock_add_hparams):
        """测试超参数记录"""
        hparams = {
            "learning_rate": 0.001,
            "batch_size": 128,
            "k_neighbors": 8
        }
        metrics = {
            "final_performance": 0.85,
            "total_episodes": 1000
        }
        
        self.writer.log_hparams(hparams, metrics)
        
        # 验证超参数被记录
        mock_add_hparams.assert_called_once()
        self.assertEqual(len(self.writer.hparams), 3)
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_scalar')
    @patch('torch.utils.tensorboard.SummaryWriter.add_text')
    def test_log_curriculum_stage_transition(self, mock_add_text, mock_add_scalar):
        """测试课程学习阶段切换记录"""
        performance_data = {
            "normalized_completion_score": 0.85,
            "per_agent_reward": 12.5
        }
        
        self.writer.log_curriculum_stage_transition(0, 1, 1000, performance_data)
        
        # 验证标量和文本被记录
        mock_add_scalar.assert_called()
        mock_add_text.assert_called()
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_scalar')
    def test_log_scale_invariant_metrics_detailed(self, mock_add_scalar):
        """测试详细尺度不变指标记录"""
        metrics = {
            "per_agent_reward": 12.5,
            "normalized_completion_score": 0.85,
            "efficiency_metric": 0.42
        }
        scenario_info = {
            "n_uavs": 5,
            "n_targets": 3
        }
        
        self.writer.log_scale_invariant_metrics_detailed(metrics, 1000, 1, scenario_info)
        
        # 验证所有指标被记录
        expected_calls = len(metrics) + len(scenario_info) + 2  # +2 for scale_factor and performance_density
        self.assertGreaterEqual(mock_add_scalar.call_count, expected_calls)
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_figure')
    @patch('torch.utils.tensorboard.SummaryWriter.add_histogram')
    @patch('torch.utils.tensorboard.SummaryWriter.add_scalar')
    def test_log_attention_weights(self, mock_add_scalar, mock_add_histogram, mock_add_figure):
        """测试注意力权重记录"""
        # 创建模拟注意力权重 [batch, heads, seq_len, seq_len]
        attention_weights = torch.rand(2, 4, 10, 10)
        
        self.writer.log_attention_weights(attention_weights, 1000, "test_layer")
        
        # 验证图表、直方图和标量被记录
        mock_add_figure.assert_called_once()
        mock_add_histogram.assert_called_once()
        mock_add_scalar.assert_called_once()
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_scalar')
    @patch('torch.utils.tensorboard.SummaryWriter.add_histogram')
    def test_log_model_gradients(self, mock_add_histogram, mock_add_scalar):
        """测试模型梯度记录"""
        # 创建模拟模型
        model = torch.nn.Linear(10, 5)
        
        # 创建模拟梯度
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        self.writer.log_model_gradients(model, 1000)
        
        # 验证梯度信息被记录
        self.assertGreater(mock_add_scalar.call_count, 0)
        self.assertGreater(mock_add_histogram.call_count, 0)
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_figure')
    def test_log_learning_curves(self, mock_add_figure):
        """测试学习曲线记录"""
        train_metrics = {
            "loss": [1.0, 0.8, 0.6, 0.4],
            "accuracy": [0.6, 0.7, 0.8, 0.9]
        }
        val_metrics = {
            "loss": [1.2, 0.9, 0.7, 0.5],
            "accuracy": [0.5, 0.65, 0.75, 0.85]
        }
        steps = [100, 200, 300, 400]
        
        self.writer.log_learning_curves(train_metrics, val_metrics, steps)
        
        # 验证学习曲线图被创建
        self.assertEqual(mock_add_figure.call_count, 2)  # loss和accuracy各一个
    
    @patch('torch.utils.tensorboard.SummaryWriter.add_text')
    def test_log_curriculum_progress_summary(self, mock_add_text):
        """测试课程学习进度摘要记录"""
        stage_summaries = {
            0: {"completed": True, "best_performance": 0.85, "total_steps": 1000, "n_uavs": 3, "n_targets": 2},
            1: {"completed": False, "best_performance": 0.75, "total_steps": 500, "n_uavs": 5, "n_targets": 3}
        }
        
        self.writer.log_curriculum_progress_summary(stage_summaries, 1500)
        
        # 验证摘要文本被记录
        mock_add_text.assert_called()
    
    def test_calculate_attention_entropy(self):
        """测试注意力熵计算"""
        # 创建均匀分布的注意力权重（高熵）
        uniform_attention = torch.ones(5, 5) / 25
        high_entropy = self.writer._calculate_attention_entropy(uniform_attention)
        
        # 创建集中分布的注意力权重（低熵）
        concentrated_attention = torch.zeros(5, 5)
        concentrated_attention[0, 0] = 1.0
        low_entropy = self.writer._calculate_attention_entropy(concentrated_attention)
        
        # 验证熵的相对大小
        self.assertGreater(high_entropy, low_entropy)


class TestTensorBoardCustomPlugin(unittest.TestCase):
    """测试TensorBoard自定义插件"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.plugin = TensorBoardCustomPlugin(self.test_dir)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.plugin.plugin_dir.exists())
    
    def test_create_curriculum_dashboard_config(self):
        """测试创建课程学习仪表板配置"""
        config_path = self.plugin.create_curriculum_dashboard_config()
        
        # 验证配置文件存在
        self.assertTrue(os.path.exists(config_path))
        
        # 验证配置内容
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.assertIn("name", config)
        self.assertIn("layout", config)
        self.assertIn("sections", config["layout"])
    
    def test_generate_custom_html_dashboard(self):
        """测试生成自定义HTML仪表板"""
        html_path = self.plugin.generate_custom_html_dashboard("test_experiment")
        
        # 验证HTML文件存在
        self.assertTrue(os.path.exists(html_path))
        
        # 验证HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        self.assertIn("test_experiment", html_content)
        self.assertIn("课程学习训练监控仪表板", html_content)
        self.assertIn("plotly", html_content.lower())


class TestTensorBoardIntegrationScenarios(unittest.TestCase):
    """TensorBoard集成场景测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_tensorboard_workflow(self):
        """测试完整的TensorBoard工作流"""
        # 1. 初始化写入器
        writer = CurriculumTensorBoardWriter(self.test_dir, "workflow_test")
        
        # 2. 记录超参数
        hparams = {"learning_rate": 0.001, "batch_size": 128}
        metrics = {"final_score": 0.85}
        
        with patch.object(writer.writer, 'add_hparams'):
            writer.log_hparams(hparams, metrics)
        
        # 3. 模拟训练过程记录
        with patch.object(writer.writer, 'add_scalar') as mock_scalar:
            for step in range(0, 1000, 100):
                stage = step // 300
                
                # 记录指标
                test_metrics = {
                    "per_agent_reward": 10 + np.random.normal(0, 1),
                    "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
                    "efficiency_metric": 0.3 + np.random.uniform(-0.05, 0.1)
                }
                scenario_info = {"n_uavs": 5, "n_targets": 3}
                
                writer.log_scale_invariant_metrics_detailed(test_metrics, step, stage, scenario_info)
                
                # 模拟阶段切换
                if step in [300, 600]:
                    with patch.object(writer.writer, 'add_text'):
                        writer.log_curriculum_stage_transition(stage-1, stage, step, test_metrics)
        
        # 4. 记录模型相关信息
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        with patch.object(writer.writer, 'add_histogram'), \
             patch.object(writer.writer, 'add_scalar'):
            writer.log_model_gradients(model, 1000)
        
        # 5. 记录注意力权重
        attention_weights = torch.rand(2, 4, 8, 8)
        with patch.object(writer.writer, 'add_figure'), \
             patch.object(writer.writer, 'add_histogram'), \
             patch.object(writer.writer, 'add_scalar'):
            writer.log_attention_weights(attention_weights, 1000)
        
        # 6. 清理
        writer.close()
        
        print("✅ 完整TensorBoard工作流测试通过")
    
    def test_plugin_integration(self):
        """测试插件集成"""
        plugin = TensorBoardCustomPlugin(self.test_dir)
        
        # 创建配置和仪表板
        config_path = plugin.create_curriculum_dashboard_config()
        html_path = plugin.generate_custom_html_dashboard("integration_test")
        
        # 验证文件存在
        self.assertTrue(os.path.exists(config_path))
        self.assertTrue(os.path.exists(html_path))
        
        print("✅ TensorBoard插件集成测试通过")


def run_tensorboard_integration_tests():
    """运行所有TensorBoard集成测试"""
    print("📊 开始TensorBoard集成测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestCurriculumTensorBoardWriter,
        TestTensorBoardCustomPlugin,
        TestTensorBoardIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print(f"\n📊 TensorBoard集成测试结果摘要:")
    print(f"   总测试数: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tensorboard_integration_tests()
    sys.exit(0 if success else 1)
