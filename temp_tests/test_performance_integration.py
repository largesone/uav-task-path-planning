"""
完整的性能监控系统集成测试
测试所有性能优化与监控组件的功能
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
import json
from unittest.mock import Mock, patch, MagicMock

# 导入性能监控组件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_monitor import PerformanceMonitor, MemorySnapshot
from benchmark_suite import BenchmarkSuite, BenchmarkResult
from training_visualizer import TrainingVisualizer
from tensorboard_plugin import TransformerGNNTensorBoardLogger
from solution_quality_analyzer import SolutionQualityAnalyzer, SolutionQualityMetrics

class TestPerformanceMonitoringComplete(unittest.TestCase):
    """完整性能监控系统测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # 初始化各个组件
        self.performance_monitor = PerformanceMonitor(max_history=100)
        self.benchmark_suite = BenchmarkSuite(output_dir=str(Path(self.temp_dir) / "benchmark"))
        self.training_visualizer = TrainingVisualizer(save_dir=str(Path(self.temp_dir) / "plots"))
        self.tensorboard_logger = TransformerGNNTensorBoardLogger(
            log_dir=str(Path(self.temp_dir) / "tensorboard")
        )
        self.quality_analyzer = SolutionQualityAnalyzer(
            results_dir=str(Path(self.temp_dir) / "quality")
        )
        
    def tearDown(self):
        """测试清理"""
        self.performance_monitor.stop_monitoring()
        self.tensorboard_logger.close()
        
    def test_performance_monitor_functionality(self):
        """测试性能监控器功能"""
        print("测试性能监控器...")
        
        # 测试内存快照记录
        self.performance_monitor.record_memory_snapshot(
            n_uavs=5, n_targets=3, stage="test_stage"
        )
        
        # 验证快照记录
        self.assertEqual(len(self.performance_monitor.memory_history), 1)
        snapshot = self.performance_monitor.memory_history[0]
        self.assertEqual(snapshot.n_uavs, 5)
        self.assertEqual(snapshot.n_targets, 3)
        self.assertEqual(snapshot.stage, "test_stage")
        
        # 测试内存使用摘要
        summary = self.performance_monitor.get_memory_usage_summary()
        self.assertIn('cpu_memory', summary)
        self.assertIn('gpu_memory', summary)
        self.assertIn('process_memory', summary)
        
        # 测试优化建议
        suggestions = self.performance_monitor.optimize_memory_usage()
        self.assertIsInstance(suggestions, list)
        self.assertTrue(len(suggestions) > 0)
        
        # 测试数据导出
        export_path = Path(self.temp_dir) / "performance_data.json"
        self.performance_monitor.export_performance_data(str(export_path))
        self.assertTrue(export_path.exists())
        
        # 验证导出数据
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn('memory_history', data)
        self.assertIn('summary', data)
        self.assertIn('optimization_suggestions', data)
        
        print("✓ 性能监控器测试通过")
        
    def test_benchmark_suite_functionality(self):
        """测试基准测试套件功能"""
        print("测试基准测试套件...")
        
        # 创建模拟模型
        mock_model = Mock()
        mock_model.compute_single_action = Mock(return_value=np.array([0.5, 0.5]))
        
        # 创建模拟环境
        mock_env = Mock()
        mock_env.reset = Mock(return_value=np.random.randn(10))
        mock_env.step = Mock(return_value=(
            np.random.randn(10),  # next_state
            1.0,                  # reward
            True,                 # done
            {'completion_rate': 0.8}  # info
        ))
        
        # 测试训练速度基准
        training_time = self.benchmark_suite.benchmark_training_speed(
            mock_model, mock_env, episodes=5
        )
        self.assertIsInstance(training_time, float)
        self.assertGreaterEqual(training_time, 0)  # 允许为0，因为模拟环境可能很快
        
        # 测试推理延迟基准
        test_states = [np.random.randn(10) for _ in range(3)]
        inference_time = self.benchmark_suite.benchmark_inference_latency(
            mock_model, test_states, runs=10
        )
        self.assertIsInstance(inference_time, float)
        self.assertGreaterEqual(inference_time, 0)  # 允许为0，因为模拟环境可能很快
        
        # 测试内存使用基准
        memory_usage = self.benchmark_suite.benchmark_memory_usage(
            mock_model, [(3, 2), (5, 3)]
        )
        self.assertIsInstance(memory_usage, dict)
        
        # 测试性能报告生成
        # 添加一些模拟结果
        self.benchmark_suite.results.append(BenchmarkResult(
            method_name="TestMethod",
            scenario_size=(3, 2),
            training_time_per_episode=0.1,
            inference_time_ms=5.0,
            memory_usage_mb=100.0,
            convergence_episodes=50,
            final_performance=0.8
        ))
        
        report = self.benchmark_suite.generate_performance_report()
        self.assertIsInstance(report, str)
        self.assertIn("性能基准测试报告", report)
        self.assertIn("TestMethod", report)
        
        print("✓ 基准测试套件测试通过")
        
    def test_training_visualizer_functionality(self):
        """测试训练可视化器功能"""
        print("测试训练可视化器...")
        
        # 模拟训练数据记录
        for step in range(20):
            metrics = {
                'per_agent_reward': 0.5 + 0.1 * step + np.random.normal(0, 0.05),
                'normalized_completion_score': 0.3 + 0.05 * step + np.random.normal(0, 0.02),
                'efficiency_metric': 0.4 + 0.03 * step + np.random.normal(0, 0.01),
                'memory_usage_mb': 500 + step * 10
            }
            stage = min(step // 5 + 1, 4)
            self.training_visualizer.log_training_step(step, metrics, stage)
        
        # 模拟阶段转换
        self.training_visualizer.log_stage_transition(1, 2, time.time() - 100)
        self.training_visualizer.log_stage_transition(2, 3, time.time() - 50)
        
        # 模拟回退事件
        self.training_visualizer.log_fallback_event(3, "性能下降", time.time() - 25)
        
        # 验证数据记录
        self.assertEqual(len(self.training_visualizer.training_data), 20)
        self.assertEqual(len(self.training_visualizer.stage_transitions), 2)
        self.assertEqual(len(self.training_visualizer.fallback_events), 1)
        
        # 测试图表生成（不实际显示）
        with patch('matplotlib.pyplot.show'):
            curriculum_fig = self.training_visualizer.plot_curriculum_progress()
            self.assertIsNotNone(curriculum_fig)
            
            performance_fig = self.training_visualizer.plot_performance_trends()
            self.assertIsNotNone(performance_fig)
            
            fallback_fig = self.training_visualizer.plot_fallback_analysis()
            self.assertIsNotNone(fallback_fig)
            
            dashboard_fig = self.training_visualizer.create_training_dashboard()
            self.assertIsNotNone(dashboard_fig)
        
        # 测试交互式报告导出
        self.training_visualizer.export_interactive_report("test_report")
        
        # 验证文件生成
        report_files = list(Path(self.training_visualizer.save_dir).glob("test_report_*"))
        self.assertGreater(len(report_files), 0)
        
        print("✓ 训练可视化器测试通过")
        
    def test_tensorboard_logger_functionality(self):
        """测试TensorBoard日志记录器功能"""
        print("测试TensorBoard日志记录器...")
        
        # 测试课程学习指标记录
        metrics = {
            'per_agent_reward': 0.75,
            'normalized_completion_score': 0.68,
            'efficiency_metric': 0.82,
            'n_uavs': 5,
            'n_targets': 3,
            'reward_std': 0.15
        }
        
        self.tensorboard_logger.log_curriculum_metrics(stage=2, metrics=metrics, step=100)
        
        # 测试注意力权重记录
        attention_weights = torch.randn(1, 4, 8, 8)  # [batch, heads, seq, seq]
        self.tensorboard_logger.log_attention_weights(attention_weights, step=100)
        
        # 测试零样本迁移记录
        source_perf = {'per_agent_reward': 0.8, 'scenario_size': (3, 2)}
        target_perf = {'per_agent_reward': 0.6, 'scenario_size': (8, 5)}
        self.tensorboard_logger.log_zero_shot_transfer(source_perf, target_perf, step=100)
        
        # 测试内存使用记录
        memory_data = {
            'cpu_memory_mb': 2048,
            'gpu_memory_mb': 4096,
            'gpu_memory_cached_mb': 1024,
            'process_memory_mb': 1536,
            'n_uavs': 5,
            'n_targets': 3
        }
        self.tensorboard_logger.log_memory_usage(memory_data, step=100)
        
        # 测试k值自适应记录
        k_values = [4, 6, 8, 6, 5]
        scenario_sizes = [(3, 2), (5, 3), (8, 5), (6, 4), (4, 3)]
        self.tensorboard_logger.log_k_value_adaptation(k_values, scenario_sizes, step=100)
        
        # 测试自定义标量仪表板创建
        self.tensorboard_logger.create_custom_scalar_dashboard()
        
        # 验证TensorBoard日志目录存在
        log_dir = Path(self.tensorboard_logger.writer.log_dir)
        self.assertTrue(log_dir.exists())
        
        print("✓ TensorBoard日志记录器测试通过")
        
    def test_solution_quality_analyzer_functionality(self):
        """测试方案质量分析器功能"""
        print("测试方案质量分析器...")
        
        # 创建模拟质量指标数据
        test_metrics = [
            SolutionQualityMetrics(
                method_name="TransformerGNN",
                scenario_size=(3, 2),
                completion_rate=0.85,
                efficiency_score=0.78,
                collision_rate=0.05,
                resource_utilization=0.82,
                convergence_time=120.5,
                zero_shot_performance=1.0,
                per_agent_reward=2.5,
                normalized_completion_score=0.75
            ),
            SolutionQualityMetrics(
                method_name="TransformerGNN",
                scenario_size=(8, 5),
                completion_rate=0.72,
                efficiency_score=0.65,
                collision_rate=0.08,
                resource_utilization=0.75,
                convergence_time=180.3,
                zero_shot_performance=0.85,
                per_agent_reward=2.1,
                normalized_completion_score=0.62
            ),
            SolutionQualityMetrics(
                method_name="FCN_Baseline",
                scenario_size=(3, 2),
                completion_rate=0.78,
                efficiency_score=0.70,
                collision_rate=0.12,
                resource_utilization=0.68,
                convergence_time=95.2,
                zero_shot_performance=1.0,
                per_agent_reward=2.2,
                normalized_completion_score=0.65
            ),
            SolutionQualityMetrics(
                method_name="FCN_Baseline",
                scenario_size=(8, 5),
                completion_rate=0.45,
                efficiency_score=0.38,
                collision_rate=0.25,
                resource_utilization=0.42,
                convergence_time=150.8,
                zero_shot_performance=0.58,
                per_agent_reward=1.3,
                normalized_completion_score=0.35
            )
        ]
        
        # 添加测试数据
        self.quality_analyzer.quality_data.extend(test_metrics)
        
        # 测试零样本迁移对比
        methods = {"TransformerGNN": None, "FCN_Baseline": None}
        transfer_df = self.quality_analyzer.compare_zero_shot_transfer(methods)
        self.assertFalse(transfer_df.empty)
        self.assertIn('zero_shot_performance', transfer_df.columns)
        
        # 测试可扩展性分析
        scalability = self.quality_analyzer.analyze_scalability()
        self.assertNotIn('error', scalability)
        self.assertIn('TransformerGNN', scalability)
        self.assertIn('FCN_Baseline', scalability)
        
        # 测试质量报告生成
        report = self.quality_analyzer.generate_quality_report()
        self.assertIsInstance(report, str)
        self.assertIn("方案质量分析报告", report)
        self.assertIn("TransformerGNN", report)
        self.assertIn("FCN_Baseline", report)
        
        # 测试图表生成（不实际显示）
        with patch('matplotlib.pyplot.show'):
            self.quality_analyzer.plot_quality_comparison()
            self.quality_analyzer.plot_zero_shot_heatmap()
        
        # 测试结果导出
        self.quality_analyzer.export_results("test_quality_analysis")
        
        # 验证导出文件
        export_files = list(Path(self.quality_analyzer.results_dir).glob("test_quality_analysis_*"))
        self.assertGreater(len(export_files), 0)
        
        print("✓ 方案质量分析器测试通过")
        
    def test_integrated_performance_monitoring_workflow(self):
        """测试集成性能监控工作流"""
        print("测试集成性能监控工作流...")
        
        # 模拟完整的训练监控流程
        
        # 1. 开始性能监控
        self.performance_monitor.start_monitoring(interval=0.1)
        time.sleep(0.3)  # 让监控运行一段时间
        
        # 2. 记录训练过程
        for step in range(10):
            # 记录训练指标
            metrics = {
                'per_agent_reward': 0.5 + 0.1 * step,
                'normalized_completion_score': 0.3 + 0.05 * step,
                'efficiency_metric': 0.4 + 0.03 * step
            }
            stage = step // 3 + 1
            
            self.training_visualizer.log_training_step(step, metrics, stage)
            self.tensorboard_logger.log_curriculum_metrics(stage, metrics, step)
            
            # 记录内存快照
            self.performance_monitor.record_memory_snapshot(
                n_uavs=3 + step // 2, 
                n_targets=2 + step // 3, 
                stage=f"stage_{stage}"
            )
        
        # 3. 停止监控
        self.performance_monitor.stop_monitoring()
        
        # 4. 生成综合报告
        performance_summary = self.performance_monitor.get_memory_usage_summary()
        training_report = "训练可视化报告已生成"
        
        # 验证监控数据
        self.assertGreater(len(self.performance_monitor.memory_history), 5)
        self.assertEqual(len(self.training_visualizer.training_data), 10)
        self.assertIn('cpu_memory', performance_summary)
        
        # 5. 导出所有结果
        export_timestamp = int(time.time())
        
        self.performance_monitor.export_performance_data(
            str(Path(self.temp_dir) / f"performance_{export_timestamp}.json")
        )
        
        self.training_visualizer.export_interactive_report(f"training_{export_timestamp}")
        
        # 验证导出文件存在
        performance_file = Path(self.temp_dir) / f"performance_{export_timestamp}.json"
        self.assertTrue(performance_file.exists())
        
        training_files = list(Path(self.training_visualizer.save_dir).glob(f"training_{export_timestamp}_*"))
        self.assertGreater(len(training_files), 0)
        
        print("✓ 集成性能监控工作流测试通过")
        
    def test_performance_optimization_recommendations(self):
        """测试性能优化建议功能"""
        print("测试性能优化建议...")
        
        # 模拟高内存使用场景
        high_memory_snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=16000,  # 16GB
            gpu_memory_mb=10000,  # 10GB
            gpu_memory_cached_mb=2000,
            process_memory_mb=8000,
            n_uavs=20,
            n_targets=15,
            stage="large_scale"
        )
        
        self.performance_monitor.memory_history.append(high_memory_snapshot)
        
        # 获取优化建议
        suggestions = self.performance_monitor.optimize_memory_usage()
        
        # 验证建议内容
        self.assertIsInstance(suggestions, list)
        self.assertTrue(any("GPU内存使用过高" in s for s in suggestions))
        self.assertTrue(any("进程内存使用较高" in s for s in suggestions))
        
        # 测试正常内存使用场景
        normal_memory_snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=2000,   # 2GB
            gpu_memory_mb=1000,   # 1GB
            gpu_memory_cached_mb=200,
            process_memory_mb=1500,
            n_uavs=5,
            n_targets=3,
            stage="small_scale"
        )
        
        # 清空历史并添加正常数据
        self.performance_monitor.memory_history.clear()
        self.performance_monitor.memory_history.append(normal_memory_snapshot)
        
        normal_suggestions = self.performance_monitor.optimize_memory_usage()
        self.assertTrue(any("内存使用正常" in s for s in normal_suggestions))
        
        print("✓ 性能优化建议测试通过")

def run_performance_monitoring_tests():
    """运行所有性能监控测试"""
    print("=" * 60)
    print("开始运行性能监控系统完整测试")
    print("=" * 60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceMonitoringComplete)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ 所有性能监控测试通过！")
        print(f"运行了 {result.testsRun} 个测试")
    else:
        print("❌ 部分测试失败")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_performance_monitoring_tests()
    exit(0 if success else 1)
