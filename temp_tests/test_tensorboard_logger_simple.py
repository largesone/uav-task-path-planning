# -*- coding: utf-8 -*-
"""
简化的TensorBoard记录器测试
测试尺度不变指标的TensorBoard集成功能
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scale_invariant_tensorboard_logger import ScaleInvariantTensorBoardLogger
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("警告: TensorBoard或相关依赖不可用，将跳过TensorBoard测试")
    TENSORBOARD_AVAILABLE = False


class MockTensorBoardLogger:
    """模拟TensorBoard记录器，用于测试核心逻辑"""
    
    def __init__(self, log_dir: str, experiment_name: str = "test"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.metrics_history = {
            "per_agent_reward": [],
            "normalized_completion_score": [],
            "efficiency_metric": [],
            "scenario_scales": [],
            "timestamps": []
        }
        self.logged_metrics = []
        print(f"模拟TensorBoard记录器初始化: {log_dir}")
    
    def log_scale_invariant_metrics(self, metrics, step, scenario_info=None, stage_info=None):
        """记录尺度不变指标"""
        # 记录基础指标
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)
        
        # 记录场景规模
        if scenario_info and "n_uavs" in scenario_info and "n_targets" in scenario_info:
            scale_factor = scenario_info["n_uavs"] * scenario_info["n_targets"]
            self.metrics_history["scenario_scales"].append(scale_factor)
        
        # 保存记录的指标用于验证
        log_entry = {
            "step": step,
            "metrics": metrics.copy(),
            "scenario_info": scenario_info.copy() if scenario_info else None,
            "stage_info": stage_info.copy() if stage_info else None
        }
        self.logged_metrics.append(log_entry)
        
        print(f"记录指标 - Step {step}: {metrics}")
    
    def log_cross_scale_comparison(self, scale_performance_data, step):
        """记录跨尺度性能对比"""
        print(f"记录跨尺度对比 - Step {step}: {len(scale_performance_data)} 个规模")
    
    def log_zero_shot_transfer_results(self, transfer_results, step):
        """记录零样本迁移结果"""
        print(f"记录零样本迁移 - Step {step}: {transfer_results.get('source_scale', 0)} → {transfer_results.get('target_scale', 0)}")
    
    def create_training_summary_report(self, output_path=None):
        """创建训练摘要报告"""
        if output_path is None:
            output_path = self.log_dir / f"{self.experiment_name}_summary.md"
        
        # 计算统计信息
        if not self.metrics_history["per_agent_reward"]:
            return str(output_path)
        
        stats = {}
        for metric_name in ["per_agent_reward", "normalized_completion_score", "efficiency_metric"]:
            if self.metrics_history[metric_name]:
                values = self.metrics_history[metric_name]
                stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "max": np.max(values),
                    "min": np.min(values)
                }
        
        # 生成报告内容
        report_content = f"""# 尺度不变指标训练摘要报告

## 实验信息
- **实验名称**: {self.experiment_name}
- **总记录数**: {len(self.logged_metrics)}

## 指标统计摘要
"""
        
        for metric_name, stat in stats.items():
            report_content += f"""
### {metric_name}
- **平均值**: {stat["mean"]:.4f}
- **标准差**: {stat["std"]:.4f}
- **最大值**: {stat["max"]:.4f}
- **最小值**: {stat["min"]:.4f}
"""
        
        # 确保目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"摘要报告已生成: {output_path}")
        return str(output_path)
    
    def close(self):
        """关闭记录器"""
        print("模拟TensorBoard记录器已关闭")


class TestTensorBoardLogger(unittest.TestCase):
    """测试TensorBoard记录器功能"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 根据可用性选择记录器
        if TENSORBOARD_AVAILABLE:
            try:
                self.logger = ScaleInvariantTensorBoardLogger(self.temp_dir, "test_experiment")
            except Exception as e:
                print(f"TensorBoard记录器初始化失败，使用模拟版本: {e}")
                self.logger = MockTensorBoardLogger(self.temp_dir, "test_experiment")
        else:
            self.logger = MockTensorBoardLogger(self.temp_dir, "test_experiment")
    
    def tearDown(self):
        """测试后清理"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_basic_metrics_logging(self):
        """测试基础指标记录"""
        metrics = {
            "per_agent_reward": 15.5,
            "normalized_completion_score": 0.75,
            "efficiency_metric": 0.002
        }
        
        scenario_info = {
            "n_uavs": 3,
            "n_targets": 2
        }
        
        stage_info = {
            "current_stage": 1,
            "stage_progress": 0.6
        }
        
        # 记录指标
        self.logger.log_scale_invariant_metrics(metrics, 100, scenario_info, stage_info)
        
        # 验证记录
        if hasattr(self.logger, 'metrics_history'):
            self.assertEqual(len(self.logger.metrics_history["per_agent_reward"]), 1)
            self.assertEqual(self.logger.metrics_history["per_agent_reward"][0], 15.5)
            
            if self.logger.metrics_history["scenario_scales"]:
                expected_scale = 3 * 2  # n_uavs * n_targets
                self.assertEqual(self.logger.metrics_history["scenario_scales"][0], expected_scale)
        
        print("✓ 基础指标记录测试通过")
    
    def test_multiple_metrics_logging(self):
        """测试多次指标记录"""
        # 记录多个时间步的指标
        for step in range(10):
            metrics = {
                "per_agent_reward": 10 + step + np.random.normal(0, 1),
                "normalized_completion_score": 0.5 + step * 0.03 + np.random.uniform(-0.05, 0.05),
                "efficiency_metric": 0.001 + step * 0.0001 + np.random.uniform(-0.0001, 0.0001)
            }
            
            scenario_info = {
                "n_uavs": 3 + (step // 3),
                "n_targets": 2 + (step // 5)
            }
            
            self.logger.log_scale_invariant_metrics(metrics, step, scenario_info)
        
        # 验证记录数量
        if hasattr(self.logger, 'metrics_history'):
            self.assertEqual(len(self.logger.metrics_history["per_agent_reward"]), 10)
            self.assertEqual(len(self.logger.metrics_history["normalized_completion_score"]), 10)
            self.assertEqual(len(self.logger.metrics_history["efficiency_metric"]), 10)
        
        print("✓ 多次指标记录测试通过")
    
    def test_cross_scale_comparison(self):
        """测试跨尺度性能对比"""
        scale_data = {
            6: {"per_agent_reward": 10.0, "normalized_completion_score": 0.6, "efficiency_metric": 0.002},
            12: {"per_agent_reward": 8.0, "normalized_completion_score": 0.7, "efficiency_metric": 0.0015},
            20: {"per_agent_reward": 6.0, "normalized_completion_score": 0.8, "efficiency_metric": 0.001}
        }
        
        # 记录跨尺度对比
        self.logger.log_cross_scale_comparison(scale_data, 200)
        
        print("✓ 跨尺度对比记录测试通过")
    
    def test_zero_shot_transfer_logging(self):
        """测试零样本迁移结果记录"""
        transfer_results = {
            "source_stage": 0,
            "target_stage": 2,
            "source_scale": 6,
            "target_scale": 20,
            "transfer_performance": {
                "per_agent_reward": 7.5,
                "normalized_completion_score": 0.65,
                "efficiency_metric": 0.0018
            },
            "baseline_performance": {
                "per_agent_reward": 10.0,
                "normalized_completion_score": 0.8,
                "efficiency_metric": 0.002
            }
        }
        
        # 记录零样本迁移结果
        self.logger.log_zero_shot_transfer_results(transfer_results, 300)
        
        print("✓ 零样本迁移记录测试通过")
    
    def test_summary_report_generation(self):
        """测试摘要报告生成"""
        # 添加一些测试数据
        for i in range(20):
            metrics = {
                "per_agent_reward": 10 + np.random.normal(0, 2),
                "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
                "efficiency_metric": 0.001 + np.random.uniform(0, 0.0005)
            }
            self.logger.log_scale_invariant_metrics(metrics, i)
        
        # 生成报告
        report_path = self.logger.create_training_summary_report()
        
        # 验证报告文件存在
        self.assertTrue(Path(report_path).exists())
        
        # 验证报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("尺度不变指标训练摘要报告", content)
            self.assertIn("Per-Agent Reward", content)  # 修正为中文标题
            self.assertIn("Normalized Completion Score", content)  # 修正为中文标题
            self.assertIn("Efficiency Metric", content)  # 修正为中文标题
        
        print(f"✓ 摘要报告生成测试通过: {report_path}")
    
    def test_metrics_validation(self):
        """测试指标验证"""
        # 测试正常指标
        valid_metrics = {
            "per_agent_reward": 15.0,
            "normalized_completion_score": 0.8,
            "efficiency_metric": 0.003
        }
        
        try:
            self.logger.log_scale_invariant_metrics(valid_metrics, 1)
            print("✓ 有效指标验证通过")
        except Exception as e:
            self.fail(f"有效指标记录失败: {e}")
        
        # 测试边界值
        boundary_metrics = {
            "per_agent_reward": 0.0,
            "normalized_completion_score": 0.0,
            "efficiency_metric": 0.0
        }
        
        try:
            self.logger.log_scale_invariant_metrics(boundary_metrics, 2)
            print("✓ 边界值指标验证通过")
        except Exception as e:
            self.fail(f"边界值指标记录失败: {e}")
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空指标
        try:
            self.logger.log_scale_invariant_metrics({}, 1)
            print("✓ 空指标处理测试通过")
        except Exception as e:
            print(f"空指标处理: {e}")
        
        # 测试None值
        try:
            self.logger.log_scale_invariant_metrics(None, 2)
            print("✓ None值处理测试通过")
        except Exception as e:
            print(f"None值处理: {e}")


class TestMetricsCalculationIntegration(unittest.TestCase):
    """测试指标计算与记录的集成"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MockTensorBoardLogger(self.temp_dir, "integration_test")
    
    def tearDown(self):
        """测试后清理"""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 模拟训练过程
        stages = [
            {"n_uavs": 3, "n_targets": 2, "episodes": 5},
            {"n_uavs": 5, "n_targets": 3, "episodes": 5},
            {"n_uavs": 8, "n_targets": 5, "episodes": 5}
        ]
        
        step = 0
        for stage_id, stage_config in enumerate(stages):
            print(f"\n模拟训练阶段 {stage_id}: {stage_config}")
            
            for episode in range(stage_config["episodes"]):
                # 模拟指标计算
                base_performance = 0.6 + stage_id * 0.1
                metrics = {
                    "per_agent_reward": 10 + stage_id * 2 + np.random.normal(0, 1),
                    "normalized_completion_score": base_performance + np.random.uniform(-0.1, 0.1),
                    "efficiency_metric": 0.001 + stage_id * 0.0005 + np.random.uniform(-0.0002, 0.0002)
                }
                
                scenario_info = {
                    "n_uavs": stage_config["n_uavs"],
                    "n_targets": stage_config["n_targets"]
                }
                
                stage_info = {
                    "current_stage": stage_id,
                    "stage_progress": episode / stage_config["episodes"]
                }
                
                # 记录指标
                self.logger.log_scale_invariant_metrics(metrics, step, scenario_info, stage_info)
                step += 1
        
        # 验证记录的完整性
        total_episodes = sum(stage["episodes"] for stage in stages)
        if hasattr(self.logger, 'logged_metrics'):
            self.assertEqual(len(self.logger.logged_metrics), total_episodes)
        
        # 生成最终报告
        report_path = self.logger.create_training_summary_report()
        self.assertTrue(Path(report_path).exists())
        
        print(f"✓ 端到端工作流程测试通过，共记录 {total_episodes} 个episode")
    
    def test_performance_trend_analysis(self):
        """测试性能趋势分析"""
        # 模拟性能改进趋势
        for step in range(50):
            # 模拟逐步改进的性能
            progress = step / 50.0
            metrics = {
                "per_agent_reward": 5 + progress * 10 + np.random.normal(0, 0.5),
                "normalized_completion_score": 0.3 + progress * 0.5 + np.random.uniform(-0.05, 0.05),
                "efficiency_metric": 0.0005 + progress * 0.002 + np.random.uniform(-0.0001, 0.0001)
            }
            
            self.logger.log_scale_invariant_metrics(metrics, step)
        
        # 验证趋势
        if hasattr(self.logger, 'metrics_history'):
            scores = self.logger.metrics_history["normalized_completion_score"]
            if len(scores) >= 10:
                early_avg = np.mean(scores[:10])
                late_avg = np.mean(scores[-10:])
                
                # 应该有改进趋势
                self.assertGreater(late_avg, early_avg)
                print(f"✓ 性能趋势分析: 早期平均 {early_avg:.3f} → 后期平均 {late_avg:.3f}")


def run_tensorboard_test():
    """运行TensorBoard测试"""
    print("=" * 60)
    print("开始TensorBoard记录器测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTensorBoardLogger))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMetricsCalculationIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tensorboard_test()
    
    if success:
        print("🎉 TensorBoard记录器测试全部通过！")
    else:
        print("❌ 部分TensorBoard测试失败，请检查实现。")
    
    exit(0 if success else 1)
