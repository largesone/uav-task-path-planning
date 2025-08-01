# -*- coding: utf-8 -*-
"""
尺度不变指标的TensorBoard日志记录器
专门用于记录和可视化尺度不变指标
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ScaleInvariantTensorBoardLogger:
    """
    尺度不变指标专用的TensorBoard记录器
    提供实时监控和历史回顾功能
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "scale_invariant_training"):
        """
        初始化TensorBoard记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 创建专用的TensorBoard日志目录
        self.tb_dir = self.log_dir / "tensorboard" / "scale_invariant" / experiment_name
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化TensorBoard写入器
        self.writer = SummaryWriter(str(self.tb_dir))
        
        # 指标历史记录
        self.metrics_history = {
            "per_agent_reward": [],
            "normalized_completion_score": [],
            "efficiency_metric": [],
            "scenario_scales": [],
            "timestamps": []
        }
        
        # 阶段性能记录
        self.stage_metrics = {}
        
        print(f"尺度不变指标TensorBoard记录器初始化完成: {self.tb_dir}")
    
    def log_scale_invariant_metrics(self, metrics: Dict[str, float], step: int, 
                                   scenario_info: Optional[Dict[str, int]] = None,
                                   stage_info: Optional[Dict[str, Any]] = None):
        """
        记录尺度不变指标
        
        Args:
            metrics: 指标字典，包含per_agent_reward, normalized_completion_score, efficiency_metric
            step: 训练步数
            scenario_info: 场景信息（n_uavs, n_targets等）
            stage_info: 阶段信息（current_stage, stage_progress等）
        """
        # 记录基础尺度不变指标
        for metric_name, value in metrics.items():
            if metric_name in ["per_agent_reward", "normalized_completion_score", "efficiency_metric"]:
                self.writer.add_scalar(f"Scale_Invariant_Metrics/{metric_name}", value, step)
                
                # 更新历史记录
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].append(value)
        
        # 记录场景信息
        if scenario_info:
            for info_name, value in scenario_info.items():
                self.writer.add_scalar(f"Scenario_Info/{info_name}", value, step)
            
            # 计算场景规模因子
            if "n_uavs" in scenario_info and "n_targets" in scenario_info:
                scale_factor = scenario_info["n_uavs"] * scenario_info["n_targets"]
                self.writer.add_scalar("Scenario_Info/scale_factor", scale_factor, step)
                self.metrics_history["scenario_scales"].append(scale_factor)
        
        # 记录阶段信息
        if stage_info:
            for info_name, value in stage_info.items():
                self.writer.add_scalar(f"Curriculum_Stage/{info_name}", value, step)
        
        # 记录时间戳
        self.metrics_history["timestamps"].append(datetime.now())
        
        # 计算并记录复合指标
        self._log_composite_metrics(metrics, step, scenario_info)
        
        # 定期创建可视化图表
        if step % 100 == 0:  # 每100步创建一次图表
            self._create_scale_invariant_visualizations(step)
    
    def log_cross_scale_comparison(self, scale_performance_data: Dict[int, Dict[str, float]], step: int):
        """
        记录跨尺度性能对比
        
        Args:
            scale_performance_data: 不同规模下的性能数据 {scale_factor: {metric: value}}
            step: 训练步数
        """
        # 记录不同规模下的性能
        for scale_factor, performance in scale_performance_data.items():
            for metric_name, value in performance.items():
                self.writer.add_scalar(f"Cross_Scale_Performance/scale_{scale_factor}_{metric_name}", value, step)
        
        # 创建跨尺度对比图表
        self._create_cross_scale_comparison_chart(scale_performance_data, step)
    
    def log_zero_shot_transfer_results(self, transfer_results: Dict[str, Any], step: int):
        """
        记录零样本迁移结果
        
        Args:
            transfer_results: 迁移结果数据
            step: 训练步数
        """
        # 记录迁移性能
        if "source_scale" in transfer_results and "target_scale" in transfer_results:
            source_scale = transfer_results["source_scale"]
            target_scale = transfer_results["target_scale"]
            
            # 记录迁移比例
            transfer_ratio = target_scale / source_scale if source_scale > 0 else 0
            self.writer.add_scalar("Zero_Shot_Transfer/scale_transfer_ratio", transfer_ratio, step)
            
            # 记录迁移性能
            if "transfer_performance" in transfer_results:
                for metric_name, value in transfer_results["transfer_performance"].items():
                    self.writer.add_scalar(f"Zero_Shot_Transfer/{metric_name}", value, step)
            
            # 记录性能保持率
            if "baseline_performance" in transfer_results and "transfer_performance" in transfer_results:
                baseline = transfer_results["baseline_performance"]
                transfer = transfer_results["transfer_performance"]
                
                for metric_name in baseline:
                    if metric_name in transfer and baseline[metric_name] > 0:
                        retention_rate = transfer[metric_name] / baseline[metric_name]
                        self.writer.add_scalar(f"Zero_Shot_Transfer/{metric_name}_retention_rate", retention_rate, step)
    
    def log_training_efficiency_metrics(self, efficiency_data: Dict[str, float], step: int):
        """
        记录训练效率指标
        
        Args:
            efficiency_data: 效率数据
            step: 训练步数
        """
        for metric_name, value in efficiency_data.items():
            self.writer.add_scalar(f"Training_Efficiency/{metric_name}", value, step)
    
    def _log_composite_metrics(self, metrics: Dict[str, float], step: int, 
                              scenario_info: Optional[Dict[str, int]] = None):
        """
        记录复合指标
        
        Args:
            metrics: 基础指标
            step: 训练步数
            scenario_info: 场景信息
        """
        # 计算综合性能指标
        if all(key in metrics for key in ["per_agent_reward", "normalized_completion_score", "efficiency_metric"]):
            # 加权综合性能 = 0.4 * NCS + 0.3 * PAR_norm + 0.3 * EM_norm
            # 先对指标进行归一化
            par_norm = np.tanh(metrics["per_agent_reward"] / 50.0)  # 假设50为合理的奖励基准
            ncs = metrics["normalized_completion_score"]  # 已经在[0,1]范围
            em_norm = np.tanh(metrics["efficiency_metric"] * 1000)  # 效率指标通常很小
            
            composite_performance = 0.4 * ncs + 0.3 * par_norm + 0.3 * em_norm
            self.writer.add_scalar("Composite_Metrics/overall_performance", composite_performance, step)
        
        # 计算性能密度（性能/规模）
        if scenario_info and "n_uavs" in scenario_info and "n_targets" in scenario_info:
            scale_factor = scenario_info["n_uavs"] * scenario_info["n_targets"]
            if scale_factor > 0 and "normalized_completion_score" in metrics:
                performance_density = metrics["normalized_completion_score"] / scale_factor
                self.writer.add_scalar("Composite_Metrics/performance_density", performance_density, step)
        
        # 计算效率-质量平衡指标
        if "normalized_completion_score" in metrics and "efficiency_metric" in metrics:
            # 平衡指标 = sqrt(完成率 * 效率)
            balance_metric = np.sqrt(metrics["normalized_completion_score"] * metrics["efficiency_metric"])
            self.writer.add_scalar("Composite_Metrics/efficiency_quality_balance", balance_metric, step)
    
    def _create_scale_invariant_visualizations(self, step: int):
        """
        创建尺度不变指标可视化图表
        
        Args:
            step: 当前训练步数
        """
        if len(self.metrics_history["per_agent_reward"]) < 10:
            return  # 数据不足，跳过可视化
        
        # 创建多子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Scale Invariant Metrics Analysis (Step {step})', fontsize=16)
        
        # 1. 指标趋势图
        ax1 = axes[0, 0]
        recent_steps = list(range(max(0, len(self.metrics_history["per_agent_reward"]) - 50), 
                                 len(self.metrics_history["per_agent_reward"])))
        
        if len(recent_steps) > 1:
            ax1.plot(recent_steps, self.metrics_history["per_agent_reward"][-50:], 
                    'b-', label='Per-Agent Reward', alpha=0.7)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(recent_steps, self.metrics_history["normalized_completion_score"][-50:], 
                         'r-', label='Normalized Completion Score', alpha=0.7)
            
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Per-Agent Reward', color='b')
            ax1_twin.set_ylabel('Completion Score', color='r')
            ax1.set_title('Metrics Trend')
            ax1.grid(True, alpha=0.3)
        
        # 2. 效率指标分布
        ax2 = axes[0, 1]
        if len(self.metrics_history["efficiency_metric"]) > 5:
            ax2.hist(self.metrics_history["efficiency_metric"][-100:], bins=20, alpha=0.7, color='green')
            ax2.set_xlabel('Efficiency Metric')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Efficiency Distribution')
            ax2.grid(True, alpha=0.3)
        
        # 3. 规模 vs 性能散点图
        ax3 = axes[1, 0]
        if (len(self.metrics_history["scenario_scales"]) > 5 and 
            len(self.metrics_history["normalized_completion_score"]) > 5):
            
            scales = self.metrics_history["scenario_scales"][-50:]
            scores = self.metrics_history["normalized_completion_score"][-50:]
            
            if len(scales) == len(scores):
                ax3.scatter(scales, scores, alpha=0.6, c=range(len(scales)), cmap='viridis')
                ax3.set_xlabel('Scenario Scale Factor')
                ax3.set_ylabel('Normalized Completion Score')
                ax3.set_title('Scale vs Performance')
                ax3.grid(True, alpha=0.3)
        
        # 4. 性能稳定性分析
        ax4 = axes[1, 1]
        if len(self.metrics_history["normalized_completion_score"]) > 20:
            scores = self.metrics_history["normalized_completion_score"]
            
            # 计算滑动窗口标准差
            window_size = 10
            rolling_std = []
            for i in range(window_size, len(scores)):
                window_std = np.std(scores[i-window_size:i])
                rolling_std.append(window_std)
            
            if rolling_std:
                ax4.plot(range(window_size, len(scores)), rolling_std, 'purple', alpha=0.7)
                ax4.set_xlabel('Training Steps')
                ax4.set_ylabel('Rolling Std')
                ax4.set_title('Performance Stability')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存到TensorBoard
        self.writer.add_figure("Scale_Invariant_Analysis/comprehensive_view", fig, step)
        plt.close(fig)
    
    def _create_cross_scale_comparison_chart(self, scale_performance_data: Dict[int, Dict[str, float]], step: int):
        """
        创建跨尺度性能对比图表
        
        Args:
            scale_performance_data: 不同规模下的性能数据
            step: 训练步数
        """
        if len(scale_performance_data) < 2:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Cross-Scale Performance Comparison (Step {step})', fontsize=16)
        
        scales = sorted(scale_performance_data.keys())
        metrics_to_plot = ["per_agent_reward", "normalized_completion_score", "efficiency_metric"]
        metric_titles = ["Per-Agent Reward", "Normalized Completion Score", "Efficiency Metric"]
        
        for i, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[i]
            values = [scale_performance_data[scale].get(metric, 0) for scale in scales]
            
            ax.plot(scales, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Scale')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(scales) > 2:
                z = np.polyfit(scales, values, 1)
                p = np.poly1d(z)
                ax.plot(scales, p(scales), "--", alpha=0.7, color='red')
        
        plt.tight_layout()
        self.writer.add_figure("Cross_Scale_Analysis/performance_comparison", fig, step)
        plt.close(fig)
    
    def create_training_summary_report(self, output_path: Optional[str] = None) -> str:
        """
        创建训练摘要报告
        
        Args:
            output_path: 输出路径，如果为None则使用默认路径
            
        Returns:
            str: 报告文件路径
        """
        if output_path is None:
            output_path = self.log_dir / f"{self.experiment_name}_summary_report.md"
        
        # 计算统计信息
        if not self.metrics_history["per_agent_reward"]:
            return str(output_path)
        
        stats = {
            "per_agent_reward": {
                "mean": np.mean(self.metrics_history["per_agent_reward"]),
                "std": np.std(self.metrics_history["per_agent_reward"]),
                "max": np.max(self.metrics_history["per_agent_reward"]),
                "min": np.min(self.metrics_history["per_agent_reward"])
            },
            "normalized_completion_score": {
                "mean": np.mean(self.metrics_history["normalized_completion_score"]),
                "std": np.std(self.metrics_history["normalized_completion_score"]),
                "max": np.max(self.metrics_history["normalized_completion_score"]),
                "min": np.min(self.metrics_history["normalized_completion_score"])
            },
            "efficiency_metric": {
                "mean": np.mean(self.metrics_history["efficiency_metric"]),
                "std": np.std(self.metrics_history["efficiency_metric"]),
                "max": np.max(self.metrics_history["efficiency_metric"]),
                "min": np.min(self.metrics_history["efficiency_metric"])
            }
        }
        
        # 生成报告内容
        report_content = f"""# 尺度不变指标训练摘要报告

## 实验信息
- **实验名称**: {self.experiment_name}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总训练步数**: {len(self.metrics_history["per_agent_reward"])}

## 指标统计摘要

### Per-Agent Reward
- **平均值**: {stats["per_agent_reward"]["mean"]:.4f}
- **标准差**: {stats["per_agent_reward"]["std"]:.4f}
- **最大值**: {stats["per_agent_reward"]["max"]:.4f}
- **最小值**: {stats["per_agent_reward"]["min"]:.4f}

### Normalized Completion Score
- **平均值**: {stats["normalized_completion_score"]["mean"]:.4f}
- **标准差**: {stats["normalized_completion_score"]["std"]:.4f}
- **最大值**: {stats["normalized_completion_score"]["max"]:.4f}
- **最小值**: {stats["normalized_completion_score"]["min"]:.4f}

### Efficiency Metric
- **平均值**: {stats["efficiency_metric"]["mean"]:.6f}
- **标准差**: {stats["efficiency_metric"]["std"]:.6f}
- **最大值**: {stats["efficiency_metric"]["max"]:.6f}
- **最小值**: {stats["efficiency_metric"]["min"]:.6f}

## 性能分析

### 收敛性分析
"""
        
        # 添加收敛性分析
        if len(self.metrics_history["normalized_completion_score"]) > 50:
            recent_performance = np.mean(self.metrics_history["normalized_completion_score"][-20:])
            early_performance = np.mean(self.metrics_history["normalized_completion_score"][:20])
            improvement = ((recent_performance - early_performance) / early_performance * 100) if early_performance > 0 else 0
            
            report_content += f"""
- **性能改进**: {improvement:.2f}%
- **最终性能**: {recent_performance:.4f}
- **初始性能**: {early_performance:.4f}
"""
        
        report_content += f"""
## TensorBoard日志位置
- **日志目录**: {self.tb_dir}
- **启动命令**: `tensorboard --logdir {self.tb_dir}`

## 建议
"""
        
        # 添加基于性能的建议
        if stats["normalized_completion_score"]["mean"] < 0.7:
            report_content += "- 完成率较低，建议调整奖励函数或增加训练时间\n"
        
        if stats["efficiency_metric"]["std"] > stats["efficiency_metric"]["mean"]:
            report_content += "- 效率指标波动较大，建议优化探索策略\n"
        
        if stats["per_agent_reward"]["std"] > 10:
            report_content += "- 奖励波动较大，建议检查环境设置和归一化机制\n"
        
        # 写入报告文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"训练摘要报告已生成: {output_path}")
        return str(output_path)
    
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()
        print("尺度不变指标TensorBoard记录器已关闭")


if __name__ == "__main__":
    # 测试代码
    logger_instance = ScaleInvariantTensorBoardLogger("./test_logs", "test_scale_invariant")
    
    # 模拟训练数据
    for step in range(200):
        metrics = {
            "per_agent_reward": 15 + np.random.normal(0, 2),
            "normalized_completion_score": 0.6 + np.random.uniform(0, 0.3),
            "efficiency_metric": 0.001 + np.random.uniform(0, 0.0005)
        }
        
        scenario_info = {
            "n_uavs": 3 + (step // 50),
            "n_targets": 2 + (step // 50)
        }
        
        stage_info = {
            "current_stage": step // 50,
            "stage_progress": (step % 50) / 50.0
        }
        
        logger_instance.log_scale_invariant_metrics(metrics, step, scenario_info, stage_info)
    
    # 生成摘要报告
    logger_instance.create_training_summary_report()
    logger_instance.close()
    
    print("测试完成")
