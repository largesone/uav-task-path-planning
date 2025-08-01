"""
TensorBoard集成模块
提供自定义的TensorBoard插件和高级可视化功能
"""

import os
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


class CurriculumTensorBoardWriter:
    """
    课程学习专用的TensorBoard写入器
    提供高级可视化和自定义图表功能
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "curriculum_training"):
        """
        初始化TensorBoard写入器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 创建TensorBoard日志目录
        self.tb_dir = self.log_dir / "tensorboard" / experiment_name
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化写入器
        self.writer = SummaryWriter(str(self.tb_dir))
        
        # 自定义标量组
        self.scalar_groups = {
            "Scale_Invariant_Metrics": ["per_agent_reward", "normalized_completion_score", "efficiency_metric"],
            "Curriculum_Progress": ["current_stage", "stage_progress", "rollback_events"],
            "Training_Dynamics": ["learning_rate", "exploration_noise", "k_neighbors"],
            "Scenario_Info": ["n_uavs", "n_targets", "scale_factor"]
        }
        
        # 记录超参数
        self.hparams = {}
        
        print(f"TensorBoard写入器初始化完成: {self.tb_dir}")
    
    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        记录超参数和对应的指标
        
        Args:
            hparams: 超参数字典
            metrics: 指标字典
        """
        self.hparams.update(hparams)
        
        # 转换超参数格式
        hparam_dict = {}
        metric_dict = {}
        
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            else:
                hparam_dict[key] = str(value)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_dict[f"hparam/{key}"] = value
        
        self.writer.add_hparams(hparam_dict, metric_dict)
        print(f"超参数已记录: {len(hparam_dict)} 个参数, {len(metric_dict)} 个指标")
    
    def log_curriculum_stage_transition(self, from_stage: int, to_stage: int, 
                                      step: int, performance_data: Dict[str, float]):
        """
        记录课程学习阶段切换
        
        Args:
            from_stage: 源阶段
            to_stage: 目标阶段
            step: 训练步数
            performance_data: 性能数据
        """
        # 记录阶段切换事件
        self.writer.add_scalar("Curriculum/Stage_Transition", to_stage, step)
        
        # 添加文本描述
        transition_text = f"阶段切换: {from_stage} → {to_stage}\n"
        transition_text += f"切换时性能:\n"
        for metric, value in performance_data.items():
            transition_text += f"  {metric}: {value:.4f}\n"
        
        self.writer.add_text("Curriculum/Transition_Details", transition_text, step)
        
        # 创建阶段切换可视化
        self._create_stage_transition_plot(from_stage, to_stage, step, performance_data)
    
    def log_scale_invariant_metrics_detailed(self, metrics: Dict[str, float], 
                                           step: int, stage: int, 
                                           scenario_info: Dict[str, int]):
        """
        详细记录尺度不变指标
        
        Args:
            metrics: 指标字典
            step: 训练步数
            stage: 当前阶段
            scenario_info: 场景信息（n_uavs, n_targets等）
        """
        # 记录基础指标
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f"Scale_Invariant/{metric_name}", value, step)
        
        # 记录场景信息
        for info_name, value in scenario_info.items():
            self.writer.add_scalar(f"Scenario/{info_name}", value, step)
        
        # 计算并记录复合指标
        if "n_uavs" in scenario_info and "n_targets" in scenario_info:
            scale_factor = scenario_info["n_uavs"] * scenario_info["n_targets"]
            self.writer.add_scalar("Scenario/scale_factor", scale_factor, step)
            
            # 记录归一化性能密度
            if "normalized_completion_score" in metrics:
                performance_density = metrics["normalized_completion_score"] / scale_factor
                self.writer.add_scalar("Advanced/performance_density", performance_density, step)
        
        # 创建多维度性能图表
        self._create_multidimensional_performance_plot(metrics, step, stage, scenario_info)
    
    def log_attention_weights(self, attention_weights: torch.Tensor, step: int, 
                            layer_name: str = "transformer"):
        """
        记录注意力权重可视化
        
        Args:
            attention_weights: 注意力权重张量 [batch, heads, seq_len, seq_len]
            step: 训练步数
            layer_name: 层名称
        """
        if attention_weights.dim() != 4:
            print(f"警告: 注意力权重维度不正确: {attention_weights.shape}")
            return
        
        # 取第一个样本的平均注意力权重
        avg_attention = attention_weights[0].mean(dim=0)  # [seq_len, seq_len]
        
        # 创建注意力热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(avg_attention.detach().cpu().numpy(), 
                   annot=False, cmap='Blues', ax=ax)
        ax.set_title(f'{layer_name} Attention Weights (Step {step})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 记录到TensorBoard
        self.writer.add_figure(f"Attention/{layer_name}_heatmap", fig, step)
        plt.close(fig)
        
        # 记录注意力统计信息
        self.writer.add_histogram(f"Attention/{layer_name}_weights", attention_weights, step)
        self.writer.add_scalar(f"Attention/{layer_name}_entropy", 
                              self._calculate_attention_entropy(avg_attention), step)
    
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """
        记录模型梯度信息
        
        Args:
            model: PyTorch模型
            step: 训练步数
        """
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录每层的梯度范数
                self.writer.add_scalar(f"Gradients/{name}_norm", param_norm, step)
                
                # 记录梯度直方图
                self.writer.add_histogram(f"Gradients/{name}", param.grad, step)
        
        total_norm = total_norm ** (1. / 2)
        
        # 记录总梯度范数
        self.writer.add_scalar("Gradients/total_norm", total_norm, step)
        self.writer.add_scalar("Gradients/param_count", param_count, step)
    
    def log_learning_curves(self, train_metrics: Dict[str, List[float]], 
                          val_metrics: Dict[str, List[float]], 
                          steps: List[int]):
        """
        记录学习曲线
        
        Args:
            train_metrics: 训练指标历史
            val_metrics: 验证指标历史
            steps: 对应的步数列表
        """
        for metric_name in train_metrics:
            if metric_name in val_metrics:
                # 创建学习曲线图
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(steps, train_metrics[metric_name], 'b-', label=f'Train {metric_name}')
                ax.plot(steps, val_metrics[metric_name], 'r-', label=f'Val {metric_name}')
                
                ax.set_xlabel('Training Steps')
                ax.set_ylabel(metric_name)
                ax.set_title(f'Learning Curve: {metric_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 记录到TensorBoard
                self.writer.add_figure(f"Learning_Curves/{metric_name}", fig, steps[-1])
                plt.close(fig)
    
    def log_curriculum_progress_summary(self, stage_summaries: Dict[int, Dict[str, Any]], 
                                      current_step: int):
        """
        记录课程学习进度摘要
        
        Args:
            stage_summaries: 各阶段摘要信息
            current_step: 当前步数
        """
        # 创建进度摘要表格
        summary_text = "## 课程学习进度摘要\n\n"
        summary_text += "| 阶段 | 完成状态 | 最佳性能 | 训练步数 | 场景规模 |\n"
        summary_text += "|------|----------|----------|----------|----------|\n"
        
        for stage_id, summary in stage_summaries.items():
            status = "✅ 完成" if summary.get("completed", False) else "🔄 进行中"
            best_perf = f"{summary.get('best_performance', 0):.3f}"
            steps = f"{summary.get('total_steps', 0):,}"
            scale = f"{summary.get('n_uavs', 0)}×{summary.get('n_targets', 0)}"
            
            summary_text += f"| {stage_id} | {status} | {best_perf} | {steps} | {scale} |\n"
        
        self.writer.add_text("Curriculum/Progress_Summary", summary_text, current_step)
        
        # 创建进度条形图
        self._create_progress_bar_chart(stage_summaries, current_step)
    
    def _create_stage_transition_plot(self, from_stage: int, to_stage: int, 
                                    step: int, performance_data: Dict[str, float]):
        """创建阶段切换可视化图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 创建阶段切换时间线
        stages = [from_stage, to_stage]
        ax.plot([0, 1], stages, 'ro-', linewidth=3, markersize=10)
        
        # 添加性能数据
        perf_text = "\n".join([f"{k}: {v:.3f}" for k, v in performance_data.items()])
        ax.text(0.5, (from_stage + to_stage) / 2, perf_text, 
               ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(min(stages) - 0.5, max(stages) + 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['From', 'To'])
        ax.set_ylabel('Stage')
        ax.set_title(f'Stage Transition at Step {step}')
        ax.grid(True, alpha=0.3)
        
        self.writer.add_figure("Curriculum/Stage_Transition_Plot", fig, step)
        plt.close(fig)
    
    def _create_multidimensional_performance_plot(self, metrics: Dict[str, float], 
                                                step: int, stage: int, 
                                                scenario_info: Dict[str, int]):
        """创建多维度性能图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Multi-dimensional Performance Analysis (Step {step}, Stage {stage})')
        
        # 1. 雷达图显示各项指标
        if len(metrics) >= 3:
            ax = axes[0, 0]
            categories = list(metrics.keys())[:6]  # 最多6个指标
            values = [metrics[cat] for cat in categories]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title('Performance Radar Chart')
        
        # 2. 场景规模 vs 性能散点图
        ax = axes[0, 1]
        if "n_uavs" in scenario_info and "n_targets" in scenario_info:
            scale_factor = scenario_info["n_uavs"] * scenario_info["n_targets"]
            main_metric = metrics.get("normalized_completion_score", 0)
            
            ax.scatter([scale_factor], [main_metric], s=100, c=stage, cmap='viridis')
            ax.set_xlabel('Scale Factor (UAVs × Targets)')
            ax.set_ylabel('Normalized Completion Score')
            ax.set_title('Scale vs Performance')
        
        # 3. 指标分布直方图
        ax = axes[1, 0]
        metric_values = list(metrics.values())
        ax.hist(metric_values, bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Metric Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Metric Distribution')
        
        # 4. 阶段进度条
        ax = axes[1, 1]
        max_stage = 4  # 假设最大4个阶段
        progress = [1 if i <= stage else 0 for i in range(max_stage)]
        colors = ['green' if p == 1 else 'lightgray' for p in progress]
        
        ax.bar(range(max_stage), [1] * max_stage, color=colors, alpha=0.7)
        ax.set_xlabel('Stage')
        ax.set_ylabel('Completion')
        ax.set_title('Curriculum Progress')
        ax.set_xticks(range(max_stage))
        
        plt.tight_layout()
        self.writer.add_figure("Analysis/Multidimensional_Performance", fig, step)
        plt.close(fig)
    
    def _create_progress_bar_chart(self, stage_summaries: Dict[int, Dict[str, Any]], 
                                 current_step: int):
        """创建进度条形图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        stages = sorted(stage_summaries.keys())
        performances = [stage_summaries[s].get('best_performance', 0) for s in stages]
        colors = ['green' if stage_summaries[s].get('completed', False) else 'orange' 
                 for s in stages]
        
        bars = ax.bar(stages, performances, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, perf in zip(bars, performances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{perf:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Curriculum Stage')
        ax.set_ylabel('Best Performance')
        ax.set_title('Curriculum Learning Progress')
        ax.set_ylim(0, 1.1)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Completed'),
                          Patch(facecolor='orange', alpha=0.7, label='In Progress')]
        ax.legend(handles=legend_elements)
        
        self.writer.add_figure("Curriculum/Progress_Bar_Chart", fig, current_step)
        plt.close(fig)
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """计算注意力权重的熵"""
        # 避免log(0)
        attention_weights = attention_weights + 1e-8
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        return entropy.mean().item()
    
    def close(self):
        """关闭写入器"""
        self.writer.close()
        print("TensorBoard写入器已关闭")


class TensorBoardCustomPlugin:
    """
    自定义TensorBoard插件
    提供课程学习专用的可视化组件
    """
    
    def __init__(self, log_dir: str):
        """
        初始化自定义插件
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.plugin_dir = self.log_dir / "plugins"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"TensorBoard自定义插件初始化: {self.plugin_dir}")
    
    def create_curriculum_dashboard_config(self) -> str:
        """
        创建课程学习仪表板配置
        
        Returns:
            配置文件路径
        """
        dashboard_config = {
            "name": "Curriculum Learning Dashboard",
            "version": "1.0.0",
            "description": "专用于课程学习的监控仪表板",
            "layout": {
                "sections": [
                    {
                        "title": "尺度不变指标",
                        "charts": [
                            {"type": "line", "metrics": ["Scale_Invariant/per_agent_reward"]},
                            {"type": "line", "metrics": ["Scale_Invariant/normalized_completion_score"]},
                            {"type": "line", "metrics": ["Scale_Invariant/efficiency_metric"]}
                        ]
                    },
                    {
                        "title": "课程学习进度",
                        "charts": [
                            {"type": "line", "metrics": ["Curriculum/current_stage"]},
                            {"type": "bar", "metrics": ["Curriculum/stage_progress"]},
                            {"type": "scatter", "metrics": ["Curriculum/rollback_events"]}
                        ]
                    },
                    {
                        "title": "场景分析",
                        "charts": [
                            {"type": "heatmap", "metrics": ["Scenario/scale_factor", "Scale_Invariant/normalized_completion_score"]},
                            {"type": "3d_scatter", "metrics": ["Scenario/n_uavs", "Scenario/n_targets", "Scale_Invariant/efficiency_metric"]}
                        ]
                    }
                ]
            }
        }
        
        config_file = self.plugin_dir / "curriculum_dashboard.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False)
        
        print(f"仪表板配置已创建: {config_file}")
        return str(config_file)
    
    def generate_custom_html_dashboard(self, experiment_name: str) -> str:
        """
        生成自定义HTML仪表板
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            HTML文件路径
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>课程学习监控仪表板 - {experiment_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>课程学习训练监控仪表板</h1>
        <p>实验: {experiment_name} | 更新时间: <span id="updateTime"></span></p>
    </div>
    
    <div class="dashboard-grid">
        <div class="metric-card">
            <div class="metric-value" id="currentStage">-</div>
            <div class="metric-label">当前阶段</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value" id="bestPerformance">-</div>
            <div class="metric-label">最佳性能</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value" id="totalSteps">-</div>
            <div class="metric-label">总训练步数</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value" id="rollbackCount">-</div>
            <div class="metric-label">回退次数</div>
        </div>
        
        <div class="chart-container">
            <div id="metricsChart"></div>
        </div>
        
        <div class="chart-container">
            <div id="stageProgressChart"></div>
        </div>
        
        <div class="chart-container">
            <div id="scaleAnalysisChart"></div>
        </div>
        
        <div class="chart-container">
            <div id="performanceTrendChart"></div>
        </div>
    </div>
    
    <script>
        // 更新时间显示
        function updateTime() {{
            document.getElementById('updateTime').textContent = new Date().toLocaleString('zh-CN');
        }}
        
        // 初始化图表
        function initializeCharts() {{
            // 尺度不变指标图表
            var metricsData = [
                {{
                    x: [1, 2, 3, 4, 5],
                    y: [0.6, 0.7, 0.75, 0.8, 0.85],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Completion Score'
                }}
            ];
            
            Plotly.newPlot('metricsChart', metricsData, {{
                title: '尺度不变指标趋势',
                xaxis: {{title: '训练步数 (×1000)'}},
                yaxis: {{title: '指标值'}}
            }});
            
            // 阶段进度图表
            var stageData = [
                {{
                    x: ['阶段0', '阶段1', '阶段2', '阶段3'],
                    y: [100, 80, 60, 20],
                    type: 'bar',
                    marker: {{color: ['green', 'green', 'orange', 'lightgray']}}
                }}
            ];
            
            Plotly.newPlot('stageProgressChart', stageData, {{
                title: '课程学习阶段进度',
                xaxis: {{title: '阶段'}},
                yaxis: {{title: '完成度 (%)'}}
            }});
        }}
        
        // 页面加载时初始化
        window.onload = function() {{
            updateTime();
            initializeCharts();
            
            // 每30秒更新一次时间
            setInterval(updateTime, 30000);
        }};
    </script>
</body>
</html>
        """
        
        html_file = self.plugin_dir / f"{experiment_name}_dashboard.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"自定义HTML仪表板已生成: {html_file}")
        return str(html_file)


if __name__ == "__main__":
    # 测试代码
    print("TensorBoard集成模块测试")
    
    # 测试TensorBoard写入器
    writer = CurriculumTensorBoardWriter("./test_tensorboard_logs", "test_experiment")
    
    # 模拟记录数据
    for step in range(100):
        stage = step // 25
        
        # 模拟指标
        metrics = {
            "per_agent_reward": 10 + np.random.normal(0, 1),
            "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
            "efficiency_metric": 0.3 + np.random.uniform(-0.05, 0.1)
        }
        
        scenario_info = {
            "n_uavs": 3 + stage,
            "n_targets": 2 + stage
        }
        
        writer.log_scale_invariant_metrics_detailed(metrics, step, stage, scenario_info)
        
        # 模拟阶段切换
        if step in [25, 50, 75]:
            writer.log_curriculum_stage_transition(stage-1, stage, step, metrics)
    
    # 记录超参数
    hparams = {
        "learning_rate": 0.001,
        "batch_size": 128,
        "k_neighbors": 8
    }
    final_metrics = {
        "final_performance": 0.85,
        "total_episodes": 1000
    }
    writer.log_hparams(hparams, final_metrics)
    
    writer.close()
    
    # 测试自定义插件
    plugin = TensorBoardCustomPlugin("./test_tensorboard_logs")
    plugin.create_curriculum_dashboard_config()
    plugin.generate_custom_html_dashboard("test_experiment")
    
    print("测试完成")
