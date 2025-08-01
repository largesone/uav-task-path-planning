"""
课程学习进度可视化模块
提供课程学习过程的实时监控和历史分析可视化功能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class CurriculumProgressVisualizer:
    """
    课程学习进度可视化器
    支持实时监控和历史分析的多种可视化功能
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "curriculum_training"):
        """
        初始化可视化器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 创建可视化输出目录
        self.viz_dir = self.log_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"课程学习进度可视化器初始化完成，输出目录: {self.viz_dir}")
    
    def load_training_history(self) -> Dict[str, Any]:
        """加载训练历史数据"""
        history_file = self.log_dir / f"{self.experiment_name}_history.json"
        
        if not history_file.exists():
            raise FileNotFoundError(f"训练历史文件不存在: {history_file}")
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        print(f"训练历史数据已加载，包含 {len(history['metrics'])} 个指标记录")
        return history
    
    def plot_scale_invariant_metrics(self, history: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        绘制尺度不变指标趋势图
        
        Args:
            history: 训练历史数据
            save_path: 保存路径，None则自动生成
            
        Returns:
            图片保存路径
        """
        if save_path is None:
            save_path = self.viz_dir / "scale_invariant_metrics.png"
        
        metrics_data = history["metrics"]
        if not metrics_data:
            print("警告: 没有找到指标数据")
            return str(save_path)
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics_data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('尺度不变指标趋势分析', fontsize=16, fontweight='bold')
        
        # Per-Agent Reward
        if 'per_agent_reward' in df.columns:
            axes[0, 0].plot(df['step'], df['per_agent_reward'], 'b-', linewidth=2)
            axes[0, 0].set_title('Per-Agent Reward')
            axes[0, 0].set_xlabel('训练步数')
            axes[0, 0].set_ylabel('平均每智能体奖励')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Normalized Completion Score
        if 'normalized_completion_score' in df.columns:
            axes[0, 1].plot(df['step'], df['normalized_completion_score'], 'g-', linewidth=2)
            axes[0, 1].set_title('Normalized Completion Score')
            axes[0, 1].set_xlabel('训练步数')
            axes[0, 1].set_ylabel('归一化完成分数')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Efficiency Metric
        if 'efficiency_metric' in df.columns:
            axes[1, 0].plot(df['step'], df['efficiency_metric'], 'r-', linewidth=2)
            axes[1, 0].set_title('Efficiency Metric')
            axes[1, 0].set_xlabel('训练步数')
            axes[1, 0].set_ylabel('效率指标')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 场景规模变化
        if 'n_uavs' in df.columns and 'n_targets' in df.columns:
            scale_factor = df['n_uavs'] * df['n_targets']
            axes[1, 1].plot(df['step'], scale_factor, 'purple', linewidth=2)
            axes[1, 1].set_title('场景规模变化')
            axes[1, 1].set_xlabel('训练步数')
            axes[1, 1].set_ylabel('规模因子 (UAVs × Targets)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"尺度不变指标图已保存到: {save_path}")
        return str(save_path)
    
    def plot_curriculum_stages(self, history: Dict[str, Any], 
                             save_path: Optional[str] = None) -> str:
        """
        绘制课程学习阶段进度图
        
        Args:
            history: 训练历史数据
            save_path: 保存路径
            
        Returns:
            图片保存路径
        """
        if save_path is None:
            save_path = self.viz_dir / "curriculum_stages.png"
        
        metrics_data = history["metrics"]
        stage_transitions = history["stage_transitions"]
        rollback_events = history["rollback_events"]
        
        if not metrics_data:
            print("警告: 没有找到指标数据")
            return str(save_path)
        
        df = pd.DataFrame(metrics_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('课程学习阶段进度分析', fontsize=16, fontweight='bold')
        
        # 上图：阶段变化和性能指标
        if 'stage' in df.columns and 'normalized_completion_score' in df.columns:
            # 绘制性能指标
            ax1.plot(df['step'], df['normalized_completion_score'], 'b-', 
                    linewidth=2, label='Normalized Completion Score')
            
            # 标记阶段切换
            for transition in stage_transitions:
                ax1.axvline(x=transition['step'], color='red', linestyle='--', 
                           alpha=0.7, label='阶段切换' if transition == stage_transitions[0] else "")
                ax1.text(transition['step'], ax1.get_ylim()[1]*0.9, 
                        f"阶段 {transition['to_stage']}", rotation=90, 
                        verticalalignment='top', fontsize=8)
            
            # 标记回退事件
            for rollback in rollback_events:
                ax1.axvline(x=rollback['step'], color='orange', linestyle=':', 
                           alpha=0.7, label='回退事件' if rollback == rollback_events[0] else "")
            
            ax1.set_title('性能指标与阶段切换')
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('归一化完成分数')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 下图：阶段分布
        if 'stage' in df.columns:
            stage_counts = df['stage'].value_counts().sort_index()
            ax2.bar(stage_counts.index, stage_counts.values, alpha=0.7, color='skyblue')
            ax2.set_title('各阶段训练步数分布')
            ax2.set_xlabel('课程学习阶段')
            ax2.set_ylabel('训练步数')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"课程学习阶段图已保存到: {save_path}")
        return str(save_path)
    
    def plot_scale_transfer_analysis(self, history: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        绘制尺度迁移能力分析图
        
        Args:
            history: 训练历史数据
            save_path: 保存路径
            
        Returns:
            图片保存路径
        """
        if save_path is None:
            save_path = self.viz_dir / "scale_transfer_analysis.png"
        
        metrics_data = history["metrics"]
        if not metrics_data:
            print("警告: 没有找到指标数据")
            return str(save_path)
        
        df = pd.DataFrame(metrics_data)
        
        # 检查必要的列
        required_cols = ['n_uavs', 'n_targets', 'normalized_completion_score', 'efficiency_metric']
        if not all(col in df.columns for col in required_cols):
            print("警告: 缺少必要的数据列进行尺度迁移分析")
            return str(save_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('尺度迁移能力分析', fontsize=16, fontweight='bold')
        
        # 计算规模因子
        df['scale_factor'] = df['n_uavs'] * df['n_targets']
        
        # 1. 性能 vs 规模因子散点图
        scatter = axes[0, 0].scatter(df['scale_factor'], df['normalized_completion_score'], 
                                   c=df['stage'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('性能 vs 场景规模')
        axes[0, 0].set_xlabel('规模因子 (UAVs × Targets)')
        axes[0, 0].set_ylabel('归一化完成分数')
        plt.colorbar(scatter, ax=axes[0, 0], label='训练阶段')
        
        # 2. 效率 vs 规模因子
        axes[0, 1].scatter(df['scale_factor'], df['efficiency_metric'], 
                          c=df['stage'], cmap='plasma', alpha=0.6)
        axes[0, 1].set_title('效率 vs 场景规模')
        axes[0, 1].set_xlabel('规模因子 (UAVs × Targets)')
        axes[0, 1].set_ylabel('效率指标')
        
        # 3. 不同UAV数量下的性能分布
        uav_groups = df.groupby('n_uavs')['normalized_completion_score']
        uav_means = uav_groups.mean()
        uav_stds = uav_groups.std()
        
        axes[1, 0].errorbar(uav_means.index, uav_means.values, 
                           yerr=uav_stds.values, marker='o', capsize=5)
        axes[1, 0].set_title('不同UAV数量下的性能')
        axes[1, 0].set_xlabel('UAV数量')
        axes[1, 0].set_ylabel('平均归一化完成分数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 不同目标数量下的性能分布
        target_groups = df.groupby('n_targets')['normalized_completion_score']
        target_means = target_groups.mean()
        target_stds = target_groups.std()
        
        axes[1, 1].errorbar(target_means.index, target_means.values, 
                           yerr=target_stds.values, marker='s', capsize=5, color='orange')
        axes[1, 1].set_title('不同目标数量下的性能')
        axes[1, 1].set_xlabel('目标数量')
        axes[1, 1].set_ylabel('平均归一化完成分数')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"尺度迁移分析图已保存到: {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self, history: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        创建交互式训练监控仪表板
        
        Args:
            history: 训练历史数据
            save_path: 保存路径
            
        Returns:
            HTML文件保存路径
        """
        if save_path is None:
            save_path = self.viz_dir / "interactive_dashboard.html"
        
        metrics_data = history["metrics"]
        if not metrics_data:
            print("警告: 没有找到指标数据")
            return str(save_path)
        
        df = pd.DataFrame(metrics_data)
        
        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Per-Agent Reward', 'Normalized Completion Score',
                          'Efficiency Metric', '场景规模变化',
                          '阶段进度', '性能热力图'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "heatmap"}]]
        )
        
        # 添加指标趋势线
        if 'per_agent_reward' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['per_agent_reward'], 
                          mode='lines', name='Per-Agent Reward',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        if 'normalized_completion_score' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['normalized_completion_score'], 
                          mode='lines', name='Completion Score',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        if 'efficiency_metric' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['efficiency_metric'], 
                          mode='lines', name='Efficiency',
                          line=dict(color='red', width=2)),
                row=2, col=1
            )
        
        # 场景规模变化
        if 'n_uavs' in df.columns and 'n_targets' in df.columns:
            scale_factor = df['n_uavs'] * df['n_targets']
            fig.add_trace(
                go.Scatter(x=df['step'], y=scale_factor, 
                          mode='lines+markers', name='Scale Factor',
                          line=dict(color='purple', width=2)),
                row=2, col=2
            )
        
        # 阶段进度
        if 'stage' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['stage'], 
                          mode='lines+markers', name='Current Stage',
                          line=dict(color='orange', width=3)),
                row=3, col=1
            )
        
        # 性能热力图
        if all(col in df.columns for col in ['n_uavs', 'n_targets', 'normalized_completion_score']):
            # 创建透视表
            pivot_table = df.pivot_table(
                values='normalized_completion_score', 
                index='n_uavs', 
                columns='n_targets', 
                aggfunc='mean'
            )
            
            fig.add_trace(
                go.Heatmap(z=pivot_table.values,
                          x=pivot_table.columns,
                          y=pivot_table.index,
                          colorscale='Viridis',
                          name='Performance Heatmap'),
                row=3, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="课程学习训练监控仪表板",
            title_x=0.5,
            height=1000,
            showlegend=True
        )
        
        # 保存为HTML文件
        fig.write_html(str(save_path))
        
        print(f"交互式仪表板已保存到: {save_path}")
        return str(save_path)
    
    def generate_training_report(self, history: Dict[str, Any]) -> str:
        """
        生成训练报告
        
        Args:
            history: 训练历史数据
            
        Returns:
            报告文件路径
        """
        report_path = self.viz_dir / "training_report.md"
        
        metrics_data = history["metrics"]
        stage_transitions = history["stage_transitions"]
        rollback_events = history["rollback_events"]
        
        # 计算统计信息
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # 基础统计
            total_steps = df['step'].max() if 'step' in df.columns else 0
            total_stages = df['stage'].nunique() if 'stage' in df.columns else 0
            
            # 性能统计
            final_performance = {}
            if 'normalized_completion_score' in df.columns:
                final_performance['completion_score'] = df['normalized_completion_score'].iloc[-1]
                final_performance['avg_completion_score'] = df['normalized_completion_score'].mean()
            
            if 'efficiency_metric' in df.columns:
                final_performance['efficiency'] = df['efficiency_metric'].iloc[-1]
                final_performance['avg_efficiency'] = df['efficiency_metric'].mean()
        else:
            total_steps = 0
            total_stages = 0
            final_performance = {}
        
        # 生成报告内容
        report_content = f"""# 课程学习训练报告

## 实验信息
- **实验名称**: {self.experiment_name}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总训练步数**: {total_steps:,}
- **课程学习阶段数**: {total_stages}

## 训练概览
- **阶段切换次数**: {len(stage_transitions)}
- **回退事件次数**: {len(rollback_events)}
- **数据记录点数**: {len(metrics_data)}

## 性能指标

### 最终性能
"""
        
        if final_performance:
            if 'completion_score' in final_performance:
                report_content += f"- **最终完成分数**: {final_performance['completion_score']:.4f}\n"
                report_content += f"- **平均完成分数**: {final_performance['avg_completion_score']:.4f}\n"
            
            if 'efficiency' in final_performance:
                report_content += f"- **最终效率指标**: {final_performance['efficiency']:.4f}\n"
                report_content += f"- **平均效率指标**: {final_performance['avg_efficiency']:.4f}\n"
        
        report_content += f"""
## 课程学习进度

### 阶段切换历史
"""
        
        for i, transition in enumerate(stage_transitions, 1):
            report_content += f"{i}. 步数 {transition['step']:,}: 阶段 {transition['from_stage']} → {transition['to_stage']} ({transition['reason']})\n"
        
        if rollback_events:
            report_content += f"""
### 回退事件
"""
            for i, rollback in enumerate(rollback_events, 1):
                report_content += f"{i}. 步数 {rollback['step']:,}: 阶段 {rollback['stage']} 回退 (性能下降 {rollback['performance_drop']:.3f})\n"
        
        report_content += f"""
## 可视化文件
- 尺度不变指标趋势图: `scale_invariant_metrics.png`
- 课程学习阶段图: `curriculum_stages.png`
- 尺度迁移分析图: `scale_transfer_analysis.png`
- 交互式仪表板: `interactive_dashboard.html`

## 建议和总结
基于训练数据分析，该课程学习实验显示了{'良好' if len(rollback_events) < 3 else '需要改进'}的训练稳定性。
{'建议继续当前的训练策略。' if len(rollback_events) < 2 else '建议调整回退阈值或学习率策略。'}
"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"训练报告已生成: {report_path}")
        return str(report_path)
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        生成所有可视化图表
        
        Returns:
            生成的文件路径字典
        """
        try:
            history = self.load_training_history()
        except FileNotFoundError:
            print("错误: 无法加载训练历史数据")
            return {}
        
        results = {}
        
        # 生成各种可视化
        results['metrics'] = self.plot_scale_invariant_metrics(history)
        results['stages'] = self.plot_curriculum_stages(history)
        results['transfer'] = self.plot_scale_transfer_analysis(history)
        results['dashboard'] = self.create_interactive_dashboard(history)
        results['report'] = self.generate_training_report(history)
        
        print(f"所有可视化已生成完成，共 {len(results)} 个文件")
        return results


if __name__ == "__main__":
    # 测试代码
    print("课程学习进度可视化模块测试")
    
    # 创建测试数据
    test_history = {
        "metrics": [
            {
                "step": i,
                "stage": i // 30,
                "n_uavs": 3 + (i // 30),
                "n_targets": 2 + (i // 30),
                "per_agent_reward": 10 + np.random.normal(0, 2),
                "normalized_completion_score": 0.7 + np.random.uniform(-0.2, 0.3),
                "efficiency_metric": 0.3 + np.random.uniform(-0.1, 0.2),
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
    
    # 保存测试数据
    os.makedirs("./test_logs", exist_ok=True)
    with open("./test_logs/test_experiment_history.json", 'w') as f:
        json.dump(test_history, f, indent=2)
    
    # 测试可视化器
    visualizer = CurriculumProgressVisualizer("./test_logs", "test_experiment")
    results = visualizer.generate_all_visualizations()
    
    print("测试完成，生成的文件:")
    for name, path in results.items():
        print(f"  {name}: {path}")
