"""
训练过程可视化系统 - 实现课程学习进度和回退事件的可视化
支持实时监控和历史回顾
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from pathlib import Path
import json

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str = "training_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.training_data = []
        self.stage_transitions = []
        self.fallback_events = []
        
    def log_training_step(self, step: int, metrics: Dict, stage: int):
        """记录训练步骤数据"""
        training_record = {
            'step': step,
            'stage': stage,
            'timestamp': datetime.now().timestamp(),
            **metrics
        }
        self.training_data.append(training_record)
        
    def log_stage_transition(self, from_stage: int, to_stage: int, timestamp: float):
        """记录阶段转换事件"""
        transition_record = {
            'from_stage': from_stage,
            'to_stage': to_stage,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        self.stage_transitions.append(transition_record)
        
    def log_fallback_event(self, stage: int, reason: str, timestamp: float):
        """记录回退事件"""
        fallback_record = {
            'stage': stage,
            'reason': reason,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        self.fallback_events.append(fallback_record)
        
    def plot_curriculum_progress(self) -> go.Figure:
        """绘制课程学习进度图"""
        if not self.training_data:
            return go.Figure().add_annotation(text="无训练数据", x=0.5, y=0.5)
        
        df = pd.DataFrame(self.training_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('课程学习阶段进度', '性能指标趋势', '奖励变化', '完成率变化'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # 阶段进度图
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['stage'], mode='lines+markers',
                      name='训练阶段', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 添加阶段转换标记
        for transition in self.stage_transitions:
            step_idx = next((i for i, record in enumerate(self.training_data) 
                           if record['timestamp'] >= transition['timestamp']), None)
            if step_idx is not None:
                step = self.training_data[step_idx]['step']
                fig.add_vline(x=step, line_dash="dash", line_color="green",
                             annotation_text=f"升级到阶段{transition['to_stage']}", row=1, col=1)
        
        # 添加回退事件标记
        for fallback in self.fallback_events:
            step_idx = next((i for i, record in enumerate(self.training_data) 
                           if record['timestamp'] >= fallback['timestamp']), None)
            if step_idx is not None:
                step = self.training_data[step_idx]['step']
                fig.add_vline(x=step, line_dash="dash", line_color="red",
                             annotation_text=f"回退: {fallback['reason']}", row=1, col=1)
        
        # 性能指标趋势
        if 'per_agent_reward' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['per_agent_reward'], mode='lines',
                          name='Per-Agent奖励', line=dict(color='orange')),
                row=1, col=2
            )
        
        if 'normalized_completion_score' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['normalized_completion_score'], mode='lines',
                          name='归一化完成分数', line=dict(color='green')),
                row=2, col=1
            )
        
        if 'efficiency_metric' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['efficiency_metric'], mode='lines',
                          name='效率指标', line=dict(color='purple')),
                row=2, col=2
            )
        
        fig.update_layout(
            title='课程学习训练进度监控',
            height=800,
            showlegend=True
        )
        
        return fig
        
    def plot_performance_trends(self) -> go.Figure:
        """绘制性能趋势图"""
        if not self.training_data:
            return go.Figure().add_annotation(text="无训练数据", x=0.5, y=0.5)
        
        df = pd.DataFrame(self.training_data)
        
        fig = go.Figure()
        
        # 按阶段分组绘制性能趋势
        stages = sorted(df['stage'].unique())
        colors = px.colors.qualitative.Set1
        
        for i, stage in enumerate(stages):
            stage_data = df[df['stage'] == stage]
            color = colors[i % len(colors)]
            
            if 'per_agent_reward' in stage_data.columns:
                fig.add_trace(go.Scatter(
                    x=stage_data['step'],
                    y=stage_data['per_agent_reward'],
                    mode='lines+markers',
                    name=f'阶段{stage} - Per-Agent奖励',
                    line=dict(color=color),
                    opacity=0.8
                ))
        
        # 添加移动平均线
        if 'per_agent_reward' in df.columns:
            window_size = max(10, len(df) // 20)
            moving_avg = df['per_agent_reward'].rolling(window=window_size, center=True).mean()
            fig.add_trace(go.Scatter(
                x=df['step'],
                y=moving_avg,
                mode='lines',
                name=f'移动平均({window_size}步)',
                line=dict(color='black', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title='性能趋势分析',
            xaxis_title='训练步数',
            yaxis_title='Per-Agent奖励',
            hovermode='x unified'
        )
        
        return fig
        
    def plot_fallback_analysis(self) -> go.Figure:
        """绘制回退事件分析图"""
        if not self.fallback_events:
            return go.Figure().add_annotation(text="无回退事件", x=0.5, y=0.5)
        
        df_fallback = pd.DataFrame(self.fallback_events)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('回退事件时间线', '回退原因统计'),
            specs=[[{"type": "scatter"}], [{"type": "bar"}]]
        )
        
        # 回退事件时间线
        fig.add_trace(
            go.Scatter(
                x=[datetime.fromtimestamp(ts) for ts in df_fallback['timestamp']],
                y=df_fallback['stage'],
                mode='markers+text',
                text=df_fallback['reason'],
                textposition='top center',
                marker=dict(size=12, color='red', symbol='x'),
                name='回退事件'
            ),
            row=1, col=1
        )
        
        # 回退原因统计
        reason_counts = df_fallback['reason'].value_counts()
        fig.add_trace(
            go.Bar(
                x=reason_counts.index,
                y=reason_counts.values,
                name='回退次数',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='回退事件分析',
            height=600,
            showlegend=True
        )
        
        return fig
        
    def create_training_dashboard(self) -> go.Figure:
        """创建训练监控仪表板"""
        if not self.training_data:
            return go.Figure().add_annotation(text="无训练数据", x=0.5, y=0.5)
        
        df = pd.DataFrame(self.training_data)
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '实时性能指标', '阶段进度',
                '奖励分布', '训练稳定性',
                '内存使用', '收敛分析'
            ),
            specs=[
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 实时性能指标
        if 'per_agent_reward' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['per_agent_reward'],
                          mode='lines', name='Per-Agent奖励'),
                row=1, col=1
            )
        
        # 阶段进度
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['stage'],
                      mode='lines+markers', name='训练阶段',
                      line=dict(shape='hv')),
            row=1, col=2
        )
        
        # 奖励分布
        if 'per_agent_reward' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['per_agent_reward'], name='奖励分布',
                           nbinsx=30),
                row=2, col=1
            )
        
        # 训练稳定性（奖励方差）
        if 'per_agent_reward' in df.columns:
            window_size = max(10, len(df) // 20)
            rolling_std = df['per_agent_reward'].rolling(window=window_size).std()
            fig.add_trace(
                go.Scatter(x=df['step'], y=rolling_std,
                          mode='lines', name='奖励标准差'),
                row=2, col=2
            )
        
        # 内存使用（如果有数据）
        if 'memory_usage_mb' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['memory_usage_mb'],
                          mode='lines', name='内存使用(MB)'),
                row=3, col=1
            )
        
        # 收敛分析
        if 'per_agent_reward' in df.columns:
            # 计算累积平均奖励
            cumulative_avg = df['per_agent_reward'].expanding().mean()
            fig.add_trace(
                go.Scatter(x=df['step'], y=cumulative_avg,
                          mode='lines', name='累积平均奖励'),
                row=3, col=2
            )
        
        fig.update_layout(
            title='训练监控仪表板',
            height=1000,
            showlegend=True
        )
        
        return fig
        
    def export_interactive_report(self, filename: str):
        """导出交互式HTML报告"""
        # 创建综合报告
        dashboard = self.create_training_dashboard()
        curriculum_progress = self.plot_curriculum_progress()
        performance_trends = self.plot_performance_trends()
        
        # 保存各个图表
        dashboard.write_html(self.save_dir / f"{filename}_dashboard.html")
        curriculum_progress.write_html(self.save_dir / f"{filename}_curriculum.html")
        performance_trends.write_html(self.save_dir / f"{filename}_trends.html")
        
        if self.fallback_events:
            fallback_analysis = self.plot_fallback_analysis()
            fallback_analysis.write_html(self.save_dir / f"{filename}_fallbacks.html")
        
        # 创建主报告页面
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>训练过程可视化报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 20px 0; }}
                .iframe-container {{ width: 100%; height: 600px; border: 1px solid #ccc; margin: 10px 0; }}
                iframe {{ width: 100%; height: 100%; border: none; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>TransformerGNN训练过程可视化报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>训练摘要</h2>
                <div class="summary">
                    <p><strong>总训练步数:</strong> {len(self.training_data)}</p>
                    <p><strong>阶段转换次数:</strong> {len(self.stage_transitions)}</p>
                    <p><strong>回退事件次数:</strong> {len(self.fallback_events)}</p>
                    <p><strong>最高训练阶段:</strong> {max([r['stage'] for r in self.training_data]) if self.training_data else 0}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>训练监控仪表板</h2>
                <div class="iframe-container">
                    <iframe src="{filename}_dashboard.html"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>课程学习进度</h2>
                <div class="iframe-container">
                    <iframe src="{filename}_curriculum.html"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>性能趋势分析</h2>
                <div class="iframe-container">
                    <iframe src="{filename}_trends.html"></iframe>
                </div>
            </div>
            
            {"<div class='section'><h2>回退事件分析</h2><div class='iframe-container'><iframe src='" + filename + "_fallbacks.html'></iframe></div></div>" if self.fallback_events else ""}
        </body>
        </html>
        """
        
        with open(self.save_dir / f"{filename}_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 导出原始数据
        export_data = {
            'training_data': self.training_data,
            'stage_transitions': self.stage_transitions,
            'fallback_events': self.fallback_events,
            'export_time': datetime.now().isoformat()
        }
        
        with open(self.save_dir / f"{filename}_data.json", 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"交互式报告已导出到: {self.save_dir / f'{filename}_report.html'}")
        print(f"原始数据已导出到: {self.save_dir / f'{filename}_data.json'}")
