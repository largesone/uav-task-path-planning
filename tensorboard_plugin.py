"""
TensorBoard自定义插件 - 支持TransformerGNN特定指标和图表展示
包括注意力可视化、课程学习进度、零样本迁移性能等
"""

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class TransformerGNNTensorBoardLogger:
    """TransformerGNN专用TensorBoard日志记录器"""
    
    def __init__(self, log_dir: str = "tensorboard_logs"):
        self.writer = SummaryWriter(log_dir)
        self.step_counter = 0
        
    def log_curriculum_metrics(self, stage: int, metrics: Dict, step: int):
        """记录课程学习指标"""
        # 记录当前阶段
        self.writer.add_scalar('Curriculum/Current_Stage', stage, step)
        
        # 记录尺度不变指标
        if 'per_agent_reward' in metrics:
            self.writer.add_scalar('Curriculum/Per_Agent_Reward', metrics['per_agent_reward'], step)
        
        if 'normalized_completion_score' in metrics:
            self.writer.add_scalar('Curriculum/Normalized_Completion_Score', metrics['normalized_completion_score'], step)
        
        if 'efficiency_metric' in metrics:
            self.writer.add_scalar('Curriculum/Efficiency_Metric', metrics['efficiency_metric'], step)
        
        # 记录场景规模信息
        if 'n_uavs' in metrics:
            self.writer.add_scalar('Curriculum/N_UAVs', metrics['n_uavs'], step)
        
        if 'n_targets' in metrics:
            self.writer.add_scalar('Curriculum/N_Targets', metrics['n_targets'], step)
        
        # 记录训练稳定性指标
        if 'reward_std' in metrics:
            self.writer.add_scalar('Curriculum/Reward_Stability', metrics['reward_std'], step)
        
        # 记录阶段特定指标
        stage_tag = f'Stage_{stage}'
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{stage_tag}/{key}', value, step)
        
    def log_attention_weights(self, attention_weights: torch.Tensor, step: int):
        """记录注意力权重可视化"""
        if attention_weights is None or attention_weights.numel() == 0:
            return
        
        # 确保张量在CPU上
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
        
        # 处理不同维度的注意力权重
        if attention_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
            # 取第一个样本和平均所有头
            attn_matrix = attention_weights[0].mean(dim=0).detach().numpy()
        elif attention_weights.dim() == 3:  # [heads, seq_len, seq_len] 或 [batch, seq_len, seq_len]
            attn_matrix = attention_weights.mean(dim=0).detach().numpy()
        elif attention_weights.dim() == 2:  # [seq_len, seq_len]
            attn_matrix = attention_weights.detach().numpy()
        else:
            return
        
        # 创建注意力热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn_matrix, annot=False, cmap='Blues', ax=ax)
        ax.set_title('Attention Weights Heatmap')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 转换为图像并记录
        self.writer.add_figure('Attention/Weights_Heatmap', fig, step)
        plt.close(fig)
        
        # 记录注意力权重统计
        self.writer.add_scalar('Attention/Mean_Weight', attn_matrix.mean(), step)
        self.writer.add_scalar('Attention/Max_Weight', attn_matrix.max(), step)
        self.writer.add_scalar('Attention/Weight_Std', attn_matrix.std(), step)
        
        # 记录注意力分布直方图
        self.writer.add_histogram('Attention/Weight_Distribution', attn_matrix.flatten(), step)
        
    def log_zero_shot_transfer(self, source_performance: Dict, target_performance: Dict, step: int):
        """记录零样本迁移性能"""
        # 计算迁移性能比率
        for metric in ['per_agent_reward', 'normalized_completion_score', 'efficiency_metric']:
            if metric in source_performance and metric in target_performance:
                source_val = source_performance[metric]
                target_val = target_performance[metric]
                
                if source_val != 0:
                    transfer_ratio = target_val / source_val
                    self.writer.add_scalar(f'ZeroShot/{metric}_transfer_ratio', transfer_ratio, step)
                
                self.writer.add_scalar(f'ZeroShot/{metric}_source', source_val, step)
                self.writer.add_scalar(f'ZeroShot/{metric}_target', target_val, step)
        
        # 记录场景规模对比
        if 'scenario_size' in source_performance and 'scenario_size' in target_performance:
            source_size = source_performance['scenario_size']
            target_size = target_performance['scenario_size']
            
            scale_factor = (target_size[0] * target_size[1]) / (source_size[0] * source_size[1])
            self.writer.add_scalar('ZeroShot/Scale_Factor', scale_factor, step)
        
        # 创建零样本迁移对比图
        metrics_names = ['per_agent_reward', 'normalized_completion_score', 'efficiency_metric']
        source_values = [source_performance.get(m, 0) for m in metrics_names]
        target_values = [target_performance.get(m, 0) for m in metrics_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax.bar(x - width/2, source_values, width, label='Source Performance', alpha=0.8)
        ax.bar(x + width/2, target_values, width, label='Target Performance', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Performance')
        ax.set_title('Zero-Shot Transfer Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45)
        ax.legend()
        
        self.writer.add_figure('ZeroShot/Performance_Comparison', fig, step)
        plt.close(fig)
        
    def log_memory_usage(self, memory_data: Dict, step: int):
        """记录内存使用情况"""
        # 记录各类内存使用
        if 'cpu_memory_mb' in memory_data:
            self.writer.add_scalar('Memory/CPU_Memory_MB', memory_data['cpu_memory_mb'], step)
        
        if 'gpu_memory_mb' in memory_data:
            self.writer.add_scalar('Memory/GPU_Memory_MB', memory_data['gpu_memory_mb'], step)
        
        if 'gpu_memory_cached_mb' in memory_data:
            self.writer.add_scalar('Memory/GPU_Cached_MB', memory_data['gpu_memory_cached_mb'], step)
        
        if 'process_memory_mb' in memory_data:
            self.writer.add_scalar('Memory/Process_Memory_MB', memory_data['process_memory_mb'], step)
        
        # 记录内存效率指标
        if 'n_uavs' in memory_data and 'n_targets' in memory_data:
            total_entities = memory_data['n_uavs'] + memory_data['n_targets']
            if total_entities > 0 and 'gpu_memory_mb' in memory_data:
                memory_per_entity = memory_data['gpu_memory_mb'] / total_entities
                self.writer.add_scalar('Memory/Memory_Per_Entity_MB', memory_per_entity, step)
        
        # 创建内存使用趋势图
        memory_types = ['CPU', 'GPU', 'GPU_Cached', 'Process']
        memory_values = [
            memory_data.get('cpu_memory_mb', 0),
            memory_data.get('gpu_memory_mb', 0),
            memory_data.get('gpu_memory_cached_mb', 0),
            memory_data.get('process_memory_mb', 0)
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax.bar(memory_types, memory_values, color=colors, alpha=0.7)
        
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage by Type')
        
        # 添加数值标签
        for bar, value in zip(bars, memory_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}MB', ha='center', va='bottom')
        
        self.writer.add_figure('Memory/Usage_Breakdown', fig, step)
        plt.close(fig)
        
    def log_network_architecture(self, model: torch.nn.Module):
        """记录网络架构图"""
        try:
            # 记录模型参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.writer.add_scalar('Model/Total_Parameters', total_params, 0)
            self.writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
            
            # 记录各层参数分布
            layer_params = {}
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]  # 获取顶层模块名
                if layer_name not in layer_params:
                    layer_params[layer_name] = 0
                layer_params[layer_name] += param.numel()
            
            # 创建参数分布饼图
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(layer_params.values(), labels=layer_params.keys(), autopct='%1.1f%%')
            ax.set_title('Model Parameters Distribution by Layer')
            
            self.writer.add_figure('Model/Parameters_Distribution', fig, 0)
            plt.close(fig)
            
            # 记录模型结构文本
            model_str = str(model)
            self.writer.add_text('Model/Architecture', model_str, 0)
            
        except Exception as e:
            print(f"记录网络架构时出错: {e}")
        
    def log_k_value_adaptation(self, k_values: List[int], scenario_sizes: List[tuple], step: int):
        """记录k值自适应情况"""
        # 记录k值统计
        self.writer.add_scalar('LocalAttention/Mean_K_Value', np.mean(k_values), step)
        self.writer.add_scalar('LocalAttention/Max_K_Value', np.max(k_values), step)
        self.writer.add_scalar('LocalAttention/Min_K_Value', np.min(k_values), step)
        
        # 记录k值分布
        self.writer.add_histogram('LocalAttention/K_Value_Distribution', np.array(k_values), step)
        
        # 创建k值与场景规模关系图
        if scenario_sizes:
            scenario_complexities = [n_uavs * n_targets for n_uavs, n_targets in scenario_sizes]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(scenario_complexities, k_values, alpha=0.6)
            ax.set_xlabel('Scenario Complexity (UAVs × Targets)')
            ax.set_ylabel('K Value')
            ax.set_title('K Value Adaptation vs Scenario Complexity')
            
            # 添加趋势线
            if len(scenario_complexities) > 1:
                z = np.polyfit(scenario_complexities, k_values, 1)
                p = np.poly1d(z)
                ax.plot(scenario_complexities, p(scenario_complexities), "r--", alpha=0.8)
            
            self.writer.add_figure('LocalAttention/K_Value_Adaptation', fig, step)
            plt.close(fig)
        
    def create_custom_scalar_dashboard(self):
        """创建自定义标量仪表板"""
        # 定义自定义标量组
        layout = {
            "课程学习监控": {
                "Per-Agent奖励": ["Multiline", ["Curriculum/Per_Agent_Reward"]],
                "完成分数": ["Multiline", ["Curriculum/Normalized_Completion_Score"]],
                "效率指标": ["Multiline", ["Curriculum/Efficiency_Metric"]],
                "训练阶段": ["Multiline", ["Curriculum/Current_Stage"]],
            },
            "零样本迁移": {
                "迁移比率": ["Multiline", [
                    "ZeroShot/per_agent_reward_transfer_ratio",
                    "ZeroShot/normalized_completion_score_transfer_ratio",
                    "ZeroShot/efficiency_metric_transfer_ratio"
                ]],
                "规模因子": ["Multiline", ["ZeroShot/Scale_Factor"]],
            },
            "内存监控": {
                "内存使用": ["Multiline", [
                    "Memory/CPU_Memory_MB",
                    "Memory/GPU_Memory_MB",
                    "Memory/Process_Memory_MB"
                ]],
                "内存效率": ["Multiline", ["Memory/Memory_Per_Entity_MB"]],
            },
            "注意力机制": {
                "注意力统计": ["Multiline", [
                    "Attention/Mean_Weight",
                    "Attention/Max_Weight",
                    "Attention/Weight_Std"
                ]],
                "K值自适应": ["Multiline", [
                    "LocalAttention/Mean_K_Value",
                    "LocalAttention/Max_K_Value",
                    "LocalAttention/Min_K_Value"
                ]],
            }
        }
        
        self.writer.add_custom_scalars(layout)
        
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()
