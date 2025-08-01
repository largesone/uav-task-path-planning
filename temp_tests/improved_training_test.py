# -*- coding: utf-8 -*-
# 文件名: improved_training_test.py
# 描述: 改进训练系统测试脚本

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from main import run_scenario
from scenarios import get_small_scenario

def test_improved_training():
    """测试改进的训练系统"""
    print("=" * 60)
    print("改进训练系统测试")
    print("=" * 60)
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = f"output/improved_training_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置参数
    config = Config()
    
    # 测试不同的网络结构
    network_types = ["SimpleNetwork", "DeepFCN", "DeepFCNResidual", "GAT"]
    results = {}
    
    for network_type in network_types:
        print(f"\n测试网络类型: {network_type}")
        print("-" * 40)
        
        # 设置网络类型
        config.NETWORK_TYPE = network_type
        
        # 获取测试场景
        uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
        scenario_name = f"improved_test_{network_type}"
        
        try:
            # 运行训练和测试
            result = run_scenario(
                config=config,
                base_uavs=uavs,
                base_targets=targets,
                obstacles=obstacles,
                scenario_name=scenario_name,
                network_type=network_type,
                output_base_dir=output_dir
            )
            
            results[network_type] = result
            
            print(f"✓ {network_type} 测试完成")
            
        except Exception as e:
            print(f"✗ {network_type} 测试失败: {e}")
            results[network_type] = None
    
    # 生成对比分析
    generate_comparison_analysis(results, output_dir)
    
    return results, output_dir

def generate_comparison_analysis(results, output_dir):
    """生成对比分析报告"""
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('改进训练系统 - 网络结构对比分析', fontsize=16, fontweight='bold')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    metrics_to_plot = [
        ('training_time', '训练时间 (秒)'),
        ('plan_generation_time', '规划时间 (秒)'),
        ('completion_rate', '完成率'),
        ('satisfied_targets_rate', '目标满足率'),
        ('resource_utilization_rate', '资源利用率'),
        ('load_balance_score', '负载均衡分数')
    ]
    
    network_names = []
    metric_values = {metric: [] for metric, _ in metrics_to_plot}
    
    # 收集数据
    for network_type, result in results.items():
        if result is None:
            continue
            
        network_names.append(network_type)
        
        # 提取指标
        training_time = result.get('training_time', 0)
        plan_generation_time = result.get('plan_generation_time', 0)
        evaluation_metrics = result.get('evaluation_metrics', {})
        
        metric_values['training_time'].append(training_time)
        metric_values['plan_generation_time'].append(plan_generation_time)
        metric_values['completion_rate'].append(evaluation_metrics.get('completion_rate', 0))
        metric_values['satisfied_targets_rate'].append(evaluation_metrics.get('satisfied_targets_rate', 0))
        metric_values['resource_utilization_rate'].append(evaluation_metrics.get('resource_utilization_rate', 0))
        metric_values['load_balance_score'].append(evaluation_metrics.get('load_balance_score', 0))
    
    # 绘制对比图表
    for i, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        
        if metric_values[metric_key]:
            bars = ax.bar(network_names, metric_values[metric_key], 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(network_names)])
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values[metric_key]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    comparison_path = os.path.join(output_dir, 'improved_training_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成文字报告
    report_path = os.path.join(output_dir, 'improved_training_analysis.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("改进训练系统分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试配置改进:\n")
        f.write("- 增加训练轮次: 1000 → 1500\n")
        f.write("- 提高学习率: 0.00001 → 0.0001\n")
        f.write("- 优化探索策略: 更平缓的衰减\n")
        f.write("- 改进早停机制: 基于资源满足率\n")
        f.write("- 优化网络结构: 添加注意力机制和SE块\n\n")
        
        f.write("网络结构对比:\n")
        f.write("-" * 30 + "\n")
        
        for network_type, result in results.items():
            if result is None:
                f.write(f"{network_type}: 测试失败\n")
                continue
                
            training_time = result.get('training_time', 0)
            evaluation_metrics = result.get('evaluation_metrics', {})
            
            f.write(f"\n{network_type}:\n")
            f.write(f"  训练时间: {training_time:.2f}秒\n")
            f.write(f"  完成率: {evaluation_metrics.get('completion_rate', 0):.3f}\n")
            f.write(f"  目标满足率: {evaluation_metrics.get('satisfied_targets_rate', 0):.3f}\n")
            f.write(f"  资源利用率: {evaluation_metrics.get('resource_utilization_rate', 0):.3f}\n")
        
        # 推荐最佳网络
        if network_names and metric_values['completion_rate']:
            best_idx = np.argmax(metric_values['completion_rate'])
            best_network = network_names[best_idx]
            best_completion = metric_values['completion_rate'][best_idx]
            
            f.write(f"\n推荐网络结构:\n")
            f.write(f"- 最佳网络: {best_network}\n")
            f.write(f"- 完成率: {best_completion:.3f}\n")
    
    print(f"\n对比分析已保存:")
    print(f"- 图表: {comparison_path}")
    print(f"- 报告: {report_path}")

if __name__ == "__main__":
    results, output_dir = test_improved_training()
    print(f"\n改进训练测试完成，结果保存在: {output_dir}")