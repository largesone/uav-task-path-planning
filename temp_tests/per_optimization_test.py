# -*- coding: utf-8 -*-
# 文件名: per_optimization_test.py
# 描述: 优先经验回放(PER)优化测试

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from main import run_scenario
from scenarios import get_small_scenario

def test_per_optimization():
    """测试优先经验回放的优化效果"""
    print("=" * 70)
    print("优先经验回放(PER)优化测试")
    print("=" * 70)
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = f"output/per_optimization_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试配置
    test_configs = {
        "标准DQN": {
            "use_per": False,
            "description": "传统随机采样经验回放"
        },
        "PER-DQN": {
            "use_per": True,
            "description": "优先经验回放，智能采样高价值经验"
        }
    }
    
    results = {}
    
    print("测试配置对比:")
    print("-" * 50)
    for name, cfg in test_configs.items():
        print(f"{name}: {cfg['description']}")
    print()
    
    # 获取测试场景
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    
    for test_name, test_config in test_configs.items():
        print(f"\n{'='*20} 测试 {test_name} {'='*20}")
        
        # 配置参数
        config = Config()
        config.NETWORK_TYPE = "DeepFCNResidual"
        
        # 设置PER参数
        config.training_config.use_prioritized_replay = test_config["use_per"]
        config.training_config.episodes = 300  # 减少轮次以快速对比
        config.training_config.log_interval = 20
        
        if test_config["use_per"]:
            print(f"PER配置:")
            print(f"  - α (优先级指数): {config.training_config.per_alpha}")
            print(f"  - β_start (重要性采样): {config.training_config.per_beta_start}")
            print(f"  - β_frames (增长帧数): {config.training_config.per_beta_frames}")
        
        scenario_name = f"per_test_{test_name.lower().replace('-', '_')}"
        
        try:
            # 运行训练和测试
            result = run_scenario(
                config=config,
                base_uavs=uavs,
                base_targets=targets,
                obstacles=obstacles,
                scenario_name=scenario_name,
                network_type=config.NETWORK_TYPE,
                output_base_dir=output_dir,
                force_retrain=True
            )
            
            results[test_name] = result
            
            print(f"✓ {test_name} 测试完成")
            
        except Exception as e:
            print(f"✗ {test_name} 测试失败: {e}")
            results[test_name] = None
    
    # 生成对比分析
    generate_per_comparison_analysis(results, output_dir, test_configs)
    
    return results, output_dir

def generate_per_comparison_analysis(results, output_dir, test_configs):
    """生成PER对比分析报告"""
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('优先经验回放(PER) vs 标准DQN - 性能对比分析', fontsize=16, fontweight='bold')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 收集对比数据
    comparison_data = {}
    
    for test_name, result in results.items():
        if result is None:
            continue
            
        # 解包结果（处理可能的元组格式）
        if isinstance(result, tuple):
            result_dict = result[0] if len(result) > 0 else {}
        else:
            result_dict = result
            
        training_time = result_dict.get('training_time', 0)
        evaluation_metrics = result_dict.get('evaluation_metrics', {})
        
        comparison_data[test_name] = {
            'training_time': training_time,
            'completion_rate': evaluation_metrics.get('completion_rate', 0),
            'satisfied_targets_rate': evaluation_metrics.get('satisfied_targets_rate', 0),
            'resource_utilization_rate': evaluation_metrics.get('resource_utilization_rate', 0),
            'total_reward_score': evaluation_metrics.get('total_reward_score', 0),
            'load_balance_score': evaluation_metrics.get('load_balance_score', 0)
        }
    
    # 绘制对比图表
    metrics = [
        ('training_time', '训练时间 (秒)'),
        ('completion_rate', '完成率'),
        ('satisfied_targets_rate', '目标满足率'),
        ('resource_utilization_rate', '资源利用率'),
        ('total_reward_score', '总奖励分数'),
        ('load_balance_score', '负载均衡分数')
    ]
    
    colors = ['#FF6B6B', '#4ECDC4']  # 红色和青色
    
    for i, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        
        test_names = list(comparison_data.keys())
        values = [comparison_data[name][metric_key] for name in test_names]
        
        if values:
            bars = ax.bar(test_names, values, color=colors[:len(test_names)], alpha=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}' if metric_key != 'training_time' else f'{value:.1f}s',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 计算改进百分比
            if len(values) == 2:
                improvement = ((values[1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                if metric_key == 'training_time':
                    improvement = -improvement  # 训练时间越短越好
                
                color = 'green' if improvement > 0 else 'red'
                ax.text(0.5, 0.95, f'改进: {improvement:+.1f}%', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=9, fontweight='bold')
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    comparison_path = os.path.join(output_dir, 'per_optimization_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成详细报告
    report_path = os.path.join(output_dir, 'per_optimization_analysis.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("优先经验回放(PER)优化分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("一、算法原理对比\n")
        f.write("-" * 40 + "\n")
        f.write("标准DQN:\n")
        f.write("  - 随机采样经验进行学习\n")
        f.write("  - 所有经验被平等对待\n")
        f.write("  - 可能浪费时间在低价值经验上\n\n")
        
        f.write("PER-DQN:\n")
        f.write("  - 基于TD误差分配优先级\n")
        f.write("  - 优先学习高价值经验\n")
        f.write("  - 使用重要性采样修正偏差\n")
        f.write("  - 提高样本效率和收敛速度\n\n")
        
        f.write("二、核心技术特性\n")
        f.write("-" * 40 + "\n")
        f.write("1. Sum Tree数据结构:\n")
        f.write("   - O(log n)的高效采样和更新\n")
        f.write("   - 支持动态优先级调整\n\n")
        
        f.write("2. 优先级计算:\n")
        f.write("   - 基于TD误差: |δ| + ε\n")
        f.write("   - α控制优先级影响程度\n")
        f.write("   - 防止优先级为0的ε值\n\n")
        
        f.write("3. 重要性采样:\n")
        f.write("   - 权重: (N * P(i))^(-β)\n")
        f.write("   - β从初始值线性增长到1.0\n")
        f.write("   - 修正非均匀采样的偏差\n\n")
        
        f.write("三、性能对比结果\n")
        f.write("-" * 40 + "\n")
        
        if len(comparison_data) >= 2:
            standard_data = comparison_data.get('标准DQN', {})
            per_data = comparison_data.get('PER-DQN', {})
            
            f.write("指标对比:\n")
            for metric_key, metric_name in metrics:
                std_val = standard_data.get(metric_key, 0)
                per_val = per_data.get(metric_key, 0)
                
                if std_val != 0:
                    improvement = ((per_val - std_val) / std_val * 100)
                    if metric_key == 'training_time':
                        improvement = -improvement
                    
                    f.write(f"  {metric_name}:\n")
                    f.write(f"    标准DQN: {std_val:.3f}\n")
                    f.write(f"    PER-DQN: {per_val:.3f}\n")
                    f.write(f"    改进幅度: {improvement:+.1f}%\n\n")
        
        f.write("四、优化效果分析\n")
        f.write("-" * 40 + "\n")
        f.write("预期优化效果:\n")
        f.write("1. 样本效率提升: 优先学习高价值经验\n")
        f.write("2. 收敛速度加快: 减少无效学习时间\n")
        f.write("3. 性能上限提高: 更好的策略质量\n")
        f.write("4. 训练稳定性: 重要性采样修正偏差\n\n")
        
        f.write("五、实施建议\n")
        f.write("-" * 40 + "\n")
        f.write("1. 参数调优:\n")
        f.write("   - α=0.6: 平衡优先级和随机性\n")
        f.write("   - β_start=0.4: 初始重要性采样强度\n")
        f.write("   - 根据具体任务调整参数\n\n")
        
        f.write("2. 监控指标:\n")
        f.write("   - TD误差分布\n")
        f.write("   - 优先级更新频率\n")
        f.write("   - 重要性采样权重\n\n")
        
        f.write("3. 适用场景:\n")
        f.write("   - 复杂决策问题\n")
        f.write("   - 稀疏奖励环境\n")
        f.write("   - 需要快速收敛的应用\n")
    
    print(f"\nPER优化分析已保存:")
    print(f"- 图表: {comparison_path}")
    print(f"- 报告: {report_path}")

if __name__ == "__main__":
    results, output_dir = test_per_optimization()
    print(f"\nPER优化测试完成，结果保存在: {output_dir}")