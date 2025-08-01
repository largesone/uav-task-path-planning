# -*- coding: utf-8 -*-
# 文件名: quick_convergence_test.py
# 描述: 简化的网络收敛性测试脚本，用于快速验证算法性能

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

# 导入核心模块
from main import run_scenario, set_chinese_font
from scenarios import get_small_scenario
from config import Config

def run_quick_test():
    """运行简化的收敛性测试"""
    print("=" * 60)
    print("简化网络收敛性测试 - 改进版")
    print("=" * 60)
    
    # 初始化配置
    config = Config()
    
    # 修改配置以加速测试并应用改进
    config.training_config.episodes = 300  # 增加训练轮次
    config.training_config.patience = 100  # 增加早停耐心值
    config.training_config.log_interval = 10  # 增加日志输出频率
    config.LEARNING_RATE = 1e-04  # 调整学习率
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = f"output/quick_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    set_chinese_font()
    
    # 获取场景数据
    print("加载小型测试场景...")
    uavs, targets, obstacles = get_small_scenario(50.0)
    print(f"场景配置: {len(uavs)} UAVs, {len(targets)} 目标, {len(obstacles)} 障碍物")
    
    # 测试网络类型
    network_types = ["SimpleNetwork", "DeepFCN"]
    results = {}
    
    for network_type in network_types:
        print(f"\n{'=' * 60}")
        print(f"测试网络类型: {network_type}")
        print(f"{'=' * 60}")
        
        # 运行场景
        final_plan, training_time, training_history, evaluation_metrics = run_scenario(
            config, uavs, targets, obstacles, "quick_test",
            network_type=network_type,
            save_visualization=True,
            show_visualization=False,
            output_base_dir=output_dir
        )
        
        # 保存结果
        results[network_type] = {
            "training_time": training_time,
            "evaluation_metrics": evaluation_metrics,
            "training_history": training_history
        }
        
        print(f"测试完成: {network_type}")
        print(f"训练时间: {training_time:.2f}秒")
        if evaluation_metrics:
            print(f"完成率: {evaluation_metrics.get('completion_rate', 0):.4f}")
            print(f"总奖励: {evaluation_metrics.get('total_reward_score', 0):.2f}")
    
    # 生成对比图表
    generate_comparison_plot(results, output_dir)
    
    print(f"\n{'=' * 60}")
    print(f"测试完成! 结果保存在: {output_dir}")
    print(f"{'=' * 60}")
    
    return results, output_dir

def generate_comparison_plot(results, output_dir):
    """生成网络性能对比图"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('网络性能对比', fontsize=16)
    
    # 奖励曲线对比
    ax = axes[0, 0]
    for network_type, result in results.items():
        history = result.get('training_history', {})
        rewards = history.get('episode_rewards', [])
        if rewards:
            ax.plot(rewards, label=network_type)
    ax.set_title('奖励曲线对比')
    ax.set_xlabel('回合')
    ax.set_ylabel('奖励')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 损失曲线对比
    ax = axes[0, 1]
    for network_type, result in results.items():
        history = result.get('training_history', {})
        losses = history.get('episode_losses', [])
        if losses:
            ax.plot(losses, label=network_type)
    ax.set_title('损失曲线对比')
    ax.set_xlabel('回合')
    ax.set_ylabel('损失')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 完成率曲线对比
    ax = axes[1, 0]
    for network_type, result in results.items():
        history = result.get('training_history', {})
        completion_rates = history.get('completion_rates', [])
        if completion_rates:
            ax.plot(completion_rates, label=network_type)
    ax.set_title('完成率曲线对比')
    ax.set_xlabel('回合')
    ax.set_ylabel('完成率')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 探索率曲线对比
    ax = axes[1, 1]
    for network_type, result in results.items():
        history = result.get('training_history', {})
        epsilon_values = history.get('epsilon_values', [])
        if epsilon_values:
            ax.plot(epsilon_values, label=network_type)
    ax.set_title('探索率曲线对比')
    ax.set_xlabel('回合')
    ax.set_ylabel('探索率')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_path = f'{output_dir}/network_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存至: {save_path}")

def analyze_convergence(results):
    """分析算法收敛性并提出改进建议"""
    print("\n" + "=" * 60)
    print("算法收敛性分析")
    print("=" * 60)
    
    for network_type, result in results.items():
        history = result.get('training_history', {})
        rewards = history.get('episode_rewards', [])
        losses = history.get('episode_losses', [])
        completion_rates = history.get('completion_rates', [])
        
        if not rewards:
            continue
        
        print(f"\n网络类型: {network_type}")
        print("-" * 30)
        
        # 奖励分析
        final_reward = rewards[-1] if rewards else 0
        max_reward = max(rewards) if rewards else 0
        reward_improvement = final_reward - rewards[0] if len(rewards) > 1 else 0
        
        print(f"奖励分析:")
        print(f"  - 最终奖励: {final_reward:.2f}")
        print(f"  - 最大奖励: {max_reward:.2f}")
        print(f"  - 奖励提升: {reward_improvement:.2f}")
        
        # 收敛性分析
        if len(rewards) > 50:
            recent_std = np.std(rewards[-30:])
            overall_std = np.std(rewards)
            stability_ratio = recent_std / overall_std if overall_std > 0 else 0
            
            print(f"收敛性分析:")
            print(f"  - 稳定性比率: {stability_ratio:.3f}")
            
            if stability_ratio < 0.3:
                print(f"  - 收敛状态: 良好 (稳定)")
            elif stability_ratio < 0.6:
                print(f"  - 收敛状态: 中等 (部分稳定)")
            else:
                print(f"  - 收敛状态: 差 (不稳定)")
        
        # 完成率分析
        final_completion = completion_rates[-1] if completion_rates else 0
        print(f"完成率分析:")
        print(f"  - 最终完成率: {final_completion:.3f}")
        
        # 提出改进建议
        print(f"改进建议:")
        if final_completion < 0.5:
            print(f"  - 完成率较低，考虑调整奖励函数或增加训练轮次")
        
        if len(rewards) > 50 and stability_ratio > 0.5:
            print(f"  - 收敛不稳定，考虑降低学习率或使用更稳定的算法如Double DQN")
        
        if max_reward > final_reward * 1.2:
            print(f"  - 最终奖励低于最大奖励，考虑增加早停耐心值或使用模型检查点")

if __name__ == "__main__":
    results, output_dir = run_quick_test()
    analyze_convergence(results)