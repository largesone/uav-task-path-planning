"""
方案质量对比分析系统 - 展示零样本迁移的性能提升
对比TransformerGNN与传统方法在不同场景下的表现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time
from collections import defaultdict

@dataclass
class SolutionQualityMetrics:
    """方案质量指标"""
    method_name: str
    scenario_size: Tuple[int, int]  # (n_uavs, n_targets)
    completion_rate: float
    efficiency_score: float
    collision_rate: float
    resource_utilization: float
    convergence_time: float
    zero_shot_performance: float
    per_agent_reward: float
    normalized_completion_score: float

class SolutionQualityAnalyzer:
    """方案质量分析器"""
    
    def __init__(self, results_dir: str = "quality_analysis"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.quality_data: List[SolutionQualityMetrics] = []
        self.baseline_performance = {}
        
    def evaluate_solution_quality(self, method, scenarios: List[Tuple[int, int]], 
                                 episodes_per_scenario: int = 50) -> List[SolutionQualityMetrics]:
        """评估方案质量"""
        results = []
        
        for n_uavs, n_targets in scenarios:
            print(f"评估方法 {method.__class__.__name__} 在场景 ({n_uavs}, {n_targets}) 下的性能...")
            
            try:
                # 创建测试环境
                from environment import UAVTaskEnv
                env = UAVTaskEnv(
                    n_uavs=n_uavs,
                    n_targets=n_targets,
                    obs_mode="graph" if "transformer" in method.__class__.__name__.lower() else "flat"
                )
                
                # 运行多个episode收集统计数据
                episode_results = []
                start_time = time.time()
                
                for episode in range(episodes_per_scenario):
                    state = env.reset()
                    done = False
                    step_count = 0
                    total_reward = 0
                    collisions = 0
                    
                    while not done and step_count < 200:
                        # 获取动作
                        if hasattr(method, 'compute_single_action'):
                            action = method.compute_single_action(state)
                        else:
                            action = method.get_action(state)
                        
                        state, reward, done, info = env.step(action)
                        total_reward += reward
                        step_count += 1
                        
                        # 统计碰撞
                        if 'collision' in info and info['collision']:
                            collisions += 1
                    
                    # 计算episode指标
                    completion_rate = info.get('completion_rate', 0.0)
                    efficiency = info.get('efficiency_score', 0.0)
                    resource_util = info.get('resource_utilization', 0.0)
                    
                    episode_results.append({
                        'completion_rate': completion_rate,
                        'efficiency_score': efficiency,
                        'collision_rate': collisions / max(step_count, 1),
                        'resource_utilization': resource_util,
                        'total_reward': total_reward,
                        'steps': step_count
                    })
                
                convergence_time = time.time() - start_time
                
                # 计算平均指标
                avg_completion = np.mean([r['completion_rate'] for r in episode_results])
                avg_efficiency = np.mean([r['efficiency_score'] for r in episode_results])
                avg_collision = np.mean([r['collision_rate'] for r in episode_results])
                avg_resource_util = np.mean([r['resource_utilization'] for r in episode_results])
                avg_reward = np.mean([r['total_reward'] for r in episode_results])
                
                # 计算归一化完成分数
                normalized_score = avg_completion * (1 - avg_collision) * avg_efficiency
                
                # 计算零样本迁移性能（相对于基线场景的性能保持率）
                baseline_key = f"{method.__class__.__name__}_baseline"
                if baseline_key not in self.baseline_performance:
                    # 第一个场景作为基线
                    self.baseline_performance[baseline_key] = normalized_score
                    zero_shot_perf = 1.0
                else:
                    zero_shot_perf = normalized_score / self.baseline_performance[baseline_key]
                
                metrics = SolutionQualityMetrics(
                    method_name=method.__class__.__name__,
                    scenario_size=(n_uavs, n_targets),
                    completion_rate=avg_completion,
                    efficiency_score=avg_efficiency,
                    collision_rate=avg_collision,
                    resource_utilization=avg_resource_util,
                    convergence_time=convergence_time,
                    zero_shot_performance=zero_shot_perf,
                    per_agent_reward=avg_reward / n_uavs,
                    normalized_completion_score=normalized_score
                )
                
                results.append(metrics)
                self.quality_data.append(metrics)
                
            except Exception as e:
                print(f"评估失败: {e}")
                continue
        
        return results
        
    def compare_zero_shot_transfer(self, methods: Dict[str, Any]) -> pd.DataFrame:
        """对比零样本迁移能力"""
        if not self.quality_data:
            print("无质量数据，请先运行评估")
            return pd.DataFrame()
        
        # 按方法和场景规模组织数据
        transfer_data = []
        
        for method_name in methods.keys():
            method_data = [m for m in self.quality_data if m.method_name == method_name]
            
            if not method_data:
                continue
            
            # 按场景规模排序
            method_data.sort(key=lambda x: x.scenario_size[0] * x.scenario_size[1])
            
            for i, metrics in enumerate(method_data):
                scenario_complexity = metrics.scenario_size[0] * metrics.scenario_size[1]
                
                transfer_data.append({
                    'method': method_name,
                    'scenario_size': f"{metrics.scenario_size[0]}×{metrics.scenario_size[1]}",
                    'scenario_complexity': scenario_complexity,
                    'zero_shot_performance': metrics.zero_shot_performance,
                    'completion_rate': metrics.completion_rate,
                    'efficiency_score': metrics.efficiency_score,
                    'per_agent_reward': metrics.per_agent_reward,
                    'normalized_completion_score': metrics.normalized_completion_score
                })
        
        df = pd.DataFrame(transfer_data)
        
        # 保存对比结果
        df.to_csv(self.results_dir / 'zero_shot_transfer_comparison.csv', index=False)
        
        return df
        
    def analyze_scalability(self) -> Dict:
        """分析可扩展性"""
        if not self.quality_data:
            return {"error": "无质量数据"}
        
        scalability_analysis = {}
        
        # 按方法分组
        methods = {}
        for metrics in self.quality_data:
            if metrics.method_name not in methods:
                methods[metrics.method_name] = []
            methods[metrics.method_name].append(metrics)
        
        for method_name, method_data in methods.items():
            # 按场景复杂度排序
            method_data.sort(key=lambda x: x.scenario_size[0] * x.scenario_size[1])
            
            complexities = [m.scenario_size[0] * m.scenario_size[1] for m in method_data]
            performances = [m.normalized_completion_score for m in method_data]
            
            # 计算性能衰减率
            if len(performances) > 1:
                performance_decay = []
                for i in range(1, len(performances)):
                    decay = (performances[0] - performances[i]) / performances[0]
                    performance_decay.append(decay)
                
                avg_decay_rate = np.mean(performance_decay)
                max_decay_rate = max(performance_decay)
            else:
                avg_decay_rate = 0.0
                max_decay_rate = 0.0
            
            # 计算复杂度容忍度（性能下降50%时的最大复杂度）
            tolerance_complexity = 0
            for i, perf in enumerate(performances):
                if perf >= performances[0] * 0.5:  # 性能保持在50%以上
                    tolerance_complexity = complexities[i]
            
            scalability_analysis[method_name] = {
                'avg_decay_rate': avg_decay_rate,
                'max_decay_rate': max_decay_rate,
                'complexity_tolerance': tolerance_complexity,
                'max_tested_complexity': max(complexities),
                'performance_range': (min(performances), max(performances))
            }
        
        return scalability_analysis
        
    def plot_quality_comparison(self):
        """绘制质量对比图表"""
        if not self.quality_data:
            print("无质量数据可绘制")
            return
        
        # 准备数据
        df = pd.DataFrame([
            {
                'method': m.method_name,
                'scenario': f"{m.scenario_size[0]}×{m.scenario_size[1]}",
                'completion_rate': m.completion_rate,
                'efficiency_score': m.efficiency_score,
                'collision_rate': m.collision_rate,
                'per_agent_reward': m.per_agent_reward,
                'zero_shot_performance': m.zero_shot_performance,
                'scenario_complexity': m.scenario_size[0] * m.scenario_size[1]
            }
            for m in self.quality_data
        ])
        
        # 创建综合对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('方案质量综合对比分析', fontsize=16)
        
        # 1. 完成率对比
        ax1 = axes[0, 0]
        sns.barplot(data=df, x='scenario', y='completion_rate', hue='method', ax=ax1)
        ax1.set_title('任务完成率对比')
        ax1.set_ylabel('完成率')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 效率分数对比
        ax2 = axes[0, 1]
        sns.barplot(data=df, x='scenario', y='efficiency_score', hue='method', ax=ax2)
        ax2.set_title('效率分数对比')
        ax2.set_ylabel('效率分数')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 碰撞率对比
        ax3 = axes[0, 2]
        sns.barplot(data=df, x='scenario', y='collision_rate', hue='method', ax=ax3)
        ax3.set_title('碰撞率对比（越低越好）')
        ax3.set_ylabel('碰撞率')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Per-Agent奖励对比
        ax4 = axes[1, 0]
        sns.barplot(data=df, x='scenario', y='per_agent_reward', hue='method', ax=ax4)
        ax4.set_title('Per-Agent奖励对比')
        ax4.set_ylabel('Per-Agent奖励')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 零样本迁移性能
        ax5 = axes[1, 1]
        sns.lineplot(data=df, x='scenario_complexity', y='zero_shot_performance', 
                    hue='method', marker='o', ax=ax5)
        ax5.set_title('零样本迁移性能')
        ax5.set_xlabel('场景复杂度')
        ax5.set_ylabel('零样本性能保持率')
        ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='基线性能')
        
        # 6. 综合性能雷达图
        ax6 = axes[1, 2]
        ax6.remove()
        ax6 = fig.add_subplot(2, 3, 6, projection='polar')
        
        # 计算各方法的平均性能
        methods = df['method'].unique()
        metrics = ['completion_rate', 'efficiency_score', 'per_agent_reward']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            values = [
                method_data['completion_rate'].mean(),
                method_data['efficiency_score'].mean(),
                method_data['per_agent_reward'].mean() / method_data['per_agent_reward'].max()  # 归一化
            ]
            values += values[:1]
            
            ax6.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax6.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(['完成率', '效率分数', '奖励(归一化)'])
        ax6.set_ylim(0, 1)
        ax6.set_title('综合性能对比', y=1.08)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_zero_shot_heatmap(self):
        """绘制零样本迁移热力图"""
        if not self.quality_data:
            print("无质量数据可绘制")
            return
        
        # 准备热力图数据
        methods = list(set(m.method_name for m in self.quality_data))
        scenarios = list(set(f"{m.scenario_size[0]}×{m.scenario_size[1]}" for m in self.quality_data))
        
        # 创建性能矩阵
        performance_matrix = np.zeros((len(methods), len(scenarios)))
        
        for i, method in enumerate(methods):
            for j, scenario in enumerate(scenarios):
                # 查找对应的性能数据
                matching_data = [
                    m for m in self.quality_data 
                    if m.method_name == method and f"{m.scenario_size[0]}×{m.scenario_size[1]}" == scenario
                ]
                if matching_data:
                    performance_matrix[i, j] = matching_data[0].zero_shot_performance
                else:
                    performance_matrix[i, j] = np.nan
        
        # 绘制热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 零样本迁移性能热力图
        ax1 = axes[0]
        sns.heatmap(performance_matrix, 
                   xticklabels=scenarios, 
                   yticklabels=methods,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlGn',
                   center=1.0,
                   ax=ax1)
        ax1.set_title('零样本迁移性能热力图')
        ax1.set_xlabel('场景规模')
        ax1.set_ylabel('方法')
        
        # 归一化完成分数热力图
        completion_matrix = np.zeros((len(methods), len(scenarios)))
        for i, method in enumerate(methods):
            for j, scenario in enumerate(scenarios):
                matching_data = [
                    m for m in self.quality_data 
                    if m.method_name == method and f"{m.scenario_size[0]}×{m.scenario_size[1]}" == scenario
                ]
                if matching_data:
                    completion_matrix[i, j] = matching_data[0].normalized_completion_score
                else:
                    completion_matrix[i, j] = np.nan
        
        ax2 = axes[1]
        sns.heatmap(completion_matrix,
                   xticklabels=scenarios,
                   yticklabels=methods,
                   annot=True,
                   fmt='.3f',
                   cmap='Blues',
                   ax=ax2)
        ax2.set_title('归一化完成分数热力图')
        ax2.set_xlabel('场景规模')
        ax2.set_ylabel('方法')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'zero_shot_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_quality_report(self) -> str:
        """生成质量分析报告"""
        if not self.quality_data:
            return "无质量数据可生成报告"
        
        report = ["# 方案质量分析报告\n"]
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"总测试结果数: {len(self.quality_data)}\n")
        
        # 方法概览
        methods = list(set(m.method_name for m in self.quality_data))
        report.append(f"测试方法数: {len(methods)}")
        report.append(f"测试方法: {', '.join(methods)}\n")
        
        # 场景概览
        scenarios = list(set(m.scenario_size for m in self.quality_data))
        report.append(f"测试场景数: {len(scenarios)}")
        report.append(f"场景规模范围: {min(scenarios)} 到 {max(scenarios)}\n")
        
        # 性能统计
        report.append("## 整体性能统计\n")
        
        df = pd.DataFrame([
            {
                'method': m.method_name,
                'completion_rate': m.completion_rate,
                'efficiency_score': m.efficiency_score,
                'collision_rate': m.collision_rate,
                'zero_shot_performance': m.zero_shot_performance,
                'per_agent_reward': m.per_agent_reward
            }
            for m in self.quality_data
        ])
        
        for method in methods:
            method_data = df[df['method'] == method]
            report.append(f"### {method}\n")
            report.append(f"- 平均完成率: {method_data['completion_rate'].mean():.3f}")
            report.append(f"- 平均效率分数: {method_data['efficiency_score'].mean():.3f}")
            report.append(f"- 平均碰撞率: {method_data['collision_rate'].mean():.3f}")
            report.append(f"- 平均零样本性能: {method_data['zero_shot_performance'].mean():.3f}")
            report.append(f"- 平均Per-Agent奖励: {method_data['per_agent_reward'].mean():.3f}\n")
        
        # 零样本迁移分析
        report.append("## 零样本迁移能力分析\n")
        
        transfer_df = self.compare_zero_shot_transfer({method: None for method in methods})
        if not transfer_df.empty:
            # 找出零样本迁移能力最强的方法
            avg_zero_shot = transfer_df.groupby('method')['zero_shot_performance'].mean()
            best_method = avg_zero_shot.idxmax()
            best_performance = avg_zero_shot.max()
            
            report.append(f"**零样本迁移能力最强**: {best_method} (平均性能保持率: {best_performance:.3f})")
            
            # 分析性能衰减
            for method in methods:
                method_transfer = transfer_df[transfer_df['method'] == method]
                if len(method_transfer) > 1:
                    performance_trend = method_transfer['zero_shot_performance'].values
                    if len(performance_trend) > 1:
                        decay_rate = (performance_trend[0] - performance_trend[-1]) / performance_trend[0]
                        report.append(f"- {method}: 性能衰减率 {decay_rate:.1%}")
        
        # 可扩展性分析
        scalability = self.analyze_scalability()
        if 'error' not in scalability:
            report.append("\n## 可扩展性分析\n")
            
            for method, analysis in scalability.items():
                report.append(f"### {method}")
                report.append(f"- 平均性能衰减率: {analysis['avg_decay_rate']:.1%}")
                report.append(f"- 最大性能衰减率: {analysis['max_decay_rate']:.1%}")
                report.append(f"- 复杂度容忍度: {analysis['complexity_tolerance']}")
                report.append(f"- 最大测试复杂度: {analysis['max_tested_complexity']}")
                report.append(f"- 性能范围: {analysis['performance_range'][0]:.3f} - {analysis['performance_range'][1]:.3f}\n")
        
        # 推荐建议
        report.append("## 推荐建议\n")
        
        if not transfer_df.empty:
            # 基于分析结果给出建议
            best_overall = df.groupby('method').agg({
                'completion_rate': 'mean',
                'efficiency_score': 'mean',
                'collision_rate': 'mean',
                'zero_shot_performance': 'mean'
            })
            
            # 综合评分（完成率 + 效率 - 碰撞率 + 零样本性能）
            best_overall['composite_score'] = (
                best_overall['completion_rate'] + 
                best_overall['efficiency_score'] + 
                best_overall['zero_shot_performance'] - 
                best_overall['collision_rate']
            )
            
            recommended_method = best_overall['composite_score'].idxmax()
            report.append(f"**推荐方法**: {recommended_method}")
            report.append("**推荐理由**: 综合考虑完成率、效率、安全性和零样本迁移能力")
            
            # 使用场景建议
            report.append("\n**使用场景建议**:")
            for method in methods:
                method_stats = best_overall.loc[method]
                if method_stats['zero_shot_performance'] > 0.8:
                    report.append(f"- {method}: 适用于需要强零样本迁移能力的场景")
                elif method_stats['completion_rate'] > 0.9:
                    report.append(f"- {method}: 适用于对任务完成率要求极高的场景")
                elif method_stats['collision_rate'] < 0.1:
                    report.append(f"- {method}: 适用于对安全性要求极高的场景")
        
        return "\n".join(report)
        
    def export_results(self, filename: str):
        """导出分析结果"""
        # 导出原始数据
        raw_data = []
        for metrics in self.quality_data:
            raw_data.append({
                'method_name': metrics.method_name,
                'scenario_size': f"{metrics.scenario_size[0]}×{metrics.scenario_size[1]}",
                'n_uavs': metrics.scenario_size[0],
                'n_targets': metrics.scenario_size[1],
                'completion_rate': metrics.completion_rate,
                'efficiency_score': metrics.efficiency_score,
                'collision_rate': metrics.collision_rate,
                'resource_utilization': metrics.resource_utilization,
                'convergence_time': metrics.convergence_time,
                'zero_shot_performance': metrics.zero_shot_performance,
                'per_agent_reward': metrics.per_agent_reward,
                'normalized_completion_score': metrics.normalized_completion_score
            })
        
        df = pd.DataFrame(raw_data)
        df.to_csv(self.results_dir / f"{filename}_raw_data.csv", index=False)
        
        # 导出分析报告
        report = self.generate_quality_report()
        with open(self.results_dir / f"{filename}_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 导出可扩展性分析
        scalability = self.analyze_scalability()
        with open(self.results_dir / f"{filename}_scalability.json", 'w', encoding='utf-8') as f:
            json.dump(scalability, f, indent=2, ensure_ascii=False)
        
        # 导出零样本迁移对比
        methods = list(set(m.method_name for m in self.quality_data))
        transfer_df = self.compare_zero_shot_transfer({method: None for method in methods})
        if not transfer_df.empty:
            transfer_df.to_csv(self.results_dir / f"{filename}_zero_shot_transfer.csv", index=False)
        
        print(f"分析结果已导出到 {self.results_dir} 目录:")
        print(f"- 原始数据: {filename}_raw_data.csv")
        print(f"- 分析报告: {filename}_report.md")
        print(f"- 可扩展性分析: {filename}_scalability.json")
        print(f"- 零样本迁移对比: {filename}_zero_shot_transfer.csv")

# 使用示例
if __name__ == "__main__":
    # 创建分析器
    analyzer = SolutionQualityAnalyzer()
    
    # 示例：对比不同方法
    # methods = {
    #     'TransformerGNN': transformer_model,
    #     'FCN_Baseline': fcn_model,
    #     'Greedy': greedy_solver
    # }
    # 
    # scenarios = [(3, 2), (5, 3), (8, 5), (12, 8)]
    # 
    # for method_name, method in methods.items():
    #     analyzer.evaluate_solution_quality(method, scenarios)
    # 
    # # 生成对比分析
    # analyzer.plot_quality_comparison()
    # analyzer.plot_zero_shot_heatmap()
    # analyzer.export_results("quality_analysis_2024")
    
    print("方案质量分析器已初始化，请使用evaluate_solution_quality方法开始评估")
