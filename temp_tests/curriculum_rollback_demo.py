"""
课程阶段与回退门限机制演示
展示完整的课程学习工作流程，包括阶段推进和智能回退
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum_stages import CurriculumStages, StageConfig
from rollback_threshold_manager import RollbackThresholdManager, PerformanceMetrics
from model_state_manager import ModelStateManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CurriculumTrainingSimulator:
    """课程学习训练模拟器"""
    
    def __init__(self):
        """初始化模拟器"""
        self.curriculum = CurriculumStages()
        self.rollback_manager = RollbackThresholdManager()
        
        # 创建临时目录用于演示
        self.temp_dir = tempfile.mkdtemp(prefix="curriculum_demo_")
        self.model_manager = ModelStateManager(checkpoint_dir=self.temp_dir)
        
        # 训练历史记录
        self.training_history = []
        self.stage_transitions = []
        
        logger.info(f"课程学习演示初始化完成，检查点目录: {self.temp_dir}")
    
    def simulate_stage_training(self, 
                               stage_config: StageConfig, 
                               episodes: int,
                               base_performance: float = 0.6,
                               performance_trend: str = "improving") -> list:
        """
        模拟单个阶段的训练过程
        
        Args:
            stage_config: 阶段配置
            episodes: 训练回合数
            base_performance: 基础性能水平
            performance_trend: 性能趋势 ("improving", "declining", "stable")
            
        Returns:
            该阶段的性能历史列表
        """
        logger.info(f"开始模拟阶段{stage_config.stage_id}训练: {stage_config.stage_name}")
        
        stage_history = []
        
        for episode in range(episodes):
            # 根据趋势生成性能数据
            if performance_trend == "improving":
                # 逐渐改善的性能
                progress = episode / episodes
                noise = np.random.normal(0, 0.05)  # 添加噪声
                ncs = base_performance + progress * 0.2 + noise
            elif performance_trend == "declining":
                # 逐渐下降的性能
                progress = episode / episodes
                noise = np.random.normal(0, 0.03)
                ncs = base_performance - progress * 0.3 + noise
            else:  # stable
                # 稳定的性能
                noise = np.random.normal(0, 0.02)
                ncs = base_performance + noise
            
            # 限制性能范围
            ncs = max(0.1, min(1.0, ncs))
            
            # 生成其他指标
            metrics = PerformanceMetrics()
            metrics.normalized_completion_score = ncs
            metrics.per_agent_reward = ncs * 15 + np.random.normal(0, 2)
            metrics.efficiency_metric = ncs * 0.8 + np.random.normal(0, 0.1)
            metrics.episode_count = episode + 1
            
            stage_history.append(metrics)
            
            # 每隔一定回合进行评估
            if (episode + 1) % stage_config.evaluation_frequency == 0:
                self.rollback_manager.update_performance(stage_config.stage_id, metrics)
                
                # 记录训练历史
                self.training_history.append({
                    "stage_id": stage_config.stage_id,
                    "episode": episode + 1,
                    "ncs": ncs,
                    "per_agent_reward": metrics.per_agent_reward,
                    "efficiency_metric": metrics.efficiency_metric
                })
                
                # 保存检查点（模拟）
                is_best = (episode + 1) % (stage_config.evaluation_frequency * 3) == 0
                if is_best:
                    self.model_manager.save_checkpoint(
                        stage_id=stage_config.stage_id,
                        model_state={"dummy_weights": f"stage_{stage_config.stage_id}_ep_{episode+1}"},
                        optimizer_state={"param_groups": [{"lr": stage_config.learning_rate}]},
                        performance_metrics={"normalized_completion_score": ncs},
                        episode_count=episode + 1,
                        is_best=is_best
                    )
        
        logger.info(f"阶段{stage_config.stage_id}训练完成，最终性能: {stage_history[-1].normalized_completion_score:.4f}")
        return stage_history
    
    def run_curriculum_training_demo(self):
        """运行完整的课程学习演示"""
        logger.info("=" * 60)
        logger.info("开始课程学习训练演示")
        logger.info("=" * 60)
        
        while not self.curriculum.is_final_stage():
            current_stage = self.curriculum.get_current_stage()
            logger.info(f"\n当前阶段: {current_stage.stage_name}")
            logger.info(f"UAV数量范围: {current_stage.n_uavs_range}")
            logger.info(f"目标数量范围: {current_stage.n_targets_range}")
            
            # 模拟训练过程
            if current_stage.stage_id == 0:
                # 第一阶段：正常改善
                stage_history = self.simulate_stage_training(
                    current_stage, 
                    episodes=2000,
                    base_performance=0.5,
                    performance_trend="improving"
                )
                
                # 确定阶段最终性能
                self.rollback_manager.finalize_stage_performance(current_stage.stage_id)
                
                # 推进到下一阶段
                self.curriculum.advance_to_next_stage()
                self.stage_transitions.append({
                    "type": "advance",
                    "from_stage": current_stage.stage_id,
                    "to_stage": current_stage.stage_id + 1,
                    "reason": "阶段训练成功完成"
                })
                
            elif current_stage.stage_id == 1:
                # 第二阶段：性能下降，触发回退
                stage_history = self.simulate_stage_training(
                    current_stage,
                    episodes=1500,
                    base_performance=0.4,  # 低性能
                    performance_trend="declining"
                )
                
                # 检查是否需要回退
                should_rollback, reason = self.rollback_manager.should_rollback(current_stage.stage_id)
                
                if should_rollback:
                    logger.warning(f"触发回退机制: {reason}")
                    
                    # 执行回退
                    rollback_data = self.model_manager.rollback_to_previous_stage(
                        current_stage_id=current_stage.stage_id,
                        target_stage_id=current_stage.stage_id - 1,
                        learning_rate_adjustment=0.5
                    )
                    
                    # 记录回退事件
                    self.rollback_manager.record_rollback(
                        from_stage=current_stage.stage_id,
                        to_stage=current_stage.stage_id - 1,
                        reason=reason,
                        learning_rate_adjustment=0.5
                    )
                    
                    # 回退课程阶段
                    self.curriculum.fallback_to_previous_stage()
                    self.stage_transitions.append({
                        "type": "rollback",
                        "from_stage": current_stage.stage_id,
                        "to_stage": current_stage.stage_id - 1,
                        "reason": reason
                    })
                    
                    # 重新训练第一阶段（使用调整后的学习率）
                    logger.info("重新训练第一阶段，使用调整后的学习率")
                    retry_stage = self.curriculum.get_current_stage()
                    retry_history = self.simulate_stage_training(
                        retry_stage,
                        episodes=1000,
                        base_performance=0.6,
                        performance_trend="improving"
                    )
                    
                    # 再次推进
                    self.rollback_manager.finalize_stage_performance(retry_stage.stage_id)
                    self.curriculum.advance_to_next_stage()
                    self.stage_transitions.append({
                        "type": "advance_after_rollback",
                        "from_stage": retry_stage.stage_id,
                        "to_stage": retry_stage.stage_id + 1,
                        "reason": "回退后重新训练成功"
                    })
                
            elif current_stage.stage_id == 2:
                # 第三阶段：成功训练
                stage_history = self.simulate_stage_training(
                    current_stage,
                    episodes=2500,
                    base_performance=0.55,
                    performance_trend="improving"
                )
                
                self.rollback_manager.finalize_stage_performance(current_stage.stage_id)
                self.curriculum.advance_to_next_stage()
                self.stage_transitions.append({
                    "type": "advance",
                    "from_stage": current_stage.stage_id,
                    "to_stage": current_stage.stage_id + 1,
                    "reason": "阶段训练成功完成"
                })
        
        # 最终阶段训练
        final_stage = self.curriculum.get_current_stage()
        logger.info(f"\n最终阶段训练: {final_stage.stage_name}")
        final_history = self.simulate_stage_training(
            final_stage,
            episodes=3000,
            base_performance=0.5,
            performance_trend="improving"
        )
        
        logger.info("=" * 60)
        logger.info("课程学习训练演示完成")
        logger.info("=" * 60)
    
    def generate_training_report(self):
        """生成训练报告"""
        logger.info("生成训练报告...")
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('课程学习训练演示报告', fontsize=16, fontweight='bold')
        
        # 提取数据
        episodes = [h["episode"] for h in self.training_history]
        stage_ids = [h["stage_id"] for h in self.training_history]
        ncs_scores = [h["ncs"] for h in self.training_history]
        rewards = [h["per_agent_reward"] for h in self.training_history]
        efficiency = [h["efficiency_metric"] for h in self.training_history]
        
        # 1. Normalized Completion Score 趋势
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green', 'orange']
        for stage_id in range(4):
            stage_episodes = [e for e, s in zip(episodes, stage_ids) if s == stage_id]
            stage_ncs = [n for n, s in zip(ncs_scores, stage_ids) if s == stage_id]
            if stage_episodes:
                ax1.plot(stage_episodes, stage_ncs, 'o-', color=colors[stage_id], 
                        label=f'阶段{stage_id}', alpha=0.7)
        
        ax1.set_xlabel('训练回合')
        ax1.set_ylabel('Normalized Completion Score')
        ax1.set_title('训练性能趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Per-Agent Reward 趋势
        ax2 = axes[0, 1]
        for stage_id in range(4):
            stage_episodes = [e for e, s in zip(episodes, stage_ids) if s == stage_id]
            stage_rewards = [r for r, s in zip(rewards, stage_ids) if s == stage_id]
            if stage_episodes:
                ax2.plot(stage_episodes, stage_rewards, 's-', color=colors[stage_id], 
                        label=f'阶段{stage_id}', alpha=0.7)
        
        ax2.set_xlabel('训练回合')
        ax2.set_ylabel('Per-Agent Reward')
        ax2.set_title('奖励趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 效率指标趋势
        ax3 = axes[1, 0]
        for stage_id in range(4):
            stage_episodes = [e for e, s in zip(episodes, stage_ids) if s == stage_id]
            stage_efficiency = [eff for eff, s in zip(efficiency, stage_ids) if s == stage_id]
            if stage_episodes:
                ax3.plot(stage_episodes, stage_efficiency, '^-', color=colors[stage_id], 
                        label=f'阶段{stage_id}', alpha=0.7)
        
        ax3.set_xlabel('训练回合')
        ax3.set_ylabel('Efficiency Metric')
        ax3.set_title('效率指标趋势')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 阶段转换时间线
        ax4 = axes[1, 1]
        transition_times = list(range(len(self.stage_transitions)))
        transition_types = [t["type"] for t in self.stage_transitions]
        
        colors_map = {
            "advance": "green",
            "rollback": "red", 
            "advance_after_rollback": "orange"
        }
        
        for i, (time, t_type) in enumerate(zip(transition_times, transition_types)):
            ax4.scatter(time, i, c=colors_map.get(t_type, "blue"), s=100, alpha=0.7)
            ax4.annotate(f'{t_type}\n({self.stage_transitions[i]["from_stage"]}→{self.stage_transitions[i]["to_stage"]})', 
                        (time, i), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('转换序号')
        ax4.set_ylabel('转换事件')
        ax4.set_title('阶段转换时间线')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        report_path = os.path.join(self.temp_dir, "curriculum_training_report.png")
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练报告图表已保存: {report_path}")
        
        # 生成文本报告
        text_report_path = os.path.join(self.temp_dir, "curriculum_training_summary.txt")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("课程学习训练演示总结报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 课程阶段配置:\n")
            for stage in self.curriculum.stages:
                f.write(f"   阶段{stage.stage_id}: {stage.stage_name}\n")
                f.write(f"   - UAV数量: {stage.n_uavs_range}\n")
                f.write(f"   - 目标数量: {stage.n_targets_range}\n")
                f.write(f"   - 学习率: {stage.learning_rate}\n\n")
            
            f.write("2. 阶段转换历史:\n")
            for i, transition in enumerate(self.stage_transitions):
                f.write(f"   转换{i+1}: {transition['type']}\n")
                f.write(f"   - 从阶段{transition['from_stage']}到阶段{transition['to_stage']}\n")
                f.write(f"   - 原因: {transition['reason']}\n\n")
            
            f.write("3. 回退机制统计:\n")
            rollback_count = len([t for t in self.stage_transitions if t["type"] == "rollback"])
            f.write(f"   - 总回退次数: {rollback_count}\n")
            f.write(f"   - 回退门限: {self.rollback_manager.performance_threshold}\n")
            f.write(f"   - 连续评估门限: {self.rollback_manager.consecutive_threshold}\n\n")
            
            f.write("4. 模型检查点统计:\n")
            for stage_id in range(4):
                summary = self.model_manager.get_stage_summary(stage_id)
                if summary["total_checkpoints"] > 0:
                    f.write(f"   阶段{stage_id}:\n")
                    f.write(f"   - 检查点数量: {summary['total_checkpoints']}\n")
                    f.write(f"   - 最佳性能: {summary['best_performance']:.4f}\n")
                    f.write(f"   - 学习率调整次数: {summary['learning_rate_adjustments']}\n\n")
        
        logger.info(f"文本报告已保存: {text_report_path}")
        
        # 显示图表
        plt.show()
    
    def cleanup(self):
        """清理临时文件"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"临时目录已清理: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")

def main():
    """主函数"""
    print("课程阶段与回退门限机制演示")
    print("=" * 50)
    
    # 创建模拟器
    simulator = CurriculumTrainingSimulator()
    
    try:
        # 运行演示
        simulator.run_curriculum_training_demo()
        
        # 生成报告
        simulator.generate_training_report()
        
        # 显示最终状态
        print("\n最终状态:")
        print(f"当前阶段: {simulator.curriculum.current_stage_id}")
        print(f"阶段转换历史: {len(simulator.stage_transitions)}次")
        print(f"训练数据点: {len(simulator.training_history)}个")
        print(f"回退事件: {len(simulator.rollback_manager.rollback_history)}次")
        
        # 显示监控状态
        monitoring_status = simulator.rollback_manager.get_monitoring_status()
        print(f"监控状态: {monitoring_status}")
        
        input("\n按回车键继续...")
        
    finally:
        # 清理资源
        simulator.cleanup()

if __name__ == "__main__":
    main()