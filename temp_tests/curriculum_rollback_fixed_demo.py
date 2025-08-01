"""
修复版课程学习回退机制演示
解决死循环问题，增加退出条件和更合理的参数设置
"""

import logging
import tempfile
import shutil
import torch
import numpy as np
from typing import Dict, Any
import sys
import os

# 添加项目根目录到路径
sys.path.append('..')

from curriculum_stages import CurriculumStages, StageConfig
from rollback_threshold_manager import RollbackThresholdManager, PerformanceMetrics
from model_state_manager import ModelStateManager

class FixedCurriculumSimulator:
    """修复版课程学习模拟器"""
    
    def __init__(self):
        self.curriculum = CurriculumStages()
        self.rollback_manager = RollbackThresholdManager(
            consecutive_evaluations_threshold=3,
            performance_drop_threshold=0.60,
            min_evaluations_before_rollback=5
        )
        
        # 创建临时目录用于检查点
        self.temp_dir = tempfile.mkdtemp(prefix="curriculum_fixed_demo_")
        self.model_manager = ModelStateManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints_per_stage=3
        )
        
        # 添加退出条件
        self.max_rollbacks_per_stage = 3  # 每个阶段最大回退次数
        self.max_total_rollbacks = 10     # 总最大回退次数
        self.stage_rollback_count = {}    # 每个阶段的回退次数
        self.total_rollback_count = 0     # 总回退次数
        
        self.logger = logging.getLogger(__name__)
    
    def simulate_stage_training(self, stage_config: StageConfig, 
                              max_episodes: int = None) -> Dict[str, Any]:
        """
        模拟单个阶段的训练过程
        
        Args:
            stage_config: 阶段配置
            max_episodes: 最大训练回合数（用于测试）
            
        Returns:
            训练结果字典
        """
        stage_id = stage_config.stage_id
        episodes = max_episodes or stage_config.max_episodes
        evaluation_freq = stage_config.evaluation_frequency
        
        self.logger.info(f"开始模拟阶段{stage_id}训练: {stage_config.stage_name}")
        
        # 模拟训练过程
        best_performance = 0.0
        episode_count = 0
        
        # 根据阶段调整性能基线（避免死循环）
        if stage_id == 0:
            # 第一阶段相对容易
            base_performance = 0.65
            performance_variance = 0.15
        elif stage_id == 1:
            # 第二阶段适中，但确保能够达到门限
            base_performance = 0.55  # 提高基础性能
            performance_variance = 0.20
        else:
            # 后续阶段更具挑战性
            base_performance = 0.45
            performance_variance = 0.25
        
        for episode in range(0, episodes, evaluation_freq):
            episode_count = episode + evaluation_freq
            
            # 模拟性能提升（随着训练进行性能逐渐提升）
            progress_factor = min(episode / (episodes * 0.7), 1.0)  # 70%处达到最佳
            
            # 生成模拟性能数据
            metrics = PerformanceMetrics()
            
            # 添加进步趋势
            current_base = base_performance + progress_factor * 0.25
            metrics.normalized_completion_score = max(0.1, 
                current_base + np.random.normal(0, performance_variance))
            metrics.per_agent_reward = np.random.uniform(0.5, 15.0)
            metrics.efficiency_metric = np.random.uniform(-0.2, 0.8)
            metrics.episode_count = episode_count
            
            # 更新性能记录
            self.rollback_manager.update_performance(stage_id, metrics)
            
            # 保存检查点（模拟）
            if metrics.normalized_completion_score > best_performance:
                best_performance = metrics.normalized_completion_score
                
                # 创建模拟的模型状态
                model_state = {"stage": stage_id, "episode": episode_count}
                optimizer_state = {"param_groups": [{"lr": stage_config.learning_rate}]}
                performance_dict = {"normalized_completion_score": metrics.normalized_completion_score}
                
                self.model_manager.save_checkpoint(
                    stage_id=stage_id,
                    model_state=model_state,
                    optimizer_state=optimizer_state,
                    performance_metrics=performance_dict,
                    episode_count=episode_count,
                    is_best=True
                )
        
        return {
            "stage_id": stage_id,
            "final_performance": best_performance,
            "episode_count": episode_count,
            "success": best_performance >= stage_config.success_threshold
        }
    
    def should_attempt_rollback(self, current_stage_id: int) -> tuple[bool, str]:
        """
        判断是否应该尝试回退（包含退出条件检查）
        
        Args:
            current_stage_id: 当前阶段ID
            
        Returns:
            (是否回退, 原因)
        """
        # 检查总回退次数限制
        if self.total_rollback_count >= self.max_total_rollbacks:
            return False, f"已达到最大总回退次数限制({self.max_total_rollbacks})"
        
        # 检查当前阶段回退次数限制
        stage_rollbacks = self.stage_rollback_count.get(current_stage_id, 0)
        if stage_rollbacks >= self.max_rollbacks_per_stage:
            return False, f"阶段{current_stage_id}已达到最大回退次数限制({self.max_rollbacks_per_stage})"
        
        # 调用原始回退判断逻辑
        should_rollback, reason = self.rollback_manager.should_rollback(current_stage_id)
        
        return should_rollback, reason
    
    def execute_rollback(self, current_stage_id: int) -> bool:
        """
        执行回退操作
        
        Args:
            current_stage_id: 当前阶段ID
            
        Returns:
            回退是否成功
        """
        if current_stage_id == 0:
            self.logger.warning("第一阶段无法回退")
            return False
        
        target_stage_id = current_stage_id - 1
        
        # 计算学习率调整因子
        stage_rollbacks = self.stage_rollback_count.get(current_stage_id, 0)
        lr_adjustment = self.rollback_manager.get_learning_rate_adjustment(stage_rollbacks + 1)
        
        # 执行模型状态回退
        rollback_data = self.model_manager.rollback_to_previous_stage(
            current_stage_id=current_stage_id,
            target_stage_id=target_stage_id,
            learning_rate_adjustment=lr_adjustment
        )
        
        if rollback_data is None:
            self.logger.error(f"回退失败：无法加载阶段{target_stage_id}的检查点")
            return False
        
        # 记录回退事件
        reason = f"连续性能不达标，第{stage_rollbacks + 1}次回退"
        self.rollback_manager.record_rollback(
            from_stage=current_stage_id,
            to_stage=target_stage_id,
            reason=reason,
            learning_rate_adjustment=lr_adjustment
        )
        
        # 更新回退计数
        self.stage_rollback_count[current_stage_id] = stage_rollbacks + 1
        self.total_rollback_count += 1
        
        # 回退课程阶段
        self.curriculum.fallback_to_previous_stage()
        
        self.logger.info(f"执行回退: 阶段{current_stage_id} -> 阶段{target_stage_id}, "
                        f"学习率调整: {lr_adjustment:.3f}")
        
        return True
    
    def run_curriculum_training_demo(self):
        """运行完整的课程学习训练演示"""
        self.logger.info("开始课程学习训练演示")
        
        max_iterations = 20  # 防止无限循环的最大迭代次数
        iteration_count = 0
        
        try:
            while not self.curriculum.is_final_stage() and iteration_count < max_iterations:
                iteration_count += 1
                current_stage = self.curriculum.get_current_stage()
                
                self.logger.info(f"\n=== 迭代 {iteration_count}: 当前阶段 {current_stage.stage_id} ===")
                self.logger.info(f"阶段名称: {current_stage.stage_name}")
                self.logger.info(f"UAV数量范围: {current_stage.n_uavs_range}")
                self.logger.info(f"目标数量范围: {current_stage.n_targets_range}")
                
                # 模拟训练（使用较少的回合数进行演示）
                training_result = self.simulate_stage_training(
                    current_stage, 
                    max_episodes=1000  # 演示用较少回合数
                )
                
                self.logger.info(f"阶段{current_stage.stage_id}训练完成，"
                               f"最终性能: {training_result['final_performance']:.4f}")
                
                # 判断是否成功
                if training_result['success']:
                    self.logger.info(f"阶段{current_stage.stage_id}训练成功！")
                    
                    # 确定阶段最终性能
                    self.rollback_manager.finalize_stage_performance(current_stage.stage_id)
                    
                    # 推进到下一阶段
                    if self.curriculum.advance_to_next_stage():
                        next_stage = self.curriculum.get_current_stage()
                        self.logger.info(f"推进到下一阶段: {next_stage.stage_name}")
                    else:
                        self.logger.info("已完成所有训练阶段！")
                        break
                else:
                    # 检查是否需要回退
                    should_rollback, reason = self.should_attempt_rollback(current_stage.stage_id)
                    
                    if should_rollback:
                        self.logger.warning(f"触发回退机制: {reason}")
                        
                        if self.execute_rollback(current_stage.stage_id):
                            self.logger.info("回退成功，重新训练上一阶段")
                        else:
                            self.logger.error("回退失败，终止训练")
                            break
                    else:
                        self.logger.warning(f"无法回退: {reason}")
                        self.logger.info("尝试继续当前阶段训练或终止")
                        
                        # 如果无法回退，可以选择继续训练或终止
                        if iteration_count >= max_iterations - 2:
                            self.logger.info("达到最大迭代次数，终止训练")
                            break
            
            # 输出最终统计信息
            self.print_final_statistics()
            
        except KeyboardInterrupt:
            self.logger.info("用户中断训练")
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {str(e)}")
        finally:
            # 清理临时目录
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"临时目录已清理: {self.temp_dir}")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        self.logger.info("\n" + "="*50)
        self.logger.info("课程学习训练统计")
        self.logger.info("="*50)
        
        # 课程进度
        progress_info = self.curriculum.get_stage_progress_info()
        self.logger.info(f"当前阶段: {progress_info['current_stage_name']}")
        self.logger.info(f"完成进度: {progress_info['progress_percentage']:.1f}%")
        
        # 回退统计
        self.logger.info(f"总回退次数: {self.total_rollback_count}")
        for stage_id, count in self.stage_rollback_count.items():
            self.logger.info(f"  阶段{stage_id}回退次数: {count}")
        
        # 监控状态
        monitoring_status = self.rollback_manager.get_monitoring_status()
        self.logger.info(f"当前监控阶段: {monitoring_status['current_stage_id']}")
        self.logger.info(f"评估次数: {monitoring_status['evaluation_count']}")
        
        # 阶段摘要
        for stage_id in range(len(self.curriculum.stages)):
            summary = self.model_manager.get_stage_summary(stage_id)
            if summary['total_checkpoints'] > 0:
                self.logger.info(f"阶段{stage_id}: 检查点{summary['total_checkpoints']}个, "
                               f"最佳性能{summary['best_performance']:.4f}")

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('temp_tests/curriculum_fixed_demo.log', encoding='utf-8')
        ]
    )
    
    # 创建并运行模拟器
    simulator = FixedCurriculumSimulator()
    simulator.run_curriculum_training_demo()

if __name__ == "__main__":
    main()
