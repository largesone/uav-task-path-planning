"""
回退门限机制管理器
监控训练性能，实现智能回退和学习率调整
"""

from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import logging
import json
from datetime import datetime

class PerformanceMetrics:
    """性能指标数据结构"""
    
    def __init__(self):
        self.normalized_completion_score: float = 0.0
        self.per_agent_reward: float = 0.0
        self.efficiency_metric: float = 0.0
        self.episode_count: int = 0
        self.timestamp: str = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "normalized_completion_score": self.normalized_completion_score,
            "per_agent_reward": self.per_agent_reward,
            "efficiency_metric": self.efficiency_metric,
            "episode_count": self.episode_count,
            "timestamp": self.timestamp
        }

class RollbackThresholdManager:
    """回退门限机制管理器"""
    
    def __init__(self, 
                 consecutive_evaluations_threshold: int = 3,
                 performance_drop_threshold: float = 0.60,
                 min_evaluations_before_rollback: int = 5):
        """
        初始化回退门限管理器
        
        Args:
            consecutive_evaluations_threshold: 连续评估周期数门限
            performance_drop_threshold: 性能下降门限（相对于上阶段最终性能）
            min_evaluations_before_rollback: 回退前的最少评估次数
        """
        self.consecutive_threshold = consecutive_evaluations_threshold
        self.performance_threshold = performance_drop_threshold
        self.min_evaluations = min_evaluations_before_rollback
        
        # 性能历史记录
        self.stage_performance_history: Dict[int, List[PerformanceMetrics]] = {}
        self.stage_final_performance: Dict[int, PerformanceMetrics] = {}
        
        # 当前阶段监控状态
        self.current_stage_id: int = 0
        self.evaluation_count: int = 0
        
        # 回退历史记录
        self.rollback_history: List[Dict] = []
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        
    def update_performance(self, 
                          stage_id: int, 
                          metrics: PerformanceMetrics) -> None:
        """
        更新性能指标
        
        Args:
            stage_id: 当前训练阶段ID
            metrics: 性能指标数据
        """
        # 如果切换到新阶段，重置监控状态
        if stage_id != self.current_stage_id:
            self._reset_stage_monitoring(stage_id)
        
        # 记录性能历史
        if stage_id not in self.stage_performance_history:
            self.stage_performance_history[stage_id] = []
        
        self.stage_performance_history[stage_id].append(metrics)
        self.evaluation_count += 1
        
        self.logger.info(f"阶段{stage_id}性能更新: NCS={metrics.normalized_completion_score:.4f}, "
                        f"PAR={metrics.per_agent_reward:.4f}, EM={metrics.efficiency_metric:.4f}")
    
    def should_rollback(self, current_stage_id: int) -> Tuple[bool, str]:
        """
        判断是否应该回退到上一阶段
        
        Args:
            current_stage_id: 当前阶段ID
            
        Returns:
            (是否回退, 回退原因)
        """
        # 第一阶段不能回退
        if current_stage_id == 0:
            return False, "第一阶段无法回退"
        
        # 评估次数不足
        if self.evaluation_count < self.min_evaluations:
            return False, f"评估次数不足，需要至少{self.min_evaluations}次评估"
        
        # 获取上一阶段的最终性能
        previous_stage_final = self.stage_final_performance.get(current_stage_id - 1)
        if not previous_stage_final:
            return False, "上一阶段最终性能数据缺失"
        
        # 获取当前阶段最近的连续评估结果
        current_stage_history = self.stage_performance_history.get(current_stage_id, [])
        if len(current_stage_history) < self.consecutive_threshold:
            return False, f"当前阶段评估数据不足，需要至少{self.consecutive_threshold}次连续评估"
        
        # 检查最近的连续评估是否都低于门限
        recent_evaluations = current_stage_history[-self.consecutive_threshold:]
        previous_ncs = previous_stage_final.normalized_completion_score
        
        consecutive_poor_count = 0
        for metrics in recent_evaluations:
            current_ncs = metrics.normalized_completion_score
            performance_ratio = current_ncs / previous_ncs if previous_ncs > 0 else 0
            
            if performance_ratio < self.performance_threshold:
                consecutive_poor_count += 1
            else:
                consecutive_poor_count = 0  # 重置计数器
        
        # 判断是否达到连续评估门限
        if consecutive_poor_count >= self.consecutive_threshold:
            avg_ratio = np.mean([m.normalized_completion_score / previous_ncs 
                               for m in recent_evaluations])
            reason = (f"连续{consecutive_poor_count}次评估性能低于上阶段最终性能的"
                     f"{self.performance_threshold*100:.0f}% (平均比例: {avg_ratio:.3f})")
            return True, reason
        
        return False, f"连续低性能评估次数: {consecutive_poor_count}/{self.consecutive_threshold}"
    
    def record_rollback(self, 
                       from_stage: int, 
                       to_stage: int, 
                       reason: str,
                       learning_rate_adjustment: float = 0.5) -> Dict:
        """
        记录回退事件
        
        Args:
            from_stage: 回退前阶段
            to_stage: 回退后阶段
            reason: 回退原因
            learning_rate_adjustment: 学习率调整因子
            
        Returns:
            回退记录字典
        """
        rollback_record = {
            "timestamp": datetime.now().isoformat(),
            "from_stage": from_stage,
            "to_stage": to_stage,
            "reason": reason,
            "learning_rate_adjustment": learning_rate_adjustment,
            "evaluation_count": self.evaluation_count
        }
        
        self.rollback_history.append(rollback_record)
        
        # 重置监控状态
        self._reset_stage_monitoring(to_stage)
        
        self.logger.info(f"记录回退事件: 从阶段{from_stage}回退到阶段{to_stage}, 原因: {reason}")
        
        return rollback_record
    
    def finalize_stage_performance(self, stage_id: int) -> None:
        """
        确定阶段最终性能（当推进到下一阶段时调用）
        
        Args:
            stage_id: 完成的阶段ID
        """
        if stage_id in self.stage_performance_history:
            # 取最近几次评估的平均值作为最终性能
            recent_metrics = self.stage_performance_history[stage_id][-5:]  # 最近5次评估
            if recent_metrics:
                final_metrics = PerformanceMetrics()
                final_metrics.normalized_completion_score = np.mean([m.normalized_completion_score for m in recent_metrics])
                final_metrics.per_agent_reward = np.mean([m.per_agent_reward for m in recent_metrics])
                final_metrics.efficiency_metric = np.mean([m.efficiency_metric for m in recent_metrics])
                final_metrics.episode_count = recent_metrics[-1].episode_count
                
                self.stage_final_performance[stage_id] = final_metrics
                
                self.logger.info(f"阶段{stage_id}最终性能确定: NCS={final_metrics.normalized_completion_score:.4f}")
    
    def get_learning_rate_adjustment(self, rollback_count: int) -> float:
        """
        根据回退次数计算学习率调整因子
        
        Args:
            rollback_count: 当前阶段的回退次数
            
        Returns:
            学习率调整因子
        """
        # 每次回退，学习率减半，但不低于原始值的0.1倍
        adjustment_factor = max(0.5 ** rollback_count, 0.1)
        return adjustment_factor
    
    def _get_recent_performance(self, stage_id: int, n_recent: int = 3) -> Optional[PerformanceMetrics]:
        """获取最近n次评估的平均性能"""
        if stage_id not in self.stage_performance_history:
            return None
        
        recent_metrics = self.stage_performance_history[stage_id][-n_recent:]
        if not recent_metrics:
            return None
        
        # 计算平均性能
        avg_metrics = PerformanceMetrics()
        avg_metrics.normalized_completion_score = np.mean([m.normalized_completion_score for m in recent_metrics])
        avg_metrics.per_agent_reward = np.mean([m.per_agent_reward for m in recent_metrics])
        avg_metrics.efficiency_metric = np.mean([m.efficiency_metric for m in recent_metrics])
        avg_metrics.episode_count = recent_metrics[-1].episode_count
        
        return avg_metrics
    
    def _reset_stage_monitoring(self, new_stage_id: int) -> None:
        """重置阶段监控状态"""
        self.current_stage_id = new_stage_id
        self.evaluation_count = 0
        
        self.logger.info(f"重置阶段{new_stage_id}监控状态")
    
    def get_monitoring_status(self) -> Dict:
        """获取当前监控状态"""
        return {
            "current_stage_id": self.current_stage_id,
            "evaluation_count": self.evaluation_count,
            "consecutive_threshold": self.consecutive_threshold,
            "performance_threshold": self.performance_threshold,
            "rollback_count": len(self.rollback_history),
            "last_rollback": self.rollback_history[-1] if self.rollback_history else None
        }
    
    def save_monitoring_data(self, filepath: str) -> None:
        """保存监控数据到文件"""
        data = {
            "stage_performance_history": {
                str(k): [m.to_dict() for m in v] 
                for k, v in self.stage_performance_history.items()
            },
            "stage_final_performance": {
                str(k): v.to_dict() 
                for k, v in self.stage_final_performance.items()
            },
            "rollback_history": self.rollback_history,
            "monitoring_status": self.get_monitoring_status()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"监控数据已保存到: {filepath}")
