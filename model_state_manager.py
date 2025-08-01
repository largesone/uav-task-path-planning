"""
模型状态管理器
实现学习率调整和模型状态恢复机制，支持课程学习的回退操作
"""

import os
import torch
import pickle
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import shutil

class ModelStateManager:
    """模型状态管理器"""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 max_checkpoints_per_stage: int = 5):
        """
        初始化模型状态管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints_per_stage: 每个阶段保留的最大检查点数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints_per_stage
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 状态记录
        self.stage_checkpoints: Dict[int, list] = {}  # 每个阶段的检查点列表
        self.best_checkpoints: Dict[int, str] = {}    # 每个阶段的最佳检查点
        self.learning_rate_history: Dict[int, list] = {}  # 学习率调整历史
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, 
                       stage_id: int,
                       model_state: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       performance_metrics: Dict[str, float],
                       episode_count: int,
                       is_best: bool = False) -> str:
        """
        保存模型检查点
        
        Args:
            stage_id: 训练阶段ID
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            performance_metrics: 性能指标
            episode_count: 训练回合数
            is_best: 是否为该阶段最佳模型
            
        Returns:
            检查点文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"stage_{stage_id}_ep_{episode_count}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # 准备检查点数据
        checkpoint_data = {
            "stage_id": stage_id,
            "episode_count": episode_count,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "performance_metrics": performance_metrics,
            "timestamp": timestamp,
            "is_best": is_best
        }
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        
        # 更新检查点记录
        if stage_id not in self.stage_checkpoints:
            self.stage_checkpoints[stage_id] = []
        
        self.stage_checkpoints[stage_id].append({
            "path": checkpoint_path,
            "episode_count": episode_count,
            "performance": performance_metrics.get("normalized_completion_score", 0.0),
            "timestamp": timestamp,
            "is_best": is_best
        })
        
        # 如果是最佳模型，更新最佳检查点记录
        if is_best:
            self.best_checkpoints[stage_id] = checkpoint_path
            self.logger.info(f"保存阶段{stage_id}最佳检查点: {checkpoint_path}")
        
        # 清理旧检查点（保留最近的几个）
        self._cleanup_old_checkpoints(stage_id)
        
        self.logger.info(f"保存检查点: {checkpoint_path}, 性能: {performance_metrics}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点数据字典，如果加载失败返回None
        """
        try:
            if not os.path.exists(checkpoint_path):
                self.logger.error(f"检查点文件不存在: {checkpoint_path}")
                return None
            
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {checkpoint_path}, 错误: {str(e)}")
            return None
    
    def get_best_checkpoint_path(self, stage_id: int) -> Optional[str]:
        """
        获取指定阶段的最佳检查点路径
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            最佳检查点路径，如果不存在返回None
        """
        return self.best_checkpoints.get(stage_id)
    
    def get_latest_checkpoint_path(self, stage_id: int) -> Optional[str]:
        """
        获取指定阶段的最新检查点路径
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            最新检查点路径，如果不存在返回None
        """
        if stage_id not in self.stage_checkpoints or not self.stage_checkpoints[stage_id]:
            return None
        
        # 按时间戳排序，返回最新的
        latest_checkpoint = max(self.stage_checkpoints[stage_id], 
                              key=lambda x: x["timestamp"])
        return latest_checkpoint["path"]
    
    def rollback_to_previous_stage(self, 
                                  current_stage_id: int,
                                  target_stage_id: int,
                                  learning_rate_adjustment: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        回退到上一阶段的最佳模型状态
        
        Args:
            current_stage_id: 当前阶段ID
            target_stage_id: 目标阶段ID
            learning_rate_adjustment: 学习率调整因子
            
        Returns:
            回退后的模型状态，如果失败返回None
        """
        # 获取目标阶段的最佳检查点
        best_checkpoint_path = self.get_best_checkpoint_path(target_stage_id)
        
        if not best_checkpoint_path:
            self.logger.error(f"阶段{target_stage_id}没有可用的最佳检查点")
            return None
        
        # 加载检查点
        checkpoint_data = self.load_checkpoint(best_checkpoint_path)
        if not checkpoint_data:
            return None
        
        # 调整学习率
        if "optimizer_state_dict" in checkpoint_data:
            self._adjust_learning_rate(checkpoint_data["optimizer_state_dict"], 
                                     learning_rate_adjustment)
        
        # 记录学习率调整历史
        if current_stage_id not in self.learning_rate_history:
            self.learning_rate_history[current_stage_id] = []
        
        self.learning_rate_history[current_stage_id].append({
            "timestamp": datetime.now().isoformat(),
            "adjustment_factor": learning_rate_adjustment,
            "rollback_from": current_stage_id,
            "rollback_to": target_stage_id
        })
        
        self.logger.info(f"成功回退到阶段{target_stage_id}，学习率调整因子: {learning_rate_adjustment}")
        
        return checkpoint_data
    
    def _adjust_learning_rate(self, 
                             optimizer_state: Dict[str, Any], 
                             adjustment_factor: float) -> None:
        """
        调整优化器状态中的学习率
        
        Args:
            optimizer_state: 优化器状态字典
            adjustment_factor: 调整因子
        """
        if "param_groups" in optimizer_state:
            for param_group in optimizer_state["param_groups"]:
                if "lr" in param_group:
                    old_lr = param_group["lr"]
                    new_lr = old_lr * adjustment_factor
                    param_group["lr"] = new_lr
                    self.logger.info(f"学习率调整: {old_lr:.6f} -> {new_lr:.6f}")
    
    def _cleanup_old_checkpoints(self, stage_id: int) -> None:
        """
        清理旧的检查点文件，保留最近的几个
        
        Args:
            stage_id: 阶段ID
        """
        if stage_id not in self.stage_checkpoints:
            return
        
        checkpoints = self.stage_checkpoints[stage_id]
        
        # 按时间戳排序
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # 保留最近的检查点和最佳检查点
        to_keep = []
        to_remove = []
        
        best_checkpoint_path = self.best_checkpoints.get(stage_id)
        
        for i, checkpoint in enumerate(checkpoints):
            # 保留最近的几个检查点或最佳检查点
            if (i < self.max_checkpoints or 
                checkpoint["is_best"] or 
                checkpoint["path"] == best_checkpoint_path):
                to_keep.append(checkpoint)
            else:
                to_remove.append(checkpoint)
        
        # 删除多余的检查点文件
        for checkpoint in to_remove:
            try:
                if os.path.exists(checkpoint["path"]):
                    os.remove(checkpoint["path"])
                    self.logger.debug(f"删除旧检查点: {checkpoint['path']}")
            except Exception as e:
                self.logger.warning(f"删除检查点失败: {checkpoint['path']}, 错误: {str(e)}")
        
        # 更新检查点列表
        self.stage_checkpoints[stage_id] = to_keep
    
    def get_stage_summary(self, stage_id: int) -> Dict[str, Any]:
        """
        获取指定阶段的摘要信息
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            阶段摘要信息
        """
        checkpoints = self.stage_checkpoints.get(stage_id, [])
        best_checkpoint = self.best_checkpoints.get(stage_id)
        lr_history = self.learning_rate_history.get(stage_id, [])
        
        summary = {
            "stage_id": stage_id,
            "total_checkpoints": len(checkpoints),
            "best_checkpoint_path": best_checkpoint,
            "learning_rate_adjustments": len(lr_history),
            "latest_checkpoint": None,
            "best_performance": 0.0
        }
        
        if checkpoints:
            # 最新检查点
            latest = max(checkpoints, key=lambda x: x["timestamp"])
            summary["latest_checkpoint"] = latest["path"]
            
            # 最佳性能
            best_perf = max(checkpoints, key=lambda x: x["performance"])
            summary["best_performance"] = best_perf["performance"]
        
        return summary
    
    def export_training_history(self, output_path: str) -> None:
        """
        导出训练历史数据
        
        Args:
            output_path: 输出文件路径
        """
        history_data = {
            "stage_checkpoints": self.stage_checkpoints,
            "best_checkpoints": self.best_checkpoints,
            "learning_rate_history": self.learning_rate_history,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(history_data, f)
        
        self.logger.info(f"训练历史已导出到: {output_path}")
