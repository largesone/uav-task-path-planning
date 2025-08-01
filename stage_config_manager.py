"""
课程学习阶段配置管理器
负责管理训练阶段的配置参数、最佳模型保存和阶段间的参数传递
"""

import os
import json
import pickle
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class StageConfig:
    """课程学习阶段配置"""
    stage_id: int
    n_uavs_range: Tuple[int, int]
    n_targets_range: Tuple[int, int]
    max_episodes: int
    success_threshold: float
    fallback_threshold: float
    learning_rate: float
    batch_size: int
    exploration_noise: float
    k_neighbors: int
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageConfig':
        """从字典创建配置对象"""
        return cls(**data)


class StageConfigManager:
    """
    课程学习阶段配置管理器
    负责管理各阶段的训练配置和最佳模型参数
    """
    
    def __init__(self, config_dir: str = "./curriculum_configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件保存目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 各阶段配置
        self.stage_configs: Dict[int, StageConfig] = {}
        
        # 最佳模型参数记录
        self.best_models: Dict[int, Dict[str, Any]] = {}
        
        # 阶段性能历史
        self.stage_performance_history: Dict[int, List[Dict[str, Any]]] = {}
        
        # 初始化默认配置
        self._initialize_default_configs()
        
        print(f"阶段配置管理器初始化完成，配置目录: {config_dir}")
    
    def _initialize_default_configs(self):
        """初始化默认的课程学习阶段配置"""
        default_configs = [
            StageConfig(
                stage_id=0,
                n_uavs_range=(2, 3),
                n_targets_range=(1, 2),
                max_episodes=1000,
                success_threshold=0.8,
                fallback_threshold=0.6,
                learning_rate=0.001,
                batch_size=64,
                exploration_noise=0.1,
                k_neighbors=4,
                description="基础协调阶段 - 少量UAV和目标"
            ),
            StageConfig(
                stage_id=1,
                n_uavs_range=(4, 6),
                n_targets_range=(3, 4),
                max_episodes=1500,
                success_threshold=0.75,
                fallback_threshold=0.55,
                learning_rate=0.0008,
                batch_size=128,
                exploration_noise=0.08,
                k_neighbors=6,
                description="中等复杂度阶段 - 增加实体数量"
            ),
            StageConfig(
                stage_id=2,
                n_uavs_range=(8, 12),
                n_targets_range=(5, 8),
                max_episodes=2000,
                success_threshold=0.7,
                fallback_threshold=0.5,
                learning_rate=0.0005,
                batch_size=256,
                exploration_noise=0.06,
                k_neighbors=8,
                description="高复杂度阶段 - 大规模协调"
            ),
            StageConfig(
                stage_id=3,
                n_uavs_range=(15, 20),
                n_targets_range=(10, 15),
                max_episodes=3000,
                success_threshold=0.65,
                fallback_threshold=0.45,
                learning_rate=0.0003,
                batch_size=512,
                exploration_noise=0.04,
                k_neighbors=12,
                description="极限场景阶段 - 最大规模测试"
            )
        ]
        
        for config in default_configs:
            self.stage_configs[config.stage_id] = config
        
        print(f"已初始化 {len(default_configs)} 个默认阶段配置")
    
    def get_stage_config(self, stage_id: int) -> Optional[StageConfig]:
        """
        获取指定阶段的配置
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            阶段配置对象，如果不存在则返回None
        """
        return self.stage_configs.get(stage_id)
    
    def update_stage_config(self, stage_id: int, **kwargs):
        """
        更新阶段配置参数
        
        Args:
            stage_id: 阶段ID
            **kwargs: 要更新的配置参数
        """
        if stage_id not in self.stage_configs:
            print(f"警告: 阶段 {stage_id} 不存在")
            return
        
        config = self.stage_configs[stage_id]
        
        # 更新配置参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"阶段 {stage_id} 配置更新: {key} = {value}")
            else:
                print(f"警告: 未知配置参数 {key}")
        
        # 保存更新后的配置
        self.save_stage_config(stage_id)
    
    def save_stage_config(self, stage_id: int):
        """
        保存阶段配置到文件
        
        Args:
            stage_id: 阶段ID
        """
        if stage_id not in self.stage_configs:
            print(f"警告: 阶段 {stage_id} 不存在")
            return
        
        config_file = self.config_dir / f"stage_{stage_id}_config.json"
        config_data = self.stage_configs[stage_id].to_dict()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"阶段 {stage_id} 配置已保存到: {config_file}")
    
    def load_stage_config(self, stage_id: int) -> bool:
        """
        从文件加载阶段配置
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            是否成功加载
        """
        config_file = self.config_dir / f"stage_{stage_id}_config.json"
        
        if not config_file.exists():
            print(f"配置文件不存在: {config_file}")
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.stage_configs[stage_id] = StageConfig.from_dict(config_data)
            print(f"阶段 {stage_id} 配置已从文件加载")
            return True
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return False
    
    def save_best_model(self, stage_id: int, model_state: Dict[str, Any], 
                       performance_metrics: Dict[str, float], 
                       training_config: Dict[str, Any]):
        """
        保存阶段最佳模型
        
        Args:
            stage_id: 阶段ID
            model_state: 模型状态字典
            performance_metrics: 性能指标
            training_config: 训练配置
        """
        best_model_info = {
            "stage_id": stage_id,
            "model_state": model_state,
            "performance_metrics": performance_metrics,
            "training_config": training_config,
            "timestamp": datetime.now().isoformat(),
            "model_size": self._calculate_model_size(model_state)
        }
        
        # 保存到内存
        self.best_models[stage_id] = best_model_info
        
        # 保存到文件
        model_file = self.config_dir / f"best_model_stage_{stage_id}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(best_model_info, f)
        
        print(f"阶段 {stage_id} 最佳模型已保存，性能: {performance_metrics}")
    
    def load_best_model(self, stage_id: int) -> Optional[Dict[str, Any]]:
        """
        加载阶段最佳模型
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            模型信息字典，如果不存在则返回None
        """
        # 先尝试从内存加载
        if stage_id in self.best_models:
            return self.best_models[stage_id]
        
        # 从文件加载
        model_file = self.config_dir / f"best_model_stage_{stage_id}.pkl"
        
        if not model_file.exists():
            print(f"最佳模型文件不存在: {model_file}")
            return None
        
        try:
            with open(model_file, 'rb') as f:
                model_info = pickle.load(f)
            
            self.best_models[stage_id] = model_info
            print(f"阶段 {stage_id} 最佳模型已从文件加载")
            return model_info
            
        except Exception as e:
            print(f"加载最佳模型失败: {e}")
            return None
    
    def record_stage_performance(self, stage_id: int, performance_metrics: Dict[str, float],
                               episode: int, step: int):
        """
        记录阶段性能历史
        
        Args:
            stage_id: 阶段ID
            performance_metrics: 性能指标
            episode: 回合数
            step: 训练步数
        """
        if stage_id not in self.stage_performance_history:
            self.stage_performance_history[stage_id] = []
        
        performance_record = {
            "episode": episode,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **performance_metrics
        }
        
        self.stage_performance_history[stage_id].append(performance_record)
        
        # 定期保存性能历史
        if len(self.stage_performance_history[stage_id]) % 100 == 0:
            self.save_stage_performance_history(stage_id)
    
    def save_stage_performance_history(self, stage_id: int):
        """
        保存阶段性能历史
        
        Args:
            stage_id: 阶段ID
        """
        
        if stage_id not in self.stage_performance_history:
            return
        
        history_file = self.config_dir / f"stage_{stage_id}_performance_history.json"
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.stage_performance_history[stage_id], f, 
                     indent=2, ensure_ascii=False)
        
        print(f"阶段 {stage_id} 性能历史已保存到: {history_file}")
    
    def get_stage_performance_summary(self, stage_id: int) -> Dict[str, Any]:
        """
        获取阶段性能摘要统计
        
        Args:
            stage_id: 阶段ID
            
        Returns:
            性能摘要字典
        """
        if stage_id not in self.stage_performance_history:
            return {}
        
        history = self.stage_performance_history[stage_id]
        if not history:
            return {}
        
        # 提取性能指标
        metrics = ['per_agent_reward', 'normalized_completion_score', 'efficiency_metric']
        summary = {
            "total_records": len(history),
            "first_episode": history[0]["episode"],
            "last_episode": history[-1]["episode"],
            "training_duration": history[-1]["step"] - history[0]["step"]
        }
        
        # 计算各指标的统计信息
        for metric in metrics:
            values = [record.get(metric, 0) for record in history if metric in record]
            if values:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
                summary[f"{metric}_min"] = np.min(values)
                summary[f"{metric}_max"] = np.max(values)
                summary[f"{metric}_final"] = values[-1]
        
        return summary
    
    def should_advance_stage(self, stage_id: int, current_performance: Dict[str, float],
                           consecutive_success_count: int = 3) -> bool:
        """
        判断是否应该进入下一阶段
        
        Args:
            stage_id: 当前阶段ID
            current_performance: 当前性能指标
            consecutive_success_count: 连续成功次数要求
            
        Returns:
            是否应该进入下一阶段
        """
        config = self.get_stage_config(stage_id)
        if not config:
            return False
        
        # 检查是否达到成功阈值
        completion_score = current_performance.get('normalized_completion_score', 0)
        
        if completion_score >= config.success_threshold:
            # 检查连续成功次数
            if stage_id in self.stage_performance_history:
                recent_records = self.stage_performance_history[stage_id][-consecutive_success_count:]
                
                if len(recent_records) >= consecutive_success_count:
                    success_count = sum(1 for record in recent_records 
                                      if record.get('normalized_completion_score', 0) >= config.success_threshold)
                    
                    return success_count >= consecutive_success_count
            
            return False
        
        return False
    
    def should_fallback_stage(self, stage_id: int, current_performance: Dict[str, float],
                            consecutive_failure_count: int = 3) -> bool:
        """
        判断是否应该回退到上一阶段
        
        Args:
            stage_id: 当前阶段ID
            current_performance: 当前性能指标
            consecutive_failure_count: 连续失败次数要求
            
        Returns:
            是否应该回退
        """
        if stage_id == 0:  # 第一阶段不能回退
            return False
        
        config = self.get_stage_config(stage_id)
        if not config:
            return False
        
        # 检查是否低于回退阈值
        completion_score = current_performance.get('normalized_completion_score', 0)
        
        if completion_score < config.fallback_threshold:
            # 检查连续失败次数
            if stage_id in self.stage_performance_history:
                recent_records = self.stage_performance_history[stage_id][-consecutive_failure_count:]
                
                if len(recent_records) >= consecutive_failure_count:
                    failure_count = sum(1 for record in recent_records 
                                      if record.get('normalized_completion_score', 0) < config.fallback_threshold)
                    
                    return failure_count >= consecutive_failure_count
        
        return False
    
    def get_adaptive_learning_rate(self, stage_id: int, performance_trend: List[float]) -> float:
        """
        根据性能趋势自适应调整学习率
        
        Args:
            stage_id: 阶段ID
            performance_trend: 最近的性能趋势列表
            
        Returns:
            调整后的学习率
        """
        config = self.get_stage_config(stage_id)
        if not config or len(performance_trend) < 2:
            return config.learning_rate if config else 0.001
        
        base_lr = config.learning_rate
        
        # 计算性能趋势
        recent_trend = np.mean(np.diff(performance_trend[-5:]))  # 最近5个点的趋势
        
        # 根据趋势调整学习率
        if recent_trend > 0.01:  # 性能持续提升
            adjusted_lr = base_lr * 1.1  # 略微增加学习率
        elif recent_trend < -0.01:  # 性能持续下降
            adjusted_lr = base_lr * 0.8  # 降低学习率
        else:  # 性能稳定
            adjusted_lr = base_lr
        
        # 限制学习率范围
        adjusted_lr = max(base_lr * 0.1, min(adjusted_lr, base_lr * 2.0))
        
        return adjusted_lr
    
    def get_adaptive_k_neighbors(self, stage_id: int, n_uavs: int, n_targets: int) -> int:
        """
        根据场景规模自适应调整k-近邻数量
        
        Args:
            stage_id: 阶段ID
            n_uavs: 无人机数量
            n_targets: 目标数量
            
        Returns:
            调整后的k值
        """
        config = self.get_stage_config(stage_id)
        base_k = config.k_neighbors if config else 8
        
        # 根据场景规模动态调整k值
        scale_factor = n_uavs * n_targets
        
        if scale_factor <= 6:  # 小规模场景
            k = max(2, min(base_k, n_targets))
        elif scale_factor <= 24:  # 中等规模场景
            k = max(4, min(base_k, n_targets // 2 + 2))
        else:  # 大规模场景
            k = max(6, min(base_k, n_targets // 3 + 4))
        
        # 确保k不超过目标数量
        k = min(k, n_targets)
        
        return k
    
    def export_all_configs(self) -> str:
        """
        导出所有阶段配置到单个文件
        
        Returns:
            导出文件路径
        """
        export_file = self.config_dir / "all_stage_configs.json"
        
        all_configs = {
            "configs": {stage_id: config.to_dict() 
                       for stage_id, config in self.stage_configs.items()},
            "export_timestamp": datetime.now().isoformat(),
            "total_stages": len(self.stage_configs)
        }
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(all_configs, f, indent=2, ensure_ascii=False)
        
        print(f"所有阶段配置已导出到: {export_file}")
        return str(export_file)
    
    def import_configs(self, config_file: str) -> bool:
        """
        从文件导入阶段配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            是否成功导入
        """
        if not os.path.exists(config_file):
            print(f"配置文件不存在: {config_file}")
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "configs" not in data:
                print("配置文件格式错误: 缺少configs字段")
                return False
            
            # 导入配置
            imported_count = 0
            for stage_id_str, config_data in data["configs"].items():
                stage_id = int(stage_id_str)
                self.stage_configs[stage_id] = StageConfig.from_dict(config_data)
                imported_count += 1
            
            print(f"成功导入 {imported_count} 个阶段配置")
            return True
            
        except Exception as e:
            print(f"导入配置文件失败: {e}")
            return False
    
    def _calculate_model_size(self, model_state: Dict[str, Any]) -> int:
        """
        计算模型大小（参数数量）
        
        Args:
            model_state: 模型状态字典
            
        Returns:
            参数总数
        """
        total_params = 0
        
        for param_name, param_tensor in model_state.items():
            if isinstance(param_tensor, torch.Tensor):
                total_params += param_tensor.numel()
        
        return total_params
    
    def get_stage_transition_recommendation(self, current_stage: int, 
                                          recent_performance: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        获取阶段切换建议
        
        Args:
            current_stage: 当前阶段
            recent_performance: 最近的性能记录列表
            
        Returns:
            切换建议字典
        """
        if not recent_performance:
            return {"action": "continue", "reason": "insufficient_data"}
        
        current_perf = recent_performance[-1]
        
        # 检查是否应该前进
        if self.should_advance_stage(current_stage, current_perf):
            next_stage_config = self.get_stage_config(current_stage + 1)
            if next_stage_config:
                return {
                    "action": "advance",
                    "target_stage": current_stage + 1,
                    "reason": "performance_threshold_met",
                    "confidence": 0.9
                }
        
        # 检查是否应该回退
        if self.should_fallback_stage(current_stage, current_perf):
            return {
                "action": "fallback",
                "target_stage": current_stage - 1,
                "reason": "performance_below_threshold",
                "confidence": 0.8
            }
        
        # 检查是否需要调整参数
        performance_trend = [p.get('normalized_completion_score', 0) for p in recent_performance[-10:]]
        if len(performance_trend) >= 5:
            trend_slope = np.polyfit(range(len(performance_trend)), performance_trend, 1)[0]
            
            if abs(trend_slope) < 0.001:  # 性能停滞
                return {
                    "action": "adjust_params",
                    "target_stage": current_stage,
                    "reason": "performance_plateau",
                    "suggestions": {
                        "learning_rate": self.get_adaptive_learning_rate(current_stage, performance_trend),
                        "exploration_noise": self.get_stage_config(current_stage).exploration_noise * 1.2
                    },
                    "confidence": 0.7
                }
        
        return {"action": "continue", "reason": "normal_progress"}
    
    def cleanup_old_files(self, keep_days: int = 30):
        """
        清理旧的配置和模型文件
        
        Args:
            keep_days: 保留天数
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (keep_days * 24 * 3600)
        
        cleaned_count = 0
        
        for file_path in self.config_dir.glob("*"):
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                
                if file_mtime < cutoff_time:
                    # 保留最佳模型和当前配置
                    if not (file_path.name.startswith("best_model_") or 
                           file_path.name.endswith("_config.json")):
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            print(f"已删除旧文件: {file_path.name}")
                        except OSError as e:
                            print(f"删除文件失败: {e}")
        
        print(f"清理完成，共删除 {cleaned_count} 个旧文件")
    
    def save_all_data(self):
        """保存所有数据到文件"""
        # 保存所有阶段配置
        for stage_id in self.stage_configs:
            self.save_stage_config(stage_id)
        
        # 保存所有性能历史
        for stage_id in self.stage_performance_history:
            self.save_stage_performance_history(stage_id)
        
        # 导出配置摘要
        self.export_all_configs()
        
        print("所有配置和数据已保存")


if __name__ == "__main__":
    # 测试代码
    print("阶段配置管理器测试")
    
    # 创建配置管理器
    manager = StageConfigManager("./test_curriculum_configs")
    
    # 测试配置获取和更新
    config = manager.get_stage_config(0)
    print(f"阶段0配置: {config}")
    
    # 更新配置
    manager.update_stage_config(0, learning_rate=0.002, batch_size=128)
    
    # 测试性能记录
    for i in range(10):
        performance = {
            "per_agent_reward": 10 + np.random.normal(0, 1),
            "normalized_completion_score": 0.7 + np.random.uniform(-0.1, 0.2),
            "efficiency_metric": 0.3 + np.random.uniform(-0.05, 0.1)
        }
        manager.record_stage_performance(0, performance, i, i*100)
    
    # 测试性能摘要
    summary = manager.get_stage_performance_summary(0)
    print(f"阶段0性能摘要: {summary}")
    
    # 测试阶段切换建议
    recent_perf = [
        {"normalized_completion_score": 0.85, "per_agent_reward": 12.0},
        {"normalized_completion_score": 0.87, "per_agent_reward": 12.5},
        {"normalized_completion_score": 0.89, "per_agent_reward": 13.0}
    ]
    recommendation = manager.get_stage_transition_recommendation(0, recent_perf)
    print(f"阶段切换建议: {recommendation}")
    
    # 保存所有数据
    manager.save_all_data()
    
    print("测试完成")
