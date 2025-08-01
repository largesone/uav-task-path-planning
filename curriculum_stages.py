"""
课程学习阶段配置模块
定义从简单到复杂的渐进式训练阶段，支持零样本迁移学习
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class StageConfig:
    """单个训练阶段的配置"""
    stage_id: int
    stage_name: str
    n_uavs_range: Tuple[int, int]  # (最小UAV数量, 最大UAV数量)
    n_targets_range: Tuple[int, int]  # (最小目标数量, 最大目标数量)
    max_episodes: int  # 该阶段最大训练回合数
    success_threshold: float  # 进入下一阶段的成功门限
    fallback_threshold: float  # 回退门限（相对于上阶段最终性能）
    learning_rate: float  # 该阶段的学习率
    evaluation_frequency: int  # 评估频率（每多少回合评估一次）
    min_episodes_before_advance: int  # 进入下一阶段前的最少训练回合数
    
    def get_random_scenario_size(self) -> Tuple[int, int]:
        """随机生成该阶段的场景规模"""
        n_uavs = np.random.randint(self.n_uavs_range[0], self.n_uavs_range[1] + 1)
        n_targets = np.random.randint(self.n_targets_range[0], self.n_targets_range[1] + 1)
        return n_uavs, n_targets

class CurriculumStages:
    """课程学习阶段管理器"""
    
    def __init__(self):
        """初始化课程阶段配置"""
        self.stages = self._define_curriculum_stages()
        self.current_stage_id = 0
        self.stage_history = []  # 记录阶段切换历史
        
    def _define_curriculum_stages(self) -> List[StageConfig]:
        """定义课程学习的各个阶段"""
        stages = [
            # 阶段1：基础协调 - 少量实体，学习基本协调机制
            StageConfig(
                stage_id=0,
                stage_name="基础协调阶段",
                n_uavs_range=(2, 3),
                n_targets_range=(1, 2),
                max_episodes=5000,
                success_threshold=0.75,  # Normalized Completion Score >= 0.75
                fallback_threshold=0.60,  # 低于上阶段最终性能的60%
                learning_rate=3e-4,
                evaluation_frequency=200,
                min_episodes_before_advance=1000
            ),
            
            # 阶段2：中等复杂度 - 增加实体数量，学习更复杂的协调
            StageConfig(
                stage_id=1,
                stage_name="中等复杂度阶段",
                n_uavs_range=(4, 6),
                n_targets_range=(3, 4),
                max_episodes=8000,
                success_threshold=0.70,
                fallback_threshold=0.60,
                learning_rate=2e-4,
                evaluation_frequency=300,
                min_episodes_before_advance=1500
            ),
            
            # 阶段3：高复杂度 - 大规模协调，测试局部注意力机制
            StageConfig(
                stage_id=2,
                stage_name="高复杂度阶段",
                n_uavs_range=(8, 12),
                n_targets_range=(5, 8),
                max_episodes=12000,
                success_threshold=0.65,
                fallback_threshold=0.60,
                learning_rate=1e-4,
                evaluation_frequency=400,
                min_episodes_before_advance=2000
            ),
            
            # 阶段4：极限场景 - 最大规模，验证零样本迁移能力
            StageConfig(
                stage_id=3,
                stage_name="极限场景阶段",
                n_uavs_range=(15, 20),
                n_targets_range=(10, 15),
                max_episodes=15000,
                success_threshold=0.60,
                fallback_threshold=0.60,
                learning_rate=5e-5,
                evaluation_frequency=500,
                min_episodes_before_advance=3000
            )
        ]
        return stages
    
    def get_current_stage(self) -> StageConfig:
        """获取当前训练阶段配置"""
        return self.stages[self.current_stage_id]
    
    def get_stage_by_id(self, stage_id: int) -> Optional[StageConfig]:
        """根据ID获取阶段配置"""
        if 0 <= stage_id < len(self.stages):
            return self.stages[stage_id]
        return None
    
    def advance_to_next_stage(self) -> bool:
        """推进到下一阶段"""
        if self.current_stage_id < len(self.stages) - 1:
            self.current_stage_id += 1
            self.stage_history.append(f"推进到阶段{self.current_stage_id}")
            return True
        return False
    
    def fallback_to_previous_stage(self) -> bool:
        """回退到上一阶段"""
        if self.current_stage_id > 0:
            self.current_stage_id -= 1
            self.stage_history.append(f"回退到阶段{self.current_stage_id}")
            return True
        return False
    
    def is_final_stage(self) -> bool:
        """判断是否为最终阶段"""
        return self.current_stage_id == len(self.stages) - 1
    
    def get_stage_progress_info(self) -> Dict:
        """获取阶段进度信息"""
        current_stage = self.get_current_stage()
        return {
            "current_stage_id": self.current_stage_id,
            "current_stage_name": current_stage.stage_name,
            "total_stages": len(self.stages),
            "progress_percentage": (self.current_stage_id + 1) / len(self.stages) * 100,
            "stage_history": self.stage_history.copy()
        }
