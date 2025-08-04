# -*- coding: utf-8 -*-
# 文件名: config.py
# 描述: 统一管理项目的所有配置参数，包括训练配置和PBRS配置
#
# PBRS (Potential-Based Reward Shaping) 配置说明:
# ===============================================
# 
# 本文件新增了完整的PBRS配置管理功能，包括：
# 
# 1. 势函数权重参数:
#    - PBRS_COMPLETION_WEIGHT: 完成度势能权重 (默认: 10.0)
#    - PBRS_DISTANCE_WEIGHT: 距离势能权重 (默认: 0.01)
#    - PBRS_COLLABORATION_WEIGHT: 协作势能权重 (默认: 5.0)
# 
# 2. 数值稳定性参数:
#    - PBRS_REWARD_CLIP_MIN/MAX: 塑形奖励裁剪范围 (默认: -50.0 ~ 50.0)
#    - PBRS_POTENTIAL_CLIP_MIN/MAX: 势函数值裁剪范围 (默认: -1000.0 ~ 1000.0)
#    - PBRS_MAX_POTENTIAL_CHANGE: 单步最大势函数变化量 (默认: 100.0)
# 
# 3. 调试和监控参数:
#    - PBRS_DEBUG_MODE: 调试模式开关 (默认: False)
#    - PBRS_LOG_POTENTIAL_VALUES: 记录势函数值 (默认: False)
#    - PBRS_LOG_REWARD_BREAKDOWN: 记录奖励组成详情 (默认: False)
# 
# 4. 性能优化参数:
#    - PBRS_ENABLE_DISTANCE_CACHE: 启用距离缓存 (默认: True)
#    - PBRS_CACHE_UPDATE_THRESHOLD: 缓存更新阈值 (默认: 0.1)
# 
# 使用示例:
# --------
# config = Config()
# config.update_pbrs_params(PBRS_DEBUG_MODE=True, PBRS_COMPLETION_WEIGHT=15.0)
# config.print_pbrs_config()
# config.save_pbrs_config("my_pbrs_config.pkl")
# 
# 验证和安全性:
# -----------
# - 所有参数都有自动验证机制
# - 无效配置会自动重置为默认值
# - 提供配置保存/加载功能
# - 支持运行时参数调整

import os
import pickle
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """训练配置类 - 统一管理所有训练参数"""
    
    # ===== 基础训练参数 =====
    episodes: int = 2000                   # 训练轮次
    learning_rate: float = 0.00005         # 降低学习率，提高数值稳定性
    gamma: float = 0.99                    # 提高折扣因子，更重视长期奖励
    batch_size: int = 64                   # 减小批次大小，提高更新频率
    memory_size: int = 15000               # 适当减小记忆库，避免过旧经验
    
    # ===== 探索策略参数 =====
    epsilon_start: float = 0.9             # 降低初始探索率
    epsilon_end: float = 0.1               # 提高最终探索率，保持适度探索
    epsilon_decay: float = 0.9995          # 放缓探索率衰减
    epsilon_min: float = 0.1               # 提高最小探索率
    
    # ===== 网络更新参数 =====
    target_update_freq: int = 20           # 降低目标网络更新频率，增加稳定性
    patience: int = 100                    # 增加早停耐心值
    log_interval: int = 20                 # 减少日志输出频率
    
    # ===== 梯度裁剪参数 =====
    use_gradient_clipping: bool = True     # 是否使用梯度裁剪
    max_grad_norm: float = 1.0             # 最大梯度范数
    
    # ===== 优先经验回放参数 =====
    use_prioritized_replay: bool = True    # 是否使用优先经验回放
    per_alpha: float = 0.6                 # 优先级指数 (0=均匀采样, 1=完全优先级采样)
    per_beta_start: float = 0.4            # 重要性采样权重初始值
    per_beta_frames: int = 100000          # β从初始值增长到1.0的帧数
    per_epsilon: float = 1e-6              # 防止优先级为0的小值
    
    # ===== 调试参数 =====
    verbose: bool = True                   # 详细输出
    debug_mode: bool = False               # 调试模式
    save_training_history: bool = True     # 保存训练历史

class Config:
    """统一管理所有算法和模拟的参数"""
    
    def __init__(self):
        # ----- 训练系统控制参数 -----
        # 训练模式选择：
        # - 'training': 训练模式，从头开始训练或继续训练
        # - 'inference': 推理模式，仅加载已训练模型进行推理
        # - 'zero_shot_train': 零样本训练模式，专用于ZeroShotGNN
        self.TRAINING_MODE = 'zero_shot_train'
        
        # 强制重新训练标志：
        # - True: 忽略已有模型，强制重新训练
        # - False: 优先加载已有模型，不存在时才训练
        self.FORCE_RETRAIN = True
        
        # 路径规划精度控制：
        # - True: 使用高精度PH-RRT算法，计算准确但耗时
        # - False: 使用快速近似算法，计算快速但精度较低
        self.USE_PHRRT_DURING_TRAINING = True          # 训练时是否使用高精度PH-RRT
        self.USE_PHRRT_DURING_PLANNING = True          # 规划时是否使用高精度PH-RRT
        
        # 模型保存/加载路径配置
        self.SAVED_MODEL_PATH = 'output/models/saved_model_final.pth'
        
        # ----- 网络结构选择参数 -----
        # 网络结构类型选择，支持以下候选项：
        # - 'SimpleNetwork': 基础全连接网络，适合简单场景，训练快速
        # - 'DeepFCN': 深度全连接网络，具有更强的表达能力
        # - 'DeepFCNResidual': 带残差连接的深度网络，缓解梯度消失问题
        # - 'ZeroShotGNN': 零样本图神经网络，具有泛化能力，适合不同规模场景
        # - 'GAT': 图注意力网络，专注于图结构数据处理
        self.NETWORK_TYPE = 'ZeroShotGNN'    # 切换到ZeroShotGNN进行稳定性调试

        # ----- 改进 ZeroShotGNN奖励函数 -----
        self.USE_IMPROVED_REWARD = True  # 启用改进版奖励函数
        
        # ----- 路径规划参数 -----
        # RRT算法核心参数：
        self.RRT_ITERATIONS = 1000          # RRT最大迭代次数，影响路径质量和计算时间
        self.RRT_STEP_SIZE = 50.0           # RRT单步扩展距离，影响路径平滑度
        self.RRT_GOAL_BIAS = 0.1            # 目标偏向概率(0-1)，越大越快收敛但可能陷入局部最优
        self.RRT_ADAPTIVE_STEP = True       # 自适应步长：True=根据环境调整，False=固定步长
        self.RRT_OBSTACLE_AWARE = True      # 障碍物感知采样：True=避开障碍物，False=随机采样
        self.RRT_MAX_ATTEMPTS = 3           # 路径规划失败时的最大重试次数
        
        # ===== PH曲线平滑参数 =====
        self.MAX_REFINEMENT_ATTEMPTS = 5    # 最大细化尝试次数
        self.BEZIER_SAMPLES = 50            # 贝塞尔曲线采样点数
        self.OBSTACLE_TOLERANCE = 50.0      # 障碍物的安全容忍距离

        # ----- 图构建参数 -----
        # 图结构离散化参数：
        self.GRAPH_N_PHI = 6                # 每个目标节点的离散化接近角度数量，影响动作空间大小

        # ----- 环境维度参数 -----
        # 环境规模限制（用于张量维度统一）：
        self.MAX_UAVS = 10                  # 支持的最大UAV数量，超出会截断
        self.MAX_TARGETS = 15               # 支持的最大目标数量，超出会截断
        self.MAP_SIZE = 1000.0              # 地图边长(米)，用于坐标归一化
        self.MAX_INTERACTION_RANGE = 2000.0 # UAV最大交互距离(米)，超出视为无效

        # ----- 模拟与评估参数 -----
        # 可视化控制：
        self.SHOW_VISUALIZATION = False     # 是否显示matplotlib可视化图表
        
        # 负载均衡参数：
        self.LOAD_BALANCE_PENALTY = 0.1     # 负载不均衡惩罚系数(0-1)，越大越重视均衡

        # ----- 奖励函数参数 -----
        self.TARGET_COMPLETION_REWARD = 1500    # 目标完成奖励
        self.MARGINAL_UTILITY_FACTOR = 1000    # 边际效用因子
        self.EFFICIENCY_REWARD_FACTOR = 500     # 效率奖励因子
        self.DISTANCE_PENALTY_FACTOR = 0.1     # 距离惩罚因子
        self.TIME_PENALTY_FACTOR = 10          # 时间惩罚因子
        self.COMPLETION_REWARD = 1000          # 完成奖励
        self.INVALID_ACTION_PENALTY = -100     # 无效动作惩罚
        self.ZERO_CONTRIBUTION_PENALTY = -50   # 零贡献惩罚
        self.DEADLOCK_PENALTY = -200           # 死锁惩罚
        self.COLLABORATION_BONUS = 200         # 协作奖励

        # ----- PBRS (Potential-Based Reward Shaping) 参数 -----
        # PBRS功能开关 (暂时禁用，回到稳定基线)
        self.ENABLE_PBRS = False                        # 暂时禁用PBRS，解决不稳定问题
        self.PBRS_TYPE = 'simple'                       # PBRS类型: 'simple'(完成目标数) 或 'progress'(资源进度)
        self.ENABLE_REWARD_LOGGING = True               # 是否保存最新的奖励组成用于调试和监控
        
        # 势函数权重参数
        self.PBRS_COMPLETION_WEIGHT = 10.0              # 完成度势能权重
        self.PBRS_DISTANCE_WEIGHT = 0.01                # 距离势能权重
        self.PBRS_COLLABORATION_WEIGHT = 5.0            # 协作势能权重
        
        # 奖励裁剪参数 (极保守版本)
        self.PBRS_REWARD_CLIP_MIN = -5.0                # 塑形奖励最小值 (极保守)
        self.PBRS_REWARD_CLIP_MAX = 5.0                 # 塑形奖励最大值 (极保守)
        self.PBRS_POTENTIAL_SCALE = 0.01                # 势函数缩放因子 (极小影响)
        self.PBRS_WARMUP_EPISODES = 100                 # PBRS预热期 (前100轮不使用)
        
        # 调试参数
        self.PBRS_DEBUG_MODE = False                    # PBRS调试模式
        self.PBRS_LOG_POTENTIAL_VALUES = False          # 是否记录势函数值
        self.PBRS_LOG_REWARD_BREAKDOWN = False          # 是否记录奖励组成详情
        
        # 数值稳定性参数
        self.PBRS_POTENTIAL_CLIP_MIN = -1000.0          # 势函数值最小值
        self.PBRS_POTENTIAL_CLIP_MAX = 1000.0           # 势函数值最大值
        self.PBRS_ENABLE_GRADIENT_CLIPPING = True       # 是否启用梯度裁剪
        self.PBRS_MAX_POTENTIAL_CHANGE = 100.0          # 单步最大势函数变化量
        
        # 缓存和性能参数
        self.PBRS_ENABLE_DISTANCE_CACHE = True          # 是否启用距离缓存
        self.PBRS_CACHE_UPDATE_THRESHOLD = 0.1          # 缓存更新阈值
        
        # ----- 紧急稳定性修复参数 -----
        # 奖励归一化优化
        self.REWARD_NORMALIZATION = True           # 启用奖励归一化
        self.REWARD_SCALE = 0.3                    # 从0.1提升到0.3 (3倍)
        
        # 数值稳定性检查
        self.ENABLE_NUMERICAL_STABILITY_CHECKS = True  # 启用数值稳定性检查
        
        # ----- 训练配置对象 -----
        self.training_config = TrainingConfig()
        
        # 根据网络类型设置优化的参数配置
        self._setup_network_specific_params()
        
        # 设置统一的训练参数访问接口
        self._setup_unified_training_params()
        
        # 验证PBRS配置
        self._validate_pbrs_on_init()
    
    def _setup_network_specific_params(self):
        """根据网络类型设置优化的参数配置"""
        
        if self.NETWORK_TYPE == 'DeepFCN':
            # DeepFCN稳定训练参数 (经过测试验证的最佳配置)
            print(f"🎯 应用DeepFCN稳定训练参数配置")
            self.training_config.learning_rate = 1e-05              # 极低学习率，避免训练震荡
            self.training_config.gradient_clip_norm = 0.5           # 严格梯度裁剪，防止梯度爆炸
            self.training_config.weight_decay = 2e-05               # 高正则化，防止过拟合和数值不稳定
            self.training_config.target_update_frequency = 1500     # 稳定的目标网络更新
            self.training_config.batch_size = 64                   # 中等批次大小，平衡稳定性和效率
            self.training_config.epsilon_decay = 0.995             # 平滑探索衰减
            self.training_config.epsilon_min = 0.05                # 保持最小探索
            
        elif self.NETWORK_TYPE == 'ZeroShotGNN':
            # ZeroShotGNN优化配置 (基于问题分析)
            print(f"🚀 应用ZeroShotGNN优化配置")
            self.training_config.learning_rate = 1e-05              # 极低学习率，借鉴DeepFCN成功经验
            self.training_config.gradient_clip_norm = 0.5           # 严格梯度裁剪，防止图网络梯度不稳定
            self.training_config.weight_decay = 2e-05               # 高正则化，增强稳定性
            self.training_config.target_update_frequency = 2000     # 更稳定的目标网络更新
            self.training_config.batch_size = 16                   # 小批次，减少计算开销
            self.training_config.epsilon_decay = 0.998             # 更慢的探索衰减，适合图网络
            self.training_config.epsilon_min = 0.1                 # 保持较高最小探索
        elif self.NETWORK_TYPE == 'SimpleNetwork':
            # SimpleNetwork基础配置
            print(f"⚡ 应用SimpleNetwork基础参数配置")
            self.training_config.learning_rate = 1e-04              # 标准学习率
            self.training_config.gradient_clip_norm = 1.0           # 标准梯度裁剪
            self.training_config.weight_decay = 1e-06               # 低正则化
            self.training_config.target_update_frequency = 500      # 频繁更新
            self.training_config.batch_size = 128                  # 大批次
            self.training_config.epsilon_decay = 0.995             # 标准衰减
            self.training_config.epsilon_min = 0.01                # 低最小探索
            
        elif self.NETWORK_TYPE == 'DeepFCNResidual':
            # DeepFCNResidual配置 (基于DeepFCN优化)
            print(f"🚀 应用DeepFCNResidual参数配置")
            self.training_config.learning_rate = 2e-05              # 略高于DeepFCN
            self.training_config.gradient_clip_norm = 0.8           # 适中梯度裁剪
            self.training_config.weight_decay = 1e-05               # 中等正则化
            self.training_config.target_update_frequency = 1200     # 适中更新频率
            self.training_config.batch_size = 64                   # 与DeepFCN相同
            self.training_config.epsilon_decay = 0.996             # 略快衰减
            self.training_config.epsilon_min = 0.05                # 标准最小探索
            
        else:
            # 默认配置
            print(f"⚠️ 使用默认参数配置 (网络类型: {self.NETWORK_TYPE})")
            self.training_config.learning_rate = 1e-04
            self.training_config.gradient_clip_norm = 1.0
            self.training_config.weight_decay = 1e-05
            self.training_config.target_update_frequency = 1000
            self.training_config.batch_size = 64
            self.training_config.epsilon_decay = 0.995
            self.training_config.epsilon_min = 0.05
        
        print(f"✅ 网络特定参数配置完成")
        print(f"   学习率: {self.training_config.learning_rate}")
        print(f"   梯度裁剪: {self.training_config.gradient_clip_norm}")
        print(f"   权重衰减: {self.training_config.weight_decay}")
        print(f"   批次大小: {self.training_config.batch_size}")
    
    def _setup_unified_training_params(self):
        """
        设置统一的训练参数访问接口
        所有训练相关参数都通过training_config统一管理，避免重复定义
        """
        # 为了向后兼容，提供属性访问接口
        pass
    
    def _validate_pbrs_on_init(self):
        """在初始化时验证PBRS配置"""
        if not self.validate_pbrs_config():
            print("⚠️  警告: PBRS配置初始化验证失败，将使用默认值")
            self.reset_pbrs_to_defaults()
    
    # ===== 统一的训练参数访问属性 =====
    @property
    def EPISODES(self):
        return self.training_config.episodes
    
    @EPISODES.setter
    def EPISODES(self, value):
        self.training_config.episodes = value
    
    @property
    def LEARNING_RATE(self):
        return self.training_config.learning_rate
    
    @LEARNING_RATE.setter
    def LEARNING_RATE(self, value):
        self.training_config.learning_rate = 1e-05
    
    @property
    def GAMMA(self):
        return self.training_config.gamma
    
    @GAMMA.setter
    def GAMMA(self, value):
        self.training_config.gamma = value
    
    @property
    def BATCH_SIZE(self):
        return self.training_config.batch_size
    
    @BATCH_SIZE.setter
    def BATCH_SIZE(self, value):
        self.training_config.batch_size = value
    
    @property
    def MEMORY_SIZE(self):
        return self.training_config.memory_size
    
    @MEMORY_SIZE.setter
    def MEMORY_SIZE(self, value):
        self.training_config.memory_size = value
    
    @property
    def MEMORY_CAPACITY(self):
        return self.training_config.memory_size
    
    @MEMORY_CAPACITY.setter
    def MEMORY_CAPACITY(self, value):
        self.training_config.memory_size = value
    
    @property
    def EPSILON_START(self):
        return self.training_config.epsilon_start
    
    @EPSILON_START.setter
    def EPSILON_START(self, value):
        self.training_config.epsilon_start = value
    
    @property
    def EPSILON_END(self):
        return self.training_config.epsilon_end
    
    @EPSILON_END.setter
    def EPSILON_END(self, value):
        self.training_config.epsilon_end = value
    
    @property
    def EPSILON_DECAY(self):
        return self.training_config.epsilon_decay
    
    @EPSILON_DECAY.setter
    def EPSILON_DECAY(self, value):
        self.training_config.epsilon_decay = value
    
    @property
    def EPSILON_MIN(self):
        return self.training_config.epsilon_min
    
    @EPSILON_MIN.setter
    def EPSILON_MIN(self, value):
        self.training_config.epsilon_min = value
    
    @property
    def TARGET_UPDATE_FREQ(self):
        return self.training_config.target_update_freq
    
    @TARGET_UPDATE_FREQ.setter
    def TARGET_UPDATE_FREQ(self, value):
        self.training_config.target_update_freq = value
    
    @property
    def PATIENCE(self):
        return self.training_config.patience
    
    @PATIENCE.setter
    def PATIENCE(self, value):
        self.training_config.patience = value
    
    @property
    def LOG_INTERVAL(self):
        return self.training_config.log_interval
    
    @LOG_INTERVAL.setter
    def LOG_INTERVAL(self, value):
        self.training_config.log_interval = value
    
    # ===== 便捷的参数修改方法 =====
    def update_training_params(self, **kwargs):
        """
        便捷的训练参数批量更新方法
        
        使用示例:
        config.update_training_params(
            episodes=1000,
            learning_rate=0.001,
            batch_size=128
        )
        """
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
                print(f"✓ 更新训练参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的训练参数 '{key}'")
    
    def get_training_summary(self):
        """获取当前训练参数摘要"""
        summary = {
            "基础参数": {
                "episodes": self.training_config.episodes,
                "learning_rate": self.training_config.learning_rate,
                "gamma": self.training_config.gamma,
                "batch_size": self.training_config.batch_size,
                "memory_size": self.training_config.memory_size,
            },
            "探索策略": {
                "epsilon_start": self.training_config.epsilon_start,
                "epsilon_end": self.training_config.epsilon_end,
                "epsilon_decay": self.training_config.epsilon_decay,
                "epsilon_min": self.training_config.epsilon_min,
            },
            "网络更新": {
                "target_update_freq": self.training_config.target_update_freq,
                "patience": self.training_config.patience,
                "log_interval": self.training_config.log_interval,
            },
            "优先经验回放": {
                "use_prioritized_replay": self.training_config.use_prioritized_replay,
                "per_alpha": self.training_config.per_alpha,
                "per_beta_start": self.training_config.per_beta_start,
                "per_beta_frames": self.training_config.per_beta_frames,
            }
        }
        return summary
    
    def print_training_config(self):
        """打印当前训练配置"""
        print("=" * 60)
        print("当前训练配置参数")
        print("=" * 60)
        
        summary = self.get_training_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        
        # 新增的训练参数
        self.use_gradient_clipping = self.training_config.use_gradient_clipping
        self.max_grad_norm = self.training_config.max_grad_norm
    
    def update_training_config(self, new_config: TrainingConfig):
        """更新训练配置"""
        self.training_config = new_config
        self._setup_backward_compatibility()
    
    def get_training_config(self) -> TrainingConfig:
        """获取当前训练配置"""
        return self.training_config
    
    def load_existing_model(self, model_path: str = None) -> bool:
        """尝试加载已存在的模型"""
        if model_path is None:
            model_path = self.SAVED_MODEL_PATH
        
        if os.path.exists(model_path):
            print(f"发现已存在的模型: {model_path}")
            return True
        return False
    
    # ===== 训练模式便捷方法 =====
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        valid_modes = ['training', 'inference', 'zero_shot_train']
        if mode not in valid_modes:
            raise ValueError(f"无效的训练模式: {mode}。有效模式: {valid_modes}")
        self.TRAINING_MODE = mode
    
    def is_training_mode(self) -> bool:
        """检查是否为训练模式"""
        return self.TRAINING_MODE == 'training'
    
    def is_inference_mode(self) -> bool:
        """检查是否为推理模式"""
        return self.TRAINING_MODE == 'inference'
    
    # ===== PBRS配置管理方法 =====
    def validate_pbrs_config(self) -> bool:
        """
        验证PBRS配置参数的有效性
        
        Returns:
            bool: 配置是否有效
        """
        validation_errors = []
        
        # 验证权重参数
        if self.PBRS_COMPLETION_WEIGHT < 0:
            validation_errors.append("PBRS_COMPLETION_WEIGHT必须为非负数")
        
        if self.PBRS_DISTANCE_WEIGHT < 0:
            validation_errors.append("PBRS_DISTANCE_WEIGHT必须为非负数")
        
        if self.PBRS_COLLABORATION_WEIGHT < 0:
            validation_errors.append("PBRS_COLLABORATION_WEIGHT必须为非负数")
        
        # 验证裁剪参数
        if self.PBRS_REWARD_CLIP_MIN >= self.PBRS_REWARD_CLIP_MAX:
            validation_errors.append("PBRS_REWARD_CLIP_MIN必须小于PBRS_REWARD_CLIP_MAX")
        
        if self.PBRS_POTENTIAL_CLIP_MIN >= self.PBRS_POTENTIAL_CLIP_MAX:
            validation_errors.append("PBRS_POTENTIAL_CLIP_MIN必须小于PBRS_POTENTIAL_CLIP_MAX")
        
        # 验证数值稳定性参数
        if self.PBRS_MAX_POTENTIAL_CHANGE <= 0:
            validation_errors.append("PBRS_MAX_POTENTIAL_CHANGE必须为正数")
        
        if self.PBRS_CACHE_UPDATE_THRESHOLD <= 0 or self.PBRS_CACHE_UPDATE_THRESHOLD >= 1:
            validation_errors.append("PBRS_CACHE_UPDATE_THRESHOLD必须在(0,1)范围内")
        
        # 输出验证结果
        if validation_errors:
            print("PBRS配置验证失败:")
            for error in validation_errors:
                print(f"  ✗ {error}")
            return False
        else:
            if self.PBRS_DEBUG_MODE:
                print("✓ PBRS配置验证通过")
            return True
    
    def get_pbrs_config_summary(self) -> Dict[str, Any]:
        """获取PBRS配置摘要"""
        return {
            "功能开关": {
                "ENABLE_PBRS": self.ENABLE_PBRS,
                "PBRS_DEBUG_MODE": self.PBRS_DEBUG_MODE,
                "PBRS_LOG_POTENTIAL_VALUES": self.PBRS_LOG_POTENTIAL_VALUES,
                "PBRS_LOG_REWARD_BREAKDOWN": self.PBRS_LOG_REWARD_BREAKDOWN,
            },
            "势函数权重": {
                "PBRS_COMPLETION_WEIGHT": self.PBRS_COMPLETION_WEIGHT,
                "PBRS_DISTANCE_WEIGHT": self.PBRS_DISTANCE_WEIGHT,
                "PBRS_COLLABORATION_WEIGHT": self.PBRS_COLLABORATION_WEIGHT,
            },
            "数值稳定性": {
                "PBRS_REWARD_CLIP_MIN": self.PBRS_REWARD_CLIP_MIN,
                "PBRS_REWARD_CLIP_MAX": self.PBRS_REWARD_CLIP_MAX,
                "PBRS_POTENTIAL_CLIP_MIN": self.PBRS_POTENTIAL_CLIP_MIN,
                "PBRS_POTENTIAL_CLIP_MAX": self.PBRS_POTENTIAL_CLIP_MAX,
                "PBRS_MAX_POTENTIAL_CHANGE": self.PBRS_MAX_POTENTIAL_CHANGE,
            },
            "性能优化": {
                "PBRS_ENABLE_DISTANCE_CACHE": self.PBRS_ENABLE_DISTANCE_CACHE,
                "PBRS_CACHE_UPDATE_THRESHOLD": self.PBRS_CACHE_UPDATE_THRESHOLD,
                "PBRS_ENABLE_GRADIENT_CLIPPING": self.PBRS_ENABLE_GRADIENT_CLIPPING,
            }
        }
    
    def print_pbrs_config(self):
        """打印PBRS配置参数"""
        print("=" * 60)
        print("PBRS (Potential-Based Reward Shaping) 配置参数")
        print("=" * 60)
        
        summary = self.get_pbrs_config_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
    
    def update_pbrs_params(self, **kwargs):
        """
        便捷的PBRS参数批量更新方法
        
        使用示例:
        config.update_pbrs_params(
            PBRS_COMPLETION_WEIGHT=15.0,
            PBRS_DEBUG_MODE=True,
            ENABLE_PBRS=False
        )
        """
        pbrs_params = [attr for attr in dir(self) if attr.startswith('PBRS_') or attr == 'ENABLE_PBRS']
        
        for key, value in kwargs.items():
            if key in pbrs_params:
                setattr(self, key, value)
                print(f"✓ 更新PBRS参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的PBRS参数 '{key}'")
        
        # 更新后重新验证配置
        if not self.validate_pbrs_config():
            print("⚠️  警告: PBRS配置验证失败，请检查参数设置")
    
    def reset_pbrs_to_defaults(self):
        """重置PBRS参数为默认值"""
        self.ENABLE_PBRS = True
        self.PBRS_COMPLETION_WEIGHT = 10.0
        self.PBRS_DISTANCE_WEIGHT = 0.01
        self.PBRS_COLLABORATION_WEIGHT = 5.0
        self.PBRS_REWARD_CLIP_MIN = -50.0
        self.PBRS_REWARD_CLIP_MAX = 50.0
        self.PBRS_DEBUG_MODE = False
        self.PBRS_LOG_POTENTIAL_VALUES = False
        self.PBRS_LOG_REWARD_BREAKDOWN = False
        self.PBRS_POTENTIAL_CLIP_MIN = -1000.0
        self.PBRS_POTENTIAL_CLIP_MAX = 1000.0
        self.PBRS_ENABLE_GRADIENT_CLIPPING = True
        self.PBRS_MAX_POTENTIAL_CHANGE = 100.0
        self.PBRS_ENABLE_DISTANCE_CACHE = True
        self.PBRS_CACHE_UPDATE_THRESHOLD = 0.1
        
        print("✓ PBRS参数已重置为默认值")
        self.validate_pbrs_config()
    
    def is_pbrs_enabled(self) -> bool:
        """检查PBRS功能是否启用"""
        return self.ENABLE_PBRS and self.validate_pbrs_config()
    
    def save_pbrs_config(self, filepath: str = "pbrs_config.pkl"):
        """
        保存PBRS配置到文件
        
        Args:
            filepath: 保存路径，默认为pbrs_config.pkl
        """
        pbrs_config = self.get_pbrs_config_summary()
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(pbrs_config, f)
            print(f"✓ PBRS配置已保存到: {filepath}")
        except Exception as e:
            print(f"✗ 保存PBRS配置失败: {e}")
    
    def load_pbrs_config(self, filepath: str = "pbrs_config.pkl"):
        """
        从文件加载PBRS配置
        
        Args:
            filepath: 配置文件路径
        """
        try:
            with open(filepath, 'rb') as f:
                pbrs_config = pickle.load(f)
            
            # 展平配置字典并更新参数
            flat_config = {}
            for category, params in pbrs_config.items():
                flat_config.update(params)
            
            self.update_pbrs_params(**flat_config)
            print(f"✓ PBRS配置已从 {filepath} 加载")
            
        except FileNotFoundError:
            print(f"✗ 配置文件不存在: {filepath}")
        except Exception as e:
            print(f"✗ 加载PBRS配置失败: {e}")
    
    # 向后兼容的方法
    @property
    def RUN_TRAINING(self) -> bool:
        """向后兼容的RUN_TRAINING属性"""
        return self.is_training_mode()
    
    @RUN_TRAINING.setter
    def RUN_TRAINING(self, value: bool):
        """向后兼容的RUN_TRAINING设置器"""
        self.TRAINING_MODE = 'training' if value else 'inference'