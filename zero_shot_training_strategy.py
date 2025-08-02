#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零样本迁移训练策略

专门为ZeroShotGNN设计的训练策略，旨在最大化零样本迁移能力
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import random
from collections import deque
import time

class ZeroShotTrainingStrategy:
    """
    零样本迁移训练策略
    
    核心思想：
    1. 多尺度训练：在不同规模的场景上训练，提高泛化能力
    2. 课程学习：从简单到复杂的渐进式训练
    3. 对抗性训练：引入噪声和扰动，增强鲁棒性
    4. 元学习：快速适应新场景的能力
    5. 正则化：防止过拟合到特定规模
    """
    
    def __init__(self, config):
        self.config = config
        self.training_phases = self._design_training_phases()
        self.current_phase = 0
        
    def _design_training_phases(self) -> List[Dict]:
        """
        设计分阶段训练策略
        
        Returns:
            List[Dict]: 训练阶段配置列表
        """
        phases = [
            # 阶段1：基础能力训练（小规模场景）
            {
                "name": "基础能力训练",
                "episodes": 500,
                "uav_range": (2, 4),
                "target_range": (2, 5),
                "learning_rate": 1e-3,
                "epsilon_start": 0.9,
                "epsilon_end": 0.3,
                "batch_size": 32,
                "focus": "基本的UAV-目标分配能力",
                "regularization": {
                    "dropout": 0.1,
                    "weight_decay": 1e-4,
                    "gradient_clip": 1.0
                }
            },
            
            # 阶段2：尺度适应训练（中等规模场景）
            {
                "name": "尺度适应训练", 
                "episodes": 800,
                "uav_range": (3, 8),
                "target_range": (4, 10),
                "learning_rate": 5e-4,
                "epsilon_start": 0.5,
                "epsilon_end": 0.2,
                "batch_size": 64,
                "focus": "适应不同规模的场景",
                "regularization": {
                    "dropout": 0.15,
                    "weight_decay": 5e-4,
                    "gradient_clip": 0.5
                },
                "multi_scale_sampling": True
            },
            
            # 阶段3：复杂性挑战训练（大规模场景）
            {
                "name": "复杂性挑战训练",
                "episodes": 1000,
                "uav_range": (6, 15),
                "target_range": (8, 20),
                "learning_rate": 2e-4,
                "epsilon_start": 0.3,
                "epsilon_end": 0.1,
                "batch_size": 128,
                "focus": "处理大规模复杂场景",
                "regularization": {
                    "dropout": 0.2,
                    "weight_decay": 1e-3,
                    "gradient_clip": 0.3
                },
                "adversarial_training": True,
                "curriculum_learning": True
            },
            
            # 阶段4：零样本泛化强化（混合规模训练）
            {
                "name": "零样本泛化强化",
                "episodes": 1200,
                "uav_range": (2, 20),
                "target_range": (2, 25),
                "learning_rate": 1e-4,
                "epsilon_start": 0.2,
                "epsilon_end": 0.05,
                "batch_size": 256,
                "focus": "最大化零样本迁移能力",
                "regularization": {
                    "dropout": 0.25,
                    "weight_decay": 2e-3,
                    "gradient_clip": 0.2
                },
                "meta_learning": True,
                "domain_randomization": True,
                "consistency_regularization": True
            }
        ]
        
        return phases
    
    def get_current_phase_config(self) -> Dict:
        """获取当前训练阶段配置"""
        if self.current_phase < len(self.training_phases):
            return self.training_phases[self.current_phase]
        else:
            # 返回最后一个阶段的配置
            return self.training_phases[-1]
    
    def should_advance_phase(self, metrics: Dict) -> bool:
        """
        判断是否应该进入下一个训练阶段
        
        Args:
            metrics: 当前训练指标
            
        Returns:
            bool: 是否应该进入下一阶段
        """
        current_config = self.get_current_phase_config()
        
        # 基于多个指标判断
        criteria = [
            metrics.get('avg_reward', 0) > current_config.get('reward_threshold', 50),
            metrics.get('completion_rate', 0) > current_config.get('completion_threshold', 0.7),
            metrics.get('episodes_completed', 0) >= current_config['episodes'] * 0.8,
            metrics.get('loss_stability', 1.0) < 0.1  # 损失稳定性
        ]
        
        # 至少满足3个条件才能进入下一阶段
        return sum(criteria) >= 3
    
    def advance_phase(self):
        """进入下一个训练阶段"""
        if self.current_phase < len(self.training_phases) - 1:
            self.current_phase += 1
            print(f"进入训练阶段 {self.current_phase + 1}: {self.get_current_phase_config()['name']}")
            return True
        return False
    
    def generate_training_scenario(self, phase_config: Dict) -> Tuple[int, int]:
        """
        根据阶段配置生成训练场景
        
        Args:
            phase_config: 阶段配置
            
        Returns:
            Tuple[int, int]: (UAV数量, 目标数量)
        """
        uav_min, uav_max = phase_config['uav_range']
        target_min, target_max = phase_config['target_range']
        
        if phase_config.get('curriculum_learning', False):
            # 课程学习：根据训练进度调整难度
            progress = min(1.0, phase_config.get('current_episode', 0) / phase_config['episodes'])
            
            # 难度渐进增加
            uav_range_size = uav_max - uav_min
            target_range_size = target_max - target_min
            
            current_uav_max = uav_min + int(uav_range_size * progress)
            current_target_max = target_min + int(target_range_size * progress)
            
            n_uavs = random.randint(uav_min, max(uav_min + 1, current_uav_max))
            n_targets = random.randint(target_min, max(target_min + 1, current_target_max))
        
        elif phase_config.get('multi_scale_sampling', False):
            # 多尺度采样：偏向于训练不同规模
            scales = ['small', 'medium', 'large']
            scale = random.choice(scales)
            
            if scale == 'small':
                n_uavs = random.randint(uav_min, uav_min + (uav_max - uav_min) // 3)
                n_targets = random.randint(target_min, target_min + (target_max - target_min) // 3)
            elif scale == 'medium':
                n_uavs = random.randint(uav_min + (uav_max - uav_min) // 3, 
                                      uav_min + 2 * (uav_max - uav_min) // 3)
                n_targets = random.randint(target_min + (target_max - target_min) // 3,
                                         target_min + 2 * (target_max - target_min) // 3)
            else:  # large
                n_uavs = random.randint(uav_min + 2 * (uav_max - uav_min) // 3, uav_max)
                n_targets = random.randint(target_min + 2 * (target_max - target_min) // 3, target_max)
        
        else:
            # 均匀随机采样
            n_uavs = random.randint(uav_min, uav_max)
            n_targets = random.randint(target_min, target_max)
        
        return n_uavs, n_targets
    
    def apply_regularization_techniques(self, model: nn.Module, phase_config: Dict) -> Dict:
        """
        应用正则化技术
        
        Args:
            model: 神经网络模型
            phase_config: 阶段配置
            
        Returns:
            Dict: 正则化配置
        """
        reg_config = phase_config.get('regularization', {})
        
        regularization_settings = {
            'dropout_rate': reg_config.get('dropout', 0.1),
            'weight_decay': reg_config.get('weight_decay', 1e-4),
            'gradient_clip_norm': reg_config.get('gradient_clip', 1.0),
        }
        
        # 应用dropout调整
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = regularization_settings['dropout_rate']
        
        return regularization_settings
    
    def apply_domain_randomization(self, scenario_params: Dict) -> Dict:
        """
        应用域随机化技术
        
        Args:
            scenario_params: 场景参数
            
        Returns:
            Dict: 随机化后的场景参数
        """
        randomized_params = scenario_params.copy()
        
        # 随机化UAV参数
        if 'uav_params' in randomized_params:
            for uav_param in randomized_params['uav_params']:
                # 随机化资源
                uav_param['resources'] = [
                    r * random.uniform(0.8, 1.2) for r in uav_param['resources']
                ]
                # 随机化速度
                uav_param['velocity_range'] = [
                    v * random.uniform(0.9, 1.1) for v in uav_param['velocity_range']
                ]
                # 随机化位置（小幅度）
                uav_param['position'] = [
                    p + random.uniform(-50, 50) for p in uav_param['position']
                ]
        
        # 随机化目标参数
        if 'target_params' in randomized_params:
            for target_param in randomized_params['target_params']:
                # 随机化资源需求
                target_param['resources'] = [
                    r * random.uniform(0.8, 1.2) for r in target_param['resources']
                ]
                # 随机化价值
                target_param['value'] = target_param['value'] * random.uniform(0.9, 1.1)
                # 随机化位置（小幅度）
                target_param['position'] = [
                    p + random.uniform(-30, 30) for p in target_param['position']
                ]
        
        return randomized_params
    
    def compute_consistency_loss(self, model: nn.Module, state_batch: torch.Tensor, 
                                augmented_state_batch: torch.Tensor) -> torch.Tensor:
        """
        计算一致性正则化损失
        
        Args:
            model: 神经网络模型
            state_batch: 原始状态批次
            augmented_state_batch: 增强状态批次
            
        Returns:
            torch.Tensor: 一致性损失
        """
        with torch.no_grad():
            original_output = model(state_batch)
        
        augmented_output = model(augmented_state_batch)
        
        # 计算KL散度作为一致性损失
        consistency_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(augmented_output, dim=1),
            torch.softmax(original_output, dim=1)
        )
        
        return consistency_loss
    
    def get_optimizer_config(self, phase_config: Dict) -> Dict:
        """
        获取优化器配置
        
        Args:
            phase_config: 阶段配置
            
        Returns:
            Dict: 优化器配置
        """
        return {
            'lr': phase_config['learning_rate'],
            'weight_decay': phase_config.get('regularization', {}).get('weight_decay', 1e-4),
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
    
    def get_training_summary(self) -> str:
        """
        获取训练策略总结
        
        Returns:
            str: 训练策略总结
        """
        summary = "零样本迁移训练策略总结:\n"
        summary += "=" * 50 + "\n"
        
        for i, phase in enumerate(self.training_phases):
            status = "✓ 已完成" if i < self.current_phase else "○ 待执行" if i == self.current_phase else "- 未开始"
            summary += f"{status} 阶段 {i+1}: {phase['name']}\n"
            summary += f"   - 训练轮数: {phase['episodes']}\n"
            summary += f"   - UAV范围: {phase['uav_range']}\n"
            summary += f"   - 目标范围: {phase['target_range']}\n"
            summary += f"   - 学习率: {phase['learning_rate']}\n"
            summary += f"   - 重点: {phase['focus']}\n"
            summary += "\n"
        
        return summary

class ZeroShotMetrics:
    """零样本迁移指标计算器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        
    def update(self, episode_metrics: Dict):
        """更新指标"""
        self.metrics_history.append(episode_metrics)
    
    def compute_transfer_score(self, test_scenarios: List[Tuple[int, int]]) -> float:
        """
        计算零样本迁移得分
        
        Args:
            test_scenarios: 测试场景列表 [(n_uavs, n_targets), ...]
            
        Returns:
            float: 迁移得分 (0-1)
        """
        if not self.metrics_history:
            return 0.0
        
        # 基于最近的性能计算迁移得分
        recent_metrics = list(self.metrics_history)[-100:]  # 最近100轮
        
        avg_reward = np.mean([m.get('reward', 0) for m in recent_metrics])
        avg_completion = np.mean([m.get('completion_rate', 0) for m in recent_metrics])
        reward_stability = 1.0 - np.std([m.get('reward', 0) for m in recent_metrics]) / (abs(avg_reward) + 1e-6)
        
        # 综合得分
        transfer_score = (
            0.4 * min(1.0, avg_reward / 100.0) +  # 奖励得分
            0.4 * avg_completion +                 # 完成率得分
            0.2 * max(0.0, reward_stability)       # 稳定性得分
        )
        
        return float(np.clip(transfer_score, 0.0, 1.0))
    
    def get_training_progress(self) -> Dict:
        """获取训练进度指标"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-50:]  # 最近50轮
        
        return {
            'avg_reward': np.mean([m.get('reward', 0) for m in recent_metrics]),
            'completion_rate': np.mean([m.get('completion_rate', 0) for m in recent_metrics]),
            'loss_stability': np.std([m.get('loss', 0) for m in recent_metrics]),
            'episodes_completed': len(self.metrics_history)
        }

def create_zero_shot_training_config(base_config) -> Dict:
    """
    创建零样本训练配置
    
    Args:
        base_config: 基础配置
        
    Returns:
        Dict: 零样本训练配置
    """
    zero_shot_config = {
        # 基础训练参数
        'total_episodes': 3500,  # 总训练轮数
        'patience': 200,         # 早停耐心值
        'log_interval': 50,      # 日志间隔
        
        # 零样本特定参数
        'enable_multi_scale_training': True,
        'enable_curriculum_learning': True,
        'enable_domain_randomization': True,
        'enable_consistency_regularization': True,
        'enable_meta_learning': False,  # 可选的元学习
        
        # 评估参数
        'evaluation_scenarios': [
            (2, 3), (4, 6), (6, 9), (8, 12), (10, 15), (12, 18), (15, 22)
        ],
        'evaluation_frequency': 100,  # 每100轮评估一次
        
        # 模型保存策略
        'save_best_transfer_model': True,
        'save_phase_checkpoints': True,
        'model_save_frequency': 200,
        
        # 高级技术
        'use_prioritized_replay': True,
        'use_double_dqn': True,
        'use_gradient_clipping': True,
        'use_learning_rate_scheduling': True,
    }
    
    # 合并基础配置
    for key, value in base_config.__dict__.items():
        if key not in zero_shot_config:
            zero_shot_config[key] = value
    
    return zero_shot_config
