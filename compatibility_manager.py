# -*- coding: utf-8 -*-
# 文件名: compatibility_manager.py
# 描述: 向后兼容性管理器，提供配置开关和兼容性保证

import os
import sys
from typing import Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
import json

from config import Config
from networks import create_network
from transformer_gnn import TransformerGNN, create_transformer_gnn_model


@dataclass
class CompatibilityConfig:
    """兼容性配置类"""
    
    # 网络架构选择
    network_mode: Literal["traditional", "transformer_gnn"] = "traditional"
    
    # 传统网络配置
    traditional_network_type: str = "DeepFCNResidual"
    
    # TransformerGNN配置
    transformer_config: Dict[str, Any] = None
    
    # 观测模式配置
    obs_mode: Literal["flat", "graph"] = "flat"
    
    # 兼容性检查
    enable_compatibility_checks: bool = True
    
    # 调试模式
    debug_mode: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if self.transformer_config is None:
            self.transformer_config = {
                "embed_dim": 128,
                "num_heads": 8,
                "num_layers": 3,
                "dropout": 0.1,
                "use_position_encoding": True,
                "use_noisy_linear": True,
                "use_local_attention": True,
                "k_adaptive": True,
                "k_min": 4,
                "k_max": 16
            }


class CompatibilityManager:
    """
    向后兼容性管理器
    
    核心功能：
    1. 提供配置开关，允许用户选择传统方法或TransformerGNN方法
    2. 确保现有FCN架构和main.py的基础run_scenario流程独立运行
    3. 管理不同网络架构的创建和配置
    4. 提供兼容性检查和验证
    5. 统一的接口，屏蔽底层实现差异
    """
    
    def __init__(self, config: Optional[CompatibilityConfig] = None):
        """
        初始化兼容性管理器
        
        Args:
            config: 兼容性配置，如果为None则使用默认配置
        """
        self.config = config or CompatibilityConfig()
        self.base_config = Config()  # 基础配置
        
        # 验证配置
        self._validate_config()
        
        # 设置环境观测模式
        self._setup_observation_mode()
        
        print(f"[CompatibilityManager] 初始化完成")
        print(f"  - 网络模式: {self.config.network_mode}")
        print(f"  - 观测模式: {self.config.obs_mode}")
        print(f"  - 传统网络类型: {self.config.traditional_network_type}")
        print(f"  - 兼容性检查: {'启用' if self.config.enable_compatibility_checks else '禁用'}")
    
    def _validate_config(self):
        """验证配置的有效性"""
        # 验证网络模式
        if self.config.network_mode not in ["traditional", "transformer_gnn"]:
            raise ValueError(f"无效的网络模式: {self.config.network_mode}")
        
        # 验证观测模式
        if self.config.obs_mode not in ["flat", "graph"]:
            raise ValueError(f"无效的观测模式: {self.config.obs_mode}")
        
        # 验证传统网络类型
        valid_traditional_types = ["SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual"]
        if self.config.traditional_network_type not in valid_traditional_types:
            raise ValueError(f"无效的传统网络类型: {self.config.traditional_network_type}")
        
        # 验证模式组合的合理性
        if self.config.network_mode == "transformer_gnn" and self.config.obs_mode == "flat":
            print("[CompatibilityManager] 警告: TransformerGNN建议使用graph观测模式以获得最佳性能")
        
        if self.config.network_mode == "traditional" and self.config.obs_mode == "graph":
            print("[CompatibilityManager] 警告: 传统网络使用graph观测模式可能导致性能下降")
    
    def _setup_observation_mode(self):
        """设置环境观测模式"""
        # 这里可以设置全局的观测模式配置
        # 实际的观测模式设置会在创建环境时进行
        pass
    
    def create_network(self, input_dim: int, hidden_dims: list, output_dim: int, 
                      obs_space=None, action_space=None) -> Any:
        """
        创建网络的统一接口
        
        Args:
            input_dim: 输入维度（传统网络使用）
            hidden_dims: 隐藏层维度列表（传统网络使用）
            output_dim: 输出维度
            obs_space: 观测空间（TransformerGNN使用）
            action_space: 动作空间（TransformerGNN使用）
            
        Returns:
            网络实例
        """
        if self.config.network_mode == "traditional":
            return self._create_traditional_network(input_dim, hidden_dims, output_dim)
        elif self.config.network_mode == "transformer_gnn":
            return self._create_transformer_gnn_network(obs_space, action_space, output_dim)
        else:
            raise ValueError(f"不支持的网络模式: {self.config.network_mode}")
    
    def _create_traditional_network(self, input_dim: int, hidden_dims: list, output_dim: int):
        """创建传统网络"""
        try:
            network = create_network(
                network_type=self.config.traditional_network_type,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim
            )
            
            if self.config.debug_mode:
                print(f"[CompatibilityManager] 创建传统网络成功: {self.config.traditional_network_type}")
                print(f"  - 输入维度: {input_dim}")
                print(f"  - 隐藏层维度: {hidden_dims}")
                print(f"  - 输出维度: {output_dim}")
            
            return network
            
        except Exception as e:
            print(f"[CompatibilityManager] 创建传统网络失败: {e}")
            raise
    
    def _create_transformer_gnn_network(self, obs_space, action_space, output_dim):
        """创建TransformerGNN网络"""
        try:
            if obs_space is None or action_space is None:
                raise ValueError("TransformerGNN需要提供obs_space和action_space")
            
            network = create_transformer_gnn_model(
                obs_space=obs_space,
                action_space=action_space,
                num_outputs=output_dim,
                model_config=self.config.transformer_config,
                name="TransformerGNN"
            )
            
            if self.config.debug_mode:
                print(f"[CompatibilityManager] 创建TransformerGNN网络成功")
                print(f"  - 观测空间: {obs_space}")
                print(f"  - 动作空间: {action_space}")
                print(f"  - 输出维度: {output_dim}")
                print(f"  - 配置: {self.config.transformer_config}")
            
            return network
            
        except Exception as e:
            print(f"[CompatibilityManager] 创建TransformerGNN网络失败: {e}")
            raise
    
    def create_environment(self, uavs, targets, graph, obstacles, config):
        """
        创建环境的统一接口
        
        Args:
            uavs: UAV列表
            targets: 目标列表
            graph: 图对象
            obstacles: 障碍物列表
            config: 环境配置
            
        Returns:
            环境实例
        """
        from environment import UAVTaskEnv
        
        try:
            env = UAVTaskEnv(
                uavs=uavs,
                targets=targets,
                graph=graph,
                obstacles=obstacles,
                config=config,
                obs_mode=self.config.obs_mode
            )
            
            if self.config.debug_mode:
                print(f"[CompatibilityManager] 创建环境成功")
                print(f"  - 观测模式: {self.config.obs_mode}")
                print(f"  - UAV数量: {len(uavs)}")
                print(f"  - 目标数量: {len(targets)}")
            
            return env
            
        except Exception as e:
            print(f"[CompatibilityManager] 创建环境失败: {e}")
            raise
    
    def create_solver(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, 
                     config, network_type=None, tensorboard_dir=None):
        """
        创建求解器的统一接口
        
        Args:
            uavs: UAV列表
            targets: 目标列表
            graph: 图对象
            obstacles: 障碍物列表
            i_dim: 输入维度
            h_dim: 隐藏层维度
            o_dim: 输出维度
            config: 配置对象
            network_type: 网络类型（可选，用于覆盖配置）
            tensorboard_dir: TensorBoard日志目录
            
        Returns:
            求解器实例
        """
        from main import GraphRLSolver
        
        try:
            # 确定使用的网络类型
            if network_type is None:
                if self.config.network_mode == "traditional":
                    network_type = self.config.traditional_network_type
                else:
                    network_type = "TransformerGNN"
            
            # 创建求解器
            solver = GraphRLSolver(
                uavs=uavs,
                targets=targets,
                graph=graph,
                obstacles=obstacles,
                i_dim=i_dim,
                h_dim=h_dim,
                o_dim=o_dim,
                config=config,
                network_type=network_type,
                tensorboard_dir=tensorboard_dir
            )
            
            if self.config.debug_mode:
                print(f"[CompatibilityManager] 创建求解器成功")
                print(f"  - 网络类型: {network_type}")
                print(f"  - 输入维度: {i_dim}")
                print(f"  - 输出维度: {o_dim}")
            
            return solver
            
        except Exception as e:
            print(f"[CompatibilityManager] 创建求解器失败: {e}")
            raise
    
    def run_compatibility_checks(self) -> Dict[str, bool]:
        """
        运行兼容性检查
        
        Returns:
            检查结果字典
        """
        if not self.config.enable_compatibility_checks:
            return {"compatibility_checks": False, "message": "兼容性检查已禁用"}
        
        results = {}
        
        try:
            # 检查1: 验证传统网络创建
            print("[CompatibilityManager] 检查传统网络创建...")
            traditional_network = create_network(
                network_type=self.config.traditional_network_type,
                input_dim=128,
                hidden_dims=[256, 128],
                output_dim=64
            )
            results["traditional_network_creation"] = True
            print("  ✓ 传统网络创建成功")
            
        except Exception as e:
            results["traditional_network_creation"] = False
            print(f"  ✗ 传统网络创建失败: {e}")
        
        try:
            # 检查2: 验证TransformerGNN创建（如果可用）
            print("[CompatibilityManager] 检查TransformerGNN创建...")
            import gymnasium as gym
            
            # 创建模拟的观测空间
            obs_space = gym.spaces.Dict({
                "uav_features": gym.spaces.Box(low=0, high=1, shape=(3, 9)),
                "target_features": gym.spaces.Box(low=0, high=1, shape=(2, 8)),
                "relative_positions": gym.spaces.Box(low=-1, high=1, shape=(3, 2, 2)),
                "distances": gym.spaces.Box(low=0, high=1, shape=(3, 2)),
                "masks": gym.spaces.Dict({
                    "uav_mask": gym.spaces.Box(low=0, high=1, shape=(3,)),
                    "target_mask": gym.spaces.Box(low=0, high=1, shape=(2,))
                })
            })
            action_space = gym.spaces.Discrete(36)
            
            transformer_network = create_transformer_gnn_model(
                obs_space=obs_space,
                action_space=action_space,
                num_outputs=36,
                model_config=self.config.transformer_config
            )
            results["transformer_gnn_creation"] = True
            print("  ✓ TransformerGNN创建成功")
            
        except Exception as e:
            results["transformer_gnn_creation"] = False
            print(f"  ✗ TransformerGNN创建失败: {e}")
        
        try:
            # 检查3: 验证环境创建
            print("[CompatibilityManager] 检查环境创建...")
            from entities import UAV, Target
            from environment import DirectedGraph
            
            # 创建模拟实体
            uavs = [UAV(i, [100*i, 100*i], [10, 10], 500, [20, 50]) for i in range(2)]
            targets = [Target(i, [200*i, 200*i], [5, 5], 100) for i in range(2)]
            
            # 创建图
            graph = DirectedGraph(uavs, targets, 6, [], self.base_config)
            
            # 测试两种观测模式
            for obs_mode in ["flat", "graph"]:
                env = self.create_environment(uavs, targets, graph, [], self.base_config)
                results[f"environment_creation_{obs_mode}"] = True
                print(f"  ✓ 环境创建成功 ({obs_mode}模式)")
            
        except Exception as e:
            results["environment_creation"] = False
            print(f"  ✗ 环境创建失败: {e}")
        
        # 计算总体兼容性状态
        all_passed = all(results.values())
        results["overall_compatibility"] = all_passed
        
        print(f"\n[CompatibilityManager] 兼容性检查完成")
        print(f"  - 总体状态: {'✓ 通过' if all_passed else '✗ 失败'}")
        print(f"  - 详细结果: {results}")
        
        return results
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        获取当前网络配置信息
        
        Returns:
            网络信息字典
        """
        info = {
            "network_mode": self.config.network_mode,
            "obs_mode": self.config.obs_mode,
            "compatibility_checks_enabled": self.config.enable_compatibility_checks,
            "debug_mode": self.config.debug_mode
        }
        
        if self.config.network_mode == "traditional":
            info["traditional_network_type"] = self.config.traditional_network_type
        else:
            info["transformer_config"] = self.config.transformer_config
        
        return info
    
    def save_config(self, filepath: str):
        """
        保存配置到文件
        
        Args:
            filepath: 配置文件路径
        """
        try:
            config_dict = {
                "network_mode": self.config.network_mode,
                "traditional_network_type": self.config.traditional_network_type,
                "transformer_config": self.config.transformer_config,
                "obs_mode": self.config.obs_mode,
                "enable_compatibility_checks": self.config.enable_compatibility_checks,
                "debug_mode": self.config.debug_mode
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"[CompatibilityManager] 配置已保存到: {filepath}")
            
        except Exception as e:
            print(f"[CompatibilityManager] 保存配置失败: {e}")
            raise
    
    @classmethod
    def load_config(cls, filepath: str) -> 'CompatibilityManager':
        """
        从文件加载配置
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            CompatibilityManager实例
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            config = CompatibilityConfig(
                network_mode=config_dict.get("network_mode", "traditional"),
                traditional_network_type=config_dict.get("traditional_network_type", "DeepFCNResidual"),
                transformer_config=config_dict.get("transformer_config"),
                obs_mode=config_dict.get("obs_mode", "flat"),
                enable_compatibility_checks=config_dict.get("enable_compatibility_checks", True),
                debug_mode=config_dict.get("debug_mode", False)
            )
            
            manager = cls(config)
            print(f"[CompatibilityManager] 配置已从文件加载: {filepath}")
            
            return manager
            
        except Exception as e:
            print(f"[CompatibilityManager] 加载配置失败: {e}")
            raise


# 全局兼容性管理器实例
_global_compatibility_manager = None


def get_compatibility_manager() -> CompatibilityManager:
    """获取全局兼容性管理器实例"""
    global _global_compatibility_manager
    if _global_compatibility_manager is None:
        _global_compatibility_manager = CompatibilityManager()
    return _global_compatibility_manager


def set_compatibility_manager(manager: CompatibilityManager):
    """设置全局兼容性管理器实例"""
    global _global_compatibility_manager
    _global_compatibility_manager = manager


def create_compatible_network(*args, **kwargs):
    """使用全局兼容性管理器创建网络的便捷函数"""
    return get_compatibility_manager().create_network(*args, **kwargs)


def create_compatible_environment(*args, **kwargs):
    """使用全局兼容性管理器创建环境的便捷函数"""
    return get_compatibility_manager().create_environment(*args, **kwargs)


def create_compatible_solver(*args, **kwargs):
    """使用全局兼容性管理器创建求解器的便捷函数"""
    return get_compatibility_manager().create_solver(*args, **kwargs)