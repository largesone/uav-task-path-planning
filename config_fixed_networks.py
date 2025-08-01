# -*- coding: utf-8 -*-
# 文件名: config_fixed_networks.py
# 描述: 修复版网络的配置文件，提供不同场景的配置模板

from typing import Dict, Any, List

class FixedNetworkConfig:
    """修复版网络配置类"""
    
    @staticmethod
    def get_semantic_extraction_config() -> Dict[str, Any]:
        """
        语义特征提取配置 - 推荐用于生产环境
        
        基于环境状态结构的精确特征分割
        """
        return {
            # 特征提取策略
            'extraction_strategy': 'semantic',
            
            # 环境配置
            'n_uavs': 3,
            'n_targets': 5,
            'uav_features_per_entity': 8,  # position(2) + heading(1) + resources(2) + max_distance(1) + velocity_range(2)
            'target_features_per_entity': 7,  # position(2) + resources(2) + value(1) + remaining_resources(2)
            
            # 网络架构
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'num_attention_layers': 2,
            
            # 位置编码
            'use_position_encoding': True,
            'max_distance': 1000.0,
            'num_distance_bins': 32,
            'num_angle_bins': 16,
            
            # 局部注意力
            'use_local_attention': True,
            'k_adaptive': True,
            'k_min': 4,
            'k_max': 16,
            'use_flash_attention': True,
            
            # 噪声探索
            'use_noisy_linear': True,
            'noisy_std_init': 0.5
        }
    
    @staticmethod
    def get_ratio_extraction_config() -> Dict[str, Any]:
        """
        比例特征提取配置 - 适用于快速原型开发
        
        改进的对半切分，支持不等比例
        """
        return {
            # 特征提取策略
            'extraction_strategy': 'ratio',
            'target_feature_ratio': 0.6,  # 目标特征占60%，UAV特征占40%
            
            # 网络架构
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 2,
            'dropout': 0.1,
            'num_attention_layers': 2,
            
            # 位置编码
            'use_position_encoding': True,
            'max_distance': 1000.0,
            'num_distance_bins': 16,
            'num_angle_bins': 8,
            
            # 局部注意力
            'use_local_attention': False,  # 简化配置
            
            # 噪声探索
            'use_noisy_linear': False  # 简化配置
        }
    
    @staticmethod
    def get_fixed_dimension_config() -> Dict[str, Any]:
        """
        固定维度特征提取配置 - 适用于特定场景优化
        
        使用预定义的特征维度
        """
        return {
            # 特征提取策略
            'extraction_strategy': 'fixed',
            'uav_feature_dim': 64,
            'target_feature_dim': 64,
            
            # 网络架构
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'num_attention_layers': 3,
            
            # 位置编码
            'use_position_encoding': True,
            'max_distance': 1000.0,
            'num_distance_bins': 32,
            'num_angle_bins': 16,
            
            # 局部注意力
            'use_local_attention': True,
            'k_adaptive': False,
            'k_fixed': 8,
            'use_flash_attention': True,
            
            # 噪声探索
            'use_noisy_linear': True,
            'noisy_std_init': 0.3
        }
    
    @staticmethod
    def get_large_scale_config() -> Dict[str, Any]:
        """
        大规模场景配置 - 适用于大量UAV和目标的场景
        
        优化内存使用和计算效率
        """
        return {
            # 特征提取策略
            'extraction_strategy': 'semantic',
            
            # 大规模环境配置
            'n_uavs': 10,
            'n_targets': 20,
            'uav_features_per_entity': 8,
            'target_features_per_entity': 7,
            
            # 网络架构 - 适度减小以控制内存
            'embedding_dim': 96,
            'num_heads': 6,
            'num_layers': 2,
            'dropout': 0.15,
            'num_attention_layers': 2,
            
            # 位置编码 - 减少分箱数量
            'use_position_encoding': True,
            'max_distance': 2000.0,
            'num_distance_bins': 16,
            'num_angle_bins': 8,
            
            # 局部注意力 - 关键优化
            'use_local_attention': True,
            'k_adaptive': True,
            'k_min': 6,
            'k_max': 12,
            'use_flash_attention': True,
            
            # 噪声探索
            'use_noisy_linear': False  # 大规模场景下禁用以提高稳定性
        }
    
    @staticmethod
    def get_small_scale_config() -> Dict[str, Any]:
        """
        小规模场景配置 - 适用于少量UAV和目标的场景
        
        最大化模型表达能力
        """
        return {
            # 特征提取策略
            'extraction_strategy': 'semantic',
            
            # 小规模环境配置
            'n_uavs': 2,
            'n_targets': 3,
            'uav_features_per_entity': 8,
            'target_features_per_entity': 7,
            
            # 网络架构 - 增大以提高表达能力
            'embedding_dim': 256,
            'num_heads': 16,
            'num_layers': 4,
            'dropout': 0.05,
            'num_attention_layers': 3,
            
            # 位置编码 - 高精度
            'use_position_encoding': True,
            'max_distance': 1000.0,
            'num_distance_bins': 64,
            'num_angle_bins': 32,
            
            # 局部注意力 - 可以使用全注意力
            'use_local_attention': False,
            
            # 噪声探索
            'use_noisy_linear': True,
            'noisy_std_init': 0.4
        }
    
    @staticmethod
    def get_rllib_integration_config() -> Dict[str, Any]:
        """
        RLlib集成配置 - 专门用于与RLlib框架集成
        
        确保与RLlib的完全兼容性
        """
        return {
            # 特征提取策略
            'extraction_strategy': 'semantic',
            
            # 环境配置
            'n_uavs': 3,
            'n_targets': 4,
            'uav_features_per_entity': 8,
            'target_features_per_entity': 7,
            
            # 网络架构 - RLlib友好配置
            'embed_dim': 128,  # 注意：RLlib使用embed_dim而不是embedding_dim
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            
            # 位置编码
            'use_position_encoding': True,
            'max_distance': 1000.0,
            'num_distance_bins': 32,
            'num_angle_bins': 16,
            
            # 局部注意力
            'use_local_attention': True,
            'k_adaptive': True,
            'k_min': 4,
            'k_max': 16,
            'use_flash_attention': True,
            
            # 噪声探索
            'use_noisy_linear': True,
            'noisy_std_init': 0.5,
            
            # RLlib特定配置
            'fcnet_hiddens': [256, 256],  # RLlib标准配置
            'fcnet_activation': 'relu',
            'use_lstm': False,
            'lstm_cell_size': 256
        }
    
    @staticmethod
    def get_config_by_scenario(scenario: str) -> Dict[str, Any]:
        """
        根据场景名称获取配置
        
        Args:
            scenario: 场景名称
            
        Returns:
            配置字典
        """
        config_map = {
            'semantic': FixedNetworkConfig.get_semantic_extraction_config,
            'ratio': FixedNetworkConfig.get_ratio_extraction_config,
            'fixed': FixedNetworkConfig.get_fixed_dimension_config,
            'large_scale': FixedNetworkConfig.get_large_scale_config,
            'small_scale': FixedNetworkConfig.get_small_scale_config,
            'rllib': FixedNetworkConfig.get_rllib_integration_config
        }
        
        if scenario not in config_map:
            raise ValueError(f"未知场景: {scenario}。支持的场景: {list(config_map.keys())}")
        
        return config_map[scenario]()
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
            
        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []
        
        # 检查必需的配置项
        required_keys = ['extraction_strategy', 'embedding_dim', 'num_heads', 'dropout']
        for key in required_keys:
            if key not in config:
                errors.append(f"缺少必需配置项: {key}")
        
        # 检查特征提取策略
        if 'extraction_strategy' in config:
            valid_strategies = ['semantic', 'ratio', 'fixed']
            if config['extraction_strategy'] not in valid_strategies:
                errors.append(f"无效的特征提取策略: {config['extraction_strategy']}。支持: {valid_strategies}")
        
        # 检查语义策略的特定配置
        if config.get('extraction_strategy') == 'semantic':
            semantic_keys = ['n_uavs', 'n_targets', 'uav_features_per_entity', 'target_features_per_entity']
            for key in semantic_keys:
                if key not in config:
                    errors.append(f"语义策略缺少配置项: {key}")
        
        # 检查固定维度策略的特定配置
        if config.get('extraction_strategy') == 'fixed':
            fixed_keys = ['uav_feature_dim', 'target_feature_dim']
            for key in fixed_keys:
                if key not in config:
                    errors.append(f"固定维度策略缺少配置项: {key}")
        
        # 检查数值范围
        if 'dropout' in config:
            if not 0 <= config['dropout'] <= 1:
                errors.append(f"dropout必须在[0,1]范围内，当前值: {config['dropout']}")
        
        if 'num_heads' in config and 'embedding_dim' in config:
            if config['embedding_dim'] % config['num_heads'] != 0:
                errors.append(f"embedding_dim ({config['embedding_dim']}) 必须能被 num_heads ({config['num_heads']}) 整除")
        
        return errors


# 预定义配置实例
SEMANTIC_CONFIG = FixedNetworkConfig.get_semantic_extraction_config()
RATIO_CONFIG = FixedNetworkConfig.get_ratio_extraction_config()
FIXED_CONFIG = FixedNetworkConfig.get_fixed_dimension_config()
LARGE_SCALE_CONFIG = FixedNetworkConfig.get_large_scale_config()
SMALL_SCALE_CONFIG = FixedNetworkConfig.get_small_scale_config()
RLLIB_CONFIG = FixedNetworkConfig.get_rllib_integration_config()
