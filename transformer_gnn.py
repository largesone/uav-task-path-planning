# -*- coding: utf-8 -*-
# 文件名: transformer_gnn.py
# 描述: TransformerGNN网络架构实现，支持相对位置编码机制和参数空间噪声探索

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

# 导入NoisyLinear相关功能
import sys
import os
from local_attention import LocalAttention, MultiScaleLocalAttention

# 添加temp_tests目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
temp_tests_dir = os.path.join(current_dir, 'temp_tests')
if os.path.exists(temp_tests_dir) and temp_tests_dir not in sys.path:
    sys.path.insert(0, temp_tests_dir)

try:
    from noisy_linear import NoisyLinear, replace_linear_with_noisy, reset_noise_in_module
except ImportError as e:
    print(f"警告：无法导入NoisyLinear: {e}")
    # 创建占位符类以避免错误
    class NoisyLinear(nn.Linear):
        pass
    def replace_linear_with_noisy(module, std_init=0.5):
        return module
    def reset_noise_in_module(module):
        pass


class PositionalEncoder(nn.Module):
    """相对位置编码器
    
    将relative_positions通过小型MLP生成位置嵌入，解决排列不变性被破坏的问题
    """
    
    def __init__(self, position_dim: int = 2, embed_dim: int = 64, hidden_dim: int = 32):
        """
        初始化位置编码器
        
        Args:
            position_dim: 位置向量维度（通常为2，表示x,y坐标）
            embed_dim: 输出嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(PositionalEncoder, self).__init__()
        
        self.position_dim = position_dim
        self.embed_dim = embed_dim
        
        # 小型MLP用于生成位置嵌入
        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 较小的初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            relative_positions: 相对位置张量，形状为 [batch_size, num_pairs, position_dim]
            
        Returns:
            位置嵌入张量，形状为 [batch_size, num_pairs, embed_dim]
        """
        # 通过MLP生成位置嵌入
        position_embeddings = self.position_mlp(relative_positions)
        
        return position_embeddings


class TransformerGNN(TorchModelV2, nn.Module):
    """
    TransformerGNN网络架构 - 支持零样本迁移的局部注意力Transformer网络
    
    核心特性：
    1. 继承Ray RLlib的TorchModelV2，完全兼容RLlib训练框架
    2. 支持图模式和扁平模式双输入格式
    3. 实体编码器分别处理UAV和目标特征
    4. 相对位置编码解决排列不变性问题
    5. 参数空间噪声探索机制
    6. 局部注意力机制避免维度爆炸
    
    设计理念：
    - 尺度不变：支持任意数量的UAV和目标
    - 零样本迁移：训练好的模型可直接应用于不同规模场景
    - 工程化：完全集成Ray RLlib生态系统
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        初始化TransformerGNN网络
        
        Args:
            obs_space: 观测空间，支持Box（扁平模式）和Dict（图模式）
            action_space: 动作空间
            num_outputs: 输出维度（动作数量）
            model_config: 模型配置字典
            name: 模型名称
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # 从配置中获取参数
        self.embed_dim = model_config.get("embed_dim", 128)
        self.num_heads = model_config.get("num_heads", 8)
        self.num_layers = model_config.get("num_layers", 3)
        self.dropout = model_config.get("dropout", 0.1)
        self.use_position_encoding = model_config.get("use_position_encoding", True)
        self.use_noisy_linear = model_config.get("use_noisy_linear", True)
        self.noisy_std_init = model_config.get("noisy_std_init", 0.5)
        
        # 局部注意力机制配置
        self.use_local_attention = model_config.get("use_local_attention", True)
        self.k_adaptive = model_config.get("k_adaptive", True)
        self.k_fixed = model_config.get("k_fixed", None)
        self.k_min = model_config.get("k_min", 4)
        self.k_max = model_config.get("k_max", 16)
        self.use_flash_attention = model_config.get("use_flash_attention", True)
        
        print(f"[TransformerGNN] 局部注意力配置: use_local_attention={self.use_local_attention}, k_adaptive={self.k_adaptive}, k_min={self.k_min}, k_max={self.k_max}")
        
        # 观测空间处理
        if hasattr(obs_space, 'original_space'):
            # 处理Dict观测空间
            self.obs_space_dict = obs_space.original_space
            self.is_dict_obs = True
            self.input_dim = None
        elif isinstance(obs_space, Dict) or hasattr(obs_space, 'spaces'):
            # 直接是Dict观测空间
            self.obs_space_dict = obs_space
            self.is_dict_obs = True
            self.input_dim = None
        elif hasattr(obs_space, 'shape'):
            # 处理Box观测空间
            self.obs_space_dict = None
            self.is_dict_obs = False
            self.input_dim = obs_space.shape[0]
        else:
            # 默认处理
            self.obs_space_dict = None
            self.is_dict_obs = False
            self.input_dim = 128  # 默认值
        
        # === 实体编码器架构设计 ===
        # 根据观测模式确定特征维度
        if self.is_dict_obs:
            # 图模式：从观测空间字典中获取精确的特征维度
            uav_features_dim = self.obs_space_dict['uav_features'].shape[-1]
            target_features_dim = self.obs_space_dict['target_features'].shape[-1]
            print(f"[TransformerGNN] 图模式初始化 - UAV特征维度: {uav_features_dim}, 目标特征维度: {target_features_dim}")
        else:
            # 扁平模式：假设观测向量的一半是UAV特征，一半是目标特征
            uav_features_dim = self.input_dim // 2
            target_features_dim = self.input_dim // 2
            print(f"[TransformerGNN] 扁平模式初始化 - 输入维度: {self.input_dim}, UAV/目标特征维度: {uav_features_dim}/{target_features_dim}")
        
        # UAV实体编码器 - 多层架构处理UAV特征
        self.uav_encoder = nn.Sequential(
            nn.Linear(uav_features_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 目标实体编码器 - 多层架构处理目标特征
        self.target_encoder = nn.Sequential(
            nn.Linear(target_features_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 位置编码器
        if self.use_position_encoding:
            self.position_encoder = PositionalEncoder(
                position_dim=2,  # x, y坐标
                embed_dim=self.embed_dim,
                hidden_dim=64
            )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, num_outputs)
        )
        
        # 值函数头
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, 1)
        )
        
        # 局部注意力机制
        if self.use_local_attention:
            self.local_attention = LocalAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                k_adaptive=self.k_adaptive,
                k_fixed=self.k_fixed,
                k_min=self.k_min,
                k_max=self.k_max,
                use_flash_attention=self.use_flash_attention
            )
            print(f"[TransformerGNN] 局部注意力机制已启用")
        else:
            self.local_attention = None
            print(f"[TransformerGNN] 局部注意力机制已禁用")



        # 初始化权重
        self._init_weights()
        
        # 将所有Linear层替换为NoisyLinear层（如果启用）
        if self.use_noisy_linear:
            self._replace_with_noisy_linear()
        
        # 存储最后的值函数输出
        self._last_value = None
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)
    
    def _replace_with_noisy_linear(self):
        """将所有Linear层替换为NoisyLinear层"""
        # 替换实体编码器中的Linear层
        self.uav_encoder = self._convert_linear_to_noisy(self.uav_encoder)
        self.target_encoder = self._convert_linear_to_noisy(self.target_encoder)
        
        # 替换位置编码器中的Linear层
        if self.use_position_encoding:
            self.position_encoder = replace_linear_with_noisy(
                self.position_encoder, self.noisy_std_init
            )
        
        # 替换输出层中的Linear层
        self.output_layer = replace_linear_with_noisy(
            self.output_layer, self.noisy_std_init
        )
        
        # 替换值函数头中的Linear层
        self.value_head = replace_linear_with_noisy(
            self.value_head, self.noisy_std_init
        )
        
        # 替换局部注意力中的Linear层
        if self.use_local_attention and self.local_attention is not None:
            self.local_attention = replace_linear_with_noisy(
                self.local_attention, self.noisy_std_init
            )
        
        # 注意：Transformer编码器层内部的Linear层由PyTorch管理，
        # 我们需要手动替换其中的Linear层
        self._replace_transformer_linear_layers()
    
    def _convert_linear_to_noisy(self, layer):
        """将单个Linear层转换为NoisyLinear层"""
        if isinstance(layer, nn.Linear):
            noisy_layer = NoisyLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                std_init=self.noisy_std_init
            )
            
            # 复制原始权重和偏置
            with torch.no_grad():
                noisy_layer.weight_mu.copy_(layer.weight)
                if layer.bias is not None:
                    noisy_layer.bias_mu.copy_(layer.bias)
            
            return noisy_layer
        else:
            return layer
    
    def _replace_transformer_linear_layers(self):
        """替换Transformer编码器中的Linear层"""
        # 注意：由于PyTorch的TransformerEncoderLayer内部结构复杂，
        # 并且替换其内部Linear层可能导致兼容性问题，
        # 我们暂时跳过Transformer内部层的替换，只替换我们自定义的层
        pass
    
    def reset_noise(self):
        """重置所有NoisyLinear层的噪声"""
        if self.use_noisy_linear:
            reset_noise_in_module(self)
            # 重置局部注意力中的噪声
            if self.use_local_attention and self.local_attention is not None:
                reset_noise_in_module(self.local_attention)
    
    def _extract_features_from_dict_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        从字典观测中提取特征 - 图模式输入解析和预处理
        
        处理图结构观测，包括：
        1. UAV和目标实体特征提取
        2. 相对位置信息处理
        3. 距离矩阵处理
        4. 掩码信息处理（用于鲁棒性）
        
        Args:
            obs_dict: 字典形式的观测，包含以下键：
                - 'uav_features': [batch_size, num_uavs, uav_feature_dim]
                - 'target_features': [batch_size, num_targets, target_feature_dim]
                - 'relative_positions': [batch_size, num_uavs, num_targets, 2]
                - 'distances': [batch_size, num_uavs, num_targets]
                - 'masks': {'uav_mask': [...], 'target_mask': [...]}
            
        Returns:
            uav_features: UAV特征张量 [batch_size, num_uavs, uav_feature_dim]
            target_features: 目标特征张量 [batch_size, num_targets, target_feature_dim]
            relative_positions: 相对位置张量 [batch_size, num_uavs, num_targets, 2]
            additional_info: 包含距离和掩码的额外信息字典
        """
        # 提取基本特征
        uav_features = obs_dict['uav_features']
        target_features = obs_dict['target_features']
        
        # 提取相对位置信息
        relative_positions = None
        if 'relative_positions' in obs_dict and self.use_position_encoding:
            relative_positions = obs_dict['relative_positions']
            print(f"[TransformerGNN] 提取相对位置信息，形状: {relative_positions.shape}")
        
        # 提取额外信息（距离矩阵和掩码）
        additional_info = {}
        
        # 距离矩阵 - 用于局部注意力机制
        if 'distances' in obs_dict:
            additional_info['distances'] = obs_dict['distances']
            print(f"[TransformerGNN] 提取距离矩阵，形状: {obs_dict['distances'].shape}")
        
        # 掩码信息 - 用于鲁棒性处理
        if 'masks' in obs_dict:
            additional_info['masks'] = obs_dict['masks']
            if 'uav_mask' in obs_dict['masks']:
                print(f"[TransformerGNN] 提取UAV掩码，形状: {obs_dict['masks']['uav_mask'].shape}")
            if 'target_mask' in obs_dict['masks']:
                print(f"[TransformerGNN] 提取目标掩码，形状: {obs_dict['masks']['target_mask'].shape}")
        
        print(f"[TransformerGNN] 图模式特征提取完成 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        return uav_features, target_features, relative_positions, additional_info
    
    def _extract_features_from_flat_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        从扁平观测中提取特征 - 扁平模式输入解析和预处理
        
        处理传统的扁平向量观测，确保向后兼容性：
        1. 将扁平向量分割为UAV和目标特征
        2. 尝试从特征中推断位置信息
        3. 生成默认的掩码信息
        
        Args:
            obs: 扁平观测张量 [batch_size, input_dim]
            
        Returns:
            uav_features: UAV特征张量 [batch_size, 1, feature_dim]
            target_features: 目标特征张量 [batch_size, 1, feature_dim]
            relative_positions: 相对位置张量（None或推断的位置）
            additional_info: 包含默认掩码的额外信息字典
        """
        batch_size = obs.shape[0]
        
        # 将观测分为UAV和目标特征
        split_point = self.input_dim // 2
        uav_features = obs[:, :split_point].unsqueeze(1)  # [batch_size, 1, feature_dim]
        target_features = obs[:, split_point:].unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        print(f"[TransformerGNN] 扁平模式特征分割 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        # 尝试从特征中推断相对位置（假设前两个维度是位置）
        relative_positions = None
        if self.use_position_encoding and uav_features.shape[-1] >= 2 and target_features.shape[-1] >= 2:
            relative_positions = self._compute_relative_positions(uav_features, target_features)
            print(f"[TransformerGNN] 从扁平特征推断相对位置，形状: {relative_positions.shape}")
        
        # 生成默认的额外信息（全部有效的掩码）
        additional_info = {
            'masks': {
                'uav_mask': torch.ones(batch_size, 1, dtype=torch.bool, device=obs.device),
                'target_mask': torch.ones(batch_size, 1, dtype=torch.bool, device=obs.device)
            }
        }
        
        return uav_features, target_features, relative_positions, additional_info
    
    def _compute_relative_positions(self, uav_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        从特征中计算相对位置（如果特征包含位置信息）
        
        Args:
            uav_features: UAV特征张量 [batch_size, num_uavs, feature_dim]
            target_features: 目标特征张量 [batch_size, num_targets, feature_dim]
            
        Returns:
            相对位置张量 [batch_size, num_uavs * num_targets, 2]
        """
        # 假设特征的前两个维度是位置坐标
        if uav_features.shape[-1] >= 2 and target_features.shape[-1] >= 2:
            uav_positions = uav_features[..., :2]  # [batch_size, num_uavs, 2]
            target_positions = target_features[..., :2]  # [batch_size, num_targets, 2]
            
            # 计算所有UAV-目标对的相对位置
            batch_size, num_uavs, _ = uav_positions.shape
            _, num_targets, _ = target_positions.shape
            
            # 扩展维度进行广播
            uav_pos_expanded = uav_positions.unsqueeze(2)  # [batch_size, num_uavs, 1, 2]
            target_pos_expanded = target_positions.unsqueeze(1)  # [batch_size, 1, num_targets, 2]
            
            # 计算相对位置 (target_pos - uav_pos)
            relative_positions = target_pos_expanded - uav_pos_expanded  # [batch_size, num_uavs, num_targets, 2]
            
            # 重塑为 [batch_size, num_pairs, 2]
            relative_positions = relative_positions.view(batch_size, num_uavs * num_targets, 2)
            
            return relative_positions
        else:
            # 如果特征不包含位置信息，返回零向量
            batch_size = uav_features.shape[0]
            num_uavs = uav_features.shape[1]
            num_targets = target_features.shape[1]
            return torch.zeros(batch_size, num_uavs * num_targets, 2, device=uav_features.device)
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        前向传播
        
        Args:
            input_dict: 输入字典，包含观测
            state: RNN状态（未使用）
            seq_lens: 序列长度（未使用）
            
        Returns:
            logits: 动作logits
            state: 更新后的状态
        """
        obs = input_dict["obs"]
        
        # 在训练模式下重置噪声（每次前向传播时）
        if self.training and self.use_noisy_linear:
            self.reset_noise()
        
        # === 图模式输入解析和预处理 ===
        if self.is_dict_obs:
            uav_features, target_features, relative_positions, additional_info = self._extract_features_from_dict_obs(obs)
        else:
            uav_features, target_features, relative_positions, additional_info = self._extract_features_from_flat_obs(obs)
        
        # 获取批次大小和实体数量信息
        batch_size = uav_features.shape[0]
        num_uavs = uav_features.shape[1]
        num_targets = target_features.shape[1]
        
        print(f"[TransformerGNN] 前向传播开始 - 批次: {batch_size}, UAV数量: {num_uavs}, 目标数量: {num_targets}")
        
        # === 实体特征编码 ===
        # 使用专门的实体编码器分别处理UAV和目标特征
        uav_embeddings = self.uav_encoder(uav_features)  # [batch_size, num_uavs, embed_dim]
        target_embeddings = self.target_encoder(target_features)  # [batch_size, num_targets, embed_dim]
        
        print(f"[TransformerGNN] 实体编码完成 - UAV嵌入: {uav_embeddings.shape}, 目标嵌入: {target_embeddings.shape}")
        
        # 应用掩码处理（鲁棒性机制）
        if 'masks' in additional_info:
            masks = additional_info['masks']
            
            # 应用UAV掩码
            if 'uav_mask' in masks:
                uav_mask = masks['uav_mask'].unsqueeze(-1)  # [batch_size, num_uavs, 1]
                uav_embeddings = uav_embeddings * uav_mask.float()
                print(f"[TransformerGNN] 应用UAV掩码，有效UAV比例: {uav_mask.float().mean().item():.3f}")
            
            # 应用目标掩码
            if 'target_mask' in masks:
                target_mask = masks['target_mask'].unsqueeze(-1)  # [batch_size, num_targets, 1]
                target_embeddings = target_embeddings * target_mask.float()
                print(f"[TransformerGNN] 应用目标掩码，有效目标比例: {target_mask.float().mean().item():.3f}")
        
        # === 局部注意力机制处理 ===
        if self.use_local_attention and self.local_attention is not None and 'distances' in additional_info:
            distances = additional_info['distances']
            print(f"[TransformerGNN] 应用局部注意力机制")
            
            # 应用局部注意力到UAV嵌入
            uav_attention_output = self.local_attention(
                uav_embeddings, target_embeddings, distances, additional_info.get('masks')
            )
            
            # 将注意力输出与原始嵌入结合（残差连接）
            uav_embeddings = uav_embeddings + uav_attention_output
            print(f"[TransformerGNN] 局部注意力应用完成，UAV嵌入已更新")
        else:
            print(f"[TransformerGNN] 跳过局部注意力（未启用或缺少距离信息）")
        
        # 合并实体嵌入
        entity_embeddings = torch.cat([uav_embeddings, target_embeddings], dim=1)  # [batch_size, num_entities, embed_dim]
        print(f"[TransformerGNN] 实体嵌入合并完成，总实体数: {entity_embeddings.shape[1]}")
        
        # === 相对位置编码处理 ===
        if self.use_position_encoding:
            if relative_positions is None:
                # 如果没有提供相对位置，尝试从特征中计算
                relative_positions = self._compute_relative_positions(uav_features, target_features)
                print(f"[TransformerGNN] 从特征计算相对位置，形状: {relative_positions.shape}")
            else:
                # 处理从观测中提供的相对位置
                # 输入形状可能是 [batch_size, num_uavs, num_targets, 2]
                if len(relative_positions.shape) == 4:
                    # 保持4D形状用于更精确的位置编码
                    print(f"[TransformerGNN] 使用4D相对位置，形状: {relative_positions.shape}")
                elif len(relative_positions.shape) == 3:
                    # 重塑为4D形状 [batch_size, num_uavs, num_targets, 2]
                    if relative_positions.shape[1] == num_uavs * num_targets:
                        relative_positions = relative_positions.view(batch_size, num_uavs, num_targets, -1)
                        print(f"[TransformerGNN] 重塑相对位置为4D，新形状: {relative_positions.shape}")
            
            if relative_positions is not None:
                # 处理4D相对位置编码
                if len(relative_positions.shape) == 4:
                    batch_size_pos, num_uavs_pos, num_targets_pos, pos_dim = relative_positions.shape
                    
                    # 为每个UAV-目标对生成独立的位置嵌入
                    # 重塑为 [batch_size * num_uavs * num_targets, pos_dim]
                    rel_pos_flat = relative_positions.view(-1, pos_dim)
                    pos_emb_flat = self.position_encoder(rel_pos_flat)  # [batch_size * num_pairs, embed_dim]
                    
                    # 重塑回 [batch_size, num_uavs, num_targets, embed_dim]
                    pos_emb_4d = pos_emb_flat.view(batch_size_pos, num_uavs_pos, num_targets_pos, self.embed_dim)
                    print(f"[TransformerGNN] 4D位置编码生成，形状: {pos_emb_4d.shape}")
                    
                    # 为每个UAV创建位置感知的嵌入
                    # 每个UAV的嵌入会根据其与不同目标的相对位置进行调整
                    uav_embeddings_with_pos = uav_embeddings.clone()
                    for i in range(num_uavs_pos):
                        # 对第i个UAV，使用其与所有目标的位置嵌入的加权平均
                        uav_pos_emb = pos_emb_4d[:, i, :, :].mean(dim=1)  # [batch_size, embed_dim]
                        uav_embeddings_with_pos[:, i, :] = uav_embeddings[:, i, :] + uav_pos_emb
                    
                    # 为每个目标创建位置感知的嵌入
                    target_embeddings_with_pos = target_embeddings.clone()
                    for j in range(num_targets_pos):
                        # 对第j个目标，使用其与所有UAV的位置嵌入的加权平均
                        target_pos_emb = pos_emb_4d[:, :, j, :].mean(dim=1)  # [batch_size, embed_dim]
                        target_embeddings_with_pos[:, j, :] = target_embeddings[:, j, :] + target_pos_emb
                    
                    # 重新合并
                    entity_embeddings = torch.cat([uav_embeddings_with_pos, target_embeddings_with_pos], dim=1)
                    print(f"[TransformerGNN] 4D位置编码已添加到实体嵌入")
                    
                else:
                    # 处理3D相对位置编码（备用方案）
                    position_embeddings = self.position_encoder(relative_positions)  # [batch_size, num_pairs, embed_dim]
                    print(f"[TransformerGNN] 3D位置编码生成，形状: {position_embeddings.shape}")
                    
                    # 将位置嵌入分配到对应的实体嵌入上
                    if position_embeddings.shape[1] == num_uavs * num_targets:
                        # 重塑位置嵌入为 [batch_size, num_uavs, num_targets, embed_dim]
                        pos_emb_reshaped = position_embeddings.view(batch_size, num_uavs, num_targets, self.embed_dim)
                        
                        # 对每个UAV，取其与所有目标的位置嵌入的平均值
                        uav_pos_emb = pos_emb_reshaped.mean(dim=2)  # [batch_size, num_uavs, embed_dim]
                        
                        # 对每个目标，取其与所有UAV的位置嵌入的平均值
                        target_pos_emb = pos_emb_reshaped.mean(dim=1)  # [batch_size, num_targets, embed_dim]
                        
                        # 将位置嵌入加到实体嵌入上
                        uav_embeddings_with_pos = uav_embeddings + uav_pos_emb
                        target_embeddings_with_pos = target_embeddings + target_pos_emb
                        
                        # 重新合并
                        entity_embeddings = torch.cat([uav_embeddings_with_pos, target_embeddings_with_pos], dim=1)
                        print(f"[TransformerGNN] 3D位置编码已添加到实体嵌入")
                    else:
                        # 备用方案：使用平均位置嵌入
                        avg_position_embedding = position_embeddings.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
                        entity_embeddings = entity_embeddings + avg_position_embedding
                        print(f"[TransformerGNN] 使用平均位置编码")
            else:
                print(f"[TransformerGNN] 跳过位置编码（无位置信息）")
        
        # === Transformer编码 ===
        transformer_output = self.transformer_encoder(entity_embeddings)  # [batch_size, num_entities, embed_dim]
        print(f"[TransformerGNN] Transformer编码完成，输出形状: {transformer_output.shape}")
        
        # === 全局池化和输出生成 ===
        # 使用平均池化聚合所有实体的信息
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, embed_dim]
        print(f"[TransformerGNN] 全局池化完成，形状: {pooled_output.shape}")
        
        # 生成动作logits
        logits = self.output_layer(pooled_output)
        print(f"[TransformerGNN] 动作logits生成，形状: {logits.shape}")
        
        # 计算值函数
        self._last_value = self.value_head(pooled_output).squeeze(-1)
        print(f"[TransformerGNN] 值函数计算完成，形状: {self._last_value.shape}")
        
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        """返回值函数输出"""
        return self._last_value


def create_transformer_gnn_model(obs_space, action_space, num_outputs, model_config, name="TransformerGNN"):
    """
    创建TransformerGNN模型的工厂函数
    
    Args:
        obs_space: 观测空间
        action_space: 动作空间
        num_outputs: 输出维度
        model_config: 模型配置
        name: 模型名称
        
    Returns:
        TransformerGNN模型实例
    """
    return TransformerGNN(obs_space, action_space, num_outputs, model_config, name)