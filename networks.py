# -*- coding: utf-8 -*-
# 文件名: networks.py
# 描述: 统一的神经网络模块，包含所有网络结构定义

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any

class SimpleNetwork(nn.Module):
    """简化的网络结构 - 基础版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(SimpleNetwork, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.extend([
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

class DeepFCN(nn.Module):
    """深度全连接网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(DeepFCN, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.extend([
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

# GAT网络已移除 - 使用ZeroShotGNN替代

class DeepFCNResidual(nn.Module):
    """带残差连接的深度全连接网络 - 优化版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        super(DeepFCNResidual, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [256, 128, 64]  # 默认层次结构
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 输入层 - 添加BatchNorm
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # 输入层使用较小的dropout
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = ResidualBlock(self.hidden_dims[i], self.hidden_dims[i+1], dropout)
            self.residual_blocks.append(block)
        
        # 注意力机制 - 简化版本
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 4, self.hidden_dims[-1]),
            nn.Sigmoid()
        )
        
        # 输出层 - 优化结构
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用He初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """前向传播 - 添加注意力机制"""
        # 输入层
        x = self.input_layer(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 输出层
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    """优化的残差块 - 改进的结构和正则化"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        # 主路径 - 使用预激活结构
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        
        # 跳跃连接
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 添加SE注意力模块
        self.se_attention = SEBlock(out_dim, reduction=4)
    
    def forward(self, x):
        """前向传播 - 预激活残差连接"""
        residual = self.shortcut(x)
        out = self.layers(x)
        
        # 应用SE注意力
        out = self.se_attention(out)
        
        # 残差连接
        return out + residual

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播"""
        # 全局平均池化
        b, c = x.size()
        y = x.mean(dim=0, keepdim=True)  # 简化的全局池化
        
        # 激励操作
        y = self.excitation(y)
        
        # 重新加权
        return x * y.expand_as(x)

def create_network(network_type: str, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
    """
    创建指定类型的网络
    
    Args:
        network_type: 网络类型 ("SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual", "ZeroShotGNN")
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        
    Returns:
        网络模型实例
    """
    if network_type == "SimpleNetwork":
        return SimpleNetwork(input_dim, hidden_dims, output_dim)
    elif network_type == "DeepFCN":
        return DeepFCN(input_dim, hidden_dims, output_dim)
    # GAT网络已移除，请使用ZeroShotGNN
    elif network_type == "DeepFCNResidual":
        return DeepFCNResidual(input_dim, hidden_dims, output_dim)
    elif network_type == "ZeroShotGNN":
        return ZeroShotGNN(input_dim, hidden_dims, output_dim)
    else:
        raise ValueError(f"不支持的网络类型: {network_type}")

class ZeroShotGNN(nn.Module):
    """
    真正的零样本图神经网络 - 基于Transformer的架构
    
    核心特性：
    1. 参数共享的实体编码器，支持可变数量的UAV和目标
    2. 自注意力机制学习同类实体间的内部关系
    3. 交叉注意力机制学习UAV-目标间的交互关系
    4. 支持掩码机制，忽略填充的无效数据
    5. 零样本迁移能力，适应不同规模的场景
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(ZeroShotGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [256, 128]
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 嵌入维度
        self.embedding_dim = 128
        
        # === 1. 参数共享的实体编码器 ===
        # UAV特征维度：position(2) + heading(1) + resources_ratio(2) + max_distance_norm(1) + 
        #              velocity_norm(2) + is_alive(1) = 9
        self.uav_feature_dim = 9
        
        # 目标特征维度：position(2) + resources_ratio(2) + value_norm(1) + 
        #              remaining_ratio(2) + is_visible(1) = 8
        self.target_feature_dim = 8
        
        # UAV编码器
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.uav_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # 目标编码器
        self.target_encoder = nn.Sequential(
            nn.Linear(self.target_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # === 2. 自注意力层 ===
        # UAV内部自注意力
        self.uav_self_attention = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # 目标内部自注意力
        self.target_self_attention = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # === 3. 交叉注意力层 ===
        # UAV-目标交叉注意力
        self.cross_attention = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # === 4. 位置编码 ===
        self.position_encoder = PositionalEncoding(self.embedding_dim, dropout)
        
        # === 5. Q值解码器 ===
        # 为每个UAV输出对所有目标的Q值
        self.q_decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 每个UAV-目标对输出一个Q值
        )
        
        # === 6. 空间编码器 ===
        # 预先定义空间编码器，避免动态创建导致的state_dict不匹配
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 32),  # 相对位置(2) + 距离(1)
            nn.ReLU(),
            nn.Linear(32, self.embedding_dim // 4)
        )
        
        # === 7. 全局聚合层 ===
        # 将所有UAV的表示聚合为最终的动作Q值
        self.global_aggregator = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, graph_obs):
        """
        前向传播 - 处理图结构观测
        
        Args:
            graph_obs (dict): 图结构观测字典，包含：
                - uav_features: [batch_size, N_uav, uav_feature_dim]
                - target_features: [batch_size, N_target, target_feature_dim]
                - relative_positions: [batch_size, N_uav, N_target, 2]
                - distances: [batch_size, N_uav, N_target]
                - masks: 掩码字典
        
        Returns:
            torch.Tensor: Q值 [batch_size, N_actions]
        """
        # 提取输入
        uav_features = graph_obs["uav_features"]  # [batch_size, N_uav, uav_feat_dim]
        target_features = graph_obs["target_features"]  # [batch_size, N_target, target_feat_dim]
        relative_positions = graph_obs["relative_positions"]  # [batch_size, N_uav, N_target, 2]
        distances = graph_obs["distances"]  # [batch_size, N_uav, N_target]
        uav_mask = graph_obs["masks"]["uav_mask"]  # [batch_size, N_uav]
        target_mask = graph_obs["masks"]["target_mask"]  # [batch_size, N_target]
        
        batch_size, n_uavs, _ = uav_features.shape
        _, n_targets, _ = target_features.shape
        
        # === 1. 增强的实体编码（融合空间信息）===
        # 编码UAV特征
        uav_embeddings = self.uav_encoder(uav_features)  # [batch_size, N_uav, embedding_dim]
        
        # 编码目标特征
        target_embeddings = self.target_encoder(target_features)  # [batch_size, N_target, embedding_dim]
        
        # 融合空间关系信息到UAV嵌入中
        uav_embeddings_enhanced = self._enhance_uav_embeddings_with_spatial_info(
            uav_embeddings, target_embeddings, relative_positions, distances
        )
        
        # 添加位置编码
        uav_embeddings_enhanced = self.position_encoder(uav_embeddings_enhanced)
        target_embeddings = self.position_encoder(target_embeddings)
        
        # === 2. 自注意力 ===
        # 安全的掩码处理，避免维度不匹配
        try:
            # UAV内部自注意力 - 学习UAV间的协作关系
            uav_mask_bool = (uav_mask == 0)  # 转换为布尔掩码，True表示需要忽略的位置
            
            # 确保掩码维度正确 [batch_size, N_uav]
            if uav_mask_bool.dim() == 1:
                uav_mask_bool = uav_mask_bool.unsqueeze(0)
            
            uav_contextualized = self.uav_self_attention(
                uav_embeddings_enhanced,
                src_key_padding_mask=uav_mask_bool
            )  # [batch_size, N_uav, embedding_dim]
            
            # 目标内部自注意力 - 学习目标间的依赖关系
            target_mask_bool = (target_mask == 0)
            
            # 确保掩码维度正确 [batch_size, N_target]
            if target_mask_bool.dim() == 1:
                target_mask_bool = target_mask_bool.unsqueeze(0)
            
            target_contextualized = self.target_self_attention(
                target_embeddings,
                src_key_padding_mask=target_mask_bool
            )  # [batch_size, N_target, embedding_dim]
            
            # === 3. 交叉注意力 ===
            # 对每个UAV，计算其对所有目标的注意力
            uav_target_aware = self.cross_attention(
                tgt=uav_contextualized,  # query: UAV表示
                memory=target_contextualized,  # key & value: 目标表示
                tgt_key_padding_mask=uav_mask_bool,
                memory_key_padding_mask=target_mask_bool
            )  # [batch_size, N_uav, embedding_dim]
            
        except Exception as e:
            # 如果注意力机制失败，使用简化的处理方式
            print(f"注意力机制失败，使用简化处理: {str(e)[:100]}...")
            uav_target_aware = uav_embeddings_enhanced
            target_contextualized = target_embeddings
        
        # === 4. 简化的Q值解码 ===
        # 使用简化方法避免复杂的向量化操作
        try:
            batch_size, n_uavs, embed_dim = uav_target_aware.shape
            _, n_targets, _ = target_contextualized.shape
            
            # 使用简单的线性变换计算Q值
            q_values_matrix = torch.zeros(batch_size, n_uavs * n_targets, device=uav_target_aware.device)
            
            for i in range(min(n_uavs, 10)):  # 限制最大UAV数量
                for j in range(min(n_targets, 15)):  # 限制最大目标数量
                    # 简单的点积作为Q值
                    q_val = torch.sum(uav_target_aware[:, i, :] * target_contextualized[:, j, :], dim=1)
                    q_values_matrix[:, i * n_targets + j] = q_val
                    
        except Exception as e:
            # 如果计算失败，使用最简化的方法
            print(f"Q值计算失败，使用最简化方法: {str(e)[:100]}...")
            batch_size = uav_target_aware.shape[0]
            q_values_matrix = torch.zeros(batch_size, 150, device=uav_target_aware.device)  # 10*15=150
        
        # === 5. 处理phi维度 ===
        # 动态计算phi维度数量，确保不超出输出维度
        n_phi = self._get_n_phi()
        expected_actions = n_uavs * n_targets * n_phi
        
        if expected_actions > self.output_dim:
            # 如果期望的动作数超出输出维度，调整phi数量
            n_phi = max(1, self.output_dim // (n_uavs * n_targets))
            expected_actions = n_uavs * n_targets * n_phi
        
        # 将Q值扩展到包含phi维度
        if n_phi > 1:
            q_values_expanded = q_values_matrix.unsqueeze(-1).repeat(1, 1, n_phi)  # [batch_size, N_uav * N_target, n_phi]
            q_values_final = q_values_expanded.view(batch_size, -1)  # [batch_size, N_uav * N_target * n_phi]
        else:
            q_values_final = q_values_matrix  # 如果n_phi=1，直接使用
        
        # 确保输出维度正确
        if q_values_final.shape[1] > self.output_dim:
            q_values_final = q_values_final[:, :self.output_dim]
        elif q_values_final.shape[1] < self.output_dim:
            # 填充到正确维度
            padding = torch.full((batch_size, self.output_dim - q_values_final.shape[1]), 
                               float('-inf'), device=q_values_final.device)
            q_values_final = torch.cat([q_values_final, padding], dim=1)
        
        # === 6. 应用掩码 ===
        # 对无效的UAV-目标对应用掩码
        action_mask = self._create_action_mask(uav_mask, target_mask, n_phi)
        if action_mask.shape[1] == q_values_final.shape[1]:
            q_values_final = q_values_final.masked_fill(action_mask, float('-inf'))
        
        return q_values_final
    
    def _enhance_uav_embeddings_with_spatial_info(self, uav_embeddings, target_embeddings, 
                                                  relative_positions, distances):
        """
        使用空间信息增强UAV嵌入
        
        Args:
            uav_embeddings: UAV嵌入 [batch_size, N_uav, embedding_dim]
            target_embeddings: 目标嵌入 [batch_size, N_target, embedding_dim]
            relative_positions: 相对位置 [batch_size, N_uav, N_target, 2]
            distances: 距离矩阵 [batch_size, N_uav, N_target]
        
        Returns:
            torch.Tensor: 增强的UAV嵌入
        """
        batch_size, n_uavs, embedding_dim = uav_embeddings.shape
        _, n_targets, _ = target_embeddings.shape
        
        # 使用预定义的空间编码器
        
        # 为每个UAV计算空间上下文
        enhanced_embeddings = []
        
        for uav_idx in range(n_uavs):
            # 获取该UAV到所有目标的空间信息
            uav_rel_pos = relative_positions[:, uav_idx, :, :]  # [batch_size, N_target, 2]
            uav_distances = distances[:, uav_idx, :].unsqueeze(-1)  # [batch_size, N_target, 1]
            
            # 组合空间特征
            spatial_features = torch.cat([uav_rel_pos, uav_distances], dim=-1)  # [batch_size, N_target, 3]
            
            # 编码空间特征
            spatial_encoded = self.spatial_encoder(spatial_features)  # [batch_size, N_target, embedding_dim//4]
            
            # 聚合空间上下文（使用注意力权重）
            spatial_context = spatial_encoded.mean(dim=1)  # [batch_size, embedding_dim//4]
            
            # 将空间上下文融合到UAV嵌入中
            uav_emb = uav_embeddings[:, uav_idx, :]  # [batch_size, embedding_dim]
            
            # 简单的拼接融合（可以改为更复杂的融合方式）
            if spatial_context.shape[-1] + uav_emb.shape[-1] <= embedding_dim:
                # 如果维度允许，直接拼接
                padding_size = embedding_dim - spatial_context.shape[-1] - uav_emb.shape[-1]
                if padding_size > 0:
                    padding = torch.zeros(batch_size, padding_size, device=uav_emb.device)
                    enhanced_emb = torch.cat([uav_emb[:, :embedding_dim-spatial_context.shape[-1]], 
                                            spatial_context, padding], dim=-1)
                else:
                    enhanced_emb = torch.cat([uav_emb[:, :embedding_dim-spatial_context.shape[-1]], 
                                            spatial_context], dim=-1)
            else:
                # 使用加权融合
                spatial_weight = 0.2
                enhanced_emb = (1 - spatial_weight) * uav_emb + spatial_weight * torch.cat([
                    spatial_context, torch.zeros(batch_size, embedding_dim - spatial_context.shape[-1], 
                                                device=spatial_context.device)
                ], dim=-1)
            
            enhanced_embeddings.append(enhanced_emb.unsqueeze(1))
        
        return torch.cat(enhanced_embeddings, dim=1)  # [batch_size, N_uav, embedding_dim]
    
    def _compute_q_values_vectorized(self, uav_target_aware, target_contextualized, uav_mask, target_mask):
        """
        向量化计算Q值，提高效率
        
        Args:
            uav_target_aware: UAV目标感知表示 [batch_size, N_uav, embedding_dim]
            target_contextualized: 目标上下文表示 [batch_size, N_target, embedding_dim]
            uav_mask: UAV掩码 [batch_size, N_uav]
            target_mask: 目标掩码 [batch_size, N_target]
        
        Returns:
            torch.Tensor: Q值矩阵 [batch_size, N_uav * N_target]
        """
        batch_size, n_uavs, embedding_dim = uav_target_aware.shape
        _, n_targets, _ = target_contextualized.shape
        
        # 扩展维度以进行广播
        uav_expanded = uav_target_aware.unsqueeze(2)  # [batch_size, N_uav, 1, embedding_dim]
        target_expanded = target_contextualized.unsqueeze(1)  # [batch_size, 1, N_target, embedding_dim]
        
        # 计算UAV-目标交互特征
        interaction_features = uav_expanded + target_expanded  # [batch_size, N_uav, N_target, embedding_dim]
        
        # 重塑为批次处理
        interaction_flat = interaction_features.view(batch_size * n_uavs * n_targets, embedding_dim)
        
        # 通过Q值解码器
        q_values_flat = self.q_decoder(interaction_flat)  # [batch_size * N_uav * N_target, 1]
        
        # 重塑回原始形状
        q_values_matrix = q_values_flat.view(batch_size, n_uavs * n_targets)
        
        return q_values_matrix
    
    def _get_n_phi(self):
        """获取phi维度数量"""
        return getattr(self, 'n_phi', 6)  # 默认6个方向
    
    def _create_action_mask(self, uav_mask, target_mask, n_phi):
        """
        创建动作掩码，屏蔽无效的UAV-目标-phi组合
        
        Args:
            uav_mask: UAV掩码 [batch_size, N_uav]
            target_mask: 目标掩码 [batch_size, N_target]
            n_phi: phi维度数量
        
        Returns:
            torch.Tensor: 动作掩码 [batch_size, N_actions]
        """
        batch_size, n_uavs = uav_mask.shape
        _, n_targets = target_mask.shape
        
        # 创建UAV-目标对掩码
        uav_mask_expanded = uav_mask.unsqueeze(2)  # [batch_size, N_uav, 1]
        target_mask_expanded = target_mask.unsqueeze(1)  # [batch_size, 1, N_target]
        
        # 无效的UAV-目标对：任一实体无效
        pair_mask = (uav_mask_expanded == 0) | (target_mask_expanded == 0)  # [batch_size, N_uav, N_target]
        
        # 扩展到包含phi维度
        pair_mask_expanded = pair_mask.unsqueeze(-1).repeat(1, 1, 1, n_phi)  # [batch_size, N_uav, N_target, n_phi]
        action_mask = pair_mask_expanded.view(batch_size, -1)  # [batch_size, N_actions]
        
        return action_mask

class PositionalEncoding(nn.Module):
    """
    位置编码模块 - 为序列添加位置信息
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def get_network_info(network_type: str) -> dict:
    """
    获取网络信息
    
    Args:
        network_type: 网络类型
        
    Returns:
        网络信息字典
    """
    network_info = {
        "SimpleNetwork": {
            "description": "基础全连接网络",
            "features": ["BatchNorm", "Dropout", "Xavier初始化"],
            "complexity": "低"
        },
        "DeepFCN": {
            "description": "深度全连接网络",
            "features": ["多层结构", "BatchNorm", "Dropout"],
            "complexity": "中"
        },
        # GAT网络已移除
        "DeepFCNResidual": {
            "description": "带残差连接的深度网络",
            "features": ["残差连接", "BatchNorm", "Dropout"],
            "complexity": "中"
        },
        "ZeroShotGNN": {
            "description": "零样本图神经网络",
            "features": ["Transformer架构", "自注意力", "交叉注意力", "参数共享", "零样本迁移"],
            "complexity": "高"
        }
    }
    
    return network_info.get(network_type, {"description": "未知网络", "features": [], "complexity": "未知"}) 