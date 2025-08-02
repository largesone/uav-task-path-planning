# -*- coding: utf-8 -*-
# 文件名: zero_shot_network.py
# 描述: 零样本迁移网络架构

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

class ZeroShotTransferNetwork(nn.Module):
    """
    零样本迁移网络
    
    核心特性：
    1. 支持可变数量的UAV和目标
    2. 基于注意力机制的实体交互建模
    3. 参数共享确保迁移能力
    4. 位置编码处理空间关系
    """
    
    def __init__(self, 
                 entity_feature_dim: int = 64,
                 hidden_dim: int = 128,
                 num_attention_heads: int = 4,
                 num_layers: int = 2,
                 max_entities: int = 50,
                 dropout: float = 0.1):
        super(ZeroShotTransferNetwork, self).__init__()
        
        self.entity_feature_dim = entity_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        self.max_entities = max_entities
        
        # UAV特征编码器
        self.uav_encoder = nn.Sequential(
            nn.Linear(8, entity_feature_dim),  # 位置(2) + 资源(2) + 速度(2) + 其他(2)
            nn.ReLU(),
            nn.Linear(entity_feature_dim, entity_feature_dim)
        )
        
        # 目标特征编码器
        self.target_encoder = nn.Sequential(
            nn.Linear(4, entity_feature_dim),  # 位置(2) + 资源需求(2)
            nn.ReLU(),
            nn.Linear(entity_feature_dim, entity_feature_dim)
        )
        
        # 位置编码
        self.position_encoding = PositionalEncoding(entity_feature_dim, max_entities)
        
        # 多层注意力机制
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(entity_feature_dim, num_attention_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(entity_feature_dim)
            for _ in range(num_layers)
        ])
        
        # 输出层 - 动态生成动作分布
        self.action_generator = ActionGenerator(entity_feature_dim, hidden_dim)
        
        print(f"[ZeroShotTransferNetwork] 初始化完成")
        print(f"  - 实体特征维度: {entity_feature_dim}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 注意力头数: {num_attention_heads}")
        print(f"  - 网络层数: {num_layers}")
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state_dict: 包含以下键的字典
                - 'uav_features': [batch_size, num_uavs, uav_feature_dim]
                - 'target_features': [batch_size, num_targets, target_feature_dim]
                - 'uav_mask': [batch_size, num_uavs] (可选)
                - 'target_mask': [batch_size, num_targets] (可选)
        
        Returns:
            action_logits: [batch_size, num_actions]
        """
        batch_size = state_dict['uav_features'].shape[0]
        num_uavs = state_dict['uav_features'].shape[1]
        num_targets = state_dict['target_features'].shape[1]
        
        # 编码UAV和目标特征
        uav_encoded = self.uav_encoder(state_dict['uav_features'])  # [B, N_u, D]
        target_encoded = self.target_encoder(state_dict['target_features'])  # [B, N_t, D]
        
        # 合并所有实体
        all_entities = torch.cat([uav_encoded, target_encoded], dim=1)  # [B, N_u+N_t, D]
        
        # 添加位置编码
        all_entities = self.position_encoding(all_entities)
        
        # 创建注意力掩码
        entity_mask = self._create_entity_mask(state_dict, num_uavs, num_targets)
        
        # 多层注意力处理
        for i, (attention, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # 自注意力
            attended = attention(all_entities, all_entities, all_entities, entity_mask)
            # 残差连接和层归一化
            all_entities = layer_norm(all_entities + attended)
        
        # 分离UAV和目标特征
        uav_final = all_entities[:, :num_uavs, :]  # [B, N_u, D]
        target_final = all_entities[:, num_uavs:, :]  # [B, N_t, D]
        
        # 生成动作分布
        action_logits = self.action_generator(uav_final, target_final)
        
        return action_logits
    
    def _create_entity_mask(self, state_dict: Dict[str, torch.Tensor], 
                           num_uavs: int, num_targets: int) -> Optional[torch.Tensor]:
        """创建实体掩码"""
        batch_size = state_dict['uav_features'].shape[0]
        total_entities = num_uavs + num_targets
        
        # 如果提供了掩码，使用它们
        if 'uav_mask' in state_dict and 'target_mask' in state_dict:
            uav_mask = state_dict['uav_mask']  # [B, N_u]
            target_mask = state_dict['target_mask']  # [B, N_t]
            entity_mask = torch.cat([uav_mask, target_mask], dim=1)  # [B, N_u+N_t]
            return entity_mask.unsqueeze(1).expand(-1, total_entities, -1)  # [B, N_u+N_t, N_u+N_t]
        
        return None

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑并通过输出层
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attention_output)
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                    V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class ActionGenerator(nn.Module):
    """动作生成器 - 基于UAV和目标特征生成动作分布"""
    
    def __init__(self, entity_feature_dim: int, hidden_dim: int):
        super(ActionGenerator, self).__init__()
        
        self.entity_feature_dim = entity_feature_dim
        self.hidden_dim = hidden_dim
        
        # UAV-目标交互评分网络
        self.interaction_scorer = nn.Sequential(
            nn.Linear(entity_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 方向选择网络（简化为4个方向）
        self.direction_selector = nn.Sequential(
            nn.Linear(entity_feature_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4个方向
        )
    
    def forward(self, uav_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        生成动作分布
        
        Args:
            uav_features: [batch_size, num_uavs, feature_dim]
            target_features: [batch_size, num_targets, feature_dim]
        
        Returns:
            action_logits: [batch_size, num_uavs * num_targets * 4]
        """
        batch_size, num_uavs, feature_dim = uav_features.shape
        num_targets = target_features.shape[1]
        
        action_logits = []
        
        # 为每个UAV-目标对生成动作分数
        for i in range(num_uavs):
            for j in range(num_targets):
                uav_feat = uav_features[:, i, :]  # [B, D]
                target_feat = target_features[:, j, :]  # [B, D]
                
                # 拼接特征
                combined_feat = torch.cat([uav_feat, target_feat], dim=1)  # [B, 2D]
                
                # 计算交互分数
                interaction_score = self.interaction_scorer(combined_feat)  # [B, 1]
                
                # 计算方向分数
                direction_scores = self.direction_selector(combined_feat)  # [B, 4]
                
                # 组合分数
                combined_scores = interaction_score + direction_scores  # [B, 4]
                action_logits.append(combined_scores)
        
        # 拼接所有动作分数
        action_logits = torch.cat(action_logits, dim=1)  # [B, num_uavs * num_targets * 4]
        
        return action_logits

def create_zero_shot_network(max_uavs: int = 10, max_targets: int = 15) -> ZeroShotTransferNetwork:
    """创建零样本迁移网络"""
    return ZeroShotTransferNetwork(
        entity_feature_dim=64,
        hidden_dim=128,
        num_attention_heads=4,
        num_layers=2,
        max_entities=max_uavs + max_targets,
        dropout=0.1
    )

# 测试代码
if __name__ == "__main__":
    print("测试零样本迁移网络...")
    
    # 创建网络
    network = create_zero_shot_network()
    
    # 创建测试数据
    batch_size = 2
    num_uavs = 3
    num_targets = 4
    
    state_dict = {
        'uav_features': torch.randn(batch_size, num_uavs, 8),
        'target_features': torch.randn(batch_size, num_targets, 4)
    }
    
    # 前向传播
    with torch.no_grad():
        output = network(state_dict)
        print(f"输出形状: {output.shape}")
        print(f"期望形状: [{batch_size}, {num_uavs * num_targets * 4}]")
        
        # 测试不同规模
        state_dict_large = {
            'uav_features': torch.randn(batch_size, 5, 8),
            'target_features': torch.randn(batch_size, 6, 4)
        }
        
        output_large = network(state_dict_large)
        print(f"大规模输出形状: {output_large.shape}")
        print(f"大规模期望形状: [{batch_size}, {5 * 6 * 4}]")
    
    print("零样本迁移网络测试完成！")