# -*- coding: utf-8 -*-
# 文件名: improved_zero_shot_network.py
# 描述: 改进的零样本迁移网络架构 - 适应不同场景规模

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

class ImprovedZeroShotGNN(nn.Module):
    """
    改进的零样本图神经网络
    
    核心改进：
    1. 自适应场景规模处理
    2. 增强的空间关系建模
    3. 多尺度特征融合
    4. 鲁棒的注意力机制
    5. 动态动作空间适配
    """
    
    def __init__(self, 
                 entity_feature_dim: int = 64,
                 hidden_dim: int = 128,
                 num_attention_heads: int = 8,
                 num_layers: int = 3,
                 max_entities: int = 50,
                 dropout: float = 0.1,
                 use_spatial_encoding: bool = True):
        super(ImprovedZeroShotGNN, self).__init__()
        
        self.entity_feature_dim = entity_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        self.max_entities = max_entities
        self.use_spatial_encoding = use_spatial_encoding
        
        # === 1. 自适应实体编码器 ===
        # UAV编码器 - 处理位置、资源、状态等信息
        self.uav_encoder = AdaptiveEntityEncoder(
            input_dim=8,  # 位置(2) + 资源(2) + 速度(2) + 朝向(1) + 状态(1)
            output_dim=entity_feature_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # 目标编码器 - 处理位置、需求、优先级等信息
        self.target_encoder = AdaptiveEntityEncoder(
            input_dim=4,  # 位置(2) + 资源需求(2)
            output_dim=entity_feature_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # === 2. 空间关系编码器 ===
        if use_spatial_encoding:
            self.spatial_encoder = SpatialRelationEncoder(
                entity_feature_dim=entity_feature_dim,
                hidden_dim=hidden_dim // 4
            )
        
        # === 3. 多层图注意力网络 ===
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(
                entity_feature_dim, 
                entity_feature_dim, 
                num_attention_heads,
                dropout
            ) for _ in range(num_layers)
        ])
        
        # === 4. 层归一化 ===
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(entity_feature_dim) for _ in range(num_layers)
        ])
        
        # === 5. 动态动作生成器 ===
        self.action_generator = DynamicActionGenerator(
            entity_feature_dim=entity_feature_dim,
            hidden_dim=hidden_dim,
            max_uavs=20,
            max_targets=30
        )
        
        # === 6. 场景复杂度估计器 ===
        self.complexity_estimator = ScenarioComplexityEstimator(
            entity_feature_dim=entity_feature_dim,
            hidden_dim=hidden_dim // 2
        )
        
        self._init_weights()
        
        print(f"[ImprovedZeroShotGNN] 初始化完成")
        print(f"  - 实体特征维度: {entity_feature_dim}")
        print(f"  - 注意力头数: {num_attention_heads}")
        print(f"  - 网络层数: {num_layers}")
        print(f"  - 空间编码: {'启用' if use_spatial_encoding else '禁用'}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state_dict: 状态字典，包含：
                - 'uav_features': [batch_size, num_uavs, uav_feature_dim]
                - 'target_features': [batch_size, num_targets, target_feature_dim]
                - 'uav_mask': [batch_size, num_uavs] (可选)
                - 'target_mask': [batch_size, num_targets] (可选)
        
        Returns:
            action_logits: [batch_size, dynamic_action_space]
        """
        batch_size = state_dict['uav_features'].shape[0]
        num_uavs = state_dict['uav_features'].shape[1]
        num_targets = state_dict['target_features'].shape[1]
        
        # === 1. 实体编码 ===
        uav_encoded = self.uav_encoder(state_dict['uav_features'])  # [B, N_u, D]
        target_encoded = self.target_encoder(state_dict['target_features'])  # [B, N_t, D]
        
        # === 2. 空间关系增强 ===
        if self.use_spatial_encoding:
            uav_encoded, target_encoded = self.spatial_encoder(
                uav_encoded, target_encoded, 
                state_dict['uav_features'][:, :, :2],  # UAV位置
                state_dict['target_features'][:, :, :2]  # 目标位置
            )
        
        # === 3. 构建统一的实体表示 ===
        # 添加实体类型标识
        uav_type_embed = torch.zeros(batch_size, num_uavs, 1, device=uav_encoded.device)
        target_type_embed = torch.ones(batch_size, num_targets, 1, device=target_encoded.device)
        
        uav_with_type = torch.cat([uav_encoded, uav_type_embed], dim=-1)
        target_with_type = torch.cat([target_encoded, target_type_embed], dim=-1)
        
        # 合并所有实体
        all_entities = torch.cat([uav_with_type, target_with_type], dim=1)  # [B, N_u+N_t, D+1]
        
        # 调整维度
        all_entities = nn.Linear(self.entity_feature_dim + 1, self.entity_feature_dim).to(all_entities.device)(all_entities)
        
        # === 4. 多层图注意力处理 ===
        entity_mask = self._create_attention_mask(state_dict, num_uavs, num_targets)
        
        for i, (attention_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # 图注意力
            attended_entities = attention_layer(all_entities, entity_mask)
            
            # 残差连接和层归一化
            all_entities = layer_norm(all_entities + attended_entities)
        
        # === 5. 分离UAV和目标表示 ===
        final_uav_features = all_entities[:, :num_uavs, :]  # [B, N_u, D]
        final_target_features = all_entities[:, num_uavs:, :]  # [B, N_t, D]
        
        # === 6. 场景复杂度评估 ===
        complexity_score = self.complexity_estimator(final_uav_features, final_target_features)
        
        # === 7. 动态动作生成 ===
        action_logits = self.action_generator(
            final_uav_features, 
            final_target_features, 
            complexity_score
        )
        
        return action_logits
    
    def _create_attention_mask(self, state_dict: Dict[str, torch.Tensor], 
                              num_uavs: int, num_targets: int) -> Optional[torch.Tensor]:
        """创建注意力掩码"""
        if 'uav_mask' in state_dict and 'target_mask' in state_dict:
            uav_mask = state_dict['uav_mask']  # [B, N_u]
            target_mask = state_dict['target_mask']  # [B, N_t]
            
            # 合并掩码
            combined_mask = torch.cat([uav_mask, target_mask], dim=1)  # [B, N_u+N_t]
            
            # 扩展为注意力掩码
            batch_size = combined_mask.shape[0]
            total_entities = combined_mask.shape[1]
            
            attention_mask = combined_mask.unsqueeze(1).expand(-1, total_entities, -1)  # [B, N_u+N_t, N_u+N_t]
            
            return attention_mask
        
        return None

class AdaptiveEntityEncoder(nn.Module):
    """自适应实体编码器"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(AdaptiveEntityEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class SpatialRelationEncoder(nn.Module):
    """空间关系编码器"""
    
    def __init__(self, entity_feature_dim: int, hidden_dim: int):
        super(SpatialRelationEncoder, self).__init__()
        
        self.entity_feature_dim = entity_feature_dim
        self.hidden_dim = hidden_dim
        
        # 距离编码器
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, entity_feature_dim)
        )
        
        # 角度编码器
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, entity_feature_dim)
        )
        
        # 融合层
        self.fusion_layer = nn.Linear(entity_feature_dim * 3, entity_feature_dim)
    
    def forward(self, uav_features: torch.Tensor, target_features: torch.Tensor,
                uav_positions: torch.Tensor, target_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            uav_features: [B, N_u, D]
            target_features: [B, N_t, D]
            uav_positions: [B, N_u, 2]
            target_positions: [B, N_t, 2]
        """
        batch_size, num_uavs, _ = uav_features.shape
        num_targets = target_features.shape[1]
        
        # 计算空间关系增强的UAV特征
        enhanced_uav_features = []
        
        for i in range(num_uavs):
            uav_pos = uav_positions[:, i:i+1, :]  # [B, 1, 2]
            uav_feat = uav_features[:, i, :]  # [B, D]
            
            # 计算到所有目标的距离和角度
            distances = torch.norm(target_positions - uav_pos, dim=-1, keepdim=True)  # [B, N_t, 1]
            
            # 计算角度
            diff = target_positions - uav_pos  # [B, N_t, 2]
            angles = torch.atan2(diff[:, :, 1:2], diff[:, :, 0:1])  # [B, N_t, 1]
            
            # 编码距离和角度
            distance_encoded = self.distance_encoder(distances)  # [B, N_t, D]
            angle_encoded = self.angle_encoder(angles)  # [B, N_t, D]
            
            # 聚合空间信息
            spatial_info = torch.mean(distance_encoded + angle_encoded, dim=1)  # [B, D]
            
            # 融合原始特征和空间信息
            enhanced_feat = self.fusion_layer(torch.cat([
                uav_feat, spatial_info, uav_feat * spatial_info
            ], dim=-1))  # [B, D]
            
            enhanced_uav_features.append(enhanced_feat)
        
        enhanced_uav_features = torch.stack(enhanced_uav_features, dim=1)  # [B, N_u, D]
        
        return enhanced_uav_features, target_features

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, dropout: float = 0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        
        # 重塑并投影
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.output_dim)
        output = self.output_proj(attended)
        
        return output

class DynamicActionGenerator(nn.Module):
    """动态动作生成器"""
    
    def __init__(self, entity_feature_dim: int, hidden_dim: int, max_uavs: int, max_targets: int):
        super(DynamicActionGenerator, self).__init__()
        
        self.entity_feature_dim = entity_feature_dim
        self.hidden_dim = hidden_dim
        self.max_uavs = max_uavs
        self.max_targets = max_targets
        
        # UAV-目标交互评分网络
        self.interaction_scorer = nn.Sequential(
            nn.Linear(entity_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4个方向的分数
        )
        
        # 复杂度自适应层
        self.complexity_adapter = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, uav_features: torch.Tensor, target_features: torch.Tensor, 
                complexity_score: torch.Tensor) -> torch.Tensor:
        """
        生成动态动作分布
        
        Args:
            uav_features: [batch_size, num_uavs, feature_dim]
            target_features: [batch_size, num_targets, feature_dim]
            complexity_score: [batch_size, 1]
        
        Returns:
            action_logits: [batch_size, num_uavs * num_targets * 4]
        """
        batch_size, num_uavs, feature_dim = uav_features.shape
        num_targets = target_features.shape[1]
        
        # 复杂度自适应权重
        complexity_weight = torch.sigmoid(self.complexity_adapter(complexity_score))  # [B, 1]
        
        action_logits = []
        
        # 为每个UAV-目标对生成动作分数
        for i in range(num_uavs):
            for j in range(num_targets):
                uav_feat = uav_features[:, i, :]  # [B, D]
                target_feat = target_features[:, j, :]  # [B, D]
                
                # 拼接特征
                combined_feat = torch.cat([uav_feat, target_feat], dim=1)  # [B, 2D]
                
                # 计算交互分数
                interaction_scores = self.interaction_scorer(combined_feat)  # [B, 4]
                
                # 应用复杂度自适应
                adapted_scores = interaction_scores * complexity_weight
                
                action_logits.append(adapted_scores)
        
        # 拼接所有动作分数
        action_logits = torch.cat(action_logits, dim=1)  # [B, num_uavs * num_targets * 4]
        
        return action_logits

class ScenarioComplexityEstimator(nn.Module):
    """场景复杂度估计器"""
    
    def __init__(self, entity_feature_dim: int, hidden_dim: int):
        super(ScenarioComplexityEstimator, self).__init__()
        
        self.complexity_net = nn.Sequential(
            nn.Linear(entity_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, uav_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        估计场景复杂度
        
        Args:
            uav_features: [batch_size, num_uavs, feature_dim]
            target_features: [batch_size, num_targets, feature_dim]
        
        Returns:
            complexity_score: [batch_size, 1]
        """
        # 聚合UAV和目标特征
        uav_global = torch.mean(uav_features, dim=1)  # [B, D]
        target_global = torch.mean(target_features, dim=1)  # [B, D]
        
        # 拼接全局特征
        global_features = torch.cat([uav_global, target_global], dim=1)  # [B, 2D]
        
        # 估计复杂度
        complexity_score = self.complexity_net(global_features)  # [B, 1]
        
        return complexity_score

def create_improved_zero_shot_network() -> ImprovedZeroShotGNN:
    """创建改进的零样本迁移网络"""
    return ImprovedZeroShotGNN(
        entity_feature_dim=64,
        hidden_dim=128,
        num_attention_heads=8,
        num_layers=3,
        max_entities=50,
        dropout=0.1,
        use_spatial_encoding=True
    )

# 测试代码
if __name__ == "__main__":
    print("测试改进的零样本迁移网络...")
    
    # 创建网络
    network = create_improved_zero_shot_network()
    
    # 创建测试数据
    batch_size = 2
    
    # 测试小规模场景
    small_state = {
        'uav_features': torch.randn(batch_size, 3, 8),
        'target_features': torch.randn(batch_size, 4, 4)
    }
    
    with torch.no_grad():
        small_output = network(small_state)
        print(f"小规模场景输出形状: {small_output.shape}")
    
    # 测试大规模场景
    large_state = {
        'uav_features': torch.randn(batch_size, 8, 8),
        'target_features': torch.randn(batch_size, 12, 4)
    }
    
    with torch.no_grad():
        large_output = network(large_state)
        print(f"大规模场景输出形状: {large_output.shape}")
    
    print("改进的零样本迁移网络测试完成！")
