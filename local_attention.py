# -*- coding: utf-8 -*-
# 文件名: local_attention.py
# 描述: 局部注意力机制实现，支持k-近邻选择和大规模场景优化

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math

try:
    # 尝试导入xformers以支持Flash Attention优化
    import xformers
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("[LocalAttention] xformers可用，将启用Flash Attention优化")
except ImportError:
    XFORMERS_AVAILABLE = False
    print("[LocalAttention] xformers不可用，使用标准注意力实现")


class LocalAttention(nn.Module):
    """
    局部注意力机制
    
    核心特性：
    1. 基于distances张量进行k-近邻选择
    2. 为每架无人机动态选择k个最近目标作为注意力计算上下文
    3. 解决维度爆炸与显存OOM问题，支持大规模场景
    4. 可选集成xformers库的分块Flash Attention优化
    5. 自适应k值：k = min(max(4, ceil(N/4)), 16)
    6. 训练期随机化：k ± 2的随机抖动
    
    设计理念：
    - 计算复杂度从O(N²)降低到O(N*k)，其中k << N
    - 保持注意力机制的表达能力，同时大幅减少内存占用
    - 支持动态k值调整，适应不同规模的场景
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        k_adaptive: bool = True,
        k_fixed: Optional[int] = None,
        k_min: int = 4,
        k_max: int = 16,
        use_flash_attention: bool = True,
        temperature: float = 1.0
    ):
        """
        初始化局部注意力机制
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比例
            k_adaptive: 是否使用自适应k值
            k_fixed: 固定k值（如果不使用自适应）
            k_min: 最小k值
            k_max: 最大k值
            use_flash_attention: 是否使用Flash Attention优化
            temperature: 注意力温度参数
        """
        super(LocalAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.k_adaptive = k_adaptive
        self.k_fixed = k_fixed
        self.k_min = k_min
        self.k_max = k_max
        self.use_flash_attention = use_flash_attention and XFORMERS_AVAILABLE
        self.temperature = temperature
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        # 查询、键、值投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
        print(f"[LocalAttention] 初始化完成 - 嵌入维度: {embed_dim}, 头数: {num_heads}, "
              f"自适应k: {k_adaptive}, Flash Attention: {self.use_flash_attention}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _compute_adaptive_k(self, num_targets: int, training: bool = True) -> int:
        """
        计算自适应k值
        
        Args:
            num_targets: 目标数量
            training: 是否在训练模式
            
        Returns:
            计算得到的k值
        """
        if not self.k_adaptive:
            return self.k_fixed if self.k_fixed is not None else min(num_targets, 8)
        
        # 基础k值：k = min(max(4, ceil(N/4)), 16)
        base_k = min(max(self.k_min, math.ceil(num_targets / 4)), self.k_max)
        
        # 训练期随机化：k ± 2的随机抖动
        if training:
            noise = torch.randint(-2, 3, (1,)).item()  # [-2, -1, 0, 1, 2]
            k = max(self.k_min, min(base_k + noise, min(self.k_max, num_targets)))
        else:
            k = min(base_k, num_targets)
        
        return k
    
    def _select_k_nearest(
        self,
        distances: torch.Tensor,
        k: int,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于距离矩阵选择k个最近的目标
        
        Args:
            distances: 距离矩阵 [batch_size, num_uavs, num_targets]
            k: 选择的近邻数量
            masks: 掩码字典，包含target_mask等
            
        Returns:
            selected_indices: 选中的目标索引 [batch_size, num_uavs, k]
            selected_distances: 选中的距离值 [batch_size, num_uavs, k]
        """
        batch_size, num_uavs, num_targets = distances.shape
        
        # 应用目标掩码（如果提供）
        masked_distances = distances.clone()
        if masks is not None and 'target_mask' in masks:
            target_mask = masks['target_mask']  # [batch_size, num_targets]
            # 确保掩码是布尔类型
            if target_mask.dtype != torch.bool:
                target_mask = target_mask.bool()
            # 将无效目标的距离设为无穷大
            invalid_mask = ~target_mask.unsqueeze(1).expand(-1, num_uavs, -1)  # [batch_size, num_uavs, num_targets]
            masked_distances[invalid_mask] = float('inf')
        
        # 确保k不超过有效目标数量
        if masks is not None and 'target_mask' in masks:
            valid_targets_per_batch = masks['target_mask'].sum(dim=1)  # [batch_size]
            min_valid_targets = valid_targets_per_batch.min().item()
            k = min(k, max(1, int(min_valid_targets)))
        else:
            k = min(k, num_targets)
        
        # 确保k是整数
        k = int(k)
        
        # 选择k个最近的目标
        selected_distances, selected_indices = torch.topk(
            masked_distances, k, dim=-1, largest=False, sorted=True
        )
        
        print(f"[LocalAttention] k-近邻选择完成 - k={k}, 距离范围: "
              f"[{selected_distances.min().item():.3f}, {selected_distances.max().item():.3f}]")
        
        return selected_indices, selected_distances
    
    def _apply_local_attention(
        self,
        uav_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        selected_indices: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        应用局部注意力计算
        
        Args:
            uav_embeddings: UAV嵌入 [batch_size, num_uavs, embed_dim]
            target_embeddings: 目标嵌入 [batch_size, num_targets, embed_dim]
            selected_indices: 选中的目标索引 [batch_size, num_uavs, k]
            masks: 掩码字典
            
        Returns:
            注意力输出 [batch_size, num_uavs, embed_dim]
        """
        batch_size, num_uavs, embed_dim = uav_embeddings.shape
        _, num_targets, _ = target_embeddings.shape
        k = selected_indices.shape[-1]
        
        # 根据选中的索引收集目标嵌入
        # selected_indices: [batch_size, num_uavs, k]
        # target_embeddings: [batch_size, num_targets, embed_dim]
        
        # 扩展target_embeddings以便索引
        target_embeddings_expanded = target_embeddings.unsqueeze(1).expand(-1, num_uavs, -1, -1)
        # [batch_size, num_uavs, num_targets, embed_dim]
        
        # 使用gather收集选中的目标嵌入
        selected_indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, -1, embed_dim)
        # [batch_size, num_uavs, k, embed_dim]
        
        selected_target_embeddings = torch.gather(
            target_embeddings_expanded, dim=2, index=selected_indices_expanded
        )
        # [batch_size, num_uavs, k, embed_dim]
        
        # 计算查询、键、值
        queries = self.q_proj(uav_embeddings)  # [batch_size, num_uavs, embed_dim]
        keys = self.k_proj(selected_target_embeddings)  # [batch_size, num_uavs, k, embed_dim]
        values = self.v_proj(selected_target_embeddings)  # [batch_size, num_uavs, k, embed_dim]
        
        # 重塑为多头注意力格式
        queries = queries.view(batch_size, num_uavs, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, num_uavs, head_dim]
        
        keys = keys.view(batch_size, num_uavs, k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        # [batch_size, num_heads, num_uavs, k, head_dim]
        
        values = values.view(batch_size, num_uavs, k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        # [batch_size, num_heads, num_uavs, k, head_dim]
        
        # 使用Flash Attention或标准注意力
        if self.use_flash_attention:
            attention_output = self._flash_attention(queries, keys, values, masks)
        else:
            attention_output = self._standard_attention(queries, keys, values, masks)
        
        # 重塑输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, num_uavs, embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attention_output)
        
        return output
    
    def _flash_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        使用xformers的Flash Attention实现
        
        Args:
            queries: [batch_size, num_heads, num_uavs, head_dim]
            keys: [batch_size, num_heads, num_uavs, k, head_dim]
            values: [batch_size, num_heads, num_uavs, k, head_dim]
            masks: 掩码字典
            
        Returns:
            注意力输出 [batch_size, num_heads, num_uavs, head_dim]
        """
        batch_size, num_heads, num_uavs, head_dim = queries.shape
        k = keys.shape[3]
        
        # 重塑为Flash Attention所需的格式
        # Flash Attention期望的输入格式: [batch_size * num_heads, seq_len, head_dim]
        queries_flat = queries.view(batch_size * num_heads, num_uavs, head_dim)
        keys_flat = keys.view(batch_size * num_heads, num_uavs * k, head_dim)
        values_flat = values.view(batch_size * num_heads, num_uavs * k, head_dim)
        
        # 创建注意力偏置矩阵（每个UAV只能看到其对应的k个目标）
        attn_bias = torch.full(
            (num_uavs, num_uavs * k), float('-inf'), 
            device=queries.device, dtype=queries.dtype
        )
        
        for i in range(num_uavs):
            start_idx = i * k
            end_idx = (i + 1) * k
            attn_bias[i, start_idx:end_idx] = 0.0
        
        attn_bias = attn_bias.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        
        try:
            # 使用xformers的memory_efficient_attention
            attention_output = xops.memory_efficient_attention(
                query=queries_flat,
                key=keys_flat,
                value=values_flat,
                attn_bias=attn_bias,
                p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(head_dim) / self.temperature
            )
            
            # 重塑回原始格式
            attention_output = attention_output.view(batch_size, num_heads, num_uavs, head_dim)
            
            print(f"[LocalAttention] Flash Attention计算完成")
            
        except Exception as e:
            print(f"[LocalAttention] Flash Attention失败，回退到标准实现: {e}")
            # 回退到标准注意力实现
            attention_output = self._standard_attention(queries, keys, values, masks)
        
        return attention_output
    
    def _standard_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        标准多头注意力实现
        
        Args:
            queries: [batch_size, num_heads, num_uavs, head_dim]
            keys: [batch_size, num_heads, num_uavs, k, head_dim]
            values: [batch_size, num_heads, num_uavs, k, head_dim]
            masks: 掩码字典
            
        Returns:
            注意力输出 [batch_size, num_heads, num_uavs, head_dim]
        """
        batch_size, num_heads, num_uavs, head_dim = queries.shape
        k = keys.shape[3]
        
        # 计算注意力分数
        # queries: [batch_size, num_heads, num_uavs, head_dim]
        # keys: [batch_size, num_heads, num_uavs, k, head_dim]
        
        # 扩展queries以匹配keys的维度
        queries_expanded = queries.unsqueeze(3)  # [batch_size, num_heads, num_uavs, 1, head_dim]
        
        # 计算点积注意力分数
        attention_scores = torch.matmul(queries_expanded, keys.transpose(-2, -1))
        # [batch_size, num_heads, num_uavs, 1, k]
        
        attention_scores = attention_scores.squeeze(3)  # [batch_size, num_heads, num_uavs, k]
        
        # 缩放
        attention_scores = attention_scores / (math.sqrt(head_dim) * self.temperature)
        
        # 应用softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用dropout
        if self.training:
            attention_weights = self.dropout_layer(attention_weights)
        
        # 计算加权值
        # attention_weights: [batch_size, num_heads, num_uavs, k]
        # values: [batch_size, num_heads, num_uavs, k, head_dim]
        
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, num_heads, num_uavs, k, 1]
        weighted_values = values * attention_weights_expanded  # [batch_size, num_heads, num_uavs, k, head_dim]
        
        # 求和得到最终输出
        attention_output = weighted_values.sum(dim=3)  # [batch_size, num_heads, num_uavs, head_dim]
        
        print(f"[LocalAttention] 标准注意力计算完成 - 注意力权重范围: "
              f"[{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
        
        return attention_output
    
    def forward(
        self,
        uav_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        distances: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        局部注意力前向传播
        
        Args:
            uav_embeddings: UAV嵌入 [batch_size, num_uavs, embed_dim]
            target_embeddings: 目标嵌入 [batch_size, num_targets, embed_dim]
            distances: 距离矩阵 [batch_size, num_uavs, num_targets]
            masks: 掩码字典，包含uav_mask, target_mask等
            
        Returns:
            局部注意力输出 [batch_size, num_uavs, embed_dim]
        """
        batch_size, num_uavs, embed_dim = uav_embeddings.shape
        _, num_targets, _ = target_embeddings.shape
        
        print(f"[LocalAttention] 前向传播开始 - UAV数量: {num_uavs}, 目标数量: {num_targets}")
        
        # 计算自适应k值
        k = self._compute_adaptive_k(num_targets, self.training)
        print(f"[LocalAttention] 自适应k值: {k}")
        
        # 选择k个最近的目标
        selected_indices, selected_distances = self._select_k_nearest(distances, k, masks)
        
        # 应用局部注意力
        attention_output = self._apply_local_attention(
            uav_embeddings, target_embeddings, selected_indices, masks
        )
        
        # 应用UAV掩码（如果提供）
        if masks is not None and 'uav_mask' in masks:
            uav_mask = masks['uav_mask'].unsqueeze(-1)  # [batch_size, num_uavs, 1]
            attention_output = attention_output * uav_mask.float()
            print(f"[LocalAttention] 应用UAV掩码，有效UAV比例: {uav_mask.float().mean().item():.3f}")
        
        print(f"[LocalAttention] 前向传播完成 - 输出形状: {attention_output.shape}")
        
        return attention_output


class MultiScaleLocalAttention(nn.Module):
    """
    多尺度局部注意力机制
    
    在不同的k值下并行计算局部注意力，然后融合结果
    这样可以同时捕获近距离和中距离的交互信息
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        k_scales: list = [4, 8, 16],
        fusion_method: str = "weighted_sum"  # "weighted_sum", "concat", "attention"
    ):
        """
        初始化多尺度局部注意力
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比例
            k_scales: 不同的k值列表
            fusion_method: 融合方法
        """
        super(MultiScaleLocalAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.k_scales = k_scales
        self.fusion_method = fusion_method
        
        # 为每个尺度创建局部注意力模块
        self.attention_modules = nn.ModuleList([
            LocalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                k_adaptive=False,
                k_fixed=k,
                use_flash_attention=True
            )
            for k in k_scales
        ])
        
        # 融合层
        if fusion_method == "weighted_sum":
            self.scale_weights = nn.Parameter(torch.ones(len(k_scales)))
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(embed_dim * len(k_scales), embed_dim)
        elif fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim, num_heads=4, dropout=dropout, batch_first=True
            )
        
        print(f"[MultiScaleLocalAttention] 初始化完成 - k尺度: {k_scales}, 融合方法: {fusion_method}")
    
    def forward(
        self,
        uav_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        distances: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        多尺度局部注意力前向传播
        
        Args:
            uav_embeddings: UAV嵌入 [batch_size, num_uavs, embed_dim]
            target_embeddings: 目标嵌入 [batch_size, num_targets, embed_dim]
            distances: 距离矩阵 [batch_size, num_uavs, num_targets]
            masks: 掩码字典
            
        Returns:
            多尺度注意力输出 [batch_size, num_uavs, embed_dim]
        """
        # 计算每个尺度的注意力输出
        scale_outputs = []
        for i, attention_module in enumerate(self.attention_modules):
            output = attention_module(uav_embeddings, target_embeddings, distances, masks)
            scale_outputs.append(output)
        
        # 融合不同尺度的输出
        if self.fusion_method == "weighted_sum":
            # 加权求和
            weights = F.softmax(self.scale_weights, dim=0)
            fused_output = sum(w * output for w, output in zip(weights, scale_outputs))
            
        elif self.fusion_method == "concat":
            # 拼接后投影
            concatenated = torch.cat(scale_outputs, dim=-1)
            fused_output = self.fusion_proj(concatenated)
            
        elif self.fusion_method == "attention":
            # 使用注意力机制融合
            stacked_outputs = torch.stack(scale_outputs, dim=2)  # [batch_size, num_uavs, num_scales, embed_dim]
            batch_size, num_uavs, num_scales, embed_dim = stacked_outputs.shape
            
            # 重塑为注意力输入格式
            stacked_outputs = stacked_outputs.view(batch_size * num_uavs, num_scales, embed_dim)
            
            # 自注意力融合
            fused_output, _ = self.fusion_attention(
                stacked_outputs, stacked_outputs, stacked_outputs
            )
            
            # 取平均作为最终输出
            fused_output = fused_output.mean(dim=1)  # [batch_size * num_uavs, embed_dim]
            fused_output = fused_output.view(batch_size, num_uavs, embed_dim)
        
        else:
            # 默认：简单平均
            fused_output = torch.stack(scale_outputs, dim=0).mean(dim=0)
        
        print(f"[MultiScaleLocalAttention] 多尺度融合完成 - 输出形状: {fused_output.shape}")
        
        return fused_output
