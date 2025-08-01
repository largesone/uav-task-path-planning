# -*- coding: utf-8 -*-
# 文件名: noisy_linear.py
# 描述: NoisyLinear层实现，用于参数空间噪声探索

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class NoisyLinear(nn.Module):
    """
    NoisyLinear层 - 参数空间噪声探索
    
    实现Noisy Networks for Exploration论文中的参数空间噪声机制。
    在训练模式下添加噪声进行探索，在eval模式下关闭噪声确保推理可复现性。
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 std_init: float = 0.5):
        """
        初始化NoisyLinear层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            bias: 是否使用偏置
            std_init: 噪声标准差初始值
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重参数：均值和标准差
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        
        # 偏置参数：均值和标准差
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
        
        # 注册噪声缓冲区（不会被保存为参数）
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        if bias:
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_buffer('bias_epsilon', None)
        
        # 初始化参数
        self.reset_parameters()
        
        # 生成初始噪声
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化网络参数"""
        # 权重均值使用Xavier初始化
        nn.init.xavier_uniform_(self.weight_mu)
        
        # 权重标准差初始化
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        # 偏置参数初始化
        if self.bias_mu is not None:
            nn.init.constant_(self.bias_mu, 0)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """重新生成噪声"""
        # 生成输入和输出的独立噪声
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # 使用外积生成权重噪声
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        
        # 生成偏置噪声
        if self.bias_epsilon is not None:
            self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        生成缩放噪声
        
        使用factorized Gaussian noise来减少计算复杂度
        """
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 输入张量
            
        Returns:
            输出张量
        """
        if self.training:
            # 训练模式：使用带噪声的权重
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = None
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # 推理模式：使用确定性权重
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)
    
    def extra_repr(self) -> str:
        """返回层的额外表示信息"""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias_mu is not None}, std_init={self.std_init}'


def replace_linear_with_noisy(module: nn.Module, std_init: float = 0.5) -> nn.Module:
    """
    递归地将模块中的所有nn.Linear层替换为NoisyLinear层
    
    Args:
        module: 要处理的模块
        std_init: NoisyLinear的噪声标准差初始值
        
    Returns:
        处理后的模块
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 替换Linear层为NoisyLinear层
            noisy_layer = NoisyLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                std_init=std_init
            )
            
            # 复制原始权重和偏置到新层的均值参数
            with torch.no_grad():
                noisy_layer.weight_mu.copy_(child.weight)
                if child.bias is not None:
                    noisy_layer.bias_mu.copy_(child.bias)
            
            setattr(module, name, noisy_layer)
        else:
            # 递归处理子模块
            replace_linear_with_noisy(child, std_init)
    
    return module


def reset_noise_in_module(module: nn.Module):
    """
    重置模块中所有NoisyLinear层的噪声
    
    Args:
        module: 要处理的模块
    """
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()