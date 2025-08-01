# -*- coding: utf-8 -*-
# 简单调试

import torch
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 直接检查TransformerGNN类
import importlib
import transformer_gnn
importlib.reload(transformer_gnn)  # 强制重新加载
from transformer_gnn import TransformerGNN

print("检查TransformerGNN类...")
print(f"模块文件路径: {transformer_gnn.__file__}")
print(f"TransformerGNN.__init__ 方法存在: {hasattr(TransformerGNN, '__init__')}")

# 检查__init__方法的源代码
import inspect
try:
    source = inspect.getsource(TransformerGNN.__init__)
    print(f"__init__方法源代码长度: {len(source)} 字符")
    
    # 检查是否包含局部注意力相关代码
    if "use_local_attention" in source:
        print("✓ 源代码包含 use_local_attention")
    else:
        print("✗ 源代码不包含 use_local_attention")
        
    if "LocalAttention" in source:
        print("✓ 源代码包含 LocalAttention")
    else:
        print("✗ 源代码不包含 LocalAttention")
        
except Exception as e:
    print(f"无法获取源代码: {e}")

# 尝试创建一个最小的实例
print("\n尝试创建最小实例...")

class MinimalSpace:
    def __init__(self, shape):
        self.shape = shape

try:
    # 最小配置
    obs_space = {'uav_features': MinimalSpace((2, 4))}
    action_space = MinimalSpace((5,))
    model_config = {"use_local_attention": True}
    
    print("调用TransformerGNN.__init__...")
    model = TransformerGNN(obs_space, action_space, 5, model_config, "test")
    print("创建成功")
    
    # 检查属性
    print(f"use_local_attention 属性: {hasattr(model, 'use_local_attention')}")
    if hasattr(model, 'use_local_attention'):
        print(f"use_local_attention 值: {model.use_local_attention}")
    
except Exception as e:
    print(f"创建失败: {e}")
    import traceback
    traceback.print_exc()