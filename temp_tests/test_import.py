# -*- coding: utf-8 -*-
# 测试导入

import sys
import os

# 添加根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from transformer_gnn import TransformerGNN
    print("成功导入TransformerGNN")
    
    # 检查是否成功导入了NoisyLinear
    from noisy_linear import NoisyLinear
    print("成功导入NoisyLinear")
    
    # 测试TransformerGNN是否能访问NoisyLinear
    import transformer_gnn
    print(f"TransformerGNN模块中的NoisyLinear: {hasattr(transformer_gnn, 'NoisyLinear')}")
    
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()