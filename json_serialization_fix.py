"""
修复JSON序列化问题的补丁
"""
import json
import numpy as np
from datetime import datetime

def make_json_serializable(obj):
    """将对象转换为JSON可序列化格式"""
    if isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def safe_json_dump(data, filepath):
    """安全的JSON保存函数"""
    try:
        serializable_data = make_json_serializable(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"JSON保存失败: {e}")
        return False
