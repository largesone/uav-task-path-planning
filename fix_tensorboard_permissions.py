"""
修复TensorBoard权限问题的工具脚本
"""
import os
import shutil
import tempfile
from pathlib import Path

def fix_tensorboard_permissions():
    """修复TensorBoard权限问题"""
    output_dir = Path("curriculum_training_output")
    
    # 如果输出目录存在，先备份然后重新创建
    if output_dir.exists():
        backup_dir = Path(f"curriculum_training_output_backup_{int(time.time())}")
        shutil.move(str(output_dir), str(backup_dir))
        print(f"已备份原目录到: {backup_dir}")
    
    # 重新创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置目录权限
    os.chmod(output_dir, 0o755)
    
    print(f"已重新创建输出目录: {output_dir}")
    return True

if __name__ == "__main__":
    import time
    fix_tensorboard_permissions()
