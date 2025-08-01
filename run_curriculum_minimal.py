"""
最小化课程学习训练脚本
只包含核心功能，避免所有可能的权限和依赖问题
"""
import os
import logging
import sys
from pathlib import Path

# 设置环境变量禁用所有可能的问题源
os.environ['DISABLE_TENSORBOARD'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU
os.environ['OMP_NUM_THREADS'] = '1'

def minimal_training():
    """最小化训练函数"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 导入必要模块
        from main import run_scenario
        
        logging.info("开始最小化课程学习训练...")
        
        # 只运行一个简单阶段
        result = run_scenario(
            scenario_type="curriculum_learning",
            num_uavs=3,
            map_size=50,
            obstacle_tolerance=0.1,
            target_distance=20,
            training_episodes=100,  # 很少的训练轮数用于测试
            network_type="DeepFCNResidual",
            enable_double_dqn=True,
            enable_gradient_clipping=True,
            exploration_decay=0.995,
            output_dir="minimal_output"
        )
        
        logging.info("最小化训练完成!")
        return result
        
    except Exception as e:
        logging.error(f"最小化训练失败: {e}")
        return None

if __name__ == "__main__":
    minimal_training()
