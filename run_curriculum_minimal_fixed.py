"""
修正版最小化课程学习训练脚本
根据实际的run_scenario函数签名调整
"""
import os
import sys
from pathlib import Path

# 设置环境变量禁用所有可能的问题源
os.environ['DISABLE_TENSORBOARD'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU
os.environ['OMP_NUM_THREADS'] = '1'

def minimal_training():
    """最小化训练函数"""
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 导入必要模块
        from main import run_scenario
        from config import Config
        from entities import UAV, Target
        
        print("开始最小化课程学习训练...")
        
        # 创建简单的配置
        config = Config()
        config.training_config.episodes = 100  # 很少的训练轮数用于测试
        
        # 创建简单的UAV和目标
        base_uavs = [
            UAV(0, [10, 10], 0, [1, 1], 100, [5, 10], 7),
            UAV(1, [20, 20], 0, [1, 1], 100, [5, 10], 7),
            UAV(2, [30, 30], 0, [1, 1], 100, [5, 10], 7)
        ]
        
        base_targets = [
            Target(0, [40, 40], [1, 1], 10),
            Target(1, [50, 50], [1, 1], 10),
            Target(2, [60, 60], [1, 1], 10)
        ]
        
        # 简单的障碍物（空列表）
        obstacles = []
        
        # 创建输出目录
        output_dir = Path("minimal_curriculum_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调用run_scenario函数（使用正确的参数）
        result = run_scenario(
            config=config,
            base_uavs=base_uavs,
            base_targets=base_targets,
            obstacles=obstacles,
            scenario_name="minimal_curriculum_test",
            network_type="DeepFCNResidual",
            save_visualization=False,
            show_visualization=False,
            save_report=False,
            force_retrain=True,
            incremental_training=False,
            output_base_dir=str(output_dir)
        )
        
        print("最小化训练完成!")
        print(f"结果: {result}")
        return result
        
    except Exception as e:
        print(f"最小化训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    minimal_training()