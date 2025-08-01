"""
完整的课程学习训练脚本（基于成功的最小化测试）
"""
import os
import sys
import time
from pathlib import Path

# 设置环境变量禁用TensorBoard
os.environ['DISABLE_TENSORBOARD'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

def run_curriculum_stage(stage_config, stage_idx):
    """运行单个课程学习阶段"""
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 导入必要模块
        from main import run_scenario
        from config import Config
        from entities import UAV, Target
        
        print(f"开始训练阶段 {stage_idx + 1}: {stage_config['name']}")
        
        # 创建配置
        config = Config()
        config.training_config.episodes = stage_config['training_episodes']
        
        # 根据阶段配置创建UAV和目标
        num_uavs = stage_config['num_uavs']
        map_size = stage_config['map_size']
        
        # 创建UAV（均匀分布在地图左侧）
        base_uavs = []
        for i in range(num_uavs):
            x = 10 + (i * 10) % (map_size // 4)
            y = 10 + (i * 15) % (map_size // 2)
            uav = UAV(i, [x, y], 0, [1, 1], 100, [5, 10], 7)
            base_uavs.append(uav)
        
        # 创建目标（均匀分布在地图右侧）
        base_targets = []
        for i in range(num_uavs):
            x = map_size - 20 + (i * 5) % 20
            y = 20 + (i * 10) % (map_size - 40)
            target = Target(i, [x, y], [1, 1], 10)
            base_targets.append(target)
        
        # 根据障碍物密度创建简单障碍物
        obstacles = []  # 暂时使用空障碍物列表
        
        # 创建阶段输出目录
        stage_output_dir = Path(f"curriculum_output_stage_{stage_idx}_{stage_config['name']}")
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调用run_scenario函数
        start_time = time.time()
        result = run_scenario(
            config=config,
            base_uavs=base_uavs,
            base_targets=base_targets,
            obstacles=obstacles,
            scenario_name=f"curriculum_stage_{stage_idx}_{stage_config['name']}",
            network_type="DeepFCNResidual",
            save_visualization=True,
            show_visualization=False,
            save_report=True,
            force_retrain=True,
            incremental_training=False,
            output_base_dir=str(stage_output_dir)
        )
        
        training_time = time.time() - start_time
        
        print(f"阶段 {stage_config['name']} 训练完成，耗时: {training_time:.2f}秒")
        
        return {
            "success": True,
            "stage_name": stage_config["name"],
            "stage_idx": stage_idx,
            "training_time": training_time,
            "result": result,
            "output_dir": str(stage_output_dir)
        }
        
    except Exception as e:
        print(f"阶段 {stage_idx} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "stage_name": stage_config["name"],
            "stage_idx": stage_idx,
            "error": str(e)
        }

def main():
    """主函数"""
    print("开始完整的课程学习训练...")
    
    # 定义课程学习阶段
    stages = [
        {
            "name": "simple",
            "num_uavs": 3,
            "map_size": 50,
            "obstacle_density": 0.1,
            "target_distance": 20,
            "training_episodes": 200
        },
        {
            "name": "medium",
            "num_uavs": 5,
            "map_size": 75,
            "obstacle_density": 0.2,
            "target_distance": 35,
            "training_episodes": 300
        },
        {
            "name": "complex",
            "num_uavs": 8,
            "map_size": 100,
            "obstacle_density": 0.3,
            "target_distance": 50,
            "training_episodes": 400
        }
    ]
    
    # 创建总输出目录
    main_output_dir = Path("curriculum_training_complete_output")
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练摘要
    training_summary = {
        "start_time": time.time(),
        "stages": [],
        "total_stages": len(stages),
        "successful_stages": 0
    }
    
    # 逐个执行训练阶段
    for i, stage in enumerate(stages):
        stage_result = run_curriculum_stage(stage, i)
        training_summary["stages"].append(stage_result)
        
        if stage_result["success"]:
            training_summary["successful_stages"] += 1
            print(f"✓ 阶段 {i+1} ({stage['name']}) 成功完成")
        else:
            print(f"✗ 阶段 {i+1} ({stage['name']}) 失败")
    
    # 完成训练
    training_summary["end_time"] = time.time()
    training_summary["total_time"] = training_summary["end_time"] - training_summary["start_time"]
    
    # 保存训练摘要
    try:
        from json_serialization_fix import safe_json_dump
        summary_file = main_output_dir / "curriculum_training_summary.json"
        safe_json_dump(training_summary, summary_file)
        print(f"训练摘要已保存到: {summary_file}")
    except Exception as e:
        print(f"保存训练摘要失败: {e}")
        # 保存文本版本
        summary_text_file = main_output_dir / "curriculum_training_summary.txt"
        with open(summary_text_file, 'w', encoding='utf-8') as f:
            f.write("课程学习训练摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"总训练时间: {training_summary['total_time']:.2f}秒\n")
            f.write(f"成功阶段: {training_summary['successful_stages']}/{training_summary['total_stages']}\n")
            f.write("\n阶段详情:\n")
            for stage in training_summary["stages"]:
                f.write(f"  {stage['stage_name']}: {'成功' if stage['success'] else '失败'}\n")
                if stage['success']:
                    f.write(f"    训练时间: {stage.get('training_time', 'N/A')}秒\n")
                    f.write(f"    输出目录: {stage.get('output_dir', 'N/A')}\n")
        print(f"文本摘要已保存到: {summary_text_file}")
    
    print(f"\n课程学习训练完成!")
    print(f"成功完成阶段: {training_summary['successful_stages']}/{training_summary['total_stages']}")
    print(f"总训练时间: {training_summary['total_time']:.2f}秒")
    
    return training_summary

if __name__ == "__main__":
    main()