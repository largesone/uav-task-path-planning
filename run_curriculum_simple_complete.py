"""
简化版课程学习训练脚本
避免复杂的继承和依赖问题
"""
import os
import logging
import time
from pathlib import Path
from json_serialization_fix import safe_json_dump, make_json_serializable

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('curriculum_simple.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_curriculum_training():
    """运行课程学习训练"""
    setup_logging()
    
    # 创建输出目录
    output_dir = Path("curriculum_training_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义训练阶段
    stages = [
        {
            "name": "simple",
            "num_uavs": 3,
            "map_size": 50,
            "obstacle_tolerance": 0.1,
            "target_distance": 20,
            "training_episodes": 500
        },
        {
            "name": "medium", 
            "num_uavs": 5,
            "map_size": 75,
            "obstacle_tolerance": 0.2,
            "target_distance": 35,
            "training_episodes": 800
        },
        {
            "name": "complex",
            "num_uavs": 8,
            "map_size": 100,
            "obstacle_tolerance": 0.3,
            "target_distance": 50,
            "training_episodes": 1200
        }
    ]
    
    training_summary = {
        "start_time": time.time(),
        "stages": [],
        "status": "started"
    }
    
    try:
        from main import run_scenario
        
        for i, stage in enumerate(stages):
            logging.info(f"开始训练阶段 {i+1}/{len(stages)}: {stage['name']}")
            
            stage_output_dir = output_dir / f"stage_{i}_{stage['name']}"
            stage_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 执行训练
                final_plan, training_time, training_history, evaluation_metrics = run_scenario(
                    scenario_type="curriculum_learning",
                    num_uavs=stage["num_uavs"],
                    map_size=stage["map_size"],
                    obstacle_tolerance=stage["obstacle_tolerance"],
                    target_distance=stage["target_distance"],
                    training_episodes=stage["training_episodes"],
                    network_type="DeepFCNResidual",
                    enable_double_dqn=True,
                    enable_gradient_clipping=True,
                    exploration_decay=0.995,
                    output_dir=str(stage_output_dir)
                )
                
                stage_result = {
                    "stage_name": stage["name"],
                    "stage_idx": i,
                    "success": True,
                    "training_time": training_time,
                    "evaluation_metrics": make_json_serializable(evaluation_metrics) if evaluation_metrics else {}
                }
                
                logging.info(f"阶段 {stage['name']} 训练完成")
                
            except Exception as e:
                logging.error(f"阶段 {stage['name']} 训练失败: {e}")
                stage_result = {
                    "stage_name": stage["name"],
                    "stage_idx": i,
                    "success": False,
                    "error": str(e)
                }
            
            training_summary["stages"].append(stage_result)
        
        training_summary["status"] = "completed"
        training_summary["end_time"] = time.time()
        training_summary["total_time"] = training_summary["end_time"] - training_summary["start_time"]
        
        # 保存摘要
        summary_file = output_dir / "training_summary.json"
        safe_json_dump(training_summary, summary_file)
        
        logging.info("课程学习训练完成!")
        return training_summary
        
    except Exception as e:
        logging.error(f"训练过程发生错误: {e}")
        training_summary["status"] = "failed"
        training_summary["error"] = str(e)
        return training_summary

if __name__ == "__main__":
    run_curriculum_training()
