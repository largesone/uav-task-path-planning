"""
无TensorBoard版本的课程学习训练
专门解决权限问题
"""
import os
import sys
import logging
import time
import tempfile
from pathlib import Path

def disable_tensorboard():
    """禁用TensorBoard相关功能"""
    # 设置环境变量禁用TensorBoard
    os.environ['DISABLE_TENSORBOARD'] = '1'
    os.environ['NO_TENSORBOARD'] = '1'

def setup_safe_environment():
    """设置安全的训练环境"""
    disable_tensorboard()
    
    # 创建临时目录作为工作目录
    temp_dir = tempfile.mkdtemp(prefix="curriculum_safe_")
    os.chdir(temp_dir)
    
    logging.info(f"使用安全工作目录: {temp_dir}")
    return temp_dir

def run_single_stage(stage_config, stage_idx):
    """运行单个训练阶段"""
    try:
        # 导入训练模块
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from main import run_scenario
        
        logging.info(f"开始训练阶段 {stage_idx}: {stage_config['name']}")
        
        # 创建阶段输出目录
        stage_dir = Path(f"stage_{stage_idx}_{stage_config['name']}")
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行训练（禁用TensorBoard）
        result = run_scenario(
            scenario_type="curriculum_learning",
            num_uavs=stage_config["num_uavs"],
            map_size=stage_config["map_size"],
            obstacle_tolerance=stage_config["obstacle_tolerance"],
            target_distance=stage_config["target_distance"],
            training_episodes=stage_config["training_episodes"],
            network_type="DeepFCNResidual",
            enable_double_dqn=True,
            enable_gradient_clipping=True,
            exploration_decay=0.995,
            output_dir=str(stage_dir),
            disable_tensorboard=True  # 明确禁用
        )
        
        return {
            "success": True,
            "result": result,
            "stage_name": stage_config["name"]
        }
        
    except Exception as e:
        logging.error(f"阶段 {stage_idx} 训练失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "stage_name": stage_config["name"]
        }

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置安全环境
    safe_dir = setup_safe_environment()
    
    # 定义训练阶段
    stages = [
        {
            "name": "simple",
            "num_uavs": 3,
            "map_size": 50,
            "obstacle_tolerance": 0.1,
            "target_distance": 20,
            "training_episodes": 300  # 减少训练轮数避免长时间运行
        },
        {
            "name": "medium",
            "num_uavs": 5,
            "map_size": 75,
            "obstacle_tolerance": 0.2,
            "target_distance": 35,
            "training_episodes": 500
        }
    ]
    
    results = []
    
    try:
        for i, stage in enumerate(stages):
            stage_result = run_single_stage(stage, i)
            results.append(stage_result)
            
            if not stage_result["success"]:
                logging.warning(f"阶段 {i} 失败，继续下一阶段")
        
        # 保存结果摘要
        summary = {
            "total_stages": len(stages),
            "successful_stages": sum(1 for r in results if r["success"]),
            "results": results,
            "safe_directory": safe_dir
        }
        
        logging.info(f"训练完成，成功阶段: {summary['successful_stages']}/{summary['total_stages']}")
        return summary
        
    except Exception as e:
        logging.error(f"训练过程发生严重错误: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    main()
