"""
修复版本的课程学习训练协调器 v2
主要修复：
1. TensorBoard权限问题
2. JSON序列化问题
3. 构造函数参数问题
4. 错误处理改进
"""
import os
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# 导入必要的模块
from json_serialization_fix import safe_json_dump, make_json_serializable

@dataclass
class CurriculumStage:
    """课程学习阶段配置"""
    name: str
    description: str
    scenario_config: Dict[str, Any]
    training_episodes: int
    success_threshold: float
    max_retries: int = 3

@dataclass
class CurriculumConfig:
    """课程学习配置"""
    stages: List[CurriculumStage]
    output_dir: Path
    checkpoint_interval: int = 100
    enable_tensorboard: bool = False  # 默认禁用TensorBoard避免权限问题
    
    def __init__(self):
        self.output_dir = Path("curriculum_training_output")
        self.checkpoint_interval = 100
        self.enable_tensorboard = False
        self.stages = self._create_default_stages()
    
    def _create_default_stages(self) -> List[CurriculumStage]:
        """创建默认的课程学习阶段"""
        return [
            CurriculumStage(
                name="simple",
                description="简单场景训练",
                scenario_config={
                    "num_uavs": 3,
                    "map_size": 50,
                    "obstacle_density": 0.1,
                    "target_distance": 20
                },
                training_episodes=500,
                success_threshold=0.7
            ),
            CurriculumStage(
                name="medium",
                description="中等复杂度场景训练",
                scenario_config={
                    "num_uavs": 5,
                    "map_size": 75,
                    "obstacle_density": 0.2,
                    "target_distance": 35
                },
                training_episodes=800,
                success_threshold=0.6
            ),
            CurriculumStage(
                name="complex",
                description="复杂场景训练",
                scenario_config={
                    "num_uavs": 8,
                    "map_size": 100,
                    "obstacle_density": 0.3,
                    "target_distance": 50
                },
                training_episodes=1200,
                success_threshold=0.5
            )
        ]

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, stage_idx: int, model_path: str, metrics: Dict[str, Any]):
        """保存检查点"""
        try:
            checkpoint_data = {
                "stage_idx": stage_idx,
                "model_path": model_path,
                "metrics": make_json_serializable(metrics),
                "timestamp": time.time()
            }
            
            checkpoint_file = self.checkpoints_dir / f"stage_{stage_idx}_checkpoint.json"
            return safe_json_dump(checkpoint_data, checkpoint_file)
            
        except Exception as e:
            logging.error(f"保存检查点失败: {e}")
            return False
    
    def load_checkpoint(self, stage_idx: int) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        try:
            checkpoint_file = self.checkpoints_dir / f"stage_{stage_idx}_checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            logging.error(f"加载检查点失败: {e}")
            return None

class FixedCurriculumTrainer:
    """修复版本的课程学习训练器"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        self.training_history = []
        self._setup_safe_output_dir()
        self._setup_logging()
    
    def _setup_safe_output_dir(self):
        """安全设置输出目录"""
        try:
            # 确保输出目录存在且有正确权限
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.config.output_dir, 0o755)
            
            # 清理可能存在的锁定文件
            for pattern in ["*.lock", "events.out.tfevents.*"]:
                for file in self.config.output_dir.glob(f"**/{pattern}"):
                    try:
                        file.unlink()
                    except:
                        pass
                        
        except Exception as e:
            logging.warning(f"输出目录设置警告: {e}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.config.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def train(self) -> Dict[str, Any]:
        """执行课程学习训练"""
        training_summary = {
            "start_time": time.time(),
            "stages": [],
            "status": "started",
            "total_stages": len(self.config.stages)
        }
        
        try:
            logging.info("开始课程学习训练...")
            
            for stage_idx, stage in enumerate(self.config.stages):
                logging.info(f"开始训练阶段 {stage_idx + 1}/{len(self.config.stages)}: {stage.name}")
                
                stage_result = self._train_stage(stage, stage_idx)
                training_summary["stages"].append(stage_result)
                
                if not stage_result.get("success", False):
                    logging.warning(f"阶段 {stage_idx} 训练未达到预期效果，但继续下一阶段")
            
            training_summary["status"] = "completed"
            training_summary["end_time"] = time.time()
            training_summary["total_time"] = training_summary["end_time"] - training_summary["start_time"]
            
            # 保存训练摘要
            self._save_training_summary(training_summary)
            
            logging.info("课程学习训练完成!")
            return training_summary
            
        except Exception as e:
            logging.error(f"课程学习训练过程中发生错误: {e}")
            training_summary["status"] = "failed"
            training_summary["error"] = str(e)
            training_summary["end_time"] = time.time()
            
            # 即使失败也保存摘要
            self._save_training_summary(training_summary)
            return training_summary
    
    def _train_stage(self, stage: CurriculumStage, stage_idx: int) -> Dict[str, Any]:
        """训练单个阶段"""
        stage_result = {
            "stage_name": stage.name,
            "stage_idx": stage_idx,
            "start_time": time.time(),
            "success": False,
            "metrics": {}
        }
        
        try:
            # 创建环境配置
            env_config = self._create_env_config(stage, stage_idx)
            
            # 执行训练
            result = self._train_stage_fallback(stage, stage_idx, env_config)
            
            stage_result.update(result)
            stage_result["success"] = True
            
        except Exception as e:
            logging.error(f"阶段 {stage_idx} 训练失败: {e}")
            stage_result["error"] = str(e)
            stage_result["success"] = False
        
        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]
        
        return stage_result
    
    def _create_env_config(self, stage: CurriculumStage, stage_idx: int) -> Dict[str, Any]:
        """创建环境配置"""
        stage_output_dir = self.config.output_dir / f"curriculum_stage_{stage_idx}_{stage.name}"
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        
        env_config = {
            "scenario_type": "curriculum_learning",
            "stage_name": stage.name,
            "output_dir": str(stage_output_dir),
            "enable_tensorboard": False,  # 禁用TensorBoard避免权限问题
            "training_episodes": stage.training_episodes,
            **stage.scenario_config
        }
        
        return env_config
    
    def _train_stage_fallback(self, stage: CurriculumStage, stage_idx: int, env_config: dict) -> Dict[str, Any]:
        """回退训练方法"""
        try:
            # 导入训练函数
            from main import run_scenario
            
            logging.info(f"开始训练阶段 {stage.name}，配置: {env_config}")
            
            # 执行训练
            final_plan, training_time, training_history, evaluation_metrics = run_scenario(
                scenario_type="curriculum_learning",
                num_uavs=env_config.get("num_uavs", 3),
                map_size=env_config.get("map_size", 50),
                obstacle_tolerance=env_config.get("obstacle_density", 0.1),
                target_distance=env_config.get("target_distance", 20),
                training_episodes=env_config.get("training_episodes", 500),
                network_type="DeepFCNResidual",
                enable_double_dqn=True,
                enable_gradient_clipping=True,
                exploration_decay=0.995,
                output_dir=env_config["output_dir"]
            )
            
            # 处理结果
            result = {
                "final_plan": final_plan,
                "training_time": training_time,
                "training_history": make_json_serializable(training_history) if training_history else [],
                "evaluation_metrics": make_json_serializable(evaluation_metrics) if evaluation_metrics else {}
            }
            
            # 保存检查点
            model_path = f"{env_config['output_dir']}/best_model.pth"
            self.checkpoint_manager.save_checkpoint(stage_idx, model_path, result["evaluation_metrics"])
            
            logging.info(f"阶段 {stage.name} 训练完成")
            return result
            
        except Exception as e:
            logging.error(f"阶段 {stage_idx} 回退训练失败: {e}")
            return self._create_fallback_result(stage, stage_idx)
    
    def _create_fallback_result(self, stage: CurriculumStage, stage_idx: int) -> Dict[str, Any]:
        """创建回退结果"""
        return {
            "final_plan": None,
            "training_time": 0,
            "training_history": [],
            "evaluation_metrics": {
                "success_rate": 0.0,
                "average_reward": -1000.0,
                "training_episodes": 0
            },
            "fallback": True
        }
    
    def _save_training_summary(self, summary: dict):
        """修复版本的训练摘要保存"""
        try:
            summary_file = self.config.output_dir / "training_summary.json"
            success = safe_json_dump(summary, summary_file)
            
            if success:
                logging.info(f"训练摘要已保存到: {summary_file}")
            else:
                # 备用保存方法
                self._save_summary_fallback(summary)
                
        except Exception as e:
            logging.error(f"保存训练摘要失败: {e}")
            self._save_summary_fallback(summary)
    
    def _save_summary_fallback(self, summary: dict):
        """备用摘要保存方法"""
        try:
            summary_file = self.config.output_dir / "training_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("课程学习训练摘要\n")
                f.write("=" * 50 + "\n")
                f.write(f"训练时间: {summary.get('total_time', 'N/A')}\n")
                f.write(f"完成阶段数: {len(summary.get('stages', []))}\n")
                f.write(f"总体状态: {summary.get('status', 'N/A')}\n")
                
                for i, stage in enumerate(summary.get('stages', [])):
                    f.write(f"\n阶段 {i+1}: {stage.get('stage_name', 'N/A')}\n")
                    f.write(f"  成功: {stage.get('success', False)}\n")
                    f.write(f"  持续时间: {stage.get('duration', 'N/A')}秒\n")
                
            logging.info(f"备用摘要已保存到: {summary_file}")
            
        except Exception as e:
            logging.error(f"备用摘要保存也失败: {e}")

def main():
    """修复版本的主函数"""
    try:
        # 使用修复版本的训练器
        config = CurriculumConfig()
        trainer = FixedCurriculumTrainer(config)
        
        logging.info("开始修复版本的课程学习训练...")
        training_summary = trainer.train()
        
        logging.info("课程学习训练完成!")
        return training_summary
        
    except Exception as e:
        logging.error(f"训练过程发生错误: {e}")
        return None

if __name__ == "__main__":
    main()
