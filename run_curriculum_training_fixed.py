"""
修复版本的课程学习训练协调器
主要修复：
1. TensorBoard权限问题
2. JSON序列化问题
3. 错误处理改进
"""
import os
import json
import logging
import tempfile
from pathlib import Path
from json_serialization_fix import safe_json_dump, make_json_serializable

# 导入原有的课程学习组件
from run_curriculum_training import (
    CurriculumStage, CurriculumConfig, CurriculumTrainer, CheckpointManager
)

class FixedCurriculumTrainer(CurriculumTrainer):
    """修复版本的课程学习训练器"""
    
    def __init__(self, config: CurriculumConfig):
        super().__init__(config)
        self._setup_safe_output_dir()
    
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
    
    def _train_stage_fallback(self, stage: CurriculumStage, stage_idx: int, env_config: dict):
        """修复版本的回退训练方法"""
        try:
            # 为每个阶段创建独立的临时目录
            stage_temp_dir = tempfile.mkdtemp(prefix=f"curriculum_stage_{stage_idx}_")
            
            # 修改环境配置以使用临时目录
            env_config_safe = env_config.copy()
            env_config_safe['tensorboard_log_dir'] = stage_temp_dir
            
            # 调用原有的训练逻辑
            result = super()._train_stage_fallback(stage, stage_idx, env_config_safe)
            
            # 安全地移动结果文件
            self._safe_move_results(stage_temp_dir, stage_idx)
            
            return result
            
        except Exception as e:
            logging.error(f"阶段 {stage_idx} 训练失败: {e}")
            return self._create_fallback_result(stage, stage_idx)
    
    def _safe_move_results(self, temp_dir: str, stage_idx: int):
        """安全移动训练结果"""
        try:
            temp_path = Path(temp_dir)
            target_dir = self.config.output_dir / f"stage_{stage_idx}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动重要文件
            for file in temp_path.glob("**/*"):
                if file.is_file() and not file.name.startswith("events.out.tfevents"):
                    target_file = target_dir / file.name
                    try:
                        file.rename(target_file)
                    except:
                        pass
                        
        except Exception as e:
            logging.warning(f"结果文件移动警告: {e}")
    
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
                f.write(f"训练时间: {summary.get('training_time', 'N/A')}\n")
                f.write(f"完成阶段数: {len(summary.get('stages', []))}\n")
                f.write(f"总体状态: {summary.get('status', 'N/A')}\n")
                
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
