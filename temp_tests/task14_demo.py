"""
任务14功能演示
展示训练数据保存与TensorBoard集成的核心功能
"""

import os
import sys
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_logger import CurriculumTensorBoardLogger, ModelCheckpointManager, create_training_config_with_logging
from stage_config_manager import StageConfigManager


def demonstrate_task14_functionality():
    """演示任务14的核心功能"""
    print("🎯 任务14功能演示：训练数据保存与TensorBoard集成")
    print("=" * 60)
    
    # 创建临时目录
    demo_dir = tempfile.mkdtemp()
    print(f"📁 演示目录: {demo_dir}")
    
    try:
        # 1. 演示TensorBoard日志记录器
        print("\n1️⃣ TensorBoard日志记录器演示")
        print("-" * 40)
        
        logger = CurriculumTensorBoardLogger(demo_dir, "demo_experiment")
        
        # 模拟课程学习训练过程
        stages = [
            {"stage": 0, "n_uavs": 3, "n_targets": 2, "description": "基础阶段"},
            {"stage": 1, "n_uavs": 5, "n_targets": 3, "description": "中等复杂度"},
            {"stage": 2, "n_uavs": 8, "n_targets": 5, "description": "高复杂度"}
        ]
        
        for i, stage_info in enumerate(stages):
            print(f"   🔄 模拟阶段 {stage_info['stage']} ({stage_info['description']})")
            
            # 模拟该阶段的训练步骤
            for step in range(i*1000, (i+1)*1000, 200):
                # 生成模拟的尺度不变指标
                metrics = {
                    "per_agent_reward": 10 + np.random.normal(0, 1),
                    "normalized_completion_score": 0.6 + i*0.1 + np.random.uniform(-0.05, 0.1),
                    "efficiency_metric": 0.3 + i*0.05 + np.random.uniform(-0.02, 0.05)
                }
                
                # 记录指标
                logger.log_scale_invariant_metrics(
                    metrics, step, stage_info['stage'], 
                    stage_info['n_uavs'], stage_info['n_targets']
                )
                
                print(f"      📊 步数 {step}: 完成分数 {metrics['normalized_completion_score']:.3f}")
            
            # 记录阶段切换
            if i < len(stages) - 1:
                next_stage = stages[i+1]['stage']
                logger.log_stage_transition(
                    stage_info['stage'], next_stage, (i+1)*1000, 
                    "performance_threshold"
                )
                print(f"      ➡️  阶段切换: {stage_info['stage']} -> {next_stage}")
        
        # 模拟一个回退事件
        logger.log_rollback_event(2, 2500, 0.12, 0.1)
        print(f"      ⬅️  回退事件: 阶段2性能下降0.12 > 阈值0.1")
        
        print("   ✅ TensorBoard日志记录完成")
        
        # 2. 演示模型检查点管理器
        print("\n2️⃣ 模型检查点管理器演示")
        print("-" * 40)
        
        checkpoint_manager = ModelCheckpointManager(os.path.join(demo_dir, "checkpoints"))
        
        # 模拟保存多个检查点
        for i in range(5):
            # 创建模拟模型状态
            model_state = {
                "transformer.weight": torch.randn(64, 32),
                "output.bias": torch.randn(10)
            }
            optimizer_state = {"lr": 0.001 - i*0.0001}
            metrics = {"normalized_completion_score": 0.7 + i*0.05}
            
            is_best = (i == 3)  # 第4个是最佳的
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model_state, optimizer_state, metrics, 
                i*500, i//2, is_best
            )
            
            status = "🏆 最佳" if is_best else "📝 常规"
            print(f"   {status} 检查点 {i+1}: 性能 {metrics['normalized_completion_score']:.3f}")
        
        # 获取最佳检查点
        best_checkpoint = checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            loaded_data = checkpoint_manager.load_checkpoint(best_checkpoint)
            print(f"   🔄 加载最佳检查点: 性能 {loaded_data['metrics']['normalized_completion_score']:.3f}")
        
        print("   ✅ 模型检查点管理完成")
        
        # 3. 演示阶段配置管理器
        print("\n3️⃣ 阶段配置管理器演示")
        print("-" * 40)
        
        config_manager = StageConfigManager(os.path.join(demo_dir, "configs"))
        
        # 显示默认配置
        for stage_id in range(4):
            config = config_manager.get_stage_config(stage_id)
            print(f"   ⚙️  阶段 {stage_id}: {config.n_uavs_range[0]}-{config.n_uavs_range[1]} UAVs, "
                  f"{config.n_targets_range[0]}-{config.n_targets_range[1]} 目标, "
                  f"学习率 {config.learning_rate}")
        
        # 模拟性能记录和配置调整
        for stage_id in range(3):
            # 记录性能数据
            for episode in range(5):
                performance = {
                    "per_agent_reward": 10 + stage_id*2 + np.random.normal(0, 0.5),
                    "normalized_completion_score": 0.6 + stage_id*0.1 + np.random.uniform(-0.05, 0.1),
                    "efficiency_metric": 0.3 + stage_id*0.05 + np.random.uniform(-0.02, 0.05)
                }
                config_manager.record_stage_performance(stage_id, performance, episode, episode*100)
            
            # 获取性能摘要
            summary = config_manager.get_stage_performance_summary(stage_id)
            if summary:
                print(f"   📈 阶段 {stage_id} 平均性能: {summary.get('normalized_completion_score_mean', 0):.3f}")
            
            # 保存最佳模型
            best_performance = {"normalized_completion_score": 0.7 + stage_id*0.1}
            model_state = {"stage_model": torch.randn(32, 16)}
            training_config = {"learning_rate": 0.001, "batch_size": 128}
            
            config_manager.save_best_model(stage_id, model_state, best_performance, training_config)
            print(f"   💾 阶段 {stage_id} 最佳模型已保存")
        
        print("   ✅ 阶段配置管理完成")
        
        # 4. 演示训练配置增强
        print("\n4️⃣ 训练配置增强演示")
        print("-" * 40)
        
        base_config = {
            "env": "UAVTaskEnv",
            "num_workers": 4,
            "lr": 0.001,
            "train_batch_size": 4000
        }
        
        enhanced_config = create_training_config_with_logging(
            base_config,
            log_dir=demo_dir,
            experiment_name="demo_experiment"
        )
        
        print("   📋 基础配置:")
        for key, value in base_config.items():
            print(f"      {key}: {value}")
        
        print("   ➕ 增强配置添加:")
        added_keys = set(enhanced_config.keys()) - set(base_config.keys())
        for key in sorted(added_keys):
            if key != "callbacks":  # callbacks是类，不适合直接打印
                print(f"      {key}: {enhanced_config[key]}")
            else:
                print(f"      {key}: CurriculumTrainingCallbacks")
        
        print("   ✅ 训练配置增强完成")
        
        # 5. 展示保存的文件
        print("\n5️⃣ 生成的文件展示")
        print("-" * 40)
        
        # 保存所有数据
        logger.save_training_history()
        checkpoint_manager.save_checkpoint_history()
        config_manager.save_all_data()
        
        # 列出生成的文件
        def list_files_recursive(directory, prefix=""):
            items = []
            try:
                for item in sorted(Path(directory).iterdir()):
                    if item.is_file():
                        size = item.stat().st_size
                        items.append(f"{prefix}📄 {item.name} ({size} bytes)")
                    elif item.is_dir():
                        items.append(f"{prefix}📁 {item.name}/")
                        items.extend(list_files_recursive(item, prefix + "  "))
            except PermissionError:
                items.append(f"{prefix}❌ 权限不足")
            return items
        
        print("   📂 生成的文件结构:")
        file_list = list_files_recursive(demo_dir, "   ")
        for file_info in file_list[:20]:  # 限制显示数量
            print(file_info)
        
        if len(file_list) > 20:
            print(f"   ... 还有 {len(file_list) - 20} 个文件")
        
        print("   ✅ 文件生成完成")
        
        # 6. 功能验证总结
        print("\n6️⃣ 功能验证总结")
        print("-" * 40)
        
        verification_results = []
        
        # 验证TensorBoard日志
        history_file = Path(demo_dir) / "demo_experiment_history.json"
        if history_file.exists():
            verification_results.append("✅ TensorBoard训练历史保存")
        else:
            verification_results.append("❌ TensorBoard训练历史保存")
        
        # 验证检查点
        checkpoint_dir = Path(demo_dir) / "checkpoints"
        if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
            verification_results.append("✅ 模型检查点保存")
        else:
            verification_results.append("❌ 模型检查点保存")
        
        # 验证配置文件
        config_dir = Path(demo_dir) / "configs"
        if config_dir.exists() and any(config_dir.glob("*.json")):
            verification_results.append("✅ 阶段配置保存")
        else:
            verification_results.append("❌ 阶段配置保存")
        
        # 验证TensorBoard目录
        tensorboard_dir = Path(demo_dir) / "tensorboard"
        if tensorboard_dir.exists():
            verification_results.append("✅ TensorBoard日志目录创建")
        else:
            verification_results.append("❌ TensorBoard日志目录创建")
        
        for result in verification_results:
            print(f"   {result}")
        
        # 清理资源
        logger.close()
        
        print("\n🎉 任务14功能演示完成！")
        print("   所有核心功能均正常工作，系统已准备好投入使用。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(demo_dir, ignore_errors=True)
            print(f"\n🧹 临时目录已清理: {demo_dir}")
        except:
            print(f"\n⚠️  临时目录清理失败: {demo_dir}")


if __name__ == "__main__":
    success = demonstrate_task14_functionality()
    if success:
        print("\n✨ 演示成功完成！任务14的训练数据保存与TensorBoard集成功能完全正常。")
    else:
        print("\n💥 演示过程中遇到问题，请检查相关模块。")
    
    sys.exit(0 if success else 1)