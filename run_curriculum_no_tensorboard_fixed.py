"""
完全禁用TensorBoard的课程学习训练脚本
专门解决阶段2 (medium) 的TensorBoard权限问题
"""
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# 强制禁用TensorBoard相关功能
os.environ['DISABLE_TENSORBOARD'] = '1'
os.environ['NO_TENSORBOARD'] = '1'
os.environ['TENSORBOARD_DISABLED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

def patch_tensorboard_imports():
    """动态禁用TensorBoard导入"""
    import sys
    
    # 创建一个虚拟的TensorBoard模块
    class MockTensorBoard:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_histogram(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    # 替换TensorBoard相关模块
    sys.modules['torch.utils.tensorboard'] = type('MockModule', (), {
        'SummaryWriter': MockTensorBoard
    })()
    sys.modules['tensorboard'] = type('MockModule', (), {})()

def create_safe_temp_directory():
    """创建安全的临时目录"""
    temp_base = tempfile.gettempdir()
    safe_temp = os.path.join(temp_base, f"curriculum_safe_{int(time.time())}")
    os.makedirs(safe_temp, exist_ok=True)
    
    # 设置目录权限
    try:
        os.chmod(safe_temp, 0o777)
    except:
        pass
    
    return safe_temp

def run_curriculum_stage_safe(stage_config, stage_idx):
    """安全运行单个课程学习阶段，完全避免TensorBoard问题"""
    try:
        # 预先禁用TensorBoard
        patch_tensorboard_imports()
        
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 创建安全的临时工作目录
        safe_temp_dir = create_safe_temp_directory()
        original_cwd = os.getcwd()
        
        try:
            # 切换到安全目录
            os.chdir(safe_temp_dir)
            
            # 导入必要模块
            from main import run_scenario
            from config import Config
            from entities import UAV, Target
            
            print(f"开始训练阶段 {stage_idx + 1}: {stage_config['name']}")
            print(f"使用安全工作目录: {safe_temp_dir}")
            
            # 创建配置并禁用TensorBoard
            config = Config()
            config.training_config.episodes = stage_config['training_episodes']
            
            # 强制禁用TensorBoard相关配置
            if hasattr(config, 'tensorboard_enabled'):
                config.tensorboard_enabled = False
            if hasattr(config, 'enable_tensorboard'):
                config.enable_tensorboard = False
            
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
            
            # 简单障碍物
            obstacles = []
            
            # 创建本地输出目录（在安全目录内）
            local_output_dir = Path(f"stage_{stage_idx}_{stage_config['name']}")
            local_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 修改GraphRLSolver以禁用TensorBoard
            def patch_solver_init():
                """动态修补求解器以禁用TensorBoard"""
                import main
                original_init = main.GraphRLSolver.__init__
                
                def new_init(self, *args, **kwargs):
                    # 强制禁用TensorBoard
                    if len(args) > 9:
                        args = list(args)
                        args[9] = None  # tensorboard_dir参数设为None
                    kwargs['tensorboard_dir'] = None
                    return original_init(self, *args, **kwargs)
                
                main.GraphRLSolver.__init__ = new_init
            
            # 应用补丁
            patch_solver_init()
            
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
                output_base_dir=str(local_output_dir)
            )
            
            training_time = time.time() - start_time
            
            # 将结果复制回原始目录
            final_output_dir = Path(original_cwd) / f"curriculum_safe_output_stage_{stage_idx}_{stage_config['name']}"
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制重要文件
            for item in local_output_dir.rglob("*"):
                if item.is_file() and not item.name.startswith("events.out.tfevents"):
                    relative_path = item.relative_to(local_output_dir)
                    target_path = final_output_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(item, target_path)
                    except:
                        pass
            
            print(f"阶段 {stage_config['name']} 训练完成，耗时: {training_time:.2f}秒")
            print(f"结果已保存到: {final_output_dir}")
            
            return {
                "success": True,
                "stage_name": stage_config["name"],
                "stage_idx": stage_idx,
                "training_time": training_time,
                "result": result,
                "output_dir": str(final_output_dir),
                "safe_temp_dir": safe_temp_dir
            }
            
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            # 清理临时目录
            try:
                shutil.rmtree(safe_temp_dir)
            except:
                print(f"警告: 无法清理临时目录 {safe_temp_dir}")
        
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
    """主函数 - 专门修复阶段2的TensorBoard问题"""
    print("开始无TensorBoard的课程学习训练...")
    
    # 只运行阶段2 (medium) 来测试修复
    stages = [
        {
            "name": "medium",
            "num_uavs": 5,
            "map_size": 75,
            "obstacle_density": 0.2,
            "target_distance": 35,
            "training_episodes": 300
        }
    ]
    
    # 创建总输出目录
    main_output_dir = Path("curriculum_safe_training_output")
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练摘要
    training_summary = {
        "start_time": time.time(),
        "stages": [],
        "total_stages": len(stages),
        "successful_stages": 0
    }
    
    # 执行训练阶段
    for i, stage in enumerate(stages):
        stage_result = run_curriculum_stage_safe(stage, i)
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
        summary_file = main_output_dir / "curriculum_safe_training_summary.json"
        safe_json_dump(training_summary, summary_file)
        print(f"训练摘要已保存到: {summary_file}")
    except Exception as e:
        print(f"保存训练摘要失败: {e}")
        # 保存文本版本
        summary_text_file = main_output_dir / "curriculum_safe_training_summary.txt"
        with open(summary_text_file, 'w', encoding='utf-8') as f:
            f.write("安全课程学习训练摘要\n")
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
    
    print(f"\n安全课程学习训练完成!")
    print(f"成功完成阶段: {training_summary['successful_stages']}/{training_summary['total_stages']}")
    print(f"总训练时间: {training_summary['total_time']:.2f}秒")
    
    return training_summary

if __name__ == "__main__":
    main()