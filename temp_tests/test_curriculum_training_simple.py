# -*- coding: utf-8 -*-
"""
文件名: test_curriculum_training_simple.py
描述: 课程学习训练协调器简化测试代码
作者: AI Assistant
日期: 2024
"""

import os
import sys
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试模块导入"""
    print("测试1: 模块导入...")
    try:
        from run_curriculum_training import (
            CurriculumStage, CurriculumConfig, CurriculumTrainer, 
            CheckpointManager, create_env_wrapper
        )
        from config import Config
        from environment import UAVTaskEnv
        from scenarios import get_small_scenario, get_balanced_scenario
        print("✓ 所有模块导入成功")
        return True, {
            'CurriculumStage': CurriculumStage,
            'CurriculumConfig': CurriculumConfig,
            'CurriculumTrainer': CurriculumTrainer,
            'CheckpointManager': CheckpointManager,
            'Config': Config,
            'get_small_scenario': get_small_scenario
        }
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False, {}

def test_config_creation(modules):
    """测试配置创建"""
    print("\n测试2: 配置创建...")
    try:
        CurriculumConfig = modules['CurriculumConfig']
        Config = modules['Config']
        
        config = CurriculumConfig()
        print(f"✓ 课程配置创建成功，包含{len(config.stages)}个阶段")
        
        base_config = Config()
        print("✓ 基础配置创建成功")
        return True, {'config': config, 'base_config': base_config}
    except Exception as e:
        print(f"✗ 配置创建失败: {e}")
        return False, {}

def test_trainer_initialization(modules, configs):
    """测试训练器初始化"""
    print("\n测试3: 训练器初始化...")
    try:
        CurriculumTrainer = modules['CurriculumTrainer']
        config = configs['config']
        base_config = configs['base_config']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            trainer = CurriculumTrainer(config, base_config)
            print("✓ 训练器初始化成功")
            return True, {'trainer': trainer}
    except Exception as e:
        print(f"✗ 训练器初始化失败: {e}")
        return False, {}

def test_env_config(modules, configs, trainer_data):
    """测试环境配置"""
    print("\n测试4: 环境配置...")
    try:
        trainer = trainer_data['trainer']
        config = configs['config']
        
        stage = config.stages[0]
        env_config = trainer._create_env_config(stage)
        
        # 检查环境配置
        required_keys = ["uavs", "targets", "obstacles", "config", "obs_mode"]
        for key in required_keys:
            if key not in env_config:
                raise ValueError(f"缺少环境配置键: {key}")
        
        print("✓ 环境配置创建成功")
        print(f"  - UAV数量: {len(env_config['uavs'])}")
        print(f"  - 目标数量: {len(env_config['targets'])}")
        print(f"  - 障碍物数量: {len(env_config['obstacles'])}")
        print(f"  - 观测模式: {env_config['obs_mode']}")
        return True
    except Exception as e:
        print(f"✗ 环境配置失败: {e}")
        return False

def test_checkpoint_manager(modules):
    """测试检查点管理器"""
    print("\n测试5: 检查点管理...")
    try:
        CheckpointManager = modules['CheckpointManager']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # 测试保存检查点
            test_result = {
                "stage_name": "test",
                "episodes_trained": 10,
                "success": True
            }
            checkpoint_manager.save_stage_checkpoint(0, test_result)
            
            # 检查文件是否存在
            checkpoint_path = os.path.join(
                checkpoint_manager.checkpoints_dir,
                "stage_0_checkpoint.json"
            )
            if os.path.exists(checkpoint_path):
                print("✓ 检查点保存成功")
                return True
            else:
                raise FileNotFoundError("检查点文件未创建")
    except Exception as e:
        print(f"✗ 检查点管理失败: {e}")
        return False

def run_basic_functionality_test():
    """运行基本功能测试"""
    print("=" * 60)
    print("课程学习训练协调器 - 基本功能测试")
    print("=" * 60)
    
    try:
        # 测试1: 导入测试
        success, modules = test_imports()
        if not success:
            return False
        
        # 测试2: 配置创建测试
        success, configs = test_config_creation(modules)
        if not success:
            return False
        
        # 测试3: 训练器初始化测试
        success, trainer_data = test_trainer_initialization(modules, configs)
        if not success:
            return False
        
        # 测试4: 环境配置测试
        success = test_env_config(modules, configs, trainer_data)
        if not success:
            return False
        
        # 测试5: 检查点管理器测试
        success = test_checkpoint_manager(modules)
        if not success:
            return False
        
        print("\n" + "=" * 60)
        print("✓ 所有基本功能测试通过!")
        print("课程学习训练协调器已准备就绪")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生未预期错误: {e}")
        return False

def main():
    """主测试函数"""
    print("课程学习训练协调器简化测试套件")
    print("=" * 50)
    
    # 运行基本功能测试
    success = run_basic_functionality_test()
    
    if success:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n✗ 测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)