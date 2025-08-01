#!/usr/bin/env python3
"""
混合经验回放机制使用示例
展示如何在实际项目中使用混合经验回放机制
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_experience_replay import MixedExperienceReplay, ExperiencePoolManager
from rllib_mixed_replay_integration import create_mixed_replay_config, CurriculumLearningCallback

def example_basic_usage():
    """基础使用示例"""
    print("📚 基础使用示例")
    print("=" * 50)
    
    # 1. 创建混合经验回放实例
    replay = MixedExperienceReplay(
        capacity_per_stage=10000,      # 每阶段容量
        current_stage_ratio=0.7,       # 当前阶段采样比例
        historical_stage_ratio=0.3,    # 历史阶段采样比例
        max_stages_to_keep=3,          # 最多保留的历史阶段数
        min_historical_samples=500     # 启用混合采样的最小历史样本数
    )
    
    print(f"✅ 创建混合经验回放实例，当前阶段: {replay.current_stage}")
    
    # 2. 第一阶段训练 - 简单场景
    print("\n🎯 第一阶段训练（简单场景）")
    replay.set_current_stage(0)
    
    # 模拟添加第一阶段的经验
    for episode in range(5):
        for step in range(20):
            experience = {
                'obs': f'stage0_obs_{episode}_{step}',
                'action': step % 3,
                'reward': 0.1 * step,
                'done': step == 19,
                'info': {'episode': episode, 'step': step}
            }
            replay.add_experience(experience)
    
    print(f"添加了 {len(replay)} 个第一阶段经验")
    
    # 第一阶段采样（只从当前阶段）
    batch = replay.sample_mixed_batch(10)
    print(f"第一阶段采样了 {len(batch)} 个经验")
    
    # 3. 第二阶段训练 - 中等复杂度
    print("\n🎯 第二阶段训练（中等复杂度）")
    replay.set_current_stage(1)
    
    # 模拟添加第二阶段的经验
    for episode in range(3):
        for step in range(30):
            experience = {
                'obs': f'stage1_obs_{episode}_{step}',
                'action': (step + episode) % 4,
                'reward': 0.15 * step + 0.1 * episode,
                'done': step == 29,
                'info': {'episode': episode, 'step': step, 'complexity': 'medium'}
            }
            replay.add_experience(experience)
    
    print(f"当前阶段有 {len(replay)} 个经验")
    
    # 第二阶段混合采样（70%当前 + 30%历史）
    batch = replay.sample_mixed_batch(20)
    print(f"第二阶段混合采样了 {len(batch)} 个经验")
    
    # 统计不同阶段的经验比例
    stage_counts = {}
    for exp in batch:
        stage_id = exp.get('stage_id', 'unknown')
        stage_counts[stage_id] = stage_counts.get(stage_id, 0) + 1
    
    print("混合采样结果:")
    for stage_id, count in stage_counts.items():
        percentage = (count / len(batch)) * 100
        print(f"  阶段 {stage_id}: {count} 个经验 ({percentage:.1f}%)")
    
    # 4. 获取统计信息
    print("\n📊 统计信息")
    stats = replay.get_statistics()
    print(f"当前阶段: {stats['current_stage']}")
    print(f"阶段缓冲区大小: {stats['stage_buffer_sizes']}")
    print(f"总添加样本数: {stats['total_samples_added']}")
    print(f"混合批次生成数: {stats['mixed_batches_generated']}")

def example_experience_pool_manager():
    """经验池管理器使用示例"""
    print("\n\n📚 经验池管理器使用示例")
    print("=" * 50)
    
    # 创建经验池管理器
    manager = ExperiencePoolManager(
        default_capacity=5000,
        max_stages=4,
        auto_cleanup=True
    )
    
    print("✅ 创建经验池管理器")
    
    # 为不同阶段创建经验池
    stages = [0, 1, 2]
    for stage_id in stages:
        pool = manager.create_pool(stage_id, capacity=3000 + stage_id * 1000)
        print(f"创建阶段 {stage_id} 经验池，容量: {pool.capacity_per_stage}")
        
        # 添加一些示例经验
        for i in range(50 + stage_id * 20):
            experience = {
                'obs': f'stage{stage_id}_sample_{i}',
                'action': i % (stage_id + 2),
                'reward': 0.1 * i * (stage_id + 1),
                'stage_info': f'complexity_level_{stage_id}'
            }
            manager.add_experience_to_stage(stage_id, experience)
    
    # 阶段间经验转移
    print("\n🔄 阶段间经验转移")
    transferred = manager.transfer_experiences(0, 1, ratio=0.1)
    print(f"从阶段0向阶段1转移了 {transferred} 个经验")
    
    # 全局统计信息
    print("\n📊 全局统计信息")
    global_stats = manager.get_global_statistics()
    print(f"活跃经验池数量: {global_stats['total_active_pools']}")
    print(f"活跃阶段: {global_stats['active_stages']}")
    
    # 内存使用估算
    memory_info = manager.get_memory_usage_estimate()
    print(f"总样本数: {memory_info['total_samples']}")
    print(f"估算内存使用: {memory_info['estimated_memory_mb']:.2f} MB")
    print(f"平均每池样本数: {memory_info['average_samples_per_pool']:.1f}")

def example_curriculum_learning():
    """课程学习使用示例"""
    print("\n\n📚 课程学习使用示例")
    print("=" * 50)
    
    # 定义课程学习配置
    curriculum_config = {
        'stages': {
            0: {  # 简单场景：2-3个UAV，1-2个目标
                'max_episodes': 1000,
                'success_threshold': 0.8,
                'rollback_threshold': 0.6,
                'description': '简单场景 - 基础协调'
            },
            1: {  # 中等场景：4-6个UAV，3-4个目标
                'max_episodes': 1500,
                'success_threshold': 0.85,
                'rollback_threshold': 0.65,
                'description': '中等场景 - 复杂协调'
            },
            2: {  # 复杂场景：8-12个UAV，5-8个目标
                'max_episodes': 2000,
                'success_threshold': 0.9,
                'rollback_threshold': 0.7,
                'description': '复杂场景 - 大规模协调'
            }
        }
    }
    
    # 创建课程学习回调
    callback = CurriculumLearningCallback(curriculum_config)
    print("✅ 创建课程学习回调")
    
    # 模拟训练过程中的阶段判断
    print("\n🎯 模拟训练过程")
    
    # 模拟不同性能水平的episode数据
    test_scenarios = [
        {'episode_reward_mean': 0.85, 'description': '高性能 - 应该推进'},
        {'episode_reward_mean': 0.75, 'description': '中等性能 - 继续训练'},
        {'episode_reward_mean': 0.55, 'description': '低性能 - 可能回退'},
    ]
    
    for scenario in test_scenarios:
        episode_data = {'episode_reward_mean': scenario['episode_reward_mean']}
        
        should_advance = callback._should_advance_stage(episode_data)
        should_rollback = callback._should_rollback_stage(episode_data)
        
        print(f"\n场景: {scenario['description']}")
        print(f"  平均奖励: {scenario['episode_reward_mean']}")
        print(f"  是否推进: {should_advance}")
        print(f"  是否回退: {should_rollback}")

def example_rllib_integration():
    """Ray RLlib集成使用示例"""
    print("\n\n📚 Ray RLlib集成使用示例")
    print("=" * 50)
    
    try:
        # 创建DQN配置
        dqn_config = create_mixed_replay_config(
            algorithm_type="DQN",
            mixed_replay_config={
                'current_stage_ratio': 0.7,
                'historical_stage_ratio': 0.3,
                'max_stages_to_keep': 3,
                'buffer_capacity': 50000
            }
        )
        print("✅ 创建DQN混合回放配置")
        
        # 创建PPO配置
        ppo_config = create_mixed_replay_config(
            algorithm_type="PPO",
            mixed_replay_config={
                'current_stage_ratio': 0.8,
                'historical_stage_ratio': 0.2,
                'max_stages_to_keep': 2,
                'buffer_capacity': 25000
            }
        )
        print("✅ 创建PPO混合回放配置")
        
        print("\n🔧 配置详情:")
        print(f"DQN混合回放配置: {dqn_config.mixed_replay_config}")
        print(f"PPO混合回放配置: {ppo_config.mixed_replay_config}")
        
    except Exception as e:
        print(f"⚠️  RLlib集成示例跳过: {e}")

if __name__ == "__main__":
    print("🚀 混合经验回放机制使用示例")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_experience_pool_manager()
        example_curriculum_learning()
        example_rllib_integration()
        
        print("\n\n🎊 所有使用示例运行完成！")
        print("\n💡 提示:")
        print("- 在实际使用中，根据具体场景调整参数")
        print("- 监控统计信息以优化性能")
        print("- 结合课程学习策略获得最佳效果")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)