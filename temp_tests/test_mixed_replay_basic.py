#!/usr/bin/env python3
"""
混合经验回放机制基础功能测试
验证核心功能是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_experience_replay import MixedExperienceReplay, ExperiencePoolManager

def test_basic_functionality():
    """测试基础功能"""
    print("🧪 测试混合经验回放基础功能...")
    
    # 创建混合经验回放实例
    replay = MixedExperienceReplay(capacity_per_stage=100)
    
    # 测试初始状态
    assert replay.current_stage == 0, f"初始阶段应为0，实际为{replay.current_stage}"
    assert len(replay) == 0, f"初始长度应为0，实际为{len(replay)}"
    
    # 添加第一阶段经验
    for i in range(10):
        experience = {
            'obs': f'obs_{i}',
            'action': i,
            'reward': i * 0.1,
            'done': False
        }
        replay.add_experience(experience)
    
    assert len(replay) == 10, f"添加10个经验后长度应为10，实际为{len(replay)}"
    
    # 测试第一阶段采样（应该只从当前阶段采样）
    batch = replay.sample_mixed_batch(5)
    assert len(batch) == 5, f"采样5个经验，实际得到{len(batch)}"
    
    print("✅ 第一阶段功能正常")
    
    # 切换到第二阶段
    replay.set_current_stage(1)
    assert replay.current_stage == 1, f"切换后阶段应为1，实际为{replay.current_stage}"
    
    # 添加第二阶段经验
    for i in range(15):
        experience = {
            'obs': f'stage1_obs_{i}',
            'action': i + 100,
            'reward': (i + 10) * 0.1,
            'done': False
        }
        replay.add_experience(experience)
    
    # 测试混合采样（应该包含两个阶段的经验）
    batch = replay.sample_mixed_batch(10)
    assert len(batch) == 10, f"混合采样10个经验，实际得到{len(batch)}"
    
    # 验证批次中包含不同阶段的经验
    stage_ids = [exp.get('stage_id', 0) for exp in batch]
    unique_stages = set(stage_ids)
    assert len(unique_stages) >= 1, "混合采样应包含至少一个阶段的经验"
    
    print("✅ 混合采样功能正常")
    
    # 测试统计信息
    stats = replay.get_statistics()
    assert 'current_stage' in stats, "统计信息应包含当前阶段"
    assert 'stage_buffer_sizes' in stats, "统计信息应包含阶段缓冲区大小"
    
    print("✅ 统计信息功能正常")
    
    print("🎉 基础功能测试通过！")

def test_experience_pool_manager():
    """测试经验池管理器"""
    print("\n🧪 测试经验池管理器...")
    
    manager = ExperiencePoolManager(default_capacity=50)
    
    # 创建经验池
    pool1 = manager.create_pool(1, capacity=30)
    assert pool1 is not None, "应该成功创建经验池"
    assert pool1.capacity_per_stage == 30, f"容量应为30，实际为{pool1.capacity_per_stage}"
    
    # 测试获取经验池
    retrieved_pool = manager.get_pool(1)
    assert retrieved_pool is pool1, "获取的经验池应该是同一个实例"
    
    # 测试全局统计
    stats = manager.get_global_statistics()
    assert 'total_active_pools' in stats, "全局统计应包含活跃池数量"
    assert stats['total_active_pools'] == 1, f"应有1个活跃池，实际为{stats['total_active_pools']}"
    
    print("✅ 经验池管理器功能正常")
    print("🎉 经验池管理器测试通过！")

def test_mixed_sampling_ratios():
    """测试混合采样比例"""
    print("\n🧪 测试混合采样比例...")
    
    replay = MixedExperienceReplay(
        capacity_per_stage=1000,
        current_stage_ratio=0.7,
        historical_stage_ratio=0.3,
        min_historical_samples=10  # 降低最小历史样本要求
    )
    
    # 添加第一阶段经验
    replay.set_current_stage(0)
    for i in range(100):
        replay.add_experience({'stage': 0, 'data': i})
    
    # 添加第二阶段经验
    replay.set_current_stage(1)
    for i in range(100):
        replay.add_experience({'stage': 1, 'data': i + 100})
    
    # 测试混合采样
    batch = replay.sample_mixed_batch(100)
    
    # 统计不同阶段的经验数量
    stage_0_count = sum(1 for exp in batch if exp.get('stage_id') == 0)
    stage_1_count = sum(1 for exp in batch if exp.get('stage_id') == 1)
    
    print(f"阶段0经验数量: {stage_0_count}")
    print(f"阶段1经验数量: {stage_1_count}")
    
    # 验证比例大致正确（允许一定误差）
    expected_stage_1 = int(100 * 0.7)  # 当前阶段70%
    expected_stage_0 = int(100 * 0.3)  # 历史阶段30%
    
    # 允许±10%的误差
    assert abs(stage_1_count - expected_stage_1) <= 10, f"当前阶段比例偏差过大: 期望~{expected_stage_1}, 实际{stage_1_count}"
    assert abs(stage_0_count - expected_stage_0) <= 10, f"历史阶段比例偏差过大: 期望~{expected_stage_0}, 实际{stage_0_count}"
    
    print("✅ 混合采样比例正确")
    print("🎉 混合采样比例测试通过！")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_experience_pool_manager()
        test_mixed_sampling_ratios()
        print("\n🎊 所有测试通过！混合经验回放机制实现成功！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)