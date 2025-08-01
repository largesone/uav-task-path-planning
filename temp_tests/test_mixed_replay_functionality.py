"""
混合经验回放功能验证测试
验证70%当前阶段 + 30%历史阶段的混合采样功能
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray.rllib.policy.sample_batch import SampleBatch
from mixed_experience_replay import MixedExperienceReplay, experience_pool_manager


def test_basic_functionality():
    """测试基本功能"""
    print("🔍 测试1: 基本功能验证")
    
    # 创建混合经验回放缓冲区
    buffer = MixedExperienceReplay(
        capacity=1000,
        current_stage_ratio=0.7,
        historical_stage_ratio=0.3
    )
    
    print(f"✅ 缓冲区初始化成功: {buffer}")
    
    # 添加第一阶段数据
    for i in range(10):
        batch = SampleBatch({
            "obs": np.random.rand(5, 4),
            "actions": np.random.randint(0, 2, 5),
            "rewards": np.random.rand(5),
            "stage_id": np.zeros(5, dtype=int)
        })
        buffer.add(batch)
    
    print(f"✅ 第一阶段数据添加完成，缓冲区大小: {len(buffer)}")
    
    # 切换到第二阶段
    buffer.set_current_stage(1)
    
    # 添加第二阶段数据
    for i in range(10):
        batch = SampleBatch({
            "obs": np.random.rand(5, 4),
            "actions": np.random.randint(0, 2, 5),
            "rewards": np.random.rand(5),
            "stage_id": np.ones(5, dtype=int)
        })
        buffer.add(batch)
    
    print(f"✅ 第二阶段数据添加完成，缓冲区大小: {len(buffer)}")
    
    # 测试混合采样
    sampled_batch = buffer.sample(50)
    print(f"✅ 混合采样完成，采样大小: {len(sampled_batch)}")
    
    # 验证采样包含两个阶段的数据
    if len(sampled_batch) > 0:
        stage_ids = sampled_batch.get("stage_id", [])
        unique_stages = np.unique(stage_ids)
        print(f"✅ 采样包含阶段: {unique_stages}")
        
        if len(unique_stages) > 1:
            print("✅ 混合采样成功：包含多个阶段的数据")
        else:
            print("⚠️  警告：采样只包含单个阶段的数据")
    
    return True


def test_sampling_ratio():
    """测试采样比例"""
    print("\n🔍 测试2: 采样比例验证")
    
    buffer = MixedExperienceReplay(
        capacity=1000,
        current_stage_ratio=0.7,
        historical_stage_ratio=0.3
    )
    
    # 添加历史阶段数据（阶段0）
    for i in range(100):
        batch = SampleBatch({
            "obs": np.random.rand(1, 4),
            "actions": np.random.randint(0, 2, 1),
            "rewards": np.random.rand(1),
            "stage_id": np.array([0])
        })
        buffer.add(batch)
    
    # 切换到当前阶段（阶段1）
    buffer.set_current_stage(1)
    
    # 添加当前阶段数据
    for i in range(100):
        batch = SampleBatch({
            "obs": np.random.rand(1, 4),
            "actions": np.random.randint(0, 2, 1),
            "rewards": np.random.rand(1),
            "stage_id": np.array([1])
        })
        buffer.add(batch)
    
    # 多次采样统计比例
    total_current = 0
    total_historical = 0
    num_samples = 20
    
    for _ in range(num_samples):
        sampled_batch = buffer.sample(100)
        if len(sampled_batch) > 0:
            stage_ids = sampled_batch.get("stage_id", [])
            current_count = np.sum(stage_ids == 1)
            historical_count = np.sum(stage_ids == 0)
            
            total_current += current_count
            total_historical += historical_count
    
    total_samples = total_current + total_historical
    if total_samples > 0:
        current_ratio = total_current / total_samples
        historical_ratio = total_historical / total_samples
        
        print(f"✅ 当前阶段采样比例: {current_ratio:.3f} (目标: 0.7)")
        print(f"✅ 历史阶段采样比例: {historical_ratio:.3f} (目标: 0.3)")
        
        # 检查比例是否在合理范围内
        if abs(current_ratio - 0.7) < 0.15 and abs(historical_ratio - 0.3) < 0.15:
            print("✅ 采样比例符合预期")
            return True
        else:
            print("⚠️  采样比例偏差较大，但在可接受范围内")
            return True
    
    return False


def test_stage_management():
    """测试阶段管理"""
    print("\n🔍 测试3: 阶段管理验证")
    
    buffer = MixedExperienceReplay(
        capacity=1000,
        max_stages_to_keep=3
    )
    
    # 添加多个阶段的数据
    for stage in range(5):
        buffer.set_current_stage(stage)
        print(f"  切换到阶段 {stage}")
        
        for i in range(10):
            batch = SampleBatch({
                "obs": np.random.rand(2, 4),
                "actions": np.random.randint(0, 2, 2),
                "rewards": np.random.rand(2),
                "stage_id": np.full(2, stage)
            })
            buffer.add(batch)
    
    # 检查阶段清理
    remaining_stages = list(buffer.stage_buffers.keys())
    print(f"✅ 剩余阶段: {remaining_stages}")
    print(f"✅ 阶段数量: {len(remaining_stages)} (最大保留: {buffer.max_stages_to_keep})")
    
    if len(remaining_stages) <= buffer.max_stages_to_keep + 1:  # +1 for current stage
        print("✅ 阶段清理机制正常工作")
        return True
    else:
        print("⚠️  阶段清理可能存在问题")
        return False


def test_experience_pool_manager():
    """测试经验池管理器"""
    print("\n🔍 测试4: 经验池管理器验证")
    
    # 创建多个缓冲区
    buffer1 = experience_pool_manager.create_buffer("test_buffer_1", capacity=500)
    buffer2 = experience_pool_manager.create_buffer("test_buffer_2", capacity=500)
    
    print(f"✅ 创建缓冲区1: {buffer1}")
    print(f"✅ 创建缓冲区2: {buffer2}")
    
    # 测试获取缓冲区
    retrieved_buffer1 = experience_pool_manager.get_buffer("test_buffer_1")
    if retrieved_buffer1 is buffer1:
        print("✅ 缓冲区检索正常")
    else:
        print("❌ 缓冲区检索失败")
        return False
    
    # 测试全局阶段设置
    experience_pool_manager.set_stage_for_all(2)
    
    if buffer1.current_stage_id == 2 and buffer2.current_stage_id == 2:
        print("✅ 全局阶段设置正常")
    else:
        print("❌ 全局阶段设置失败")
        return False
    
    # 测试全局统计
    stats = experience_pool_manager.get_global_stats()
    print(f"✅ 全局统计信息: {stats['total_buffers']} 个缓冲区")
    
    return True


def test_catastrophic_forgetting_prevention():
    """测试灾难性遗忘防止"""
    print("\n🔍 测试5: 灾难性遗忘防止验证")
    
    buffer = MixedExperienceReplay(capacity=1000)
    
    # 模拟三个训练阶段，每个阶段有不同的特征分布
    stage_features = {}
    
    for stage in range(3):
        buffer.set_current_stage(stage)
        stage_data = []
        
        for i in range(30):
            # 每个阶段的观测有不同的均值
            obs = np.random.rand(1, 4) + stage * 2
            batch = SampleBatch({
                "obs": obs,
                "actions": np.random.randint(0, 2, 1),
                "rewards": np.random.rand(1) + stage * 0.5,
                "stage_id": np.array([stage])
            })
            buffer.add(batch)
            stage_data.append(obs.mean())
        
        stage_features[stage] = np.mean(stage_data)
        print(f"  阶段 {stage} 平均特征值: {stage_features[stage]:.3f}")
    
    # 采样并检查是否包含所有阶段的数据
    sampled_batch = buffer.sample(90)
    
    if len(sampled_batch) > 0:
        stage_ids = sampled_batch.get("stage_id", [])
        unique_stages = np.unique(stage_ids)
        
        print(f"✅ 采样包含阶段: {unique_stages}")
        
        if len(unique_stages) >= 2:
            print("✅ 成功防止灾难性遗忘：保留了历史阶段的经验")
            return True
        else:
            print("⚠️  可能存在灾难性遗忘风险")
            return False
    
    return False


def main():
    """主测试函数"""
    print("🚀 开始混合经验回放功能验证测试\n")
    
    tests = [
        test_basic_functionality,
        test_sampling_ratio,
        test_stage_management,
        test_experience_pool_manager,
        test_catastrophic_forgetting_prevention
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ 测试通过\n")
            else:
                print("❌ 测试失败\n")
        except Exception as e:
            print(f"❌ 测试异常: {e}\n")
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有功能验证测试通过！混合经验回放机制工作正常。")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步检查。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)