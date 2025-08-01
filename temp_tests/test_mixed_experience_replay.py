"""
混合经验回放机制测试
验证混合采样的正确性和有效性
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import logging

from ray.rllib.policy.sample_batch import SampleBatch

# 导入待测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_experience_replay import MixedExperienceReplay, ExperiencePoolManager
from rllib_mixed_replay_integration import MixedReplayDQN, CurriculumLearningCallback


class TestMixedExperienceReplay(unittest.TestCase):
    """
    混合经验回放缓冲区测试
    """
    
    def setUp(self):
        """测试初始化"""
        self.buffer = MixedExperienceReplay(
            capacity=1000,
            current_stage_ratio=0.7,
            historical_stage_ratio=0.3
        )
        
        # 创建测试数据
        self.test_batch_stage0 = SampleBatch({
            "obs": np.random.rand(10, 4),
            "actions": np.random.randint(0, 2, 10),
            "rewards": np.random.rand(10),
            "stage_id": np.zeros(10, dtype=int)
        })
        
        self.test_batch_stage1 = SampleBatch({
            "obs": np.random.rand(15, 4),
            "actions": np.random.randint(0, 2, 15),
            "rewards": np.random.rand(15),
            "stage_id": np.ones(15, dtype=int)
        })
    
    def test_buffer_initialization(self):
        """测试缓冲区初始化"""
        self.assertEqual(self.buffer.capacity, 1000)
        self.assertEqual(self.buffer.current_stage_ratio, 0.7)
        self.assertEqual(self.buffer.historical_stage_ratio, 0.3)
        self.assertEqual(self.buffer.current_stage_id, 0)
        self.assertEqual(len(self.buffer), 0)
    
    def test_add_experience_single_stage(self):
        """测试单阶段经验添加"""
        # 添加第一阶段经验
        self.buffer.add(self.test_batch_stage0)
        
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.stage_sizes[0], 1)
        self.assertIn(0, self.buffer.stage_buffers)
    
    def test_add_experience_multiple_stages(self):
        """测试多阶段经验添加"""
        # 添加第一阶段经验
        self.buffer.add(self.test_batch_stage0)
        
        # 切换到第二阶段
        self.buffer.set_current_stage(1)
        self.buffer.add(self.test_batch_stage1)
        
        self.assertEqual(len(self.buffer), 2)
        self.assertEqual(self.buffer.stage_sizes[0], 1)
        self.assertEqual(self.buffer.stage_sizes[1], 1)
        self.assertEqual(self.buffer.current_stage_id, 1)
    
    def test_single_stage_sampling(self):
        """测试单阶段采样"""
        # 只有第一阶段数据
        self.buffer.add(self.test_batch_stage0)
        
        sampled_batch = self.buffer.sample(5)
        
        self.assertIsInstance(sampled_batch, SampleBatch)
        # 单阶段时应该直接从当前阶段采样
    
    def test_mixed_stage_sampling(self):
        """测试混合阶段采样"""
        # 添加多阶段数据
        for _ in range(5):
            self.buffer.add(self.test_batch_stage0)
        
        self.buffer.set_current_stage(1)
        for _ in range(5):
            self.buffer.add(self.test_batch_stage1)
        
        # 测试混合采样
        sampled_batch = self.buffer.sample(10)
        
        self.assertIsInstance(sampled_batch, SampleBatch)
        # 验证采样比例（这里需要更详细的验证逻辑）
    
    def test_stage_switching(self):
        """测试阶段切换"""
        initial_stage = self.buffer.current_stage_id
        
        self.buffer.set_current_stage(2)
        
        self.assertEqual(self.buffer.current_stage_id, 2)
        self.assertNotEqual(self.buffer.current_stage_id, initial_stage)
    
    def test_buffer_size_maintenance(self):
        """测试缓冲区大小维护"""
        # 创建小容量缓冲区
        small_buffer = MixedExperienceReplay(capacity=10)
        
        # 添加超过容量的数据
        for i in range(15):
            batch = SampleBatch({
                "obs": np.random.rand(1, 4),
                "actions": np.random.randint(0, 2, 1),
                "rewards": np.random.rand(1),
                "stage_id": np.array([0])
            })
            small_buffer.add(batch)
        
        # 验证总大小不超过容量
        self.assertLessEqual(len(small_buffer), small_buffer.capacity)
    
    def test_old_stage_cleanup(self):
        """测试旧阶段数据清理"""
        buffer = MixedExperienceReplay(capacity=1000, max_stages_to_keep=2)
        
        # 添加多个阶段的数据
        for stage in range(5):
            buffer.set_current_stage(stage)
            for _ in range(3):
                batch = SampleBatch({
                    "obs": np.random.rand(2, 4),
                    "actions": np.random.randint(0, 2, 2),
                    "rewards": np.random.rand(2),
                    "stage_id": np.full(2, stage)
                })
                buffer.add(batch)
        
        # 验证只保留最近的阶段
        self.assertLessEqual(len(buffer.stage_buffers), buffer.max_stages_to_keep)
    
    def test_sampling_statistics(self):
        """测试采样统计信息"""
        # 添加多阶段数据
        for _ in range(3):
            self.buffer.add(self.test_batch_stage0)
        
        self.buffer.set_current_stage(1)
        for _ in range(3):
            self.buffer.add(self.test_batch_stage1)
        
        # 执行采样
        self.buffer.sample(10)
        
        stats = self.buffer.get_stats()
        
        self.assertIn('total_experiences', stats)
        self.assertIn('current_stage_id', stats)
        self.assertIn('sampling_stats', stats)
        self.assertEqual(stats['current_stage_id'], 1)
        self.assertEqual(stats['total_experiences'], 6)


class TestExperiencePoolManager(unittest.TestCase):
    """
    经验池管理器测试
    """
    
    def setUp(self):
        """测试初始化"""
        self.manager = ExperiencePoolManager()
    
    def test_create_buffer(self):
        """测试创建缓冲区"""
        buffer = self.manager.create_buffer("test_buffer", capacity=500)
        
        self.assertIsInstance(buffer, MixedExperienceReplay)
        self.assertEqual(buffer.capacity, 500)
        self.assertIn("test_buffer", self.manager.replay_buffers)
    
    def test_get_buffer(self):
        """测试获取缓冲区"""
        # 创建缓冲区
        original_buffer = self.manager.create_buffer("test_buffer")
        
        # 获取缓冲区
        retrieved_buffer = self.manager.get_buffer("test_buffer")
        
        self.assertIs(original_buffer, retrieved_buffer)
        
        # 测试获取不存在的缓冲区
        non_existent = self.manager.get_buffer("non_existent")
        self.assertIsNone(non_existent)
    
    def test_set_stage_for_all(self):
        """测试为所有缓冲区设置阶段"""
        # 创建多个缓冲区
        buffer1 = self.manager.create_buffer("buffer1")
        buffer2 = self.manager.create_buffer("buffer2")
        
        # 设置阶段
        self.manager.set_stage_for_all(2)
        
        self.assertEqual(buffer1.current_stage_id, 2)
        self.assertEqual(buffer2.current_stage_id, 2)
    
    def test_global_stats(self):
        """测试全局统计信息"""
        # 创建缓冲区并添加数据
        buffer = self.manager.create_buffer("test_buffer")
        test_batch = SampleBatch({
            "obs": np.random.rand(5, 4),
            "actions": np.random.randint(0, 2, 5),
            "rewards": np.random.rand(5)
        })
        buffer.add(test_batch)
        
        stats = self.manager.get_global_stats()
        
        self.assertIn('total_buffers', stats)
        self.assertIn('buffer_stats', stats)
        self.assertEqual(stats['total_buffers'], 1)
        self.assertIn('test_buffer', stats['buffer_stats'])


class TestCurriculumLearningCallback(unittest.TestCase):
    """
    课程学习回调函数测试
    """
    
    def setUp(self):
        """测试初始化"""
        self.curriculum_config = {
            'stages': {
                0: {'max_episodes': 100, 'success_threshold': 0.8, 'rollback_threshold': 0.6},
                1: {'max_episodes': 150, 'success_threshold': 0.85, 'rollback_threshold': 0.65},
                2: {'max_episodes': 200, 'success_threshold': 0.9, 'rollback_threshold': 0.7}
            }
        }
        self.callback = CurriculumLearningCallback(self.curriculum_config)
    
    def test_stage_advancement_conditions(self):
        """测试阶段推进条件"""
        # 模拟达到推进条件
        self.callback.stage_episodes = 100
        episode_data = {'episode_reward_mean': 0.85}
        
        should_advance = self.callback._should_advance_stage(episode_data)
        self.assertTrue(should_advance)
        
        # 模拟未达到推进条件
        episode_data = {'episode_reward_mean': 0.75}
        should_advance = self.callback._should_advance_stage(episode_data)
        self.assertFalse(should_advance)
    
    def test_stage_rollback_conditions(self):
        """测试阶段回退条件"""
        self.callback.current_stage = 1
        
        # 模拟需要回退的情况
        episode_data = {'episode_reward_mean': 0.5}
        should_rollback = self.callback._should_rollback_stage(episode_data)
        self.assertTrue(should_rollback)
        
        # 模拟不需要回退的情况
        episode_data = {'episode_reward_mean': 0.7}
        should_rollback = self.callback._should_rollback_stage(episode_data)
        self.assertFalse(should_rollback)
    
    @patch('mixed_experience_replay.experience_pool_manager')
    def test_stage_advancement(self, mock_manager):
        """测试阶段推进"""
        mock_algorithm = Mock()
        mock_algorithm.set_curriculum_stage = Mock()
        
        initial_stage = self.callback.current_stage
        self.callback._advance_stage(mock_algorithm)
        
        self.assertEqual(self.callback.current_stage, initial_stage + 1)
        mock_algorithm.set_curriculum_stage.assert_called_once_with(initial_stage + 1)
        mock_manager.set_stage_for_all.assert_called_once_with(initial_stage + 1)
    
    @patch('mixed_experience_replay.experience_pool_manager')
    def test_stage_rollback(self, mock_manager):
        """测试阶段回退"""
        mock_algorithm = Mock()
        mock_algorithm.set_curriculum_stage = Mock()
        
        self.callback.current_stage = 2
        self.callback._rollback_stage(mock_algorithm)
        
        self.assertEqual(self.callback.current_stage, 1)
        mock_algorithm.set_curriculum_stage.assert_called_once_with(1)
        mock_manager.set_stage_for_all.assert_called_once_with(1)


class TestMixedReplayIntegration(unittest.TestCase):
    """
    混合回放集成测试
    """
    
    def test_sampling_ratio_verification(self):
        """验证混合采样比例的正确性"""
        buffer = MixedExperienceReplay(
            capacity=1000,
            current_stage_ratio=0.7,
            historical_stage_ratio=0.3
        )
        
        # 添加第一阶段数据（标记为历史）
        for i in range(50):
            batch = SampleBatch({
                "obs": np.random.rand(1, 4),
                "actions": np.random.randint(0, 2, 1),
                "rewards": np.random.rand(1),
                "stage_id": np.array([0]),
                "batch_id": np.array([f"stage0_batch_{i}"])  # 用于追踪
            })
            buffer.add(batch)
        
        # 切换到第二阶段
        buffer.set_current_stage(1)
        
        # 添加第二阶段数据（当前阶段）
        for i in range(50):
            batch = SampleBatch({
                "obs": np.random.rand(1, 4),
                "actions": np.random.randint(0, 2, 1),
                "rewards": np.random.rand(1),
                "stage_id": np.array([1]),
                "batch_id": np.array([f"stage1_batch_{i}"])  # 用于追踪
            })
            buffer.add(batch)
        
        # 执行多次采样，验证比例
        total_samples = 0
        current_stage_samples = 0
        historical_stage_samples = 0
        
        for _ in range(10):  # 多次采样取平均
            sampled_batch = buffer.sample(100)
            if len(sampled_batch) > 0:
                stage_ids = sampled_batch.get("stage_id", [])
                current_count = np.sum(stage_ids == 1)
                historical_count = np.sum(stage_ids == 0)
                
                total_samples += len(stage_ids)
                current_stage_samples += current_count
                historical_stage_samples += historical_count
        
        if total_samples > 0:
            current_ratio = current_stage_samples / total_samples
            historical_ratio = historical_stage_samples / total_samples
            
            # 允许一定的误差范围
            self.assertAlmostEqual(current_ratio, 0.7, delta=0.1)
            self.assertAlmostEqual(historical_ratio, 0.3, delta=0.1)
    
    def test_catastrophic_forgetting_prevention(self):
        """测试灾难性遗忘防止机制"""
        buffer = MixedExperienceReplay(capacity=1000)
        
        # 模拟多个训练阶段
        stage_data = {}
        for stage in range(3):
            buffer.set_current_stage(stage)
            stage_batches = []
            
            for i in range(20):
                batch = SampleBatch({
                    "obs": np.random.rand(1, 4) + stage,  # 每个阶段有不同的特征
                    "actions": np.random.randint(0, 2, 1),
                    "rewards": np.random.rand(1) + stage * 0.1,
                    "stage_id": np.array([stage])
                })
                buffer.add(batch)
                stage_batches.append(batch)
            
            stage_data[stage] = stage_batches
        
        # 验证历史阶段数据仍然可以被采样到
        sampled_batch = buffer.sample(60)  # 采样较大批次
        
        if len(sampled_batch) > 0:
            stage_ids = sampled_batch.get("stage_id", [])
            unique_stages = np.unique(stage_ids)
            
            # 应该包含多个阶段的数据
            self.assertGreater(len(unique_stages), 1, "应该包含多个阶段的数据以防止灾难性遗忘")


def run_mixed_replay_tests():
    """
    运行所有混合经验回放测试
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestMixedExperienceReplay))
    test_suite.addTest(unittest.makeSuite(TestExperiencePoolManager))
    test_suite.addTest(unittest.makeSuite(TestCurriculumLearningCallback))
    test_suite.addTest(unittest.makeSuite(TestMixedReplayIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("开始混合经验回放机制测试...")
    result = run_mixed_replay_tests()
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！混合经验回放机制实现正确。")
    else:
        print(f"\n❌ 测试失败：{len(result.failures)} 个失败，{len(result.errors)} 个错误")
        
        for test, traceback in result.failures + result.errors:
            print(f"\n失败测试: {test}")
            print(f"错误信息: {traceback}")
