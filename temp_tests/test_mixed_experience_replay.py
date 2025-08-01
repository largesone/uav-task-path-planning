"""
混合经验回放机制测试

该测试文件验证MixedExperienceReplay类的正确性和有效性，包括：
1. 基本功能测试
2. 混合采样策略测试
3. 多阶段管理测试
4. 统计信息验证
"""

import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_experience_replay import MixedExperienceReplay, RLlibMixedReplayBuffer


class TestMixedExperienceReplay(unittest.TestCase):
    """混合经验回放机制测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.replay = MixedExperienceReplay(
            capacity_per_stage=1000,
            current_stage_ratio=0.7,
            historical_stage_ratio=0.3,
            max_stages_to_keep=3,
            min_historical_samples=50
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.replay.current_stage, 1)
        self.assertEqual(self.replay.capacity_per_stage, 1000)
        self.assertEqual(self.replay.current_stage_ratio, 0.7)
        self.assertEqual(self.replay.historical_stage_ratio, 0.3)
        
        stats = self.replay.get_statistics()
        self.assertEqual(stats['current_stage'], 1)
        self.assertEqual(stats['total_samples_added'], 0)
    
    def test_add_experience_single_stage(self):
        """测试单阶段经验添加"""
        # 添加一些经验到第一阶段
        for i in range(100):
            experience = {
                'obs': np.random.random(10),
                'action': np.random.randint(0, 4),
                'reward': np.random.random(),
                'done': False
            }
            self.replay.add_experience(experience)
        
        stats = self.replay.get_statistics()
        self.assertEqual(stats['total_samples_added'], 100)
        self.assertEqual(stats['stage_buffer_sizes'][1], 100)
        self.assertEqual(len(stats['samples_per_stage']), 1)
    
    def test_stage_switching(self):
        """测试阶段切换"""
        # 在第一阶段添加经验
        for i in range(50):
            experience = {'data': f'stage1_exp_{i}'}
            self.replay.add_experience(experience)
        
        # 切换到第二阶段
        self.replay.set_current_stage(2)
        self.assertEqual(self.replay.current_stage, 2)
        
        # 在第二阶段添加经验
        for i in range(30):
            experience = {'data': f'stage2_exp_{i}'}
            self.replay.add_experience(experience)
        
        stats = self.replay.get_statistics()
        self.assertEqual(stats['stage_buffer_sizes'][1], 50)
        self.assertEqual(stats['stage_buffer_sizes'][2], 30)
        self.assertEqual(stats['total_samples_added'], 80)
    
    def test_single_stage_sampling(self):
        """测试第一阶段的纯当前阶段采样"""
        # 添加经验到第一阶段
        experiences = []
        for i in range(100):
            exp = {'id': i, 'value': np.random.random()}
            experiences.append(exp)
            self.replay.add_experience(exp)
        
        # 在第一阶段采样应该只从当前阶段采样
        batch = self.replay.sample_mixed_batch(20)
        self.assertEqual(len(batch), 20)
        
        # 验证所有样本都来自第一阶段
        for sample in batch:
            self.assertEqual(sample['stage_id'], 1)
    
    def test_mixed_sampling_strategy(self):
        """测试混合采样策略"""
        # 第一阶段添加经验
        for i in range(100):
            exp = {'stage': 1, 'id': i, 'value': np.random.random()}
            self.replay.add_experience(exp, stage_id=1)
        
        # 切换到第二阶段并添加经验
        self.replay.set_current_stage(2)
        for i in range(80):
            exp = {'stage': 2, 'id': i, 'value': np.random.random()}
            self.replay.add_experience(exp, stage_id=2)
        
        # 测试混合采样
        batch_size = 100
        batch = self.replay.sample_mixed_batch(batch_size)
        
        # 验证批次大小
        self.assertEqual(len(batch), batch_size)
        
        # 统计不同阶段的样本数量
        stage1_count = sum(1 for sample in batch if sample['stage_id'] == 1)
        stage2_count = sum(1 for sample in batch if sample['stage_id'] == 2)
        
        # 验证比例大致正确（允许一定误差）
        expected_stage2 = int(batch_size * 0.7)  # 当前阶段70%
        expected_stage1 = batch_size - expected_stage2  # 历史阶段30%
        
        # 允许±5的误差
        self.assertAlmostEqual(stage2_count, expected_stage2, delta=5)
        self.assertAlmostEqual(stage1_count, expected_stage1, delta=5)
    
    def test_insufficient_historical_data(self):
        """测试历史数据不足时的处理"""
        # 第一阶段添加少量经验
        for i in range(10):  # 少于min_historical_samples
            exp = {'id': i, 'value': np.random.random()}
            self.replay.add_experience(exp, stage_id=1)
        
        # 切换到第二阶段
        self.replay.set_current_stage(2)
        for i in range(50):
            exp = {'id': i, 'value': np.random.random()}
            self.replay.add_experience(exp, stage_id=2)
        
        # 由于历史数据不足，应该只从当前阶段采样
        batch = self.replay.sample_mixed_batch(30)
        
        # 验证所有样本都来自当前阶段
        for sample in batch:
            self.assertEqual(sample['stage_id'], 2)
    
    def test_stage_cleanup(self):
        """测试阶段清理功能"""
        # 添加多个阶段的数据
        for stage in range(1, 6):  # 5个阶段
            self.replay.set_current_stage(stage)
            for i in range(20):
                exp = {'stage': stage, 'id': i}
                self.replay.add_experience(exp)
        
        # 设置max_stages_to_keep=3，应该只保留最近3个阶段
        self.replay.max_stages_to_keep = 3
        self.replay.set_current_stage(6)  # 触发清理
        
        stats = self.replay.get_statistics()
        
        # 应该只保留阶段4, 5, 6
        expected_stages = {4, 5, 6}
        actual_stages = set(stats['stage_buffer_sizes'].keys())
        self.assertEqual(actual_stages, expected_stages)
    
    def test_statistics_accuracy(self):
        """测试统计信息的准确性"""
        # 添加不同阶段的经验
        total_added = 0
        
        for stage in [1, 2, 3]:
            self.replay.set_current_stage(stage)
            stage_count = 30 + stage * 10  # 不同阶段添加不同数量
            
            for i in range(stage_count):
                exp = {'stage': stage, 'id': i}
                self.replay.add_experience(exp)
                total_added += 1
        
        stats = self.replay.get_statistics()
        
        # 验证总数
        self.assertEqual(stats['total_samples_added'], total_added)
        
        # 验证各阶段数量
        self.assertEqual(stats['samples_per_stage'][1], 40)
        self.assertEqual(stats['samples_per_stage'][2], 50)
        self.assertEqual(stats['samples_per_stage'][3], 60)
        
        # 验证缓冲区大小
        self.assertEqual(stats['stage_buffer_sizes'][1], 40)
        self.assertEqual(stats['stage_buffer_sizes'][2], 50)
        self.assertEqual(stats['stage_buffer_sizes'][3], 60)
    
    def test_empty_buffer_handling(self):
        """测试空缓冲区的处理"""
        # 从空缓冲区采样
        batch = self.replay.sample_mixed_batch(10)
        self.assertEqual(len(batch), 0)
        
        # 添加少量经验后采样
        for i in range(5):
            exp = {'id': i}
            self.replay.add_experience(exp)
        
        # 请求的样本数量大于可用数量
        batch = self.replay.sample_mixed_batch(10)
        self.assertEqual(len(batch), 5)  # 应该返回所有可用的样本
    
    def test_capacity_limit(self):
        """测试容量限制"""
        # 添加超过容量的经验
        capacity = self.replay.capacity_per_stage
        for i in range(capacity + 100):
            exp = {'id': i, 'value': i}
            self.replay.add_experience(exp)
        
        # 验证缓冲区大小不超过容量
        stats = self.replay.get_statistics()
        self.assertEqual(stats['stage_buffer_sizes'][1], capacity)
        
        # 验证保留的是最新的经验（由于使用deque，旧的会被自动移除）
        batch = self.replay.sample_mixed_batch(capacity)
        ids = [sample['id'] for sample in batch]
        
        # 最新的经验ID应该在100到capacity+99之间
        min_id = min(ids)
        max_id = max(ids)
        self.assertGreaterEqual(min_id, 100)
        self.assertLessEqual(max_id, capacity + 99)


class TestRLlibMixedReplayBuffer(unittest.TestCase):
    """RLlib兼容混合回放缓冲区测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.buffer = RLlibMixedReplayBuffer(capacity=1000)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.buffer.mixed_replay)
        stats = self.buffer.get_statistics()
        self.assertEqual(stats['current_stage'], 1)
    
    def test_add_and_sample(self):
        """测试添加和采样功能"""
        # 添加经验
        for i in range(50):
            experience = {
                'obs': np.random.random(10),
                'actions': np.random.randint(0, 4),
                'rewards': np.random.random(),
                'dones': False
            }
            self.buffer.add(experience)
        
        # 采样
        batch = self.buffer.sample(20)
        
        # 验证批次不为空且格式正确
        self.assertIsNotNone(batch)
        if hasattr(batch, 'data') and batch.data:
            # 如果是有效的SampleBatch
            self.assertGreater(len(batch), 0)
    
    def test_stage_management(self):
        """测试阶段管理"""
        # 第一阶段添加经验
        for i in range(30):
            exp = {'data': f'stage1_{i}'}
            self.buffer.add(exp)
        
        # 切换阶段
        self.buffer.set_current_stage(2)
        
        # 验证阶段切换
        stats = self.buffer.get_statistics()
        self.assertEqual(stats['current_stage'], 2)


def run_mixed_replay_tests():
    """运行所有混合经验回放测试"""
    print("开始运行混合经验回放机制测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMixedExperienceReplay))
    suite.addTests(loader.loadTestsFromTestCase(TestRLlibMixedReplayBuffer))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print(f"\n测试摘要:")
    print(f"运行测试数量: {result.testsRun}")
    print(f"失败数量: {len(result.failures)}")
    print(f"错误数量: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n测试结果: {'通过' if success else '失败'}")
    
    return success


if __name__ == '__main__':
    run
