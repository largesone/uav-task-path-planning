#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁棒输入掩码机制测试
测试任务3的实现：实现鲁棒的输入掩码机制

测试内容：
1. UAV的is_alive位（0/1）标识通信/感知状态
2. Target的is_visible位（0/1）标识可见性状态  
3. 掩码位与masks的结合使用，为TransformerGNN提供失效节点屏蔽能力
4. 部分可观测情况下的状态生成正确性验证

需求覆盖: 3.1, 3.2, 3.3, 3.4
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph
from config import Config


class TestRobustInputMasking(unittest.TestCase):
    """鲁棒输入掩码机制测试类"""
    
    def setUp(self):
        """测试环境初始化"""
        self.config = Config()
        
        # 创建测试UAV
        self.uavs = [
            UAV(1, [100, 200], 0.0, [50, 30], 1000, [10, 20], 15.0),
            UAV(2, [300, 400], 0.5, [40, 35], 1200, [8, 18], 12.0),
            UAV(3, [500, 600], 1.0, [0, 0], 800, [5, 15], 10.0)  # 无资源UAV
        ]
        
        # 创建测试目标
        self.targets = [
            Target(1, [150, 250], [60, 40], 100),
            Target(2, [350, 450], [45, 25], 80),
            Target(3, [550, 650], [0, 0], 50)  # 无需求目标
        ]
        
        # 创建图和环境
        self.obstacles = []
        self.graph = DirectedGraph(self.uavs, self.targets, 8, self.obstacles, self.config)
        self.env = UAVTaskEnv(self.uavs, self.targets, self.graph, self.obstacles, 
                             self.config, obs_mode="graph")
    
    def test_uav_alive_status_basic(self):
        """测试UAV存活状态基础功能"""
        print("\n=== 测试UAV存活状态基础功能 ===")
        
        # 测试有资源的UAV
        alive_status_1 = self.env._calculate_uav_alive_status(self.uavs[0], 0)
        self.assertEqual(alive_status_1, 1.0, "有资源的UAV应该处于存活状态")
        print(f"UAV 1 存活状态: {alive_status_1} (预期: 1.0)")
        
        # 测试无资源的UAV
        alive_status_3 = self.env._calculate_uav_alive_status(self.uavs[2], 2)
        self.assertEqual(alive_status_3, 0.0, "无资源的UAV应该处于失效状态")
        print(f"UAV 3 存活状态: {alive_status_3} (预期: 0.0)")
    
    def test_uav_alive_status_with_failures(self):
        """测试UAV存活状态在失效场景下的表现"""
        print("\n=== 测试UAV存活状态失效场景 ===")
        
        # 设置通信失效率
        self.config.UAV_COMM_FAILURE_RATE = 0.5
        
        # 多次测试以验证随机性
        alive_counts = 0
        test_iterations = 100
        
        for i in range(test_iterations):
            self.env.step_count = i  # 改变step_count以影响随机种子
            alive_status = self.env._calculate_uav_alive_status(self.uavs[0], 0)
            if alive_status > 0.5:
                alive_counts += 1
        
        alive_rate = alive_counts / test_iterations
        print(f"通信失效率0.5下，UAV存活率: {alive_rate:.2f}")
        
        # 验证存活率在合理范围内（考虑随机性）
        self.assertGreater(alive_rate, 0.3, "存活率应该大于30%")
        self.assertLess(alive_rate, 0.7, "存活率应该小于70%")
    
    def test_target_visibility_status_basic(self):
        """测试目标可见性状态基础功能"""
        print("\n=== 测试目标可见性状态基础功能 ===")
        
        # 测试有剩余资源的目标
        visibility_1 = self.env._calculate_target_visibility_status(self.targets[0], 0)
        self.assertGreater(visibility_1, 0.0, "有剩余资源的目标应该可见")
        print(f"目标 1 可见性: {visibility_1:.3f}")
        
        # 测试无剩余资源的目标
        visibility_3 = self.env._calculate_target_visibility_status(self.targets[2], 2)
        self.assertEqual(visibility_3, 0.0, "无剩余资源的目标应该不可见")
        print(f"目标 3 可见性: {visibility_3:.3f} (预期: 0.0)")
    
    def test_target_visibility_with_distance(self):
        """测试目标可见性受距离影响"""
        print("\n=== 测试目标可见性距离影响 ===")
        
        # 设置感知范围
        self.config.MAX_SENSING_RANGE = 500.0
        
        # 计算UAV到目标的距离
        uav_pos = np.array(self.uavs[0].current_position)
        target_pos = np.array(self.targets[0].position)
        distance = np.linalg.norm(target_pos - uav_pos)
        print(f"UAV 1 到目标 1 的距离: {distance:.2f}m")
        
        # 测试可见性
        visibility = self.env._calculate_target_visibility_status(self.targets[0], 0)
        print(f"目标 1 可见性: {visibility:.3f}")
        
        # 验证距离衰减效果
        expected_visibility = max(0.0, 1.0 - (distance / self.config.MAX_SENSING_RANGE) ** 2)
        self.assertAlmostEqual(visibility, expected_visibility, places=2, 
                              msg="可见性应该根据距离衰减")
    
    def test_robust_masks_generation(self):
        """测试鲁棒掩码生成"""
        print("\n=== 测试鲁棒掩码生成 ===")
        
        masks = self.env._calculate_robust_masks()
        
        # 验证掩码结构
        required_keys = [
            'uav_mask', 'target_mask', 'uav_resource_mask', 'uav_communication_mask',
            'target_resource_mask', 'target_visibility_mask', 'interaction_mask',
            'active_uav_count', 'visible_target_count', 'total_interactions'
        ]
        
        for key in required_keys:
            self.assertIn(key, masks, f"掩码应该包含 {key}")
        
        print("掩码结构验证通过")
        
        # 验证掩码维度
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        self.assertEqual(masks['uav_mask'].shape, (n_uavs,), "UAV掩码维度错误")
        self.assertEqual(masks['target_mask'].shape, (n_targets,), "目标掩码维度错误")
        self.assertEqual(masks['interaction_mask'].shape, (n_uavs, n_targets), "交互掩码维度错误")
        
        print(f"掩码维度验证通过: UAV({n_uavs}), Target({n_targets})")
        
        # 打印掩码详情
        print(f"UAV有效掩码: {masks['uav_mask']}")
        print(f"目标有效掩码: {masks['target_mask']}")
        print(f"活跃UAV数量: {masks['active_uav_count']}")
        print(f"可见目标数量: {masks['visible_target_count']}")
        print(f"总交互数量: {masks['total_interactions']}")
    
    def test_graph_state_with_masking(self):
        """测试图状态生成包含掩码信息"""
        print("\n=== 测试图状态生成包含掩码信息 ===")
        
        # 重置环境并获取图状态
        state = self.env.reset()
        
        # 验证状态结构
        required_keys = ['uav_features', 'target_features', 'relative_positions', 
                        'distances', 'masks']
        
        for key in required_keys:
            self.assertIn(key, state, f"图状态应该包含 {key}")
        
        print("图状态结构验证通过")
        
        # 验证特征维度
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        self.assertEqual(state['uav_features'].shape, (n_uavs, 9), "UAV特征维度错误")
        self.assertEqual(state['target_features'].shape, (n_targets, 8), "目标特征维度错误")
        
        print(f"特征维度验证通过: UAV特征({n_uavs}, 9), 目标特征({n_targets}, 8)")
        
        # 验证is_alive和is_visible位
        for i in range(n_uavs):
            is_alive = state['uav_features'][i, 8]  # 第9个特征是is_alive
            self.assertIn(is_alive, [0.0, 1.0], f"UAV {i} 的is_alive位应该是0或1")
            print(f"UAV {i+1} is_alive: {is_alive}")
        
        for i in range(n_targets):
            is_visible = state['target_features'][i, 7]  # 第8个特征是is_visible
            self.assertGreaterEqual(is_visible, 0.0, f"目标 {i} 的is_visible位应该>=0")
            self.assertLessEqual(is_visible, 1.0, f"目标 {i} 的is_visible位应该<=1")
            print(f"目标 {i+1} is_visible: {is_visible:.3f}")
    
    def test_partial_observability_scenarios(self):
        """测试部分可观测场景下的状态生成"""
        print("\n=== 测试部分可观测场景 ===")
        
        # 场景1: 高通信失效率
        print("\n场景1: 高通信失效率")
        self.config.UAV_COMM_FAILURE_RATE = 0.8
        self.config.TARGET_OCCLUSION_RATE = 0.0
        
        state1 = self.env._get_graph_state()
        masks1 = state1['masks']
        
        print(f"通信失效率80%下，活跃UAV数量: {masks1['active_uav_count']}")
        print(f"UAV通信掩码: {masks1['uav_communication_mask']}")
        
        # 场景2: 高目标遮挡率
        print("\n场景2: 高目标遮挡率")
        self.config.UAV_COMM_FAILURE_RATE = 0.0
        self.config.TARGET_OCCLUSION_RATE = 0.7
        
        state2 = self.env._get_graph_state()
        masks2 = state2['masks']
        
        print(f"目标遮挡率70%下，可见目标数量: {masks2['visible_target_count']}")
        print(f"目标可见性掩码: {masks2['target_visibility_mask']}")
        
        # 场景3: 综合失效场景
        print("\n场景3: 综合失效场景")
        self.config.UAV_COMM_FAILURE_RATE = 0.5
        self.config.TARGET_OCCLUSION_RATE = 0.5
        
        state3 = self.env._get_graph_state()
        masks3 = state3['masks']
        
        print(f"综合失效下，总交互数量: {masks3['total_interactions']}")
        print(f"交互掩码非零元素: {np.count_nonzero(masks3['interaction_mask'])}")
    
    def test_environment_complexity_calculation(self):
        """测试环境复杂度计算"""
        print("\n=== 测试环境复杂度计算 ===")
        
        # 测试UAV环境复杂度
        complexity_uav1 = self.env._calculate_environment_complexity(self.uavs[0])
        complexity_uav2 = self.env._calculate_environment_complexity(self.uavs[1])
        
        print(f"UAV 1 环境复杂度: {complexity_uav1:.3f}")
        print(f"UAV 2 环境复杂度: {complexity_uav2:.3f}")
        
        # 验证复杂度范围
        self.assertGreaterEqual(complexity_uav1, 0.5, "环境复杂度应该>=0.5")
        self.assertLessEqual(complexity_uav1, 2.0, "环境复杂度应该<=2.0")
        
        # 测试目标环境复杂度
        complexity_target1 = self.env._calculate_target_environment_complexity(self.targets[0])
        complexity_target2 = self.env._calculate_target_environment_complexity(self.targets[1])
        
        print(f"目标 1 环境复杂度: {complexity_target1:.3f}")
        print(f"目标 2 环境复杂度: {complexity_target2:.3f}")
        
        # 验证复杂度范围
        self.assertGreaterEqual(complexity_target1, 0.5, "目标环境复杂度应该>=0.5")
        self.assertLessEqual(complexity_target1, 2.0, "目标环境复杂度应该<=2.0")
    
    def test_masking_integration_with_transformer_gnn(self):
        """测试掩码与TransformerGNN的集成"""
        print("\n=== 测试掩码与TransformerGNN集成 ===")
        
        # 获取图状态
        state = self.env._get_graph_state()
        masks = state['masks']
        
        # 模拟TransformerGNN的注意力计算
        uav_features = state['uav_features']
        target_features = state['target_features']
        interaction_mask = masks['interaction_mask']
        
        print(f"UAV特征形状: {uav_features.shape}")
        print(f"目标特征形状: {target_features.shape}")
        print(f"交互掩码形状: {interaction_mask.shape}")
        
        # 验证掩码可以正确屏蔽失效节点
        uav_mask = masks['uav_mask']
        target_mask = masks['target_mask']
        
        # 检查失效UAV的特征是否被正确标记
        for i, mask_val in enumerate(uav_mask):
            is_alive = uav_features[i, 8]
            if mask_val == 0:
                self.assertEqual(is_alive, 0.0, f"失效UAV {i} 的is_alive位应该为0")
                print(f"UAV {i+1} 已失效，is_alive: {is_alive}")
            else:
                self.assertEqual(is_alive, 1.0, f"有效UAV {i} 的is_alive位应该为1")
                print(f"UAV {i+1} 正常工作，is_alive: {is_alive}")
        
        # 检查不可见目标的特征是否被正确标记
        for i, mask_val in enumerate(target_mask):
            is_visible = target_features[i, 7]
            if mask_val == 0:
                self.assertEqual(is_visible, 0.0, f"不可见目标 {i} 的is_visible位应该为0")
                print(f"目标 {i+1} 不可见，is_visible: {is_visible}")
            else:
                self.assertGreater(is_visible, 0.0, f"可见目标 {i} 的is_visible位应该>0")
                print(f"目标 {i+1} 可见，is_visible: {is_visible:.3f}")
    
    def test_masking_consistency(self):
        """测试掩码一致性"""
        print("\n=== 测试掩码一致性 ===")
        
        masks = self.env._calculate_robust_masks()
        
        # 验证基础掩码与详细掩码的一致性
        uav_effective = masks['uav_resource_mask'] & masks['uav_communication_mask']
        target_effective = masks['target_resource_mask'] & masks['target_visibility_mask']
        
        np.testing.assert_array_equal(masks['uav_mask'], uav_effective, 
                                    "UAV有效掩码应该等于资源掩码与通信掩码的交集")
        np.testing.assert_array_equal(masks['target_mask'], target_effective,
                                    "目标有效掩码应该等于资源掩码与可见性掩码的交集")
        
        print("掩码一致性验证通过")
        
        # 验证统计信息的正确性
        expected_active_uavs = np.sum(masks['uav_mask'])
        expected_visible_targets = np.sum(masks['target_mask'])
        expected_interactions = np.sum(masks['interaction_mask'])
        
        self.assertEqual(masks['active_uav_count'], expected_active_uavs, 
                        "活跃UAV统计错误")
        self.assertEqual(masks['visible_target_count'], expected_visible_targets,
                        "可见目标统计错误")
        self.assertEqual(masks['total_interactions'], expected_interactions,
                        "总交互统计错误")
        
        print("统计信息一致性验证通过")


def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("鲁棒输入掩码机制综合测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRobustInputMasking)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
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
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\n测试成功率: {success_rate:.1%}")
    
    if success_rate == 1.0:
        print("✅ 所有测试通过！鲁棒输入掩码机制实现正确。")
    else:
        print("❌ 部分测试失败，需要检查实现。")
    
    return result


if __name__ == "__main__":
    # 运行综合测试
    run_comprehensive_test()
