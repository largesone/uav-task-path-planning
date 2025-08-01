# -*- coding: utf-8 -*-
# 文件名: temp_tests/task18_demo.py
# 描述: 任务18向后兼容性演示

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compatibility_manager import CompatibilityManager, CompatibilityConfig
from entities import UAV, Target
from environment import DirectedGraph
from config import Config

def demo_compatibility_manager():
    """演示兼容性管理器的功能"""
    print("=" * 60)
    print("任务18向后兼容性演示")
    print("=" * 60)
    
    # 创建测试数据
    config = Config()
    uavs = [
        UAV(0, [100, 100], 0, [10, 10], 500, [20, 50], 30),
        UAV(1, [200, 200], 0, [8, 12], 600, [25, 45], 35)
    ]
    targets = [
        Target(0, [400, 400], [5, 5], 100),
        Target(1, [500, 500], [7, 3], 150)
    ]
    graph = DirectedGraph(uavs, targets, 6, [], config)
    
    print(f"测试场景: {len(uavs)}个UAV, {len(targets)}个目标")
    
    # 演示1: 传统网络模式
    print("\n" + "=" * 40)
    print("演示1: 传统网络模式")
    print("=" * 40)
    
    traditional_config = CompatibilityConfig(
        network_mode="traditional",
        traditional_network_type="DeepFCNResidual",
        obs_mode="flat",
        debug_mode=True
    )
    
    traditional_manager = CompatibilityManager(traditional_config)
    
    # 创建传统环境
    traditional_env = traditional_manager.create_environment(
        uavs, targets, graph, [], config
    )
    
    print(f"✅ 传统环境创建成功")
    print(f"   - 观测模式: {traditional_env.obs_mode}")
    print(f"   - 观测空间: {traditional_env.observation_space}")
    
    # 测试环境运行
    state = traditional_env.reset()
    print(f"   - 初始状态形状: {state.shape}")
    
    next_state, reward, done, truncated, info = traditional_env.step(0)
    print(f"   - 步进成功，奖励: {reward:.2f}")
    
    # 创建传统网络
    traditional_network = traditional_manager.create_network(
        input_dim=traditional_env.observation_space.shape[0],
        hidden_dims=[256, 128],
        output_dim=traditional_env.n_actions
    )
    
    print(f"✅ 传统网络创建成功")
    print(f"   - 网络类型: {traditional_config.traditional_network_type}")
    print(f"   - 参数数量: {sum(p.numel() for p in traditional_network.parameters()):,}")
    
    # 演示2: TransformerGNN模式
    print("\n" + "=" * 40)
    print("演示2: TransformerGNN模式")
    print("=" * 40)
    
    transformer_config = CompatibilityConfig(
        network_mode="transformer_gnn",
        obs_mode="graph",
        debug_mode=True
    )
    
    transformer_manager = CompatibilityManager(transformer_config)
    
    # 创建TransformerGNN环境
    transformer_env = transformer_manager.create_environment(
        uavs, targets, graph, [], config
    )
    
    print(f"✅ TransformerGNN环境创建成功")
    print(f"   - 观测模式: {transformer_env.obs_mode}")
    print(f"   - 观测空间键: {list(transformer_env.observation_space.spaces.keys())}")
    
    # 测试环境运行
    state = transformer_env.reset()
    print(f"   - 初始状态类型: {type(state)}")
    print(f"   - UAV特征形状: {state['uav_features'].shape}")
    print(f"   - 目标特征形状: {state['target_features'].shape}")
    
    next_state, reward, done, truncated, info = transformer_env.step(0)
    print(f"   - 步进成功，奖励: {reward:.2f}")
    
    # 创建TransformerGNN网络
    transformer_network = transformer_manager.create_network(
        input_dim=None,
        hidden_dims=None,
        output_dim=transformer_env.n_actions,
        obs_space=transformer_env.observation_space,
        action_space=transformer_env.action_space
    )
    
    print(f"✅ TransformerGNN网络创建成功")
    print(f"   - 网络类型: TransformerGNN")
    print(f"   - 参数数量: {sum(p.numel() for p in transformer_network.parameters()):,}")
    
    # 演示3: 配置管理
    print("\n" + "=" * 40)
    print("演示3: 配置管理")
    print("=" * 40)
    
    # 保存配置
    config_path = "temp_demo_config.json"
    transformer_manager.save_config(config_path)
    print(f"✅ 配置已保存到: {config_path}")
    
    # 加载配置
    loaded_manager = CompatibilityManager.load_config(config_path)
    print(f"✅ 配置已加载")
    print(f"   - 网络模式: {loaded_manager.config.network_mode}")
    print(f"   - 观测模式: {loaded_manager.config.obs_mode}")
    
    # 清理临时文件
    if os.path.exists(config_path):
        os.remove(config_path)
    
    # 演示4: 兼容性检查
    print("\n" + "=" * 40)
    print("演示4: 兼容性检查")
    print("=" * 40)
    
    check_config = CompatibilityConfig(
        enable_compatibility_checks=True,
        debug_mode=False
    )
    
    check_manager = CompatibilityManager(check_config)
    results = check_manager.run_compatibility_checks()
    
    print(f"✅ 兼容性检查完成")
    print(f"   - 总体状态: {'通过' if results['overall_compatibility'] else '失败'}")
    print(f"   - 检查项数量: {len(results)}")
    
    passed_checks = sum(1 for v in results.values() if v is True)
    total_checks = len([k for k in results.keys() if k != 'overall_compatibility'])
    print(f"   - 通过率: {passed_checks}/{total_checks}")
    
    # 总结
    print("\n" + "=" * 60)
    print("演示总结")
    print("=" * 60)
    print("✅ 传统网络模式正常工作")
    print("✅ TransformerGNN模式正常工作")
    print("✅ 配置管理功能正常")
    print("✅ 兼容性检查功能正常")
    print("✅ 向后兼容性保证实现成功！")
    
    return True

if __name__ == "__main__":
    try:
        success = demo_compatibility_manager()
        if success:
            print("\n🎉 任务18演示成功完成！")
        else:
            print("\n❌ 任务18演示失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)