# -*- coding: utf-8 -*-
# 文件名: test_refactored_system.py
# 描述: 测试重构后的系统

import time
from main import run_scenario
from scenarios import get_small_scenario, get_strategic_trap_scenario
from config import Config

def test_simple_network():
    """测试SimpleNetwork"""
    print("=" * 50)
    print("测试 SimpleNetwork")
    print("=" * 50)
    
    config = Config()
    uavs, targets, obstacles = get_small_scenario(50.0)
    
    try:
        final_plan, training_time, training_history, evaluation_metrics = run_scenario(
            config, uavs, targets, obstacles, "simple_test",
            network_type="SimpleNetwork",
            save_visualization=True,
            show_visualization=False
        )
        
        print(f"测试成功!")
        print(f"训练时间: {training_time:.2f}秒")
        if evaluation_metrics:
            print(f"完成率: {evaluation_metrics.get('completion_rate', 0):.4f}")
            print(f"总奖励: {evaluation_metrics.get('total_reward_score', 0):.2f}")
        
        return True
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

def test_deep_fcn():
    """测试DeepFCN"""
    print("=" * 50)
    print("测试 DeepFCN")
    print("=" * 50)
    
    config = Config()
    uavs, targets, obstacles = get_small_scenario(50.0)
    
    try:
        final_plan, training_time, training_history, evaluation_metrics = run_scenario(
            config, uavs, targets, obstacles, "deep_fcn_test",
            network_type="DeepFCN",
            save_visualization=True,
            show_visualization=False
        )
        
        print(f"测试成功!")
        print(f"训练时间: {training_time:.2f}秒")
        if evaluation_metrics:
            print(f"完成率: {evaluation_metrics.get('completion_rate', 0):.4f}")
            print(f"总奖励: {evaluation_metrics.get('total_reward_score', 0):.2f}")
        
        return True
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("重构系统测试")
    print("=" * 50)
    
    # 测试SimpleNetwork
    simple_success = test_simple_network()
    
    # 测试DeepFCN
    deep_fcn_success = test_deep_fcn()
    
    # 输出结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"SimpleNetwork: {'成功' if simple_success else '失败'}")
    print(f"DeepFCN: {'成功' if deep_fcn_success else '失败'}")
    
    if simple_success and deep_fcn_success:
        print("\n✅ 重构成功! 系统运行正常。")
    else:
        print("\n❌ 重构存在问题，需要进一步调试。")

if __name__ == "__main__":
    main() 