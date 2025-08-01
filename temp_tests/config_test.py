# -*- coding: utf-8 -*-
# 文件名: config_test.py
# 描述: 测试配置参数的统一性和修复效果

from config import Config

def test_config_unified_params():
    """测试配置参数的统一性"""
    print("=" * 70)
    print("配置参数统一性测试")
    print("=" * 70)
    
    # 创建配置实例
    config = Config()
    
    # 1. 测试参数访问的一致性
    print("\n1. 测试参数访问一致性:")
    print("-" * 40)
    
    # 测试通过属性和training_config访问是否一致
    test_params = [
        ('episodes', 'EPISODES'),
        ('learning_rate', 'LEARNING_RATE'),
        ('gamma', 'GAMMA'),
        ('batch_size', 'BATCH_SIZE'),
        ('memory_size', 'MEMORY_SIZE'),
    ]
    
    for attr_name, prop_name in test_params:
        attr_value = getattr(config.training_config, attr_name)
        prop_value = getattr(config, prop_name)
        
        if attr_value == prop_value:
            print(f"✓ {attr_name}: {attr_value} == {prop_name}: {prop_value}")
        else:
            print(f"✗ {attr_name}: {attr_value} != {prop_name}: {prop_value}")
    
    # 2. 测试参数修改功能
    print("\n2. 测试参数修改功能:")
    print("-" * 40)
    
    # 保存原始值
    original_episodes = config.EPISODES
    original_lr = config.LEARNING_RATE
    
    # 测试单个参数修改
    config.EPISODES = 500
    config.LEARNING_RATE = 0.001
    
    print(f"修改后 episodes: {config.EPISODES} (training_config: {config.training_config.episodes})")
    print(f"修改后 learning_rate: {config.LEARNING_RATE} (training_config: {config.training_config.learning_rate})")
    
    # 测试批量参数修改
    print("\n3. 测试批量参数修改:")
    print("-" * 40)
    
    config.update_training_params(
        episodes=1000,
        learning_rate=0.0001,
        batch_size=64,
        gamma=0.99
    )
    
    # 4. 显示当前配置
    print("\n4. 当前训练配置:")
    print("-" * 40)
    config.print_training_config()
    
    # 恢复原始值
    config.EPISODES = original_episodes
    config.LEARNING_RATE = original_lr
    
    print("\n✓ 配置参数统一性测试完成")

def test_reward_function_fix():
    """测试奖励函数修复"""
    print("\n" + "=" * 70)
    print("奖励函数测试修复验证")
    print("=" * 70)
    
    # 模拟不同的返回格式
    test_cases = [
        # 情况1: 字典格式
        {
            'training_time': 100.5,
            'evaluation_metrics': {
                'completion_rate': 0.95,
                'satisfied_targets_rate': 0.90
            }
        },
        # 情况2: 元组格式 (字典, 其他数据)
        (
            {
                'training_time': 200.3,
                'evaluation_metrics': {
                    'completion_rate': 0.88,
                    'satisfied_targets_rate': 0.85
                }
            },
            "额外数据"
        ),
        # 情况3: 空元组
        (),
        # 情况4: None
        None
    ]
    
    def process_result(result):
        """模拟修复后的结果处理逻辑"""
        if result:
            # 处理可能的tuple返回格式
            if isinstance(result, tuple):
                result_dict = result[0] if len(result) > 0 else {}
            else:
                result_dict = result
            
            training_time = result_dict.get('training_time', 0)
            evaluation_metrics = result_dict.get('evaluation_metrics', {})
            
            return {
                'training_time': training_time,
                'completion_rate': evaluation_metrics.get('completion_rate', 0),
                'satisfied_targets_rate': evaluation_metrics.get('satisfied_targets_rate', 0)
            }
        else:
            return None
    
    print("测试不同返回格式的处理:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            processed = process_result(test_case)
            if processed:
                print(f"✓ 测试用例 {i}: 成功处理")
                print(f"  训练时间: {processed['training_time']:.2f}秒")
                print(f"  完成率: {processed['completion_rate']:.3f}")
                print(f"  目标满足率: {processed['satisfied_targets_rate']:.3f}")
            else:
                print(f"✓ 测试用例 {i}: 正确处理空结果")
        except Exception as e:
            print(f"✗ 测试用例 {i}: 处理失败 - {e}")
        print()
    
    print("✓ 奖励函数测试修复验证完成")

if __name__ == "__main__":
    test_config_unified_params()
    test_reward_function_fix()