# -*- coding: utf-8 -*-
# 文件名: test_complex_scenario.py
# 描述: 测试复杂场景下的资源分配问题

import numpy as np
from collections import defaultdict
from scenarios import get_new_experimental_scenario
from main import calibrate_resource_assignments, Config, run_scenario

def test_complex_scenario():
    """测试复杂场景下的资源分配问题"""
    print("=== 复杂场景资源分配测试 ===")
    
    # 创建复杂场景
    config = Config()
    
    uavs, targets, obstacles = get_new_experimental_scenario('complex')
    
    print(f"测试场景: {len(uavs)} 架无人机, {len(targets)} 个目标")
    
    # 打印目标和无人机的资源信息
    print("\n目标资源需求:")
    for target in targets:
        print(f"  目标 {target.id}: {target.resources}")
    
    print("\n无人机初始资源:")
    for uav in uavs:
        print(f"  UAV {uav.id}: {uav.initial_resources}")
    
    # 创建模拟的任务分配（模拟用户报告的问题）
    task_assignments = defaultdict(list)
    
    # 获取实际的无人机和目标ID
    uav_ids = [u.id for u in uavs]
    target_ids = [t.id for t in targets]
    
    print(f"实际无人机ID: {uav_ids}")
    print(f"实际目标ID: {target_ids}")
    
    # 模拟正常的分配
    if len(uav_ids) >= 1 and len(target_ids) >= 1:
        task_assignments[uav_ids[0]].append((target_ids[0], 0))  # UAV 1 -> 目标 1
    
    if len(uav_ids) >= 2 and len(target_ids) >= 2:
        task_assignments[uav_ids[1]].append((target_ids[1], 1))  # UAV 2 -> 目标 2
    
    if len(uav_ids) >= 3 and len(target_ids) >= 1:
        task_assignments[uav_ids[2]].append((target_ids[0], 0))  # UAV 3 -> 目标 1 (重复)
    
    if len(uav_ids) >= 4 and len(target_ids) >= 2:
        task_assignments[uav_ids[3]].append((target_ids[1], 1))  # UAV 4 -> 目标 2 (重复)
    
    print("\n原始任务分配:")
    for uav_id, tasks in task_assignments.items():
        if tasks:
            print(f"  UAV {uav_id}: {tasks}")
    
    # 执行校准
    calibrated_assignments = calibrate_resource_assignments(task_assignments, uavs, targets)
    
    print("\n校准后的任务分配:")
    for uav_id, tasks in calibrated_assignments.items():
        if tasks:
            print(f"  UAV {uav_id}: {tasks}")
    
    # 验证结果
    original_count = sum(len(tasks) for tasks in task_assignments.values())
    calibrated_count = sum(len(tasks) for tasks in calibrated_assignments.values())
    
    print(f"\n验证结果:")
    print(f"  原始分配数量: {original_count}")
    print(f"  校准后数量: {calibrated_count}")
    print(f"  移除的无效分配: {original_count - calibrated_count}")
    
    # 检查UAV 10是否被移除
    if 10 in task_assignments and 10 not in calibrated_assignments:
        print("✓ UAV 10的无效分配已被正确移除")
    elif 10 in calibrated_assignments:
        print("✗ UAV 10的无效分配未被移除")
    
    print("测试完成!")

if __name__ == "__main__":
    test_complex_scenario() 