# -*- coding: utf-8 -*-
# 文件名: test_resource_calibration.py
# 描述: 测试资源分配校准功能

import numpy as np
from collections import defaultdict
from scenarios import get_new_experimental_scenario
from main import calibrate_resource_assignments, Config

def test_resource_calibration():
    """测试资源分配校准功能"""
    print("=== 资源分配校准测试 ===")
    
    # 创建测试场景
    config = Config()
    uavs, targets, obstacles = get_new_experimental_scenario('complex')
    
    print(f"测试场景: {len(uavs)} 架无人机, {len(targets)} 个目标")
    
    # 创建模拟的任务分配（包含一些无效分配）
    task_assignments = defaultdict(list)
    
    # 获取实际的无人机和目标ID
    uav_ids = [u.id for u in uavs]
    target_ids = [t.id for t in targets]
    
    print(f"实际无人机ID: {uav_ids}")
    print(f"实际目标ID: {target_ids}")
    
    # 模拟一些正常的分配
    if len(uav_ids) >= 1 and len(target_ids) >= 1:
        task_assignments[uav_ids[0]].append((target_ids[0], 0))  # 第一个UAV -> 第一个目标
    
    if len(uav_ids) >= 2 and len(target_ids) >= 2:
        task_assignments[uav_ids[1]].append((target_ids[1], 1))  # 第二个UAV -> 第二个目标
    
    # 模拟一些无效分配（重复分配）
    if len(uav_ids) >= 3 and len(target_ids) >= 1:
        task_assignments[uav_ids[2]].append((target_ids[0], 0))  # 第三个UAV -> 第一个目标（重复）
    
    if len(uav_ids) >= 4 and len(target_ids) >= 2:
        task_assignments[uav_ids[3]].append((target_ids[1], 1))  # 第四个UAV -> 第二个目标（重复）
    
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
    
    # 检查是否还有无效分配
    uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
    target_needs = {t.id: t.resources.copy().astype(float) for t in targets}
    
    for uav_id, tasks in calibrated_assignments.items():
        for target_id, phi_idx in tasks:
            contribution = np.minimum(uav_resources[uav_id], target_needs[target_id])
            if np.any(contribution > 1e-6):
                uav_resources[uav_id] -= contribution
                target_needs[target_id] -= contribution
            else:
                print(f"警告: 校准后仍存在无效分配 - UAV {uav_id} -> 目标 {target_id}")
    
    print("测试完成!")

if __name__ == "__main__":
    test_resource_calibration() 