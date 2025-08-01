# -*- coding: utf-8 -*-
# 文件名: simplified_reward.py
# 描述: 简化的奖励函数，减少相互竞争的目标，并进行归一化处理

import numpy as np

def calculate_simplified_reward(target, uav, actual_contribution, path_len, 
                              was_satisfied, travel_time, done):
    """
    简化的奖励函数，重点关注目标资源满足和死锁避免
    
    Args:
        target: 目标对象
        uav: UAV对象
        actual_contribution: 实际资源贡献
        path_len: 路径长度
        was_satisfied: 之前是否已满足目标
        travel_time: 旅行时间
        done: 是否完成所有目标
        
    Returns:
        float: 归一化的奖励值
    """
    # 1. 任务完成奖励 (最高优先级)
    if done:
        return 10.0  # 归一化后的最高奖励
    
    # 2. 目标满足奖励
    now_satisfied = np.all(target.remaining_resources <= 0)
    new_satisfied = int(now_satisfied and not was_satisfied)
    target_completion_reward = 5.0 if new_satisfied else 0.0
    
    # 3. 资源贡献奖励 (核心奖励)
    # 计算贡献比例而不是绝对值
    target_initial_total = np.sum(target.resources)
    contribution_ratio = np.sum(actual_contribution) / target_initial_total if target_initial_total > 0 else 0
    contribution_reward = contribution_ratio * 3.0  # 最高3分
    
    # 4. 零贡献惩罚 (避免死锁)
    if np.all(actual_contribution <= 0):
        return -5.0  # 严重惩罚零贡献动作
    
    # 5. 距离惩罚 (简化版)
    # 使用相对距离而不是绝对距离
    max_distance = 1000.0  # 假设的最大距离
    distance_ratio = min(path_len / max_distance, 1.0)
    distance_penalty = -distance_ratio * 1.0  # 最多-1分
    
    # 总奖励 (归一化到[-5, 10]范围)
    total_reward = target_completion_reward + contribution_reward + distance_penalty
    
    return float(total_reward)