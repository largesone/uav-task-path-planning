# 批处理算法场景验证使用说明

## 功能概述

批处理验证脚本用于验证算法在不同场景下的求解结果，支持：
- 多个场景的批量测试
- 多种网络架构的对比
- 结果保存为CSV格式
- 可选择特定场景进行测试

## 文件结构

```
temp_tests/
├── batch_scenario_validation.py    # 主验证脚本
├── quick_batch_test.py            # 快速测试脚本
└── BATCH_VALIDATION_README.md     # 使用说明
```

## 使用方法

### 1. 快速测试

```bash
cd temp_tests
python quick_batch_test.py
```

### 2. 完整批处理验证

```bash
cd temp_tests
python batch_scenario_validation.py --scenarios experimental balanced complex --networks DeepFCNResidual --episodes 200
```

### 3. 命令行参数

- `--scenarios`: 要测试的场景类型列表
  - 可用场景: `experimental`, `balanced`, `complex`, `small`, `complex_v4`, `strategic_trap`
  - 默认: `['experimental', 'balanced', 'complex']`

- `--networks`: 要测试的网络类型列表
  - 可用网络: `DeepFCNResidual`, `SimpleNetwork`, `DeepFCN`, `GAT`
  - 默认: `['DeepFCNResidual']`

- `--episodes`: 训练轮数
  - 默认: 200

- `--force-retrain`: 强制重新训练（不使用已保存的模型）
  - 默认: False

- `--output-dir`: 输出目录
  - 默认: `output`

### 4. 使用示例

#### 测试单个场景
```bash
python batch_scenario_validation.py --scenarios experimental --episodes 100
```

#### 测试多个场景
```bash
python batch_scenario_validation.py --scenarios experimental balanced complex --episodes 200
```

#### 测试多种网络架构
```bash
python batch_scenario_validation.py --scenarios experimental --networks DeepFCNResidual SimpleNetwork --episodes 150
```

#### 强制重新训练
```bash
python batch_scenario_validation.py --scenarios experimental --force-retrain
```

## 输出结果

### 1. CSV文件格式

结果保存为CSV文件，包含以下字段：

| 字段名 | 描述 | 示例 |
|--------|------|------|
| scenario | 场景名称 | experimental_场景 |
| solver | 求解器类型 | RL |
| config | 配置信息 | DeepFCNResidual_200ep |
| obstacle_mode | 障碍物模式 | present/absent |
| num_uavs | UAV数量 | 4 |
| num_targets | 目标数量 | 2 |
| num_obstacles | 障碍物数量 | 0 |
| training_time | 训练时间(秒) | 24.55 |
| planning_time | 规划时间(秒) | 0.0 |
| total_time | 总时间(秒) | 32.42 |
| total_reward_score | 总奖励分数 | 2900.00 |
| completion_rate | 完成率 | 0.667 |
| satisfied_targets_count | 满足目标数量 | 1 |
| total_targets | 总目标数量 | 2 |
| satisfied_targets_rate | 目标满足率 | 0.500 |
| resource_utilization_rate | 资源利用率 | 0.833 |
| resource_penalty | 资源惩罚 | 0.000 |
| sync_feasibility_rate | 同步可行性率 | 0.800 |
| load_balance_score | 负载均衡分数 | 0.750 |
| total_distance | 总距离 | 7958.61 |
| is_deadlocked | 是否死锁 | 0 |
| deadlocked_uav_count | 死锁UAV数量 | 0 |

### 2. 输出文件位置

- CSV文件: `output/batch_validation_results_YYYYMMDD_HHMMSS.csv`
- 模型文件: `output/batch_test/场景名_网络类型/`
- 可视化文件: `output/batch_test/场景名_网络类型/`

## 支持的场景类型

### 1. 内置场景函数
- `experimental`: 试验场景（4UAV, 2目标, 0障碍）
- `balanced`: 平衡场景（10UAV, 5目标, 资源平衡）
- `complex`: 复杂场景（6UAV, 5目标, 4障碍）
- `small`: 小规模场景（3UAV, 2目标, 1障碍）
- `complex_v4`: 复杂场景v4（8UAV, 6目标, 5障碍）
- `strategic_trap`: 战略陷阱场景（5UAV, 4目标, 3障碍）

### 2. PKL文件场景
- `collaborative`: 协作场景
- `mixed`: 混合场景
- `resource_starvation`: 资源紧缺场景
- `uav_sweep_5_targets`: UAV扫描5目标场景
- `uav_sweep_10_targets`: UAV扫描10目标场景

## 性能指标说明

### 1. 完成率 (completion_rate)
综合目标满足、资源满足、资源利用率的加权平均

### 2. 资源利用率 (resource_utilization_rate)
实际使用的资源与总可用资源的比例

### 3. 目标满足率 (satisfied_targets_rate)
完全满足资源需求的目标数量与总目标数量的比例

### 4. 总奖励分数 (total_reward_score)
算法获得的累计奖励分数

### 5. 总距离 (total_distance)
所有UAV的总飞行距离

## 注意事项

1. **模型复用**: 默认会尝试加载已训练的模型，避免重复训练
2. **内存管理**: 大量场景测试时注意内存使用
3. **时间估算**: 每个场景的训练时间取决于网络复杂度和训练轮数
4. **错误处理**: 脚本包含错误处理机制，单个场景失败不会影响整体测试

## 故障排除

### 1. 导入错误
确保在正确的conda环境中运行：
```bash
conda activate ray312_py312
```

### 2. 内存不足
减少训练轮数或同时测试的场景数量：
```bash
python batch_scenario_validation.py --scenarios experimental --episodes 50
```

### 3. 模型加载失败
使用强制重新训练：
```bash
python batch_scenario_validation.py --scenarios experimental --force-retrain
```

### 4. 场景加载失败
检查场景文件是否存在，或使用内置场景函数。

## 扩展功能

### 1. 添加新场景
在`scenarios.py`中添加新的场景函数，然后在`BatchScenarioValidator`的`scenario_functions`字典中注册。

### 2. 添加新网络架构
在`networks.py`中添加新的网络架构，然后在命令行参数中指定。

### 3. 自定义评估指标
在`run_single_scenario`方法中添加新的评估指标计算逻辑。 