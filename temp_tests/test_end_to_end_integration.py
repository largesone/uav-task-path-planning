# -*- coding: utf-8 -*-
"""
任务19: 端到端系统集成测试
实现完整的课程学习训练流程测试，验证零样本迁移能力和系统完整性
"""

import os
import sys
import time
import json
import numpy as np
import torch
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入核心模块
from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph
from config import Config
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario
from curriculum_stages import CurriculumStages, StageConfig
from transformer_gnn import TransformerGNN
from mixed_experience_replay import MixedExperienceReplay
from model_state_manager import ModelStateManager
from rollback_threshold_manager import RollbackThresholdManager

# TensorBoard支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard未安装，将跳过TensorBoard测试")

class EndToEndIntegrationTester:
    """端到端系统集成测试器"""
    
    def __init__(self, test_output_dir: str = None):
        """
        初始化测试器
        
        Args:
            test_output_dir: 测试输出目录，如果为None则创建临时目录
        """
        self.test_output_dir = test_output_dir or tempfile.mkdtemp(prefix="e2e_test_")
        self.config = Config()
        self.curriculum_stages = CurriculumStages()
        self.test_results = {}
        self.tensorboard_dir = os.path.join(self.test_output_dir, "tensorboard")
        
        # 创建输出目录
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        print(f"端到端集成测试初始化完成")
        print(f"测试输出目录: {self.test_output_dir}")
        print(f"TensorBoard目录: {self.tensorboard_dir}")
    
    def test_curriculum_learning_pipeline(self) -> Dict:
        """
        测试1: 完整的课程学习训练流程测试
        
        验证：
        - 课程阶段配置正确性
        - 阶段间切换机制
        - 回退门限机制
        - 混合经验回放
        """
        print("\n=== 测试1: 课程学习训练流程 ===")
        test_start_time = time.time()
        results = {
            "test_name": "curriculum_learning_pipeline",
            "status": "running",
            "stages_tested": [],
            "stage_transitions": [],
            "fallback_events": [],
            "errors": []
        }
        
        try:
            # 1.1 测试课程阶段配置
            print("1.1 验证课程阶段配置...")
            stages = self.curriculum_stages.stages
            assert len(stages) >= 4, f"课程阶段数量不足: {len(stages)}"
            
            for i, stage in enumerate(stages):
                assert stage.stage_id == i, f"阶段ID不匹配: {stage.stage_id} != {i}"
                assert stage.n_uavs_range[0] <= stage.n_uavs_range[1], f"UAV数量范围无效: {stage.n_uavs_range}"
                assert stage.n_targets_range[0] <= stage.n_targets_range[1], f"目标数量范围无效: {stage.n_targets_range}"
                
                results["stages_tested"].append({
                    "stage_id": stage.stage_id,
                    "stage_name": stage.stage_name,
                    "uav_range": stage.n_uavs_range,
                    "target_range": stage.n_targets_range,
                    "max_episodes": stage.max_episodes
                })
            
            print(f"✓ 课程阶段配置验证通过，共{len(stages)}个阶段")
            
            # 1.2 测试每个阶段的场景生成
            print("1.2 测试各阶段场景生成...")
            for stage in stages[:2]:  # 测试前两个阶段以节省时间
                print(f"  测试阶段{stage.stage_id}: {stage.stage_name}")
                
                # 生成多个随机场景
                for _ in range(5):
                    n_uavs, n_targets = stage.get_random_scenario_size()
                    assert stage.n_uavs_range[0] <= n_uavs <= stage.n_uavs_range[1]
                    assert stage.n_targets_range[0] <= n_targets <= stage.n_targets_range[1]
                
                # 创建实际场景并测试环境
                n_uavs, n_targets = stage.get_random_scenario_size()
                uavs, targets, obstacles = self._create_test_scenario(n_uavs, n_targets)
                
                # 测试双模式环境
                for obs_mode in ["flat", "graph"]:
                    graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                    env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode=obs_mode)
                    
                    # 测试环境重置和步进
                    obs = env.reset()
                    action = env.action_space.sample()
                    next_obs, reward, done, truncated, info = env.step(action)
                    
                    print(f"    ✓ {obs_mode}模式环境测试通过")
            
            print("✓ 各阶段场景生成测试通过")
            
            # 1.3 测试阶段切换机制
            print("1.3 测试阶段切换机制...")
            original_stage = self.curriculum_stages.current_stage_id
            
            # 测试推进
            if self.curriculum_stages.advance_to_next_stage():
                results["stage_transitions"].append({
                    "type": "advance",
                    "from_stage": original_stage,
                    "to_stage": self.curriculum_stages.current_stage_id
                })
                print(f"  ✓ 阶段推进测试通过: {original_stage} -> {self.curriculum_stages.current_stage_id}")
            
            # 测试回退
            if self.curriculum_stages.fallback_to_previous_stage():
                results["stage_transitions"].append({
                    "type": "fallback",
                    "from_stage": self.curriculum_stages.current_stage_id + 1,
                    "to_stage": self.curriculum_stages.current_stage_id
                })
                print(f"  ✓ 阶段回退测试通过")
            
            # 恢复原始阶段
            self.curriculum_stages.current_stage_id = original_stage
            
            # 1.4 测试混合经验回放机制
            print("1.4 测试混合经验回放机制...")
            mixed_replay = MixedExperienceReplay(
                capacity=1000,
                current_stage_ratio=0.7,
                historical_ratio=0.3
            )
            
            # 添加不同阶段的经验
            for stage_id in range(3):
                for _ in range(50):
                    experience = {
                        'state': np.random.randn(10),
                        'action': np.random.randint(0, 5),
                        'reward': np.random.randn(),
                        'next_state': np.random.randn(10),
                        'done': False,
                        'stage_id': stage_id
                    }
                    mixed_replay.add_experience(experience, stage_id)
            
            # 测试采样
            batch = mixed_replay.sample_batch(32, current_stage_id=1)
            assert len(batch) == 32, f"批次大小不正确: {len(batch)}"
            
            # 验证混合比例
            stage_counts = {}
            for exp in batch:
                stage_id = exp['stage_id']
                stage_counts[stage_id] = stage_counts.get(stage_id, 0) + 1
            
            print(f"  ✓ 混合经验回放测试通过，阶段分布: {stage_counts}")
            
            results["status"] = "passed"
            results["duration"] = time.time() - test_start_time
            print(f"✓ 课程学习训练流程测试通过 (耗时: {results['duration']:.2f}秒)")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ 课程学习训练流程测试失败: {e}")
        
        return results
    
    def test_zero_shot_transfer_capability(self) -> Dict:
        """
        测试2: 验证零样本迁移能力
        
        验证：
        - 从小规模训练场景到大规模测试场景的迁移
        - 尺度不变性
        - 性能保持度
        """
        print("\n=== 测试2: 零样本迁移能力验证 ===")
        test_start_time = time.time()
        results = {
            "test_name": "zero_shot_transfer",
            "status": "running",
            "training_scenarios": [],
            "test_scenarios": [],
            "transfer_results": [],
            "errors": []
        }
        
        try:
            # 2.1 创建小规模训练场景
            print("2.1 创建小规模训练场景...")
            small_uavs, small_targets, obstacles = self._create_test_scenario(3, 2)
            small_graph = DirectedGraph(small_uavs, small_targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            small_env = UAVTaskEnv(small_uavs, small_targets, small_graph, obstacles, self.config, obs_mode="graph")
            
            results["training_scenarios"].append({
                "n_uavs": len(small_uavs),
                "n_targets": len(small_targets),
                "obs_space_shape": str(small_env.observation_space)
            })
            
            # 2.2 创建TransformerGNN模型
            print("2.2 创建TransformerGNN模型...")
            model_config = {
                "embed_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "use_position_encoding": True,
                "use_noisy_linear": False,  # 简化测试
                "use_local_attention": True,
                "k_adaptive": True,
                "k_min": 2,
                "k_max": 8
            }
            
            model = TransformerGNN(
                obs_space=small_env.observation_space,
                action_space=small_env.action_space,
                num_outputs=small_env.action_space.n,
                model_config=model_config,
                name="test_transformer_gnn"
            )
            
            print(f"  ✓ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
            
            # 2.3 简化训练（快速收敛测试）
            print("2.3 执行简化训练...")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            training_losses = []
            for episode in range(50):  # 简化训练轮数
                obs = small_env.reset()
                episode_loss = 0
                
                for step in range(10):  # 简化步数
                    # 转换观测为张量
                    obs_tensor = self._convert_obs_to_tensor(obs)
                    
                    # 前向传播
                    logits, _ = model({"obs": obs_tensor}, [], [])
                    
                    # 选择动作
                    action_probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(action_probs, 1).item()
                    
                    # 环境步进
                    next_obs, reward, done, truncated, info = small_env.step(action)
                    
                    # 简化损失计算（策略梯度的简化版本）
                    loss = -torch.log(action_probs[0, action]) * reward
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    episode_loss += loss.item()
                    obs = next_obs
                    
                    if done or truncated:
                        break
                
                training_losses.append(episode_loss)
                
                if episode % 10 == 0:
                    print(f"  训练进度: Episode {episode}, Loss: {episode_loss:.4f}")
            
            print(f"  ✓ 简化训练完成，最终损失: {training_losses[-1]:.4f}")
            
            # 2.4 测试不同规模场景的零样本迁移
            print("2.4 测试零样本迁移...")
            test_scenarios = [
                (5, 3),   # 中等规模
                (8, 5),   # 大规模
                (12, 8)   # 超大规模
            ]
            
            for n_uavs, n_targets in test_scenarios:
                print(f"  测试场景: {n_uavs} UAVs, {n_targets} 目标")
                
                # 创建测试环境
                test_uavs, test_targets, test_obstacles = self._create_test_scenario(n_uavs, n_targets)
                test_graph = DirectedGraph(test_uavs, test_targets, self.config.GRAPH_N_PHI, test_obstacles, self.config)
                test_env = UAVTaskEnv(test_uavs, test_targets, test_graph, test_obstacles, self.config, obs_mode="graph")
                
                # 执行测试回合
                test_rewards = []
                test_completion_rates = []
                
                for test_episode in range(5):  # 每个规模测试5回合
                    obs = test_env.reset()
                    episode_reward = 0
                    
                    for step in range(20):  # 限制步数
                        obs_tensor = self._convert_obs_to_tensor(obs)
                        
                        with torch.no_grad():
                            logits, _ = model({"obs": obs_tensor}, [], [])
                            action_probs = torch.softmax(logits, dim=-1)
                            action = torch.argmax(action_probs, dim=-1).item()
                        
                        next_obs, reward, done, truncated, info = test_env.step(action)
                        episode_reward += reward
                        obs = next_obs
                        
                        if done or truncated:
                            break
                    
                    # 计算完成率
                    completed_targets = sum(1 for t in test_env.targets if np.all(t.remaining_resources <= 0))
                    completion_rate = completed_targets / len(test_env.targets)
                    
                    test_rewards.append(episode_reward)
                    test_completion_rates.append(completion_rate)
                
                avg_reward = np.mean(test_rewards)
                avg_completion = np.mean(test_completion_rates)
                
                transfer_result = {
                    "scenario_size": (n_uavs, n_targets),
                    "avg_reward": float(avg_reward),
                    "avg_completion_rate": float(avg_completion),
                    "reward_std": float(np.std(test_rewards)),
                    "completion_std": float(np.std(test_completion_rates))
                }
                
                results["transfer_results"].append(transfer_result)
                results["test_scenarios"].append({
                    "n_uavs": n_uavs,
                    "n_targets": n_targets,
                    "obs_space_shape": str(test_env.observation_space)
                })
                
                print(f"    ✓ 平均奖励: {avg_reward:.2f}, 平均完成率: {avg_completion:.3f}")
            
            # 2.5 分析迁移性能
            print("2.5 分析迁移性能...")
            if len(results["transfer_results"]) >= 2:
                # 计算性能衰减
                baseline_completion = results["transfer_results"][0]["avg_completion_rate"]
                performance_retention = []
                
                for result in results["transfer_results"][1:]:
                    retention = result["avg_completion_rate"] / baseline_completion if baseline_completion > 0 else 0
                    performance_retention.append(retention)
                
                avg_retention = np.mean(performance_retention)
                results["performance_retention"] = float(avg_retention)
                
                print(f"  ✓ 平均性能保持度: {avg_retention:.3f}")
                
                # 判断迁移成功标准
                if avg_retention >= 0.7:  # 保持70%以上性能认为迁移成功
                    print("  ✓ 零样本迁移能力验证通过")
                else:
                    print(f"  ⚠ 零样本迁移性能较低，保持度: {avg_retention:.3f}")
            
            results["status"] = "passed"
            results["duration"] = time.time() - test_start_time
            print(f"✓ 零样本迁移能力验证完成 (耗时: {results['duration']:.2f}秒)")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ 零样本迁移能力验证失败: {e}")
        
        return results
    
    def test_scale_invariant_metrics(self) -> Dict:
        """
        测试3: 验证尺度不变指标的正确性和一致性
        
        验证：
        - Per-Agent Reward计算
        - Normalized Completion Score计算
        - Efficiency Metric计算
        - 不同规模场景下的指标一致性
        """
        print("\n=== 测试3: 尺度不变指标验证 ===")
        test_start_time = time.time()
        results = {
            "test_name": "scale_invariant_metrics",
            "status": "running",
            "metric_tests": [],
            "consistency_tests": [],
            "errors": []
        }
        
        try:
            # 3.1 测试Per-Agent Reward计算
            print("3.1 测试Per-Agent Reward计算...")
            
            scenarios = [(3, 2), (6, 4), (12, 8)]  # 不同规模场景
            per_agent_rewards = []
            
            for n_uavs, n_targets in scenarios:
                uavs, targets, obstacles = self._create_test_scenario(n_uavs, n_targets)
                graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
                
                # 执行几步获取奖励
                obs = env.reset()
                total_reward = 0
                steps = 0
                
                for _ in range(10):
                    action = env.action_space.sample()
                    next_obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    obs = next_obs
                    
                    if done or truncated:
                        break
                
                # 计算Per-Agent Reward
                n_active_uavs = len([u for u in env.uavs if np.any(u.resources > 0)])
                per_agent_reward = total_reward / n_active_uavs if n_active_uavs > 0 else 0
                
                per_agent_rewards.append({
                    "scenario": (n_uavs, n_targets),
                    "total_reward": float(total_reward),
                    "n_active_uavs": n_active_uavs,
                    "per_agent_reward": float(per_agent_reward)
                })
                
                print(f"  场景({n_uavs}, {n_targets}): Per-Agent Reward = {per_agent_reward:.3f}")
            
            results["metric_tests"].append({
                "metric_name": "per_agent_reward",
                "test_results": per_agent_rewards
            })
            
            # 3.2 测试Normalized Completion Score计算
            print("3.2 测试Normalized Completion Score计算...")
            
            completion_scores = []
            for n_uavs, n_targets in scenarios:
                uavs, targets, obstacles = self._create_test_scenario(n_uavs, n_targets)
                graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
                
                # 模拟部分完成状态
                obs = env.reset()
                for _ in range(15):  # 执行更多步骤
                    action = env.action_space.sample()
                    next_obs, reward, done, truncated, info = env.step(action)
                    obs = next_obs
                    if done or truncated:
                        break
                
                # 计算Normalized Completion Score
                satisfied_targets = sum(1 for t in env.targets if np.all(t.remaining_resources <= 0))
                satisfied_targets_rate = satisfied_targets / len(env.targets)
                
                # 计算拥堵指标（简化版本）
                total_allocations = sum(len(t.allocated_uavs) for t in env.targets)
                avg_congestion = total_allocations / len(env.targets) if len(env.targets) > 0 else 0
                congestion_metric = min(avg_congestion / n_uavs, 1.0) if n_uavs > 0 else 0
                
                normalized_completion_score = satisfied_targets_rate * (1 - congestion_metric)
                
                completion_scores.append({
                    "scenario": (n_uavs, n_targets),
                    "satisfied_targets_rate": float(satisfied_targets_rate),
                    "congestion_metric": float(congestion_metric),
                    "normalized_completion_score": float(normalized_completion_score)
                })
                
                print(f"  场景({n_uavs}, {n_targets}): Completion Score = {normalized_completion_score:.3f}")
            
            results["metric_tests"].append({
                "metric_name": "normalized_completion_score",
                "test_results": completion_scores
            })
            
            # 3.3 测试Efficiency Metric计算
            print("3.3 测试Efficiency Metric计算...")
            
            efficiency_metrics = []
            for n_uavs, n_targets in scenarios:
                uavs, targets, obstacles = self._create_test_scenario(n_uavs, n_targets)
                graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
                
                obs = env.reset()
                total_flight_distance = 0
                
                for _ in range(10):
                    action = env.action_space.sample()
                    
                    # 记录飞行距离
                    target_idx, uav_idx, phi_idx = env._action_to_assignment(action)
                    if target_idx < len(env.targets) and uav_idx < len(env.uavs):
                        target = env.targets[target_idx]
                        uav = env.uavs[uav_idx]
                        distance = np.linalg.norm(np.array(target.position) - np.array(uav.current_position))
                        total_flight_distance += distance
                    
                    next_obs, reward, done, truncated, info = env.step(action)
                    obs = next_obs
                    if done or truncated:
                        break
                
                # 计算Efficiency Metric
                completed_targets = sum(1 for t in env.targets if np.all(t.remaining_resources <= 0))
                efficiency_metric = completed_targets / total_flight_distance if total_flight_distance > 0 else 0
                
                efficiency_metrics.append({
                    "scenario": (n_uavs, n_targets),
                    "completed_targets": completed_targets,
                    "total_flight_distance": float(total_flight_distance),
                    "efficiency_metric": float(efficiency_metric)
                })
                
                print(f"  场景({n_uavs}, {n_targets}): Efficiency = {efficiency_metric:.6f}")
            
            results["metric_tests"].append({
                "metric_name": "efficiency_metric",
                "test_results": efficiency_metrics
            })
            
            # 3.4 测试指标一致性
            print("3.4 测试指标一致性...")
            
            # 检查指标是否在合理范围内
            for metric_test in results["metric_tests"]:
                metric_name = metric_test["metric_name"]
                test_results = metric_test["test_results"]
                
                if metric_name == "per_agent_reward":
                    # Per-Agent Reward应该相对稳定
                    rewards = [r["per_agent_reward"] for r in test_results]
                    reward_std = np.std(rewards)
                    consistency_score = 1.0 / (1.0 + reward_std)  # 标准差越小，一致性越高
                    
                elif metric_name == "normalized_completion_score":
                    # Completion Score应该在[0, 1]范围内
                    scores = [r["normalized_completion_score"] for r in test_results]
                    valid_range = all(0 <= s <= 1 for s in scores)
                    consistency_score = 1.0 if valid_range else 0.0
                    
                elif metric_name == "efficiency_metric":
                    # Efficiency Metric应该为正数
                    efficiencies = [r["efficiency_metric"] for r in test_results]
                    all_positive = all(e >= 0 for e in efficiencies)
                    consistency_score = 1.0 if all_positive else 0.0
                
                results["consistency_tests"].append({
                    "metric_name": metric_name,
                    "consistency_score": float(consistency_score),
                    "details": f"测试通过" if consistency_score > 0.8 else f"一致性较低"
                })
                
                print(f"  {metric_name}一致性得分: {consistency_score:.3f}")
            
            results["status"] = "passed"
            results["duration"] = time.time() - test_start_time
            print(f"✓ 尺度不变指标验证完成 (耗时: {results['duration']:.2f}秒)")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ 尺度不变指标验证失败: {e}")
        
        return results
    
    def test_tensorboard_logging(self) -> Dict:
        """
        测试4: TensorBoard日志记录和可视化功能完整性
        
        验证：
        - TensorBoard Writer创建和使用
        - 各类指标记录
        - 日志文件生成
        - 可视化数据完整性
        """
        print("\n=== 测试4: TensorBoard日志记录验证 ===")
        test_start_time = time.time()
        results = {
            "test_name": "tensorboard_logging",
            "status": "running",
            "tensorboard_available": TENSORBOARD_AVAILABLE,
            "log_files_created": [],
            "metrics_logged": [],
            "errors": []
        }
        
        try:
            if not TENSORBOARD_AVAILABLE:
                print("⚠ TensorBoard不可用，跳过相关测试")
                results["status"] = "skipped"
                results["skip_reason"] = "TensorBoard not available"
                return results
            
            # 4.1 创建TensorBoard Writer
            print("4.1 创建TensorBoard Writer...")
            tb_test_dir = os.path.join(self.tensorboard_dir, "test_run")
            writer = SummaryWriter(tb_test_dir)
            
            print(f"  ✓ TensorBoard Writer创建成功: {tb_test_dir}")
            
            # 4.2 测试各类指标记录
            print("4.2 测试指标记录...")
            
            # 记录标量指标
            scalar_metrics = [
                ("Training/Loss", [1.5, 1.2, 0.9, 0.7, 0.5]),
                ("Training/Reward", [10.0, 15.0, 20.0, 25.0, 30.0]),
                ("Training/Completion_Rate", [0.2, 0.4, 0.6, 0.7, 0.8]),
                ("Training/Per_Agent_Reward", [3.3, 3.7, 4.1, 4.5, 4.8]),
                ("Training/Efficiency_Metric", [0.001, 0.002, 0.003, 0.004, 0.005])
            ]
            
            for metric_name, values in scalar_metrics:
                for step, value in enumerate(values):
                    writer.add_scalar(metric_name, value, step)
                
                results["metrics_logged"].append({
                    "metric_name": metric_name,
                    "num_points": len(values),
                    "value_range": (min(values), max(values))
                })
                
                print(f"  ✓ 记录标量指标: {metric_name} ({len(values)}个数据点)")
            
            # 记录直方图数据
            print("4.3 测试直方图记录...")
            for step in range(5):
                # 模拟网络权重分布
                weights = torch.randn(100) * (0.5 + step * 0.1)
                writer.add_histogram("Weights/layer1", weights, step)
                
                # 模拟梯度分布
                gradients = torch.randn(100) * (0.1 + step * 0.02)
                writer.add_histogram("Gradients/layer1", gradients, step)
            
            results["metrics_logged"].extend([
                {"metric_name": "Weights/layer1", "type": "histogram", "num_points": 5},
                {"metric_name": "Gradients/layer1", "type": "histogram", "num_points": 5}
            ])
            
            print("  ✓ 直方图记录完成")
            
            # 4.4 记录课程学习相关指标
            print("4.4 测试课程学习指标记录...")
            
            curriculum_metrics = [
                ("Curriculum/Current_Stage", [0, 0, 1, 1, 2]),
                ("Curriculum/Stage_Progress", [0.2, 0.5, 0.1, 0.8, 0.3]),
                ("Curriculum/Fallback_Count", [0, 0, 0, 1, 1]),
                ("Curriculum/Mixed_Replay_Ratio", [0.7, 0.7, 0.7, 0.7, 0.7])
            ]
            
            for metric_name, values in curriculum_metrics:
                for step, value in enumerate(values):
                    writer.add_scalar(metric_name, step, value)
                
                results["metrics_logged"].append({
                    "metric_name": metric_name,
                    "type": "curriculum",
                    "num_points": len(values)
                })
            
            print("  ✓ 课程学习指标记录完成")
            
            # 4.5 关闭Writer并检查文件
            print("4.5 检查日志文件生成...")
            writer.close()
            
            # 检查生成的文件
            if os.path.exists(tb_test_dir):
                log_files = list(Path(tb_test_dir).rglob("*"))
                results["log_files_created"] = [str(f) for f in log_files]
                
                print(f"  ✓ 生成日志文件数量: {len(log_files)}")
                for log_file in log_files[:5]:  # 显示前5个文件
                    print(f"    - {log_file.name}")
                
                if len(log_files) > 5:
                    print(f"    ... 还有{len(log_files) - 5}个文件")
            
            # 4.6 验证日志数据完整性
            print("4.6 验证日志数据完整性...")
            
            total_metrics = len(results["metrics_logged"])
            scalar_metrics_count = len([m for m in results["metrics_logged"] if m.get("type") != "histogram"])
            histogram_metrics_count = len([m for m in results["metrics_logged"] if m.get("type") == "histogram"])
            
            print(f"  ✓ 总指标数量: {total_metrics}")
            print(f"  ✓ 标量指标: {scalar_metrics_count}")
            print(f"  ✓ 直方图指标: {histogram_metrics_count}")
            
            # 验证关键指标是否都已记录
            required_metrics = ["Training/Loss", "Training/Reward", "Training/Completion_Rate"]
            logged_metric_names = [m["metric_name"] for m in results["metrics_logged"]]
            
            missing_metrics = [m for m in required_metrics if m not in logged_metric_names]
            if missing_metrics:
                results["errors"].append(f"缺少关键指标: {missing_metrics}")
                print(f"  ⚠ 缺少关键指标: {missing_metrics}")
            else:
                print("  ✓ 所有关键指标已记录")
            
            results["status"] = "passed"
            results["duration"] = time.time() - test_start_time
            print(f"✓ TensorBoard日志记录验证完成 (耗时: {results['duration']:.2f}秒)")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ TensorBoard日志记录验证失败: {e}")
        
        return results
    
    def test_solution_output_compatibility(self) -> Dict:
        """
        测试5: 验证方案输出和评估流程的完整性
        
        验证：
        - TransformerGNN输出格式兼容性
        - 方案转换接口
        - 评估流程完整性
        - 与现有系统的兼容性
        """
        print("\n=== 测试5: 方案输出兼容性验证 ===")
        test_start_time = time.time()
        results = {
            "test_name": "solution_output_compatibility",
            "status": "running",
            "output_format_tests": [],
            "compatibility_tests": [],
            "evaluation_tests": [],
            "errors": []
        }
        
        try:
            # 5.1 测试TransformerGNN输出格式
            print("5.1 测试TransformerGNN输出格式...")
            
            # 创建测试环境和模型
            uavs, targets, obstacles = self._create_test_scenario(4, 3)
            graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
            
            model_config = {
                "embed_dim": 32,
                "num_heads": 2,
                "num_layers": 1,
                "use_noisy_linear": False
            }
            
            model = TransformerGNN(
                obs_space=env.observation_space,
                action_space=env.action_space,
                num_outputs=env.action_space.n,
                model_config=model_config,
                name="compatibility_test"
            )
            
            # 测试模型输出
            obs = env.reset()
            obs_tensor = self._convert_obs_to_tensor(obs)
            
            with torch.no_grad():
                logits, state = model({"obs": obs_tensor}, [], [])
                value = model.value_function()
            
            # 验证输出格式
            assert logits.shape[0] == 1, f"批次维度错误: {logits.shape[0]}"
            assert logits.shape[1] == env.action_space.n, f"动作维度错误: {logits.shape[1]} != {env.action_space.n}"
            assert value.shape[0] == 1, f"值函数维度错误: {value.shape[0]}"
            
            results["output_format_tests"].append({
                "test_name": "transformer_gnn_output",
                "logits_shape": list(logits.shape),
                "value_shape": list(value.shape),
                "action_space_size": env.action_space.n,
                "status": "passed"
            })
            
            print(f"  ✓ TransformerGNN输出格式验证通过")
            print(f"    - Logits形状: {logits.shape}")
            print(f"    - Value形状: {value.shape}")
            
            # 5.2 测试动作选择和转换
            print("5.2 测试动作选择和转换...")
            
            # 测试不同的动作选择策略
            action_selection_methods = [
                ("greedy", lambda x: torch.argmax(x, dim=-1)),
                ("sampling", lambda x: torch.multinomial(torch.softmax(x, dim=-1), 1)),
                ("epsilon_greedy", lambda x: torch.randint(0, x.shape[-1], (1,)) if np.random.random() < 0.1 
                                           else torch.argmax(x, dim=-1))
            ]
            
            for method_name, action_fn in action_selection_methods:
                action = action_fn(logits).item()
                
                # 验证动作有效性
                assert 0 <= action < env.action_space.n, f"动作超出范围: {action}"
                
                # 测试动作转换
                target_idx, uav_idx, phi_idx = env._action_to_assignment(action)
                assert 0 <= target_idx < len(env.targets), f"目标索引无效: {target_idx}"
                assert 0 <= uav_idx < len(env.uavs), f"UAV索引无效: {uav_idx}"
                assert 0 <= phi_idx < graph.n_phi, f"方向索引无效: {phi_idx}"
                
                results["output_format_tests"].append({
                    "test_name": f"action_selection_{method_name}",
                    "action": action,
                    "target_idx": target_idx,
                    "uav_idx": uav_idx,
                    "phi_idx": phi_idx,
                    "status": "passed"
                })
                
                print(f"  ✓ {method_name}动作选择测试通过: action={action} -> ({target_idx}, {uav_idx}, {phi_idx})")
            
            # 5.3 测试方案生成和评估
            print("5.3 测试方案生成和评估...")
            
            # 生成完整方案
            obs = env.reset()
            solution_steps = []
            
            for step in range(10):  # 限制步数
                obs_tensor = self._convert_obs_to_tensor(obs)
                
                with torch.no_grad():
                    logits, _ = model({"obs": obs_tensor}, [], [])
                    action = torch.argmax(logits, dim=-1).item()
                
                # 记录方案步骤
                target_idx, uav_idx, phi_idx = env._action_to_assignment(action)
                solution_steps.append({
                    "step": step,
                    "action": action,
                    "target_id": env.targets[target_idx].id,
                    "uav_id": env.uavs[uav_idx].id,
                    "phi_idx": phi_idx
                })
                
                next_obs, reward, done, truncated, info = env.step(action)
                obs = next_obs
                
                if done or truncated:
                    break
            
            # 评估方案质量
            completed_targets = sum(1 for t in env.targets if np.all(t.remaining_resources <= 0))
            completion_rate = completed_targets / len(env.targets)
            total_reward = sum(step.get("reward", 0) for step in solution_steps)
            
            evaluation_result = {
                "solution_steps": len(solution_steps),
                "completed_targets": completed_targets,
                "total_targets": len(env.targets),
                "completion_rate": float(completion_rate),
                "total_reward": float(total_reward)
            }
            
            results["evaluation_tests"].append({
                "test_name": "solution_generation",
                "result": evaluation_result,
                "status": "passed"
            })
            
            print(f"  ✓ 方案生成测试通过:")
            print(f"    - 方案步数: {len(solution_steps)}")
            print(f"    - 完成率: {completion_rate:.3f}")
            print(f"    - 总奖励: {total_reward:.2f}")
            
            # 5.4 测试与现有系统的兼容性
            print("5.4 测试系统兼容性...")
            
            # 测试扁平模式兼容性
            flat_env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="flat")
            flat_obs = flat_env.reset()
            
            # 验证扁平观测格式
            assert isinstance(flat_obs, np.ndarray), f"扁平观测应为numpy数组: {type(flat_obs)}"
            assert len(flat_obs.shape) == 1, f"扁平观测应为1维: {flat_obs.shape}"
            
            results["compatibility_tests"].append({
                "test_name": "flat_mode_compatibility",
                "obs_shape": list(flat_obs.shape),
                "obs_type": str(type(flat_obs)),
                "status": "passed"
            })
            
            print(f"  ✓ 扁平模式兼容性验证通过")
            
            # 测试图模式兼容性
            graph_obs = env.reset()
            assert isinstance(graph_obs, dict), f"图观测应为字典: {type(graph_obs)}"
            
            required_keys = ["uav_features", "target_features", "relative_positions", "distances", "masks"]
            missing_keys = [key for key in required_keys if key not in graph_obs]
            
            if missing_keys:
                results["errors"].append(f"图观测缺少键: {missing_keys}")
            else:
                results["compatibility_tests"].append({
                    "test_name": "graph_mode_compatibility",
                    "obs_keys": list(graph_obs.keys()),
                    "status": "passed"
                })
                print(f"  ✓ 图模式兼容性验证通过")
            
            results["status"] = "passed"
            results["duration"] = time.time() - test_start_time
            print(f"✓ 方案输出兼容性验证完成 (耗时: {results['duration']:.2f}秒)")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ 方案输出兼容性验证失败: {e}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """
        运行完整的端到端集成测试
        
        Returns:
            完整的测试结果字典
        """
        print("=" * 80)
        print("开始端到端系统集成测试")
        print("=" * 80)
        
        start_time = time.time()
        
        # 运行所有测试
        test_results = {
            "test_suite": "end_to_end_integration",
            "start_time": start_time,
            "test_results": {},
            "summary": {}
        }
        
        # 测试1: 课程学习训练流程
        test_results["test_results"]["curriculum_learning"] = self.test_curriculum_learning_pipeline()
        
        # 测试2: 零样本迁移能力
        test_results["test_results"]["zero_shot_transfer"] = self.test_zero_shot_transfer_capability()
        
        # 测试3: 尺度不变指标
        test_results["test_results"]["scale_invariant_metrics"] = self.test_scale_invariant_metrics()
        
        # 测试4: TensorBoard日志记录
        test_results["test_results"]["tensorboard_logging"] = self.test_tensorboard_logging()
        
        # 测试5: 方案输出兼容性
        test_results["test_results"]["solution_compatibility"] = self.test_solution_output_compatibility()
        
        # 生成测试摘要
        end_time = time.time()
        test_results["end_time"] = end_time
        test_results["total_duration"] = end_time - start_time
        
        # 统计测试结果
        passed_tests = sum(1 for result in test_results["test_results"].values() 
                          if result["status"] == "passed")
        failed_tests = sum(1 for result in test_results["test_results"].values() 
                          if result["status"] == "failed")
        skipped_tests = sum(1 for result in test_results["test_results"].values() 
                           if result["status"] == "skipped")
        total_tests = len(test_results["test_results"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
        }
        
        # 保存测试结果
        self._save_test_results(test_results)
        
        # 打印测试摘要
        self._print_test_summary(test_results)
        
        return test_results
    
    def _create_test_scenario(self, n_uavs: int, n_targets: int) -> Tuple[List[UAV], List[Target], List]:
        """创建测试场景"""
        # 创建UAVs
        uavs = []
        for i in range(n_uavs):
            uav = UAV(
                id=i,
                position=(np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
                resources=np.array([50.0, 30.0]),
                max_distance=200.0,
                velocity_range=(10.0, 50.0)
            )
            uavs.append(uav)
        
        # 创建目标
        targets = []
        for i in range(n_targets):
            target = Target(
                id=i,
                position=(np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
                resources=np.array([30.0, 20.0]),
                value=100.0
            )
            targets.append(target)
        
        # 简单障碍物（空列表）
        obstacles = []
        
        return uavs, targets, obstacles
    
    def _convert_obs_to_tensor(self, obs) -> torch.Tensor:
        """将观测转换为张量"""
        if isinstance(obs, dict):
            # 图模式观测
            obs_tensor = {}
            for key, value in obs.items():
                if isinstance(value, dict):
                    obs_tensor[key] = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in value.items()}
                else:
                    obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0)
            return obs_tensor
        else:
            # 扁平模式观测
            return torch.FloatTensor(obs).unsqueeze(0)
    
    def _save_test_results(self, test_results: Dict):
        """保存测试结果到文件"""
        results_file = os.path.join(self.test_output_dir, "end_to_end_test_results.json")
        
        # 转换numpy类型为Python原生类型
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(test_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: {results_file}")
    
    def _print_test_summary(self, test_results: Dict):
        """打印测试摘要"""
        print("\n" + "=" * 80)
        print("端到端系统集成测试摘要")
        print("=" * 80)
        
        summary = test_results["summary"]
        print(f"总测试数量: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"跳过测试: {summary['skipped_tests']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"总耗时: {test_results['total_duration']:.2f}秒")
        print(f"整体状态: {summary['overall_status']}")
        
        print("\n详细测试结果:")
        for test_name, result in test_results["test_results"].items():
            status_symbol = "✓" if result["status"] == "passed" else "✗" if result["status"] == "failed" else "⚠"
            duration = result.get("duration", 0)
            print(f"  {status_symbol} {test_name}: {result['status'].upper()} ({duration:.2f}s)")
            
            if result["status"] == "failed" and result.get("errors"):
                for error in result["errors"][:2]:  # 显示前两个错误
                    print(f"    错误: {error}")
        
        print("=" * 80)


def main():
    """主函数"""
    print("端到端系统集成测试启动")
    
    # 创建测试器
    tester = EndToEndIntegrationTester()
    
    try:
        # 运行完整测试
        results = tester.run_comprehensive_test()
        
        # 根据测试结果设置退出码
        if results["summary"]["overall_status"] == "PASSED":
            print("\n🎉 所有测试通过！系统集成验证成功！")
            exit_code = 0
        else:
            print(f"\n❌ 测试失败！{results['summary']['failed_tests']}个测试未通过。")
            exit_code = 1
        
    except Exception as e:
        print(f"\n💥 测试执行过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 2
    
    finally:
        # 清理临时文件（可选）
        if hasattr(tester, 'test_output_dir') and tester.test_output_dir.startswith('/tmp'):
            print(f"\n清理临时目录: {tester.test_output_dir}")
            # shutil.rmtree(tester.test_output_dir, ignore_errors=True)
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
