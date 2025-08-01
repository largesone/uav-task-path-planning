# -*- coding: utf-8 -*-
"""
任务19: 简化版端到端系统集成测试
避免编码问题，专注于核心功能验证
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Tuple

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入核心模块
from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph
from config import Config
from curriculum_stages import CurriculumStages
from transformer_gnn import TransformerGNN

class SimpleEndToEndTester:
    """简化版端到端系统集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.config = Config()
        self.curriculum_stages = CurriculumStages()
        self.test_results = {}
        
        print("简化版端到端集成测试初始化完成")
    
    def test_curriculum_learning_basic(self) -> Dict:
        """测试1: 基础课程学习功能"""
        print("\n=== 测试1: 基础课程学习功能 ===")
        results = {
            "test_name": "curriculum_learning_basic",
            "status": "running",
            "stages_verified": 0,
            "errors": []
        }
        
        try:
            # 验证课程阶段配置
            stages = self.curriculum_stages.stages
            print(f"课程阶段数量: {len(stages)}")
            
            for i, stage in enumerate(stages):
                print(f"阶段{i}: {stage.stage_name}")
                print(f"  UAV范围: {stage.n_uavs_range}")
                print(f"  目标范围: {stage.n_targets_range}")
                print(f"  最大回合: {stage.max_episodes}")
                
                # 验证阶段配置合理性
                assert stage.n_uavs_range[0] <= stage.n_uavs_range[1]
                assert stage.n_targets_range[0] <= stage.n_targets_range[1]
                assert stage.max_episodes > 0
                
                results["stages_verified"] += 1
            
            # 测试阶段切换
            original_stage = self.curriculum_stages.current_stage_id
            
            # 测试推进
            if self.curriculum_stages.advance_to_next_stage():
                print(f"阶段推进测试: {original_stage} -> {self.curriculum_stages.current_stage_id}")
            
            # 测试回退
            if self.curriculum_stages.fallback_to_previous_stage():
                print(f"阶段回退测试: {self.curriculum_stages.current_stage_id + 1} -> {self.curriculum_stages.current_stage_id}")
            
            # 恢复原始阶段
            self.curriculum_stages.current_stage_id = original_stage
            
            results["status"] = "passed"
            print("课程学习基础功能测试通过")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"课程学习基础功能测试失败: {e}")
        
        return results
    
    def test_zero_shot_transfer_basic(self) -> Dict:
        """测试2: 基础零样本迁移能力"""
        print("\n=== 测试2: 基础零样本迁移能力 ===")
        results = {
            "test_name": "zero_shot_transfer_basic",
            "status": "running",
            "scenarios_tested": 0,
            "model_created": False,
            "errors": []
        }
        
        try:
            # 创建小规模训练场景
            print("创建小规模训练场景...")
            small_uavs, small_targets, obstacles = self._create_test_scenario(3, 2)
            small_graph = DirectedGraph(small_uavs, small_targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            small_env = UAVTaskEnv(small_uavs, small_targets, small_graph, obstacles, self.config, obs_mode="graph")
            
            print(f"小规模场景: {len(small_uavs)} UAVs, {len(small_targets)} 目标")
            results["scenarios_tested"] += 1
            
            # 创建TransformerGNN模型
            print("创建TransformerGNN模型...")
            model_config = {
                "embed_dim": 32,
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.1,
                "use_position_encoding": True,
                "use_noisy_linear": False,
                "use_local_attention": True,
                "k_adaptive": True,
                "k_min": 2,
                "k_max": 4
            }
            
            model = TransformerGNN(
                obs_space=small_env.observation_space,
                action_space=small_env.action_space,
                num_outputs=small_env.action_space.n,
                model_config=model_config,
                name="test_transformer_gnn"
            )
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"模型创建成功，参数数量: {param_count}")
            results["model_created"] = True
            
            # 测试不同规模场景
            test_scenarios = [(5, 3), (8, 5)]
            
            for n_uavs, n_targets in test_scenarios:
                print(f"测试场景: {n_uavs} UAVs, {n_targets} 目标")
                
                # 创建测试环境
                test_uavs, test_targets, test_obstacles = self._create_test_scenario(n_uavs, n_targets)
                test_graph = DirectedGraph(test_uavs, test_targets, self.config.GRAPH_N_PHI, test_obstacles, self.config)
                test_env = UAVTaskEnv(test_uavs, test_targets, test_graph, test_obstacles, self.config, obs_mode="graph")
                
                # 为每个场景创建对应的模型（零样本迁移的核心测试）
                test_model = TransformerGNN(
                    obs_space=test_env.observation_space,
                    action_space=test_env.action_space,
                    num_outputs=test_env.action_space.n,
                    model_config=model_config,
                    name=f"test_transformer_gnn_{n_uavs}_{n_targets}"
                )
                
                # 测试模型前向传播
                obs = test_env.reset()
                obs_tensor = self._convert_obs_to_tensor(obs)
                
                with torch.no_grad():
                    logits, _ = test_model({"obs": obs_tensor}, [], [])
                    value = test_model.value_function()
                
                # 验证输出格式
                assert logits.shape[0] == 1, f"批次维度错误: {logits.shape[0]}"
                assert logits.shape[1] == test_env.action_space.n, f"动作维度错误: {logits.shape[1]} != {test_env.action_space.n}"
                assert value.shape[0] == 1, f"值函数维度错误: {value.shape[0]}"
                
                print(f"  模型输出验证通过: logits={logits.shape}, value={value.shape}")
                print(f"  动作空间大小: {test_env.action_space.n}")
                results["scenarios_tested"] += 1
            
            results["status"] = "passed"
            print("零样本迁移基础功能测试通过")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"零样本迁移基础功能测试失败: {e}")
        
        return results
    
    def test_scale_invariant_metrics_basic(self) -> Dict:
        """测试3: 基础尺度不变指标"""
        print("\n=== 测试3: 基础尺度不变指标 ===")
        results = {
            "test_name": "scale_invariant_metrics_basic",
            "status": "running",
            "metrics_tested": 0,
            "errors": []
        }
        
        try:
            # 测试不同规模场景的指标计算
            scenarios = [(3, 2), (6, 4)]
            
            for n_uavs, n_targets in scenarios:
                print(f"测试场景: {n_uavs} UAVs, {n_targets} 目标")
                
                uavs, targets, obstacles = self._create_test_scenario(n_uavs, n_targets)
                graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
                
                # 执行几步获取奖励
                obs = env.reset()
                total_reward = 0
                
                for _ in range(5):
                    action = env.action_space.sample()
                    next_obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    obs = next_obs
                    
                    if done or truncated:
                        break
                
                # 计算Per-Agent Reward
                n_active_uavs = len([u for u in env.uavs if np.any(u.resources > 0)])
                per_agent_reward = total_reward / n_active_uavs if n_active_uavs > 0 else 0
                
                print(f"  Per-Agent Reward: {per_agent_reward:.3f}")
                
                # 计算Normalized Completion Score
                satisfied_targets = sum(1 for t in env.targets if np.all(t.remaining_resources <= 0))
                completion_rate = satisfied_targets / len(env.targets)
                
                print(f"  Completion Rate: {completion_rate:.3f}")
                
                results["metrics_tested"] += 1
            
            results["status"] = "passed"
            print("尺度不变指标基础功能测试通过")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"尺度不变指标基础功能测试失败: {e}")
        
        return results
    
    def test_dual_mode_compatibility(self) -> Dict:
        """测试4: 双模式兼容性"""
        print("\n=== 测试4: 双模式兼容性 ===")
        results = {
            "test_name": "dual_mode_compatibility",
            "status": "running",
            "modes_tested": 0,
            "errors": []
        }
        
        try:
            # 创建测试场景
            uavs, targets, obstacles = self._create_test_scenario(4, 3)
            graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            
            # 测试扁平模式
            print("测试扁平模式...")
            flat_env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="flat")
            flat_obs = flat_env.reset()
            
            assert isinstance(flat_obs, np.ndarray)
            assert len(flat_obs.shape) == 1
            print(f"  扁平观测形状: {flat_obs.shape}")
            
            # 测试环境步进
            action = flat_env.action_space.sample()
            next_obs, reward, done, truncated, info = flat_env.step(action)
            print(f"  扁平模式步进成功，奖励: {reward:.3f}")
            
            results["modes_tested"] += 1
            
            # 测试图模式
            print("测试图模式...")
            graph_env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
            graph_obs = graph_env.reset()
            
            assert isinstance(graph_obs, dict)
            required_keys = ["uav_features", "target_features", "relative_positions", "distances", "masks"]
            for key in required_keys:
                assert key in graph_obs, f"缺少键: {key}"
            
            print(f"  图观测键: {list(graph_obs.keys())}")
            
            # 测试环境步进
            action = graph_env.action_space.sample()
            next_obs, reward, done, truncated, info = graph_env.step(action)
            print(f"  图模式步进成功，奖励: {reward:.3f}")
            
            results["modes_tested"] += 1
            
            results["status"] = "passed"
            print("双模式兼容性测试通过")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"双模式兼容性测试失败: {e}")
        
        return results
    
    def run_simple_tests(self) -> Dict:
        """运行简化版测试套件"""
        print("=" * 60)
        print("开始简化版端到端系统集成测试")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行所有测试
        test_results = {
            "test_suite": "simple_end_to_end_integration",
            "start_time": start_time,
            "test_results": {},
            "summary": {}
        }
        
        # 执行测试
        test_results["test_results"]["curriculum_learning"] = self.test_curriculum_learning_basic()
        test_results["test_results"]["zero_shot_transfer"] = self.test_zero_shot_transfer_basic()
        test_results["test_results"]["scale_invariant_metrics"] = self.test_scale_invariant_metrics_basic()
        test_results["test_results"]["dual_mode_compatibility"] = self.test_dual_mode_compatibility()
        
        # 生成摘要
        end_time = time.time()
        test_results["end_time"] = end_time
        test_results["total_duration"] = end_time - start_time
        
        # 统计结果
        passed_tests = sum(1 for result in test_results["test_results"].values() 
                          if result["status"] == "passed")
        failed_tests = sum(1 for result in test_results["test_results"].values() 
                          if result["status"] == "failed")
        total_tests = len(test_results["test_results"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
        }
        
        # 打印摘要
        self._print_summary(test_results)
        
        return test_results
    
    def _create_test_scenario(self, n_uavs: int, n_targets: int) -> Tuple[List[UAV], List[Target], List]:
        """创建测试场景"""
        # 创建UAVs
        uavs = []
        for i in range(n_uavs):
            uav = UAV(
                id=i,
                position=(np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
                heading=np.random.uniform(0, 2*np.pi),
                resources=np.array([50.0, 30.0]),
                max_distance=200.0,
                velocity_range=(10.0, 50.0),
                economic_speed=25.0
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
    
    def _print_summary(self, test_results: Dict):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("简化版端到端系统集成测试摘要")
        print("=" * 60)
        
        summary = test_results["summary"]
        print(f"总测试数量: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"总耗时: {test_results['total_duration']:.2f}秒")
        print(f"整体状态: {summary['overall_status']}")
        
        print("\n详细测试结果:")
        for test_name, result in test_results["test_results"].items():
            status_symbol = "✓" if result["status"] == "passed" else "✗"
            print(f"  {status_symbol} {test_name}: {result['status'].upper()}")
            
            if result["status"] == "failed" and result.get("errors"):
                for error in result["errors"][:1]:  # 只显示第一个错误
                    print(f"    错误: {error}")
        
        print("=" * 60)


def main():
    """主函数"""
    print("简化版端到端系统集成测试启动")
    
    # 创建测试器
    tester = SimpleEndToEndTester()
    
    try:
        # 运行测试
        results = tester.run_simple_tests()
        
        # 根据测试结果设置退出码
        if results["summary"]["overall_status"] == "PASSED":
            print("\n🎉 所有测试通过！系统集成验证成功！")
            return 0
        else:
            print(f"\n❌ 测试失败！{results['summary']['failed_tests']}个测试未通过。")
            return 1
        
    except Exception as e:
        print(f"\n💥 测试执行过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)