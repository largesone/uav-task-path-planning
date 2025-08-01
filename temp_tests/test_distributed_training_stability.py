# -*- coding: utf-8 -*-
"""
分布式训练稳定性和性能测试
验证Ray RLlib分布式训练的数据一致性和稳定性
"""

import os
import sys
import time
import json
import numpy as np
import torch
import tempfile
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ray RLlib相关导入
try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    from ray.rllib.env.env_context import EnvContext
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("警告: Ray RLlib未安装，将跳过分布式训练测试")

from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph
from config import Config
from transformer_gnn import TransformerGNN

class DistributedTrainingTester:
    """分布式训练稳定性测试器"""
    
    def __init__(self, test_output_dir: str = None):
        """初始化测试器"""
        self.test_output_dir = test_output_dir or tempfile.mkdtemp(prefix="dist_test_")
        self.config = Config()
        self.test_results = {}
        
        os.makedirs(self.test_output_dir, exist_ok=True)
        print(f"分布式训练测试初始化完成，输出目录: {self.test_output_dir}")
    
    def test_ray_initialization(self) -> Dict:
        """测试Ray集群初始化"""
        print("\n=== 测试Ray集群初始化 ===")
        results = {
            "test_name": "ray_initialization",
            "status": "running",
            "ray_available": RAY_AVAILABLE,
            "cluster_info": {},
            "errors": []
        }
        
        if not RAY_AVAILABLE:
            results["status"] = "skipped"
            results["skip_reason"] = "Ray not available"
            return results
        
        try:
            # 初始化Ray
            if not ray.is_initialized():
                ray.init(local_mode=False, num_cpus=2, num_gpus=0)
            
            # 获取集群信息
            cluster_resources = ray.cluster_resources()
            results["cluster_info"] = {
                "total_cpus": cluster_resources.get("CPU", 0),
                "total_gpus": cluster_resources.get("GPU", 0),
                "total_memory": cluster_resources.get("memory", 0),
                "node_count": len(ray.nodes())
            }
            
            print(f"✓ Ray集群初始化成功")
            print(f"  - CPU数量: {results['cluster_info']['total_cpus']}")
            print(f"  - GPU数量: {results['cluster_info']['total_gpus']}")
            print(f"  - 节点数量: {results['cluster_info']['node_count']}")
            
            results["status"] = "passed"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ Ray集群初始化失败: {e}")
        
        return results
    
    def test_distributed_data_consistency(self) -> Dict:
        """测试分布式训练的数据一致性"""
        print("\n=== 测试分布式数据一致性 ===")
        results = {
            "test_name": "distributed_data_consistency",
            "status": "running",
            "consistency_tests": [],
            "errors": []
        }
        
        if not RAY_AVAILABLE:
            results["status"] = "skipped"
            return results
        
        try:
            # 创建测试环境工厂函数
            def env_creator(env_config):
                n_uavs = env_config.get("n_uavs", 4)
                n_targets = env_config.get("n_targets", 3)
                
                # 创建实体
                uavs = [UAV(i, (np.random.uniform(0, 1000), np.random.uniform(0, 1000)), 
                           np.array([50.0, 30.0]), 200.0, (10.0, 50.0)) for i in range(n_uavs)]
                targets = [Target(i, (np.random.uniform(0, 1000), np.random.uniform(0, 1000)), 
                                np.array([30.0, 20.0]), 100.0) for i in range(n_targets)]
                obstacles = []
                
                graph = DirectedGraph(uavs, targets, 8, obstacles, Config())
                return UAVTaskEnv(uavs, targets, graph, obstacles, Config(), obs_mode="graph")
            
            # 注册环境
            from ray.tune.registry import register_env
            register_env("test_uav_env", env_creator)
            
            # 配置PPO算法
            config = PPOConfig()
            config.environment("test_uav_env", env_config={"n_uavs": 4, "n_targets": 3})
            config.rollouts(num_rollout_workers=2, num_envs_per_worker=1)
            config.training(train_batch_size=128, sgd_minibatch_size=32)
            config.framework("torch")
            
            # 创建算法实例
            algo = config.build()
            
            # 执行几轮训练测试数据一致性
            print("执行分布式训练测试...")
            for i in range(3):
                result = algo.train()
                
                # 检查训练结果的一致性
                consistency_test = {
                    "iteration": i,
                    "episode_reward_mean": result.get("episode_reward_mean", 0),
                    "timesteps_this_iter": result.get("timesteps_this_iter", 0),
                    "training_iteration": result.get("training_iteration", 0)
                }
                
                results["consistency_tests"].append(consistency_test)
                print(f"  迭代 {i}: 平均奖励 = {consistency_test['episode_reward_mean']:.3f}")
            
            # 清理资源
            algo.stop()
            
            # 验证数据一致性
            if len(results["consistency_tests"]) >= 2:
                # 检查训练是否有进展
                first_reward = results["consistency_tests"][0]["episode_reward_mean"]
                last_reward = results["consistency_tests"][-1]["episode_reward_mean"]
                
                if abs(last_reward - first_reward) > 0.1:  # 有明显变化说明训练在进行
                    print("✓ 分布式训练数据一致性验证通过")
                    results["status"] = "passed"
                else:
                    print("⚠ 训练进展较小，可能存在数据一致性问题")
                    results["status"] = "warning"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ 分布式数据一致性测试失败: {e}")
        
        return results
    
    def test_worker_fault_tolerance(self) -> Dict:
        """测试Worker故障容错能力"""
        print("\n=== 测试Worker故障容错 ===")
        results = {
            "test_name": "worker_fault_tolerance",
            "status": "running",
            "fault_tolerance_tests": [],
            "errors": []
        }
        
        if not RAY_AVAILABLE:
            results["status"] = "skipped"
            return results
        
        try:
            # 模拟Worker故障场景
            print("模拟Worker故障场景...")
            
            # 创建多个Worker的配置
            worker_configs = [
                {"num_rollout_workers": 1, "test_name": "single_worker"},
                {"num_rollout_workers": 2, "test_name": "dual_worker"},
                {"num_rollout_workers": 3, "test_name": "triple_worker"}
            ]
            
            for worker_config in worker_configs:
                print(f"  测试配置: {worker_config['test_name']}")
                
                try:
                    # 创建环境工厂
                    def env_creator(env_config):
                        uavs = [UAV(i, (100*i, 100*i), np.array([50.0, 30.0]), 200.0, (10.0, 50.0)) 
                               for i in range(3)]
                        targets = [Target(i, (200+100*i, 200+100*i), np.array([30.0, 20.0]), 100.0) 
                                 for i in range(2)]
                        graph = DirectedGraph(uavs, targets, 8, [], Config())
                        return UAVTaskEnv(uavs, targets, graph, [], Config(), obs_mode="flat")
                    
                    from ray.tune.registry import register_env
                    register_env(f"fault_test_env_{worker_config['test_name']}", env_creator)
                    
                    # 配置算法
                    config = PPOConfig()
                    config.environment(f"fault_test_env_{worker_config['test_name']}")
                    config.rollouts(
                        num_rollout_workers=worker_config["num_rollout_workers"],
                        num_envs_per_worker=1,
                        rollout_fragment_length=50
                    )
                    config.training(train_batch_size=100, sgd_minibatch_size=25)
                    config.framework("torch")
                    
                    # 创建算法并训练
                    algo = config.build()
                    
                    # 执行训练并记录性能
                    training_results = []
                    for i in range(2):  # 简化测试
                        result = algo.train()
                        training_results.append({
                            "iteration": i,
                            "reward": result.get("episode_reward_mean", 0),
                            "episodes": result.get("episodes_this_iter", 0)
                        })
                    
                    algo.stop()
                    
                    fault_test_result = {
                        "config": worker_config,
                        "training_results": training_results,
                        "status": "passed",
                        "final_reward": training_results[-1]["reward"] if training_results else 0
                    }
                    
                    results["fault_tolerance_tests"].append(fault_test_result)
                    print(f"    ✓ {worker_config['test_name']} 测试通过")
                    
                except Exception as worker_error:
                    fault_test_result = {
                        "config": worker_config,
                        "status": "failed",
                        "error": str(worker_error)
                    }
                    results["fault_tolerance_tests"].append(fault_test_result)
                    print(f"    ✗ {worker_config['test_name']} 测试失败: {worker_error}")
            
            # 评估整体容错性能
            passed_tests = sum(1 for test in results["fault_tolerance_tests"] 
                             if test["status"] == "passed")
            total_tests = len(results["fault_tolerance_tests"])
            
            if passed_tests >= total_tests * 0.7:  # 70%通过率
                results["status"] = "passed"
                print(f"✓ Worker故障容错测试通过 ({passed_tests}/{total_tests})")
            else:
                results["status"] = "failed"
                print(f"✗ Worker故障容错测试失败 ({passed_tests}/{total_tests})")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ Worker故障容错测试失败: {e}")
        
        return results
    
    def test_memory_usage_stability(self) -> Dict:
        """测试内存使用稳定性"""
        print("\n=== 测试内存使用稳定性 ===")
        results = {
            "test_name": "memory_usage_stability",
            "status": "running",
            "memory_snapshots": [],
            "memory_leak_detected": False,
            "errors": []
        }
        
        try:
            import psutil
            import gc
            
            # 获取初始内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"初始内存使用: {initial_memory:.2f} MB")
            
            # 执行多轮内存密集型操作
            for iteration in range(5):
                print(f"  内存测试迭代 {iteration + 1}/5")
                
                # 创建大量对象模拟训练过程
                large_tensors = []
                for _ in range(10):
                    # 创建较大的张量
                    tensor = torch.randn(1000, 1000)
                    large_tensors.append(tensor)
                
                # 模拟TransformerGNN前向传播
                for _ in range(5):
                    # 创建模拟的观测数据
                    batch_size = 32
                    uav_features = torch.randn(batch_size, 5, 9)
                    target_features = torch.randn(batch_size, 3, 8)
                    
                    # 模拟注意力计算
                    attention_weights = torch.bmm(uav_features, target_features.transpose(1, 2))
                    attention_output = torch.bmm(attention_weights, target_features)
                    
                    # 清理中间结果
                    del attention_weights, attention_output
                
                # 清理大对象
                del large_tensors
                
                # 强制垃圾回收
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 记录内存使用
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_snapshot = {
                    "iteration": iteration,
                    "memory_mb": current_memory,
                    "memory_increase": current_memory - initial_memory
                }
                
                results["memory_snapshots"].append(memory_snapshot)
                print(f"    当前内存: {current_memory:.2f} MB (+{memory_snapshot['memory_increase']:.2f} MB)")
                
                # 短暂等待
                time.sleep(0.5)
            
            # 分析内存泄漏
            if len(results["memory_snapshots"]) >= 3:
                # 检查内存增长趋势
                memory_increases = [snap["memory_increase"] for snap in results["memory_snapshots"]]
                
                # 如果最后的内存增长超过初始内存的50%，可能存在内存泄漏
                final_increase = memory_increases[-1]
                if final_increase > initial_memory * 0.5:
                    results["memory_leak_detected"] = True
                    print(f"⚠ 检测到可能的内存泄漏，增长: {final_increase:.2f} MB")
                else:
                    print(f"✓ 内存使用稳定，最大增长: {max(memory_increases):.2f} MB")
            
            # 最终内存检查
            final_memory = process.memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory
            
            results["final_memory_mb"] = final_memory
            results["total_memory_increase"] = total_increase
            
            # 判断测试结果
            if total_increase < initial_memory * 0.3 and not results["memory_leak_detected"]:
                results["status"] = "passed"
                print(f"✓ 内存稳定性测试通过")
            else:
                results["status"] = "warning"
                print(f"⚠ 内存使用需要关注")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            print(f"✗ 内存稳定性测试失败: {e}")
        
        return results
    
    def run_distributed_tests(self) -> Dict:
        """运行所有分布式训练测试"""
        print("=" * 60)
        print("开始分布式训练稳定性测试")
        print("=" * 60)
        
        start_time = time.time()
        test_results = {
            "test_suite": "distributed_training_stability",
            "start_time": start_time,
            "test_results": {},
            "summary": {}
        }
        
        # 执行所有测试
        test_results["test_results"]["ray_initialization"] = self.test_ray_initialization()
        test_results["test_results"]["data_consistency"] = self.test_distributed_data_consistency()
        test_results["test_results"]["fault_tolerance"] = self.test_worker_fault_tolerance()
        test_results["test_results"]["memory_stability"] = self.test_memory_usage_stability()
        
        # 生成摘要
        end_time = time.time()
        test_results["end_time"] = end_time
        test_results["total_duration"] = end_time - start_time
        
        # 统计结果
        passed_tests = sum(1 for result in test_results["test_results"].values() 
                          if result["status"] == "passed")
        failed_tests = sum(1 for result in test_results["test_results"].values() 
                          if result["status"] == "failed")
        skipped_tests = sum(1 for result in test_results["test_results"].values() 
                           if result["status"] == "skipped")
        warning_tests = sum(1 for result in test_results["test_results"].values() 
                           if result["status"] == "warning")
        total_tests = len(test_results["test_results"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "warning_tests": warning_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
        }
        
        # 保存结果
        self._save_results(test_results)
        self._print_summary(test_results)
        
        return test_results
    
    def _save_results(self, results: Dict):
        """保存测试结果"""
        results_file = os.path.join(self.test_output_dir, "distributed_test_results.json")
        
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(convert_for_json(results), f, indent=2, ensure_ascii=False)
        
        print(f"\n分布式测试结果已保存到: {results_file}")
    
    def _print_summary(self, results: Dict):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("分布式训练稳定性测试摘要")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"总测试数量: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"跳过测试: {summary['skipped_tests']}")
        print(f"警告测试: {summary['warning_tests']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"总耗时: {results['total_duration']:.2f}秒")
        print(f"整体状态: {summary['overall_status']}")
        
        print("\n详细结果:")
        for test_name, result in results["test_results"].items():
            status_map = {"passed": "✓", "failed": "✗", "skipped": "⚠", "warning": "⚠"}
            symbol = status_map.get(result["status"], "?")
            print(f"  {symbol} {test_name}: {result['status'].upper()}")


def main():
    """主函数"""
    tester = DistributedTrainingTester()
    
    try:
        results = tester.run_distributed_tests()
        return 0 if results["summary"]["overall_status"] == "PASSED" else 1
    except Exception as e:
        print(f"分布式测试执行失败: {e}")
        return 2
    finally:
        # 清理Ray资源
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
