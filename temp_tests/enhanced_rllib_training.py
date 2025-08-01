# -*- coding: utf-8 -*-
# 文件名: enhanced_rllib_training.py
# 描述: 增强的RAY训练脚本，包含详细的训练曲线记录和可视化功能

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.tune.registry import register_env

# 导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from migration_temp.rllib_files.rllib_env import UAVTaskEnvRLlib
from migration_temp.rllib_files.rllib_networks import create_network, get_network_info
from scenarios import get_strategic_trap_scenario, get_simple_convergence_test_scenario
from config import Config

class EnhancedRLlibTrainer:
    """增强的RLlib训练器，包含详细的训练曲线记录"""
    
    def __init__(self, output_dir: str = "output/enhanced_rllib"):
        self.output_dir = output_dir
        self.training_history = defaultdict(list)
        self.evaluation_history = defaultdict(list)
        self.config = Config()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/curves", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def env_creator(self, env_config: Dict[str, Any]) -> UAVTaskEnvRLlib:
        """环境创建函数"""
        scenario_name = env_config.get("scenario_name", "strategic_trap")
        
        if scenario_name == "strategic_trap":
            uavs, targets, obstacles = get_strategic_trap_scenario(50.0)
        elif scenario_name == "simple_convergence":
            uavs, targets, obstacles = get_simple_convergence_test_scenario(50.0)
        else:
            raise ValueError(f"未知场景: {scenario_name}")
        
        return UAVTaskEnvRLlib(uavs, targets, obstacles, self.config)
    
    def train_with_curves(self, 
                         algorithm: str = "DQN",
                         scenario_name: str = "strategic_trap",
                         num_episodes: int = 1000,
                         network_type: str = "DeepFCN") -> Dict[str, Any]:
        """
        训练模型并记录详细的训练曲线
        
        Args:
            algorithm: 算法名称 ("DQN", "PPO", "A3C")
            scenario_name: 场景名称
            num_episodes: 训练轮次
            network_type: 网络类型 ("DeepFCN", "GAT", "DeepFCN_Residual")
            
        Returns:
            训练结果字典
        """
        print(f"=== 开始增强RAY训练 ===")
        print(f"算法: {algorithm}")
        print(f"场景: {scenario_name}")
        print(f"网络类型: {network_type}")
        print(f"训练轮次: {num_episodes}")
        print("-" * 50)
        
        # 注册环境
        register_env("uav_task_env", self.env_creator)
        
        # 创建算法配置
        if algorithm == "DQN":
            config = DQNConfig()
            config = config.environment("uav_task_env", env_config={"scenario_name": scenario_name})
            config = config.framework("torch")
            config = config.training(
                train_batch_size=128,
                lr=0.001,
                gamma=0.98,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.999,
                target_network_update_freq=10,
                num_atoms=1,
                v_min=-10.0,
                v_max=10.0,
                double_q=True,
                dueling=True,
                hiddens=[256, 128, 64],
                post_fcnet_hiddens=[32]
            )
        elif algorithm == "PPO":
            config = PPOConfig()
            config = config.environment("uav_task_env", env_config={"scenario_name": scenario_name})
            config = config.framework("torch")
            config = config.training(
                train_batch_size=4000,
                lr=0.0003,
                gamma=0.99,
                lambda_=0.95,
                kl_coeff=0.2,
                num_sgd_iter=30,
                sgd_minibatch_size=128,
                model={"fcnet_hiddens": [256, 128, 64]}
            )
        elif algorithm == "A3C":
            config = A3CConfig()
            config = config.environment("uav_task_env", env_config={"scenario_name": scenario_name})
            config = config.framework("torch")
            config = config.training(
                lr=0.001,
                gamma=0.99,
                model={"fcnet_hiddens": [256, 128, 64]}
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 设置训练参数
        config = config.resources(num_gpus=0, num_cpus=4)
        config = config.rollouts(num_rollout_workers=2)
        config = config.evaluation(evaluation_interval=50, evaluation_duration=10)
        
        # 自定义回调函数来记录训练曲线
        def custom_callback(info):
            """自定义回调函数，记录训练曲线"""
            episode_reward_mean = info.get("episode_reward_mean", 0)
            episode_len_mean = info.get("episode_len_mean", 0)
            custom_metrics = info.get("custom_metrics", {})
            
            # 记录训练历史
            self.training_history["episode_reward_mean"].append(episode_reward_mean)
            self.training_history["episode_len_mean"].append(episode_len_mean)
            self.training_history["episode"].append(info.get("training_iteration", 0))
            
            # 记录自定义指标
            for key, value in custom_metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            return info
        
        config = config.callbacks(custom_callback)
        
        # 开始训练
        start_time = time.time()
        
        # 使用tune运行训练
        analysis = tune.run(
            algorithm,
            config=config,
            stop={"training_iteration": num_episodes // 10},  # 每10个episode记录一次
            checkpoint_freq=10,
            checkpoint_at_end=True,
            local_dir=self.output_dir,
            name=f"{algorithm}_{scenario_name}_{network_type}",
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # 获取最佳检查点
        best_checkpoint = analysis.get_best_checkpoint(metric="episode_reward_mean", mode="max")
        
        # 保存训练曲线
        self._save_training_curves(algorithm, scenario_name, network_type)
        
        # 生成训练报告
        self._generate_training_report(algorithm, scenario_name, network_type, 
                                     training_time, best_checkpoint)
        
        return {
            "algorithm": algorithm,
            "scenario_name": scenario_name,
            "network_type": network_type,
            "training_time": training_time,
            "best_checkpoint": best_checkpoint,
            "training_history": dict(self.training_history),
            "analysis": analysis
        }
    
    def _save_training_curves(self, algorithm: str, scenario_name: str, network_type: str):
        """保存训练曲线"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algorithm} - {scenario_name} - {network_type} 训练曲线', fontsize=16)
        
        # 奖励曲线
        if "episode_reward_mean" in self.training_history:
            axes[0, 0].plot(self.training_history["episode"], 
                           self.training_history["episode_reward_mean"], 
                           'b-', linewidth=2)
            axes[0, 0].set_title('平均奖励曲线')
            axes[0, 0].set_xlabel('训练轮次')
            axes[0, 0].set_ylabel('平均奖励')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 探索率曲线
        if "epsilon" in self.training_history:
            axes[0, 1].plot(self.training_history["episode"], 
                           self.training_history["epsilon"], 
                           'r-', linewidth=2)
            axes[0, 1].set_title('探索率曲线')
            axes[0, 1].set_xlabel('训练轮次')
            axes[0, 1].set_ylabel('探索率')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 损失曲线
        if "learner/default_policy/learner_stats/cur_lr" in self.training_history:
            axes[1, 0].plot(self.training_history["episode"], 
                           self.training_history["learner/default_policy/learner_stats/cur_lr"], 
                           'g-', linewidth=2)
            axes[1, 0].set_title('学习率曲线')
            axes[1, 0].set_xlabel('训练轮次')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 完成率曲线
        if "custom_metrics/completion_rate_mean" in self.training_history:
            axes[1, 1].plot(self.training_history["episode"], 
                           self.training_history["custom_metrics/completion_rate_mean"], 
                           'm-', linewidth=2)
            axes[1, 1].set_title('任务完成率曲线')
            axes[1, 1].set_xlabel('训练轮次')
            axes[1, 1].set_ylabel('完成率')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        curve_path = f"{self.output_dir}/curves/{algorithm}_{scenario_name}_{network_type}_{timestamp}.png"
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        data_path = f"{self.output_dir}/curves/{algorithm}_{scenario_name}_{network_type}_{timestamp}.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.training_history), f, indent=2, ensure_ascii=False)
        
        print(f"训练曲线已保存至: {curve_path}")
        print(f"训练数据已保存至: {data_path}")
    
    def _generate_training_report(self, algorithm: str, scenario_name: str, network_type: str,
                                training_time: float, best_checkpoint: str):
        """生成训练报告"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_path = f"{self.output_dir}/reports/{algorithm}_{scenario_name}_{network_type}_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"RAY训练报告 - {timestamp}\n")
            f.write("=" * 60 + "\n")
            f.write(f"算法: {algorithm}\n")
            f.write(f"场景: {scenario_name}\n")
            f.write(f"网络类型: {network_type}\n")
            f.write(f"训练时间: {training_time:.2f}秒\n")
            f.write(f"最佳检查点: {best_checkpoint}\n")
            f.write(f"训练轮次: {len(self.training_history.get('episode', []))}\n")
            
            if "episode_reward_mean" in self.training_history:
                rewards = self.training_history["episode_reward_mean"]
                f.write(f"最终平均奖励: {rewards[-1]:.2f}\n")
                f.write(f"最高平均奖励: {max(rewards):.2f}\n")
                f.write(f"最低平均奖励: {min(rewards):.2f}\n")
            
            if "episode_len_mean" in self.training_history:
                lengths = self.training_history["episode_len_mean"]
                f.write(f"平均回合长度: {lengths[-1]:.2f}\n")
            
            f.write("\n训练历史摘要:\n")
            f.write("-" * 30 + "\n")
            for key, values in self.training_history.items():
                if values:
                    f.write(f"{key}: {len(values)} 个数据点\n")
        
        print(f"训练报告已保存至: {report_path}")
    
    def evaluate_model(self, checkpoint_path: str, algorithm: str, 
                      scenario_name: str = "strategic_trap", 
                      num_episodes: int = 50) -> Dict[str, Any]:
        """评估模型性能"""
        print(f"\n=== 开始模型评估 ===")
        print(f"检查点: {checkpoint_path}")
        print(f"算法: {algorithm}")
        print(f"场景: {scenario_name}")
        print(f"评估轮次: {num_episodes}")
        print("-" * 50)
        
        # 恢复算法
        if algorithm == "DQN":
            algo = DQNConfig().build()
        elif algorithm == "PPO":
            algo = PPOConfig().build()
        elif algorithm == "A3C":
            algo = A3CConfig().build()
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 加载检查点
        algo.restore(checkpoint_path)
        
        # 创建环境
        env = self.env_creator({"scenario_name": scenario_name})
        
        # 运行评估
        episode_rewards = []
        episode_lengths = []
        completion_rates = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = algo.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 计算完成率
            completion_rate = env.get_completion_rate() if hasattr(env, 'get_completion_rate') else 0.0
            completion_rates.append(completion_rate)
        
        # 计算统计信息
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        avg_completion_rate = np.mean(completion_rates)
        
        evaluation_results = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "avg_completion_rate": avg_completion_rate,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "completion_rates": completion_rates
        }
        
        print(f"评估完成!")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均回合长度: {avg_length:.2f}")
        print(f"平均完成率: {avg_completion_rate:.2f}")
        
        return evaluation_results

def main():
    """主函数"""
    print("增强RAY训练系统")
    print("=" * 50)
    
    # 创建训练器
    trainer = EnhancedRLlibTrainer()
    
    # 测试不同算法和网络结构
    algorithms = ["DQN", "PPO"]
    network_types = ["DeepFCN", "GAT", "DeepFCN_Residual"]
    scenarios = ["strategic_trap", "simple_convergence"]
    
    results = {}
    
    for algorithm in algorithms:
        for network_type in network_types:
            for scenario in scenarios:
                print(f"\n{'='*60}")
                print(f"测试配置: {algorithm} + {network_type} + {scenario}")
                print(f"{'='*60}")
                
                try:
                    # 训练模型
                    result = trainer.train_with_curves(
                        algorithm=algorithm,
                        scenario_name=scenario,
                        num_episodes=500,  # 减少轮次用于快速测试
                        network_type=network_type
                    )
                    
                    # 评估模型
                    if result["best_checkpoint"]:
                        eval_result = trainer.evaluate_model(
                            result["best_checkpoint"],
                            algorithm,
                            scenario
                        )
                        result["evaluation"] = eval_result
                    
                    results[f"{algorithm}_{network_type}_{scenario}"] = result
                    
                except Exception as e:
                    print(f"训练失败: {e}")
                    continue
    
    # 生成综合报告
    print(f"\n{'='*60}")
    print("训练完成！生成综合报告...")
    print(f"{'='*60}")
    
    # 保存所有结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = f"{trainer.output_dir}/reports/comprehensive_results_{timestamp}.json"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"综合结果已保存至: {results_path}")
    print("增强RAY训练系统运行完成！")

if __name__ == "__main__":
    main() 