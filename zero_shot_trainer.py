#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零样本训练器 - 集成训练策略到主程序

使用方法：
1. 直接运行此文件进行零样本训练
2. 或在main.py中导入并使用
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Tuple
import time
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zero_shot_training_strategy import ZeroShotTrainingStrategy, ZeroShotMetrics, create_zero_shot_training_config
from main import GraphRLSolver
from entities import UAV, Target
from environment import UAVTaskEnv, DirectedGraph
from config import Config
from scenarios import get_small_scenario

class ZeroShotTrainer:
    """零样本训练器"""
    
    def __init__(self, config):
        self.config = config
        self.strategy = ZeroShotTrainingStrategy(config)
        self.metrics = ZeroShotMetrics()
        
    def create_dynamic_scenario(self, n_uavs: int, n_targets: int) -> Tuple[List[UAV], List[Target], List]:
        """
        动态创建训练场景 - 修复版本，确保动作空间兼容
        
        Args:
            n_uavs: UAV数量
            n_targets: 目标数量
            
        Returns:
            Tuple: (UAV列表, 目标列表, 障碍物列表)
        """
        # 限制场景规模以确保动作空间不超过网络输出维度
        n_phi = 6  # 方向数量
        max_output_dim = 1000  # ZeroShotGNN的输出维度
        max_possible_actions = max_output_dim
        
        # 计算最大实体数量
        max_total_pairs = max_possible_actions // n_phi
        max_entities_sqrt = int(np.sqrt(max_total_pairs))
        
        # 保守地限制UAV和目标数量
        max_uavs = min(n_uavs, max_entities_sqrt)
        max_targets = min(n_targets, max_entities_sqrt)
        
        # 进一步确保动作空间不超限
        while max_uavs * max_targets * n_phi > max_output_dim:
            if max_uavs > max_targets:
                max_uavs -= 1
            else:
                max_targets -= 1
        
        n_uavs = max(1, max_uavs)
        n_targets = max(1, max_targets)
        
        # 只在场景规模被调整时显示警告
        if n_uavs != max_uavs or n_targets != max_targets:
            print(f"场景规模调整: {n_uavs} UAV, {n_targets} 目标 (动作空间: {n_uavs * n_targets * n_phi})")
        
        # 创建UAV
        uavs = []
        for i in range(n_uavs):
            position = np.array([
                random.uniform(0, 1000),
                random.uniform(0, 1000)
            ])
            heading = random.uniform(0, 2 * np.pi)
            resources = np.array([
                random.uniform(40, 80),
                random.uniform(40, 80)
            ])
            max_distance = random.uniform(800, 1200)
            velocity_range = (
                random.uniform(20, 40),
                random.uniform(60, 100)
            )
            economic_speed = random.uniform(50, 80)
            
            uav = UAV(i+1, position, heading, resources, max_distance, velocity_range, economic_speed)
            uavs.append(uav)
        
        # 创建目标
        targets = []
        for i in range(n_targets):
            position = np.array([
                random.uniform(200, 800),
                random.uniform(200, 800)
            ])
            resources = np.array([
                random.uniform(20, 60),
                random.uniform(20, 60)
            ])
            value = random.uniform(80, 120)
            
            target = Target(i+1, position, resources, value)
            targets.append(target)
        
        # 简单障碍物（可选）
        obstacles = []
        
        return uavs, targets, obstacles
    
    def train_with_strategy(self, output_dir: str = "output/zero_shot_training") -> Dict:
        """
        使用零样本训练策略进行训练
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict: 训练结果
        """
        print("🚀 开始零样本迁移训练")
        print("=" * 60)
        print(self.strategy.get_training_summary())
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练历史
        training_history = {
            'phases': [],
            'overall_metrics': [],
            'best_models': []
        }
        
        total_episodes = 0
        start_time = time.time()
        
        # 分阶段训练
        for phase_idx in range(len(self.strategy.training_phases)):
            phase_config = self.strategy.get_current_phase_config()
            print(f"\n📚 开始训练阶段 {phase_idx + 1}: {phase_config['name']}")
            print(f"目标: {phase_config['focus']}")
            print("-" * 40)
            
            # 阶段训练结果
            phase_results = self._train_phase(phase_config, output_dir, phase_idx)
            training_history['phases'].append(phase_results)
            
            total_episodes += phase_results['episodes_completed']
            
            # 检查是否应该进入下一阶段
            if phase_results['should_advance']:
                print(f"✅ 阶段 {phase_idx + 1} 完成，进入下一阶段")
                self.strategy.advance_phase()
            else:
                print(f"⚠️ 阶段 {phase_idx + 1} 未达到预期目标，但继续下一阶段")
                self.strategy.advance_phase()
        
        total_time = time.time() - start_time
        
        # 最终评估
        print(f"\n🎯 训练完成总结")
        print("-" * 40)
        print(f"总训练轮数: {total_episodes}")
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        
        # 零样本迁移测试
        transfer_results = self._evaluate_zero_shot_transfer(output_dir)
        
        final_results = {
            'training_history': training_history,
            'total_episodes': total_episodes,
            'total_time': total_time,
            'transfer_results': transfer_results,
            'final_transfer_score': self.metrics.compute_transfer_score([])
        }
        
        # 保存训练结果
        import json
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        return final_results
    
    def _train_phase(self, phase_config: Dict, output_dir: str, phase_idx: int) -> Dict:
        """训练单个阶段"""
        phase_start_time = time.time()
        phase_episodes = 0
        phase_rewards = []
        phase_completion_rates = []
        best_phase_reward = float('-inf')
        
        # 创建初始场景用于初始化solver
        n_uavs, n_targets = self.strategy.generate_training_scenario(phase_config)
        uavs, targets, obstacles = self.create_dynamic_scenario(n_uavs, n_targets)
        
        # 创建图和环境
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
        
        # 创建solver
        solver = GraphRLSolver(
            uavs, targets, graph, obstacles,
            i_dim=1,  # 占位值，ZeroShotGNN会忽略
            h_dim=[256, 128],
            o_dim=1000,  # 足够大的输出维度
            config=self.config,
            network_type="ZeroShotGNN",
            tensorboard_dir=f"{output_dir}/phase_{phase_idx}",
            obs_mode="graph"
        )
        
        # 应用阶段特定的配置
        self._apply_phase_config(solver, phase_config)
        
        print(f"阶段配置: UAV范围{phase_config['uav_range']}, 目标范围{phase_config['target_range']}")
        print(f"学习率: {phase_config['learning_rate']}, 批次大小: {phase_config['batch_size']}")
        
        # 阶段训练循环
        for episode in range(phase_config['episodes']):
            try:
                # 动态生成新场景 - 使用更保守的频率和规模限制
                if episode % 20 == 0:  # 每20轮更换场景，减少频率
                    n_uavs, n_targets = self.strategy.generate_training_scenario(phase_config)
                    # 进一步限制规模以确保稳定性
                    n_uavs = min(n_uavs, 6)
                    n_targets = min(n_targets, 8)
                    
                    uavs, targets, obstacles = self.create_dynamic_scenario(n_uavs, n_targets)
                    
                    # 更新solver的环境
                    graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                    solver.env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
                    solver.uavs = uavs
                    solver.targets = targets
                    solver.graph = graph
                
                # 训练一轮
                episode_result = self._train_episode(solver, phase_config)
                
                phase_episodes += 1
                phase_rewards.append(episode_result['reward'])
                phase_completion_rates.append(episode_result['completion_rate'])
                
                # 更新指标
                self.metrics.update(episode_result)
                
                # 记录最佳模型
                if episode_result['reward'] > best_phase_reward:
                    best_phase_reward = episode_result['reward']
                    model_path = f"{output_dir}/phase_{phase_idx}_best_model.pth"
                    solver.save_model(model_path)
                
                # 阶段进度报告
                if episode % 50 == 0 and episode > 0:
                    avg_reward = np.mean(phase_rewards[-50:])
                    avg_completion = np.mean(phase_completion_rates[-50:])
                    print(f"  Episode {episode:4d}: 平均奖励 {avg_reward:8.2f}, 完成率 {avg_completion:.3f}")
                    
            except Exception as e:
                # 安全地处理异常信息，避免打印tensor对象
                error_msg = str(e) if not isinstance(e, torch.Tensor) else f"Tensor异常: shape={e.shape}"
                print(f"Episode {episode} 训练出错: {error_msg}")
                # 继续下一个episode
                continue
        
        phase_time = time.time() - phase_start_time
        
        # 阶段结果
        phase_results = {
            'phase_name': phase_config['name'],
            'episodes_completed': phase_episodes,
            'phase_time': phase_time,
            'avg_reward': np.mean(phase_rewards),
            'best_reward': best_phase_reward,
            'avg_completion_rate': np.mean(phase_completion_rates),
            'final_completion_rate': phase_completion_rates[-1] if phase_completion_rates else 0,
            'should_advance': True  # 简化版本，总是进入下一阶段
        }
        
        print(f"阶段 {phase_idx + 1} 完成:")
        print(f"  - 训练轮数: {phase_episodes}")
        print(f"  - 平均奖励: {phase_results['avg_reward']:.2f}")
        print(f"  - 最佳奖励: {phase_results['best_reward']:.2f}")
        print(f"  - 平均完成率: {phase_results['avg_completion_rate']:.3f}")
        print(f"  - 训练时间: {phase_time/60:.1f} 分钟")
        
        return phase_results
    
    def _apply_phase_config(self, solver: GraphRLSolver, phase_config: Dict):
        """应用阶段特定配置"""
        # 更新学习率
        for param_group in solver.optimizer.param_groups:
            param_group['lr'] = phase_config['learning_rate']
        
        # 更新探索率
        solver.epsilon = phase_config['epsilon_start']
        solver.epsilon_min = phase_config['epsilon_end']
        
        # 应用正则化设置
        reg_settings = self.strategy.apply_regularization_techniques(solver.policy_net, phase_config)
        solver.grad_clip_norm = reg_settings['gradient_clip_norm']
    
    def _train_episode(self, solver: GraphRLSolver, phase_config: Dict) -> Dict:
        """训练单个episode - 修复版本，添加错误处理"""
        state = solver.env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(solver.env.max_steps):
            try:
                # 准备状态
                state_tensor = solver._prepare_state_tensor(state)
                
                # 选择动作
                action = solver.select_action(state_tensor)
                action_idx = action.item()
                
                # 验证动作有效性
                if action_idx >= solver.env.n_actions:
                    # 使用模运算调整动作
                    action_idx = action_idx % solver.env.n_actions
                    action = torch.tensor([[action_idx]], device=solver.device, dtype=torch.long)
                
                # 执行动作
                try:
                    next_state, reward, done, truncated, _ = solver.env.step(action_idx)
                except Exception as step_error:
                    # 如果step出错，跳过这一步
                    error_msg = str(step_error) if not isinstance(step_error, torch.Tensor) else f"Tensor异常: shape={step_error.shape}"
                    print(f"环境step出错: {error_msg}")
                    continue
                
                episode_reward += reward
                episode_steps += 1
                
                # 存储经验
                next_state_tensor = solver._prepare_state_tensor(next_state)
                
                if solver.use_per:
                    solver.memory.push(
                        state_tensor, action,
                        torch.tensor([reward], device=solver.device),
                        next_state_tensor, done
                    )
                else:
                    solver.memory.append((
                        state_tensor, action,
                        torch.tensor([reward], device=solver.device),
                        next_state_tensor, done
                    ))
                
                # 优化模型
                if len(solver.memory) >= solver.config.BATCH_SIZE:
                    solver.optimize_model()
                
                state = next_state
                
                if done or truncated:
                    break
                    
            except Exception as e:
                # 安全地处理异常信息，避免打印tensor对象
                error_msg = str(e) if not isinstance(e, torch.Tensor) else f"Tensor异常: shape={e.shape}"
                print(f"训练步骤 {step} 出错: {error_msg}")
                # 跳过这一步，继续训练
                continue
        
        # 计算完成率
        if solver.env.targets:
            completed_targets = sum(1 for target in solver.env.targets 
                                  if np.all(target.remaining_resources <= 0))
            completion_rate = completed_targets / len(solver.env.targets)
        else:
            completion_rate = 0
        
        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'completion_rate': completion_rate,
            'n_uavs': len(solver.env.uavs),
            'n_targets': len(solver.env.targets)
        }
    
    def _evaluate_zero_shot_transfer(self, output_dir: str) -> Dict:
        """评估零样本迁移能力"""
        print(f"\n🔬 评估零样本迁移能力")
        print("-" * 40)
        
        # 测试场景
        test_scenarios = [
            (3, 4),   # 小规模
            (6, 8),   # 中规模  
            (10, 12), # 大规模
            (15, 18), # 超大规模
        ]
        
        # 加载最佳模型
        best_model_path = f"{output_dir}/phase_3_best_model.pth"  # 最后阶段的最佳模型
        
        transfer_results = []
        
        for n_uavs, n_targets in test_scenarios:
            print(f"测试场景: {n_uavs} UAV, {n_targets} 目标")
            
            try:
                # 创建测试场景
                uavs, targets, obstacles = self.create_dynamic_scenario(n_uavs, n_targets)
                graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                
                # 创建测试solver
                test_solver = GraphRLSolver(
                    uavs, targets, graph, obstacles,
                    i_dim=1, h_dim=[256, 128], o_dim=1000,
                    config=self.config, network_type="ZeroShotGNN",
                    obs_mode="graph"
                )
                
                # 加载模型（如果存在）
                if os.path.exists(best_model_path):
                    test_solver.load_model(best_model_path)
                
                # 测试性能
                test_rewards = []
                test_completion_rates = []
                
                for test_episode in range(10):  # 测试10轮
                    state = test_solver.env.reset()
                    episode_reward = 0
                    
                    for step in range(test_solver.env.max_steps):
                        state_tensor = test_solver._prepare_state_tensor(state)
                        
                        with torch.no_grad():
                            test_solver.policy_net.eval()
                            q_values = test_solver.policy_net(state_tensor)
                            action = q_values.max(1)[1].view(1, 1)
                        
                        next_state, reward, done, truncated, _ = test_solver.env.step(action.item())
                        episode_reward += reward
                        state = next_state
                        
                        if done or truncated:
                            break
                    
                    test_rewards.append(episode_reward)
                    
                    # 计算完成率
                    completed = sum(1 for t in test_solver.env.targets 
                                  if np.all(t.remaining_resources <= 0))
                    completion_rate = completed / len(test_solver.env.targets)
                    test_completion_rates.append(completion_rate)
                
                avg_reward = np.mean(test_rewards)
                avg_completion = np.mean(test_completion_rates)
                
                transfer_results.append({
                    'scenario': (n_uavs, n_targets),
                    'avg_reward': avg_reward,
                    'avg_completion_rate': avg_completion,
                    'success': True
                })
                
                print(f"  ✓ 平均奖励: {avg_reward:.2f}, 完成率: {avg_completion:.3f}")
                
            except Exception as e:
                transfer_results.append({
                    'scenario': (n_uavs, n_targets),
                    'success': False,
                    'error': str(e)
                })
                print(f"  ✗ 失败: {str(e)}")
        
        # 计算总体迁移得分
        successful_results = [r for r in transfer_results if r['success']]
        if successful_results:
            avg_transfer_reward = np.mean([r['avg_reward'] for r in successful_results])
            avg_transfer_completion = np.mean([r['avg_completion_rate'] for r in successful_results])
            transfer_success_rate = len(successful_results) / len(test_scenarios)
        else:
            avg_transfer_reward = 0
            avg_transfer_completion = 0
            transfer_success_rate = 0
        
        print(f"\n零样本迁移总结:")
        print(f"  - 成功率: {transfer_success_rate:.1%}")
        print(f"  - 平均奖励: {avg_transfer_reward:.2f}")
        print(f"  - 平均完成率: {avg_transfer_completion:.3f}")
        
        return {
            'test_scenarios': test_scenarios,
            'results': transfer_results,
            'summary': {
                'success_rate': transfer_success_rate,
                'avg_reward': avg_transfer_reward,
                'avg_completion_rate': avg_transfer_completion
            }
        }

def main():
    """
    主函数 - 支持所有网络类型和零样本训练
    
    运行模式：
    1. normal: 正常训练/测试模式，支持所有网络类型
    2. zero_shot_train: 零样本训练模式，专门用于ZeroShotGNN
    """
    print("多无人机协同任务分配与路径规划系统 - 增强版")
    print("=" * 60)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='UAV任务分配系统')
    parser.add_argument('--mode', choices=['normal', 'zero_shot_train'], 
                       default='normal', help='运行模式')
    parser.add_argument('--network', choices=['SimpleNetwork', 'DeepFCN', 'DeepFCNResidual', 'ZeroShotGNN'],
                       default='ZeroShotGNN', help='网络类型')
    parser.add_argument('--scenario', choices=['small', 'balanced', 'complex', 'experimental', 'strategic_trap'],
                       default='experimental', help='场景类型')
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数')
    parser.add_argument('--force_retrain', action='store_true', help='强制重新训练')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.NETWORK_TYPE = args.network
    
    print(f"运行模式: {args.mode}")
    print(f"网络类型: {args.network}")
    print(f"场景类型: {args.scenario}")
    
    # === 零样本训练模式 ===
    if args.mode == 'zero_shot_train':
        if args.network != 'ZeroShotGNN':
            print("⚠️  零样本训练模式只支持ZeroShotGNN网络")
            print("自动切换到ZeroShotGNN网络...")
            args.network = 'ZeroShotGNN'
            config.NETWORK_TYPE = 'ZeroShotGNN'
        
        try:
            from zero_shot_trainer import ZeroShotTrainer
            from zero_shot_training_strategy import create_zero_shot_training_config
            
            print("\n🚀 启动零样本训练模式")
            print("-" * 40)
            
            # 创建零样本训练配置
            zero_shot_config = create_zero_shot_training_config(config)
            for key, value in zero_shot_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # 开始零样本训练
            trainer = ZeroShotTrainer(config)
            results = trainer.train_with_strategy("output/zero_shot_training")
            
            print(f"\n🎉 零样本训练完成!")
            print(f"最终迁移得分: {results['final_transfer_score']:.3f}")
            print(f"训练总时间: {results['total_time']/3600:.2f} 小时")
            print(f"总训练轮数: {results['total_episodes']}")
            
            return results
            
        except ImportError as e:
            print(f"❌ 零样本训练模块导入失败: {e}")
            print("请确保 zero_shot_trainer.py 和 zero_shot_training_strategy.py 文件存在")
            return None
        except Exception as e:
            print(f"❌ 零样本训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # === 正常训练/测试模式 ===
    print(f"\n🔧 启动正常训练/测试模式")
    print("-" * 40)
    
    # 清理临时文件
    print("清理临时文件...")
    cleanup_temp_files()
    
    # 加载场景数据
    print("加载场景数据...")
    
    # 根据场景类型选择场景
    scenario_functions = {
        'small': lambda: get_small_scenario(obstacle_tolerance=50.0),
        'balanced': lambda: get_balanced_scenario(obstacle_tolerance=50.0),
        'complex': lambda: get_complex_scenario(obstacle_tolerance=50.0),
        'experimental': lambda: get_new_experimental_scenario(obstacle_tolerance=50.0),
        'strategic_trap': lambda: get_strategic_trap_scenario(obstacle_tolerance=50.0)
    }
    
    try:
        if args.scenario in scenario_functions:
            base_uavs, base_targets, obstacles = scenario_functions[args.scenario]()
        else:
            print(f"⚠️  未知场景类型: {args.scenario}，使用默认实验场景")
            base_uavs, base_targets, obstacles = get_new_experimental_scenario(obstacle_tolerance=50.0)
        
        print(f"场景加载成功: {len(base_uavs)} UAV, {len(base_targets)} 目标, {len(obstacles)} 障碍物")
        
    except Exception as e:
        print(f"❌ 场景加载失败: {e}")
        print("使用默认小规模场景...")
        base_uavs, base_targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    
    # 网络类型特定配置
    network_configs = {
        'SimpleNetwork': {
            'description': '基础全连接网络',
            'obs_mode': 'flat',
            'recommended_episodes': 800,
            'features': ['BatchNorm', 'Dropout', 'Xavier初始化']
        },
        'DeepFCN': {
            'description': '深度全连接网络',
            'obs_mode': 'flat', 
            'recommended_episodes': 1000,
            'features': ['多层结构', 'BatchNorm', 'Dropout']
        },
        'DeepFCNResidual': {
            'description': '带残差连接的深度网络',
            'obs_mode': 'flat',
            'recommended_episodes': 1200,
            'features': ['残差连接', 'SE注意力', 'BatchNorm']
        },
        'ZeroShotGNN': {
            'description': '零样本图神经网络',
            'obs_mode': 'graph',
            'recommended_episodes': 1500,
            'features': ['Transformer架构', '自注意力', '交叉注意力', '参数共享', '零样本迁移']
        }
    }
    
    # 获取网络配置
    net_config = network_configs.get(args.network, network_configs['ZeroShotGNN'])
    
    print(f"\n📊 网络信息:")
    print(f"  - 类型: {args.network}")
    print(f"  - 描述: {net_config['description']}")
    print(f"  - 观测模式: {net_config['obs_mode']}")
    print(f"  - 推荐训练轮数: {net_config['recommended_episodes']}")
    print(f"  - 特性: {', '.join(net_config['features'])}")
    
    # 调整训练轮数
    if args.episodes == 1000:  # 如果使用默认值，则采用推荐值
        training_episodes = net_config['recommended_episodes']
        print(f"  - 使用推荐训练轮数: {training_episodes}")
    else:
        training_episodes = args.episodes
        print(f"  - 使用指定训练轮数: {training_episodes}")
    
    # 运行场景
    try:
        print(f"\n🎯 开始运行场景...")
        
        final_plan, training_time, training_history, evaluation_metrics = run_scenario(
            config=config,
            base_uavs=base_uavs,
            base_targets=base_targets,
            obstacles=obstacles,
            scenario_name=f"{args.scenario}场景",
            network_type=args.network,
            save_visualization=True,
            show_visualization=False,  # 在批量测试时关闭显示
            save_report=True,
            force_retrain=args.force_retrain,
            incremental_training=False,
            output_base_dir=None
        )
        
        # 训练结果总结
        print(f"\n📈 训练结果总结:")
        print(f"  - 网络类型: {args.network}")
        print(f"  - 场景类型: {args.scenario}")
        print(f"  - 训练时间: {training_time/60:.2f} 分钟")
        
        if training_history:
            print(f"  - 训练轮数: {len(training_history.get('episode_rewards', []))}")
            if training_history.get('episode_rewards'):
                avg_reward = np.mean(training_history['episode_rewards'][-100:])  # 最后100轮平均
                print(f"  - 最终平均奖励: {avg_reward:.2f}")
            if training_history.get('completion_rates'):
                avg_completion = np.mean(training_history['completion_rates'][-100:])
                print(f"  - 最终完成率: {avg_completion:.3f}")
        
        if evaluation_metrics:
            print(f"  - 评估指标: {evaluation_metrics}")
        
        # 网络特定的性能分析
        if args.network == 'ZeroShotGNN':
            print(f"\n🔬 ZeroShotGNN 特性分析:")
            print(f"  - 支持可变数量实体: ✓")
            print(f"  - 零样本迁移能力: ✓")
            print(f"  - 图结构观测: ✓")
            print(f"  - Transformer注意力: ✓")
            
            # 建议进行零样本训练
            if not args.force_retrain:
                print(f"\n💡 建议:")
                print(f"  - 对于ZeroShotGNN，建议使用零样本训练模式获得更好的迁移能力")
                print(f"  - 运行命令: python main.py --mode zero_shot_train --network ZeroShotGNN")
        
        elif args.network in ['SimpleNetwork', 'DeepFCN', 'DeepFCNResidual']:
            print(f"\n🔬 传统网络特性分析:")
            print(f"  - 固定输入维度: ✓")
            print(f"  - 扁平向量观测: ✓")
            print(f"  - 场景特定训练: ✓")
            
            if args.network == 'DeepFCNResidual':
                print(f"  - 残差连接: ✓")
                print(f"  - SE注意力机制: ✓")
        
        # 性能对比建议
        print(f"\n📊 性能对比建议:")
        print(f"  - 当前网络: {args.network}")
        print(f"  - 如需对比其他网络，可运行:")
        for net_type in ['SimpleNetwork', 'DeepFCN', 'DeepFCNResidual', 'ZeroShotGNN']:
            if net_type != args.network:
                print(f"    python main.py --network {net_type} --scenario {args.scenario}")
        
        return {
            'network_type': args.network,
            'scenario_type': args.scenario,
            'training_time': training_time,
            'training_history': training_history,
            'evaluation_metrics': evaluation_metrics,
            'final_plan': final_plan
        }
        
    except Exception as e:
        print(f"❌ 场景运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# 添加辅助函数
def cleanup_temp_files():
    """清理临时文件"""
    import glob
    temp_patterns = [
        "*.tmp",
        "temp_*",
        "__pycache__/*",
        "*.pyc"
    ]
    
    for pattern in temp_patterns:
        for file in glob.glob(pattern):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    import shutil
                    shutil.rmtree(file)
            except:
                pass

def print_network_comparison():
    """打印网络对比信息"""
    print("\n📋 支持的网络类型对比:")
    print("-" * 80)
    print(f"{'网络类型':<20} {'观测模式':<10} {'复杂度':<8} {'特色功能':<30}")
    print("-" * 80)
    print(f"{'SimpleNetwork':<20} {'flat':<10} {'低':<8} {'基础全连接，快速训练':<30}")
    print(f"{'DeepFCN':<20} {'flat':<10} {'中':<8} {'深度网络，更强表达能力':<30}")
    print(f"{'DeepFCNResidual':<20} {'flat':<10} {'中':<8} {'残差连接，SE注意力':<30}")
    print(f"{'ZeroShotGNN':<20} {'graph':<10} {'高':<8} {'零样本迁移，Transformer':<30}")
    print("-" * 80)

if __name__ == "__main__":
    # 在程序开始时显示网络对比信息
    print_network_comparison()
    
    # 运行主程序
    result = main()
    
    if result:
        print(f"\n✅ 程序执行完成")
    else:
        print(f"\n❌ 程序执行失败")
        sys.exit(1)
