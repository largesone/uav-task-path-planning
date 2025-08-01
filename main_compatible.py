# -*- coding: utf-8 -*-
# 文件名: main_compatible.py
# 描述: 集成向后兼容性管理器的多无人机协同任务分配与路径规划系统
#      支持传统FCN网络和新的TransformerGNN网络的无缝切换

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from collections import deque, defaultdict
import os
import time
import pickle
import random
import json
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
import glob

# TensorBoard支持 - 可选依赖 
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("警告: TensorBoard未安装，将跳过TensorBoard功能")
    print("安装命令: pip install tensorboard")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# --- 本地模块导入 ---
from entities import UAV, Target
from path_planning import PHCurveRRTPlanner
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario, get_new_experimental_scenario, get_complex_scenario_v4, get_strategic_trap_scenario
from config import Config
from evaluate import evaluate_plan
from networks import create_network, get_network_info
from environment import UAVTaskEnv, DirectedGraph

# === 新增：向后兼容性管理器导入 ===
from compatibility_manager import (
    CompatibilityManager, 
    CompatibilityConfig,
    get_compatibility_manager,
    set_compatibility_manager
)

# 允许多个OpenMP库共存，解决某些环境下的冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        pass

# 初始化配置类
config = Config()

# === 新增：全局兼容性配置 ===
# 用户可以通过修改这些参数来选择使用传统方法或TransformerGNN方法
COMPATIBILITY_CONFIG = CompatibilityConfig(
    network_mode="traditional",  # 可选: "traditional" 或 "transformer_gnn"
    traditional_network_type="DeepFCNResidual",  # 传统网络类型
    obs_mode="flat",  # 可选: "flat" 或 "graph"
    enable_compatibility_checks=True,  # 是否启用兼容性检查
    debug_mode=False  # 调试模式
)

# 初始化全局兼容性管理器
set_compatibility_manager(CompatibilityManager(COMPATIBILITY_CONFIG))

# JSON序列化辅助函数（保持不变）
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

# 字体设置函数（保持不变）
def set_chinese_font(preferred_fonts=None, manual_font_path=None):
    """设置matplotlib支持中文显示的字体"""
    if manual_font_path and os.path.exists(manual_font_path):
        try:
            font_prop = FontProperties(fname=manual_font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功加载手动指定的字体: {manual_font_path}")
            return True
        except Exception as e:
            print(f"加载手动指定字体失败: {e}")
    
    if preferred_fonts is None:
        preferred_fonts = ['Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'KaiTi', 'FangSong']
    
    try:
        for font in preferred_fonts:
            if findfont(FontProperties(family=font)):
                plt.rcParams["font.family"] = font
                plt.rcParams['axes.unicode_minus'] = False
                return True
    except Exception:
        pass
    
    return False

# === 修改后的GraphRLSolver类 ===
class CompatibleGraphRLSolver:
    """兼容的图强化学习求解器 - 支持传统网络和TransformerGNN的无缝切换"""
    
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, 
                 network_type=None, tensorboard_dir=None, compatibility_manager=None):
        """
        初始化兼容求解器
        
        Args:
            uavs: UAV列表
            targets: 目标列表
            graph: 图对象
            obstacles: 障碍物列表
            i_dim: 输入维度
            h_dim: 隐藏层维度
            o_dim: 输出维度
            config: 配置对象
            network_type: 网络类型（可选，用于覆盖兼容性管理器配置）
            tensorboard_dir: TensorBoard日志目录
            compatibility_manager: 兼容性管理器（可选）
        """
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取兼容性管理器
        self.compatibility_manager = compatibility_manager or get_compatibility_manager()
        
        # 确定网络类型和模式
        if network_type:
            # 如果指定了网络类型，临时覆盖配置
            if network_type == "TransformerGNN":
                self.network_mode = "transformer_gnn"
                self.obs_mode = "graph"  # TransformerGNN建议使用graph模式
            else:
                self.network_mode = "traditional"
                self.obs_mode = "flat"  # 传统网络使用flat模式
        else:
            # 使用兼容性管理器的配置
            self.network_mode = self.compatibility_manager.config.network_mode
            self.obs_mode = self.compatibility_manager.config.obs_mode
        
        print(f"[CompatibleGraphRLSolver] 初始化")
        print(f"  - 网络模式: {self.network_mode}")
        print(f"  - 观测模式: {self.obs_mode}")
        print(f"  - 设备: {self.device}")
        
        # 创建环境
        self.env = self.compatibility_manager.create_environment(
            uavs, targets, graph, obstacles, config
        )
        
        # 创建网络
        if self.network_mode == "traditional":
            self._init_traditional_network(i_dim, h_dim, o_dim)
        else:
            self._init_transformer_gnn_network(o_dim)
        
        # TensorBoard支持
        self.tensorboard_dir = tensorboard_dir
        self.writer = None
        if tensorboard_dir and TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(tensorboard_dir)
                print(f"TensorBoard日志将保存至: {tensorboard_dir}")
            except Exception as e:
                print(f"TensorBoard初始化失败: {e}")
        
        # 初始化训练参数
        self._init_training_params()
        
        # 动作映射
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}
        self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
        
        print(f"[CompatibleGraphRLSolver] 初始化完成")
    
    def _init_traditional_network(self, i_dim, h_dim, o_dim):
        """初始化传统网络"""
        network_type = self.compatibility_manager.config.traditional_network_type
        
        self.policy_net = self.compatibility_manager.create_network(
            input_dim=i_dim,
            hidden_dims=h_dim if isinstance(h_dim, list) else [h_dim],
            output_dim=o_dim
        ).to(self.device)
        
        self.target_net = self.compatibility_manager.create_network(
            input_dim=i_dim,
            hidden_dims=h_dim if isinstance(h_dim, list) else [h_dim],
            output_dim=o_dim
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        print(f"  - 传统网络类型: {network_type}")
        print(f"  - 网络参数: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def _init_transformer_gnn_network(self, o_dim):
        """初始化TransformerGNN网络"""
        self.policy_net = self.compatibility_manager.create_network(
            input_dim=None,
            hidden_dims=None,
            output_dim=o_dim,
            obs_space=self.env.observation_space,
            action_space=self.env.action_space
        ).to(self.device)
        
        # TransformerGNN通常不需要单独的目标网络，因为它本身就包含了价值函数
        self.target_net = None
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        print(f"  - TransformerGNN网络")
        print(f"  - 网络参数: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def _init_training_params(self):
        """初始化训练参数"""
        # 使用优先经验回放缓冲区
        self.use_per = config.training_config.use_prioritized_replay
        if self.use_per:
            from main import PrioritizedReplayBuffer
            self.memory = PrioritizedReplayBuffer(
                capacity=config.MEMORY_CAPACITY,
                alpha=config.training_config.per_alpha,
                beta_start=config.training_config.per_beta_start,
                beta_frames=config.training_config.per_beta_frames
            )
        else:
            self.memory = deque(maxlen=config.MEMORY_CAPACITY)
        
        self.epsilon = config.training_config.epsilon_start
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.15
        
        # 训练统计
        self.step_count = 0
        self.update_count = 0
    
    def select_action(self, state):
        """选择动作 - 兼容不同网络类型"""
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.env.n_actions)]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            if self.network_mode == "traditional":
                # 传统网络：直接前向传播
                self.policy_net.eval()
                q_values = self.policy_net(state)
                self.policy_net.train()
                return q_values.max(1)[1].view(1, 1)
            else:
                # TransformerGNN：使用RLlib接口
                self.policy_net.eval()
                logits, _ = self.policy_net.forward({"obs": state}, [], [])
                self.policy_net.train()
                return logits.max(1)[1].view(1, 1)
    
    def optimize_model(self):
        """优化模型 - 兼容不同网络类型"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        
        # 采样经验
        if self.use_per:
            transitions, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = random.sample(self.memory, self.config.BATCH_SIZE)
            indices = None
            weights = torch.ones(self.config.BATCH_SIZE).to(self.device)
        
        # 解包批次数据
        batch = list(zip(*transitions))
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_states_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)
        
        # 计算当前Q值
        if self.network_mode == "traditional":
            current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        else:
            # TransformerGNN
            logits, _ = self.policy_net.forward({"obs": state_batch}, [], [])
            current_q_values = logits.gather(1, action_batch)
        
        # 计算目标Q值
        next_q_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        
        with torch.no_grad():
            if self.network_mode == "traditional" and self.target_net is not None:
                # 传统网络使用目标网络
                next_actions = self.policy_net(next_states_batch).max(1)[1].unsqueeze(1)
                next_q_values[~done_batch] = self.target_net(next_states_batch[~done_batch]).gather(1, next_actions[~done_batch]).squeeze(1)
            else:
                # TransformerGNN或无目标网络的情况
                if self.network_mode == "traditional":
                    next_q_values[~done_batch] = self.policy_net(next_states_batch[~done_batch]).max(1)[0]
                else:
                    logits, _ = self.policy_net.forward({"obs": next_states_batch[~done_batch]}, [], [])
                    next_q_values[~done_batch] = logits.max(1)[0]
        
        expected_q_values = reward_batch + (self.config.GAMMA * next_q_values)
        
        # 计算TD误差
        td_errors = (current_q_values.squeeze() - expected_q_values).detach()
        
        # 计算损失
        if self.use_per:
            elementwise_loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none')
            loss = (elementwise_loss.squeeze() * weights).mean()
        else:
            loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        self.update_count += 1
        
        # 更新PER优先级
        if self.use_per and indices is not None:
            priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Training/Loss', loss.item(), self.update_count)
            self.writer.add_scalar('Training/Mean_Q_Value', current_q_values.mean().item(), self.update_count)
        
        # 更新目标网络
        if self.target_net is not None and self.update_count % self.config.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train(self, episodes, patience, log_interval, model_save_path):
        """训练模型 - 兼容不同网络类型"""
        start_time = time.time()
        best_reward = float('-inf')
        patience_counter = 0
        
        # 初始化训练历史
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_values = []
        self.completion_rates = []
        
        print(f"开始训练 - 网络模式: {self.network_mode}, 观测模式: {self.obs_mode}")
        
        for i_episode in tqdm(range(episodes), desc=f"训练进度"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            
            # 将状态转换为张量
            if isinstance(state, dict):
                # 图模式状态
                state_tensor = self._convert_dict_state_to_tensor(state)
            else:
                # 扁平模式状态
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            for step in range(self.env.max_steps):
                action = self.select_action(state_tensor)
                
                next_state, reward, done, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                self.step_count += 1
                
                # 转换下一状态
                if isinstance(next_state, dict):
                    next_state_tensor = self._convert_dict_state_to_tensor(next_state)
                else:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # 存储经验
                if self.use_per:
                    self.memory.push(
                        state_tensor,
                        action,
                        torch.tensor([reward], device=self.device),
                        next_state_tensor,
                        done
                    )
                else:
                    self.memory.append((
                        state_tensor,
                        action,
                        torch.tensor([reward], device=self.device),
                        next_state_tensor,
                        done
                    ))
                
                # 优化模型
                if len(self.memory) >= self.config.BATCH_SIZE:
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                
                state_tensor = next_state_tensor
                
                if done or truncated:
                    break
            
            # 更新探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 记录训练历史
            avg_episode_loss = episode_loss / max(loss_count, 1)
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_episode_loss)
            self.epsilon_values.append(self.epsilon)
            
            # 计算完成率
            completion_rate = self._calculate_completion_rate()
            self.completion_rates.append(completion_rate)
            
            # TensorBoard记录
            if self.writer:
                self.writer.add_scalar('Episode/Reward', episode_reward, i_episode)
                self.writer.add_scalar('Episode/Loss', avg_episode_loss, i_episode)
                self.writer.add_scalar('Episode/Epsilon', self.epsilon, i_episode)
                self.writer.add_scalar('Episode/Completion_Rate', completion_rate, i_episode)
            
            # 日志输出
            if i_episode % (log_interval * 5) == 0 and i_episode > 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_completion = np.mean(self.completion_rates[-log_interval:])
                print(f"Episode {i_episode:4d}: 平均奖励 {avg_reward:8.2f}, 完成率 {avg_completion:6.3f}")
            
            # 早停检查
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                # 保存最佳模型
                self.save_model(model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"早停触发于第 {i_episode} 回合")
                break
        
        training_time = time.time() - start_time
        
        if self.writer:
            self.writer.close()
        
        print(f"训练完成 - 耗时: {training_time:.2f}秒, 最佳奖励: {best_reward:.2f}")
        return training_time
    
    def _convert_dict_state_to_tensor(self, state_dict):
        """将字典状态转换为张量"""
        tensor_dict = {}
        for key, value in state_dict.items():
            if key == "masks":
                tensor_dict[key] = {
                    sub_key: torch.tensor(sub_value, dtype=torch.float32, device=self.device)
                    for sub_key, sub_value in value.items()
                }
            else:
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        return tensor_dict
    
    def _calculate_completion_rate(self):
        """计算完成率"""
        if not self.env.targets:
            return 0.0
        
        completed_targets = sum(1 for target in self.env.targets if np.all(target.remaining_resources <= 0))
        return completed_targets / len(self.env.targets)
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict() if self.target_net else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_mode': self.network_mode,
            'obs_mode': self.obs_mode,
            'config': self.compatibility_manager.get_network_info()
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        if self.target_net and checkpoint['target_net_state_dict']:
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# === 兼容的run_scenario函数 ===
def run_scenario_compatible(scenario_func, scenario_name, config_override=None):
    """
    兼容的场景运行函数 - 支持传统方法和TransformerGNN方法的无缝切换
    
    Args:
        scenario_func: 场景生成函数
        scenario_name: 场景名称
        config_override: 配置覆盖（可选）
    
    Returns:
        运行结果字典
    """
    print(f"\n{'='*60}")
    print(f"运行场景: {scenario_name}")
    print(f"{'='*60}")
    
    # 获取兼容性管理器
    compatibility_manager = get_compatibility_manager()
    
    # 应用配置覆盖
    if config_override:
        if isinstance(config_override, dict):
            # 从字典创建新配置
            new_config = CompatibilityConfig(**config_override)
            compatibility_manager = CompatibilityManager(new_config)
        elif isinstance(config_override, CompatibilityConfig):
            compatibility_manager = CompatibilityManager(config_override)
    
    print(f"兼容性配置:")
    print(f"  - 网络模式: {compatibility_manager.config.network_mode}")
    print(f"  - 观测模式: {compatibility_manager.config.obs_mode}")
    if compatibility_manager.config.network_mode == "traditional":
        print(f"  - 传统网络类型: {compatibility_manager.config.traditional_network_type}")
    
    # 运行兼容性检查
    if compatibility_manager.config.enable_compatibility_checks:
        print("\n运行兼容性检查...")
        check_results = compatibility_manager.run_compatibility_checks()
        if not check_results.get("overall_compatibility", False):
            print("⚠️  兼容性检查未完全通过，但继续运行...")
    
    try:
        # 生成场景
        uavs, targets, obstacles = scenario_func()
        
        # 创建图
        graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
        
        # 计算状态和动作维度
        env_temp = compatibility_manager.create_environment(uavs, targets, graph, obstacles, config)
        
        if compatibility_manager.config.obs_mode == "flat":
            i_dim = env_temp.observation_space.shape[0]
        else:
            i_dim = 128  # TransformerGNN不直接使用这个维度
        
        h_dim = [256, 128]
        o_dim = env_temp.n_actions
        
        print(f"\n场景信息:")
        print(f"  - UAV数量: {len(uavs)}")
        print(f"  - 目标数量: {len(targets)}")
        print(f"  - 障碍物数量: {len(obstacles)}")
        print(f"  - 输入维度: {i_dim}")
        print(f"  - 输出维度: {o_dim}")
        
        # 创建求解器
        tensorboard_dir = f"runs/{scenario_name}_{compatibility_manager.config.network_mode}_{int(time.time())}"
        
        solver = CompatibleGraphRLSolver(
            uavs=uavs,
            targets=targets,
            graph=graph,
            obstacles=obstacles,
            i_dim=i_dim,
            h_dim=h_dim,
            o_dim=o_dim,
            config=config,
            tensorboard_dir=tensorboard_dir,
            compatibility_manager=compatibility_manager
        )
        
        # 训练模型
        model_save_path = f"output/models/{scenario_name}_{compatibility_manager.config.network_mode}_model.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        print(f"\n开始训练...")
        training_time = solver.train(
            episodes=config.EPISODES,
            patience=config.PATIENCE,
            log_interval=config.LOG_INTERVAL,
            model_save_path=model_save_path
        )
        
        # 评估结果
        final_completion_rate = solver._calculate_completion_rate()
        avg_reward = np.mean(solver.episode_rewards[-20:]) if len(solver.episode_rewards) >= 20 else 0
        
        # 生成方案
        plan_info = generate_plan_from_solver(solver)
        
        # 评估方案
        evaluation_result = evaluate_plan(plan_info, uavs, targets, obstacles, config)
        
        # 构建结果
        result = {
            "scenario_name": scenario_name,
            "network_mode": compatibility_manager.config.network_mode,
            "obs_mode": compatibility_manager.config.obs_mode,
            "training_time": training_time,
            "final_completion_rate": final_completion_rate,
            "average_reward": avg_reward,
            "total_episodes": len(solver.episode_rewards),
            "plan_info": plan_info,
            "evaluation": evaluation_result,
            "model_path": model_save_path,
            "tensorboard_dir": tensorboard_dir,
            "compatibility_info": compatibility_manager.get_network_info()
        }
        
        print(f"\n{'='*60}")
        print(f"场景 {scenario_name} 运行完成")
        print(f"{'='*60}")
        print(f"网络模式: {result['network_mode']}")
        print(f"训练时间: {result['training_time']:.2f}秒")
        print(f"最终完成率: {result['final_completion_rate']:.3f}")
        print(f"平均奖励: {result['average_reward']:.2f}")
        print(f"总训练轮次: {result['total_episodes']}")
        print(f"模型保存路径: {result['model_path']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 场景运行失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "scenario_name": scenario_name,
            "error": str(e),
            "success": False
        }

def generate_plan_from_solver(solver):
    """从求解器生成方案信息"""
    # 这里应该实现从训练好的求解器生成任务分配方案的逻辑
    # 为了兼容性，返回基本的方案信息
    plan_info = {
        "assignments": [],
        "total_distance": 0,
        "completion_time": 0,
        "method": solver.network_mode
    }
    
    # 简化的方案生成逻辑
    for i, uav in enumerate(solver.uavs):
        for j, target in enumerate(solver.targets):
            if len(target.allocated_uavs) > 0:
                plan_info["assignments"].append({
                    "uav_id": uav.id,
                    "target_id": target.id,
                    "distance": solver.graph.get_dist(uav.id, target.id)
                })
    
    return plan_info

# === 主函数 ===
def main():
    """主函数 - 演示向后兼容性"""
    print("多无人机协同任务分配系统 - 向后兼容版本")
    print("支持传统FCN网络和TransformerGNN网络的无缝切换")
    
    # 设置中文字体
    set_chinese_font()
    
    # 演示不同配置的运行
    scenarios_to_test = [
        (get_small_scenario, "小规模场景"),
        (get_balanced_scenario, "平衡场景")
    ]
    
    configurations_to_test = [
        {
            "network_mode": "traditional",
            "traditional_network_type": "DeepFCNResidual",
            "obs_mode": "flat",
            "enable_compatibility_checks": True
        },
        {
            "network_mode": "transformer_gnn",
            "obs_mode": "graph",
            "enable_compatibility_checks": True
        }
    ]
    
    results = []
    
    for scenario_func, scenario_name in scenarios_to_test:
        for config_override in configurations_to_test:
            print(f"\n{'='*80}")
            print(f"测试配置: {config_override}")
            print(f"{'='*80}")
            
            result = run_scenario_compatible(
                scenario_func=scenario_func,
                scenario_name=f"{scenario_name}_{config_override['network_mode']}",
                config_override=config_override
            )
            
            results.append(result)
    
    # 输出总结
    print(f"\n{'='*80}")
    print("向后兼容性测试总结")
    print(f"{'='*80}")
    
    for result in results:
        if result.get("success", True):
            print(f"✅ {result['scenario_name']}")
            print(f"   网络模式: {result['network_mode']}")
            print(f"   完成率: {result['final_completion_rate']:.3f}")
            print(f"   训练时间: {result['training_time']:.2f}秒")
        else:
            print(f"❌ {result['scenario_name']}: {result.get('error', '未知错误')}")
    
    print(f"\n向后兼容性验证完成！")

if __name__ == "__main__":
    main()
