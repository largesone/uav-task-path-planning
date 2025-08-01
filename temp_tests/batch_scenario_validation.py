# -*- coding: utf-8 -*-
# 文件名: batch_scenario_validation.py
# 描述: 批处理算法场景验证脚本，用于验证算法在不同场景下的求解结果

import os
import sys
import time
import csv
import pickle
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 添加父目录到路径，确保能正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from main import run_scenario
from config import Config
from scenarios import (
    get_balanced_scenario, get_small_scenario, get_complex_scenario, 
    get_new_experimental_scenario, get_complex_scenario_v4, get_strategic_trap_scenario
)

class BatchScenarioValidator:
    """批处理场景验证器"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化批处理验证器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.results = []
        self.scenario_functions = {
            "balanced": get_balanced_scenario,
            "small": get_small_scenario,
            "complex": get_complex_scenario,
            "experimental": get_new_experimental_scenario,
            "complex_v4": get_complex_scenario_v4,
            "strategic_trap": get_strategic_trap_scenario,
            "uav_sweep_5_targets": get_new_experimental_scenario,  # 使用默认的4UAV, 2目标
            "uav_sweep_10_targets": get_new_experimental_scenario   # 使用默认的4UAV, 2目标
        }
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def load_pkl_scenario(self, scenario_path: str) -> Tuple[List, List, List]:
        """
        从PKL文件加载场景
        
        Args:
            scenario_path: 场景文件路径
            
        Returns:
            uavs, targets, obstacles
        """
        try:
            # 导入必要的类，确保pickle能正确反序列化
            import sys
            import os
            
            # 确保entities模块在sys.modules中
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.insert(0, parent_dir)
            
            # 强制导入entities模块
            import entities
            from entities import UAV, Target
            
            # 导入障碍物类
            from path_planning import CircularObstacle, PolygonalObstacle
            
            # 确保所有类在全局命名空间中可用
            globals()['UAV'] = UAV
            globals()['Target'] = Target
            globals()['CircularObstacle'] = CircularObstacle
            globals()['PolygonalObstacle'] = PolygonalObstacle
            
            with open(scenario_path, 'rb') as f:
                scenario_data = pickle.load(f)
            
            uavs = scenario_data['uavs']
            targets = scenario_data['targets']
            obstacles = scenario_data.get('obstacles', [])
            
            return uavs, targets, obstacles
        except Exception as e:
            print(f"加载场景文件失败: {scenario_path}, 错误: {e}")
            return None, None, None
    
    def get_scenario_info(self, uavs: List, targets: List, obstacles: List) -> Dict[str, Any]:
        """
        获取场景基本信息
        
        Args:
            uavs: UAV列表
            targets: 目标列表
            obstacles: 障碍物列表
            
        Returns:
            场景信息字典
        """
        return {
            'num_uavs': len(uavs),
            'num_targets': len(targets),
            'num_obstacles': len(obstacles),
            'total_uav_resources': sum(sum(uav.initial_resources) for uav in uavs),
            'total_target_demand': sum(sum(target.resources) for target in targets)
        }
    
    def run_single_scenario(self, scenario_name: str, scenario_type: str, 
                          network_type: str = "DeepFCNResidual", 
                          episodes: int = 200, force_retrain: bool = False,
                          pkl_filename: str = None) -> Dict[str, Any]:
        """
        运行单个场景
        
        Args:
            scenario_name: 场景名称
            scenario_type: 场景类型
            network_type: 网络类型
            episodes: 训练轮数
            force_retrain: 是否强制重新训练
            pkl_filename: PKL文件名（用于特定场景）
            
        Returns:
            结果字典
        """
        print(f"\n=== 运行场景: {scenario_name} ({scenario_type}) ===")
        if pkl_filename:
            print(f"使用PKL文件: {pkl_filename}")
        
        # 创建配置
        config = Config()
        config.NETWORK_TYPE = network_type
        config.training_config.episodes = episodes
        
        # 加载场景
        if scenario_type in self.scenario_functions and pkl_filename is None:
            # 使用内置场景函数
            uavs, targets, obstacles = self.scenario_functions[scenario_type](50.0)
        else:
            # 尝试从PKL文件加载 - 修复文件路径格式
            # 获取项目根目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            if pkl_filename:
                # 使用指定的PKL文件
                scenario_path = os.path.join(project_root, "scenarios", scenario_type, pkl_filename)
            elif scenario_type == "uav_sweep_5_targets":
                # 使用第一个可用的场景文件
                scenario_path = os.path.join(project_root, "scenarios", scenario_type, "uavs_04_targets_05.pkl")
            elif scenario_type == "uav_sweep_10_targets":
                scenario_path = os.path.join(project_root, "scenarios", scenario_type, "uavs_06_targets_10.pkl")
            else:
                # 尝试通用格式
                scenario_path = os.path.join(project_root, "scenarios", scenario_type, "scenario_1.pkl")
            
            uavs, targets, obstacles = self.load_pkl_scenario(scenario_path)
            if uavs is None:
                print(f"无法加载场景: {scenario_type}")
                return None
        
        # 获取场景信息
        scenario_info = self.get_scenario_info(uavs, targets, obstacles)
        
        # 运行场景
        start_time = time.time()
        try:
            final_plan, training_time, training_history, evaluation_metrics = run_scenario(
                config=config,
                base_uavs=uavs,
                base_targets=targets,
                obstacles=obstacles,
                scenario_name=scenario_name,
                force_retrain=force_retrain,
                save_visualization=False,
                show_visualization=False,
                output_base_dir=f"{self.output_dir}/batch_test"
            )
            total_time = time.time() - start_time
            
            # 计算额外指标
            total_distance = 0
            if final_plan:
                for uav_id, tasks in final_plan.items():
                    for task in tasks:
                        # 计算路径长度（简化计算）
                        uav = next(u for u in uavs if u.id == uav_id)
                        target = next(t for t in targets if t.id == task['target_id'])
                        distance = ((target.position[0] - uav.position[0])**2 + 
                                  (target.position[1] - uav.position[1])**2)**0.5
                        total_distance += distance
            
            # 构建结果
            result = {
                'scenario': scenario_name,
                'solver': 'RL',
                'config': f'{network_type}_{episodes}ep',
                'obstacle_mode': 'present' if obstacles else 'absent',
                'num_uavs': scenario_info['num_uavs'],
                'num_targets': scenario_info['num_targets'],
                'num_obstacles': scenario_info['num_obstacles'],
                'training_time': training_time,
                'planning_time': 0.0,  # RL算法没有单独的规划时间
                'total_time': total_time,
                'total_reward_score': evaluation_metrics.get('total_reward_score', 0),
                'completion_rate': evaluation_metrics.get('completion_rate', 0),
                'satisfied_targets_count': evaluation_metrics.get('satisfied_targets_count', 0),
                'total_targets': scenario_info['num_targets'],
                'satisfied_targets_rate': evaluation_metrics.get('satisfied_targets_rate', 0),
                'resource_utilization_rate': evaluation_metrics.get('resource_utilization_rate', 0),
                'resource_penalty': evaluation_metrics.get('resource_penalty', 0),
                'sync_feasibility_rate': evaluation_metrics.get('sync_feasibility_rate', 0),
                'load_balance_score': evaluation_metrics.get('load_balance_score', 0),
                'total_distance': total_distance,
                'is_deadlocked': 0,  # 简化处理
                'deadlocked_uav_count': 0
            }
            
            print(f"场景 {scenario_name} 运行完成")
            print(f"  训练时间: {training_time:.2f}秒")
            print(f"  完成率: {result['completion_rate']:.3f}")
            print(f"  资源利用率: {result['resource_utilization_rate']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"运行场景 {scenario_name} 时出错: {e}")
            return None
    
    def run_batch_validation(self, scenario_types: List[str], 
                           network_types: List[str] = ["DeepFCNResidual"],
                           episodes: int = 200, force_retrain: bool = False) -> None:
        """
        运行批处理验证
        
        Args:
            scenario_types: 场景类型列表
            network_types: 网络类型列表
            episodes: 训练轮数
            force_retrain: 是否强制重新训练
        """
        print(f"开始批处理验证...")
        print(f"场景类型: {scenario_types}")
        print(f"网络类型: {network_types}")
        print(f"训练轮数: {episodes}")
        
        for scenario_type in scenario_types:
            for network_type in network_types:
                # 检查是否需要遍历PKL文件
                if scenario_type in ["uav_sweep_5_targets", "uav_sweep_10_targets"]:
                    # 获取该场景目录下的所有PKL文件
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(current_dir)
                    scenario_dir = os.path.join(project_root, "scenarios", scenario_type)
                    
                    if os.path.exists(scenario_dir):
                        pkl_files = [f for f in os.listdir(scenario_dir) if f.endswith('.pkl')]
                        pkl_files.sort()  # 按文件名排序
                        
                        print(f"\n发现 {len(pkl_files)} 个PKL文件在 {scenario_type} 目录中")
                        
                        for pkl_file in pkl_files:
                            # 生成场景名称，包含PKL文件名
                            scenario_name = f"{scenario_type}_{pkl_file.replace('.pkl', '')}"
                            
                            result = self.run_single_scenario(
                                scenario_name=scenario_name,
                                scenario_type=scenario_type,
                                network_type=network_type,
                                episodes=episodes,
                                force_retrain=force_retrain,
                                pkl_filename=pkl_file
                            )
                            
                            if result:
                                self.results.append(result)
                    else:
                        print(f"场景目录不存在: {scenario_dir}")
                else:
                    # 使用内置场景函数
                    scenario_name = f"{scenario_type}_场景"
                    
                    result = self.run_single_scenario(
                        scenario_name=scenario_name,
                        scenario_type=scenario_type,
                        network_type=network_type,
                        episodes=episodes,
                        force_retrain=force_retrain
                    )
                    
                    if result:
                        self.results.append(result)
        
        # 保存结果
        self.save_results()
    
    def save_results(self) -> None:
        """保存结果到CSV文件"""
        if not self.results:
            print("没有结果可保存")
            return
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_validation_results_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存为CSV
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存至: {filepath}")
        print(f"共处理 {len(self.results)} 个场景")
        
        # 显示统计信息
        if len(self.results) > 0:
            print("\n=== 统计信息 ===")
            print(f"平均训练时间: {df['training_time'].mean():.2f}秒")
            print(f"平均完成率: {df['completion_rate'].mean():.3f}")
            print(f"平均资源利用率: {df['resource_utilization_rate'].mean():.3f}")
            print(f"平均总时间: {df['total_time'].mean():.2f}秒")

def main():
    """主函数""" 
    #  'experimental', 'balanced', 'complex'
    parser = argparse.ArgumentParser(description='批处理算法场景验证')
    parser.add_argument('--scenarios', nargs='+', 
                       default=['uav_sweep_5_targets'],
                       help='要测试的场景类型列表')
    parser.add_argument('--networks', nargs='+', 
                       default=['DeepFCNResidual'],
                       help='要测试的网络类型列表')
    parser.add_argument('--episodes', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--force-retrain', action='store_true',
                       help='强制重新训练')
    parser.add_argument('--output-dir', default='output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = BatchScenarioValidator(output_dir=args.output_dir)
    
    # 运行批处理验证
    validator.run_batch_validation(
        scenario_types=args.scenarios,
        network_types=args.networks,
        episodes=args.episodes
    )
    # ,
        # force_retrain=args.force_retrain
if __name__ == "__main__":
    main() 