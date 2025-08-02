#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroShotGNN可扩展性分析

评估训练好的ZeroShotGNN网络能够支持的最大无人机和目标数量
"""

import numpy as np
import torch
import time
import psutil
import gc
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class ScalabilityAnalyzer:
    """ZeroShotGNN可扩展性分析器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 性能基准
        self.memory_limit_gb = 8.0  # GPU/CPU内存限制
        self.time_limit_seconds = 10.0  # 单次推理时间限制
        self.batch_size = 1  # 推理批次大小
        
    def analyze_theoretical_limits(self) -> Dict:
        """
        分析理论极限
        
        基于网络架构和参数数量估算理论支持的最大规模
        """
        # 获取模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        embedding_dim = self.model.embedding_dim
        
        # 理论分析
        analysis = {
            'model_parameters': total_params,
            'embedding_dimension': embedding_dim,
            'theoretical_limits': {},
            'memory_analysis': {},
            'computational_complexity': {}
        }
        
        # === 内存分析 ===
        # 模型参数内存 (假设float32)
        model_memory_mb = total_params * 4 / (1024 * 1024)
        
        # 单个实体的内存占用
        uav_feature_size = 9 * 4  # 9个特征 * 4字节
        target_feature_size = 8 * 4  # 8个特征 * 4字节
        
        # 中间激活的内存占用（粗略估算）
        def estimate_activation_memory(n_uavs, n_targets):
            # UAV嵌入: batch_size * n_uavs * embedding_dim * 4
            uav_embedding_mb = self.batch_size * n_uavs * embedding_dim * 4 / (1024 * 1024)
            
            # 目标嵌入: batch_size * n_targets * embedding_dim * 4
            target_embedding_mb = self.batch_size * n_targets * embedding_dim * 4 / (1024 * 1024)
            
            # 注意力矩阵: batch_size * n_uavs * n_targets * 4 (简化)
            attention_mb = self.batch_size * n_uavs * n_targets * 4 / (1024 * 1024)
            
            # Q值矩阵: batch_size * n_uavs * n_targets * n_phi * 4
            n_phi = 6  # 假设6个方向
            q_values_mb = self.batch_size * n_uavs * n_targets * n_phi * 4 / (1024 * 1024)
            
            return uav_embedding_mb + target_embedding_mb + attention_mb + q_values_mb
        
        analysis['memory_analysis'] = {
            'model_memory_mb': model_memory_mb,
            'uav_feature_size_bytes': uav_feature_size,
            'target_feature_size_bytes': target_feature_size,
            'activation_memory_estimator': estimate_activation_memory
        }
        
        # === 计算复杂度分析 ===
        # Transformer的复杂度主要是O(n²)的注意力计算
        def estimate_flops(n_uavs, n_targets):
            # 自注意力: O(n_uavs²) + O(n_targets²)
            self_attention_flops = n_uavs**2 * embedding_dim + n_targets**2 * embedding_dim
            
            # 交叉注意力: O(n_uavs * n_targets)
            cross_attention_flops = n_uavs * n_targets * embedding_dim
            
            # Q值计算: O(n_uavs * n_targets)
            q_computation_flops = n_uavs * n_targets * embedding_dim
            
            return self_attention_flops + cross_attention_flops + q_computation_flops
        
        analysis['computational_complexity'] = {
            'flops_estimator': estimate_flops,
            'complexity_order': 'O(n_uavs² + n_targets² + n_uavs * n_targets)'
        }
        
        # === 理论极限估算 ===
        memory_limit_mb = self.memory_limit_gb * 1024
        available_memory_mb = memory_limit_mb - model_memory_mb
        
        # 基于内存限制的理论极限
        max_entities_memory = int(np.sqrt(available_memory_mb * 1024 * 1024 / (embedding_dim * 4 * 2)))
        
        # 基于计算时间限制的理论极限（粗略估算）
        # 假设现代GPU/CPU每秒可以处理10^9次浮点运算
        flops_per_second = 1e9
        max_flops = flops_per_second * self.time_limit_seconds
        max_entities_compute = int(np.sqrt(max_flops / (embedding_dim * 3)))
        
        analysis['theoretical_limits'] = {
            'max_entities_by_memory': max_entities_memory,
            'max_entities_by_computation': max_entities_compute,
            'estimated_max_uavs': min(max_entities_memory, max_entities_compute) // 2,
            'estimated_max_targets': min(max_entities_memory, max_entities_compute) // 2,
            'conservative_estimate': {
                'max_uavs': min(max_entities_memory, max_entities_compute) // 3,
                'max_targets': min(max_entities_memory, max_entities_compute) // 2
            }
        }
        
        return analysis
    
    def benchmark_performance(self, test_scenarios: List[Tuple[int, int]]) -> Dict:
        """
        基准性能测试
        
        Args:
            test_scenarios: 测试场景列表 [(n_uavs, n_targets), ...]
            
        Returns:
            Dict: 性能测试结果
        """
        results = {
            'scenarios': [],
            'successful_scenarios': [],
            'failed_scenarios': [],
            'performance_metrics': {}
        }
        
        for n_uavs, n_targets in test_scenarios:
            print(f"测试场景: {n_uavs} UAVs, {n_targets} 目标")
            
            try:
                # 创建测试数据
                test_data = self._create_test_data(n_uavs, n_targets)
                
                # 性能测试
                perf_metrics = self._run_performance_test(test_data, n_uavs, n_targets)
                
                scenario_result = {
                    'n_uavs': n_uavs,
                    'n_targets': n_targets,
                    'success': True,
                    'metrics': perf_metrics
                }
                
                results['scenarios'].append(scenario_result)
                results['successful_scenarios'].append((n_uavs, n_targets))
                
                print(f"  ✓ 成功 - 推理时间: {perf_metrics['inference_time']:.3f}s, "
                      f"内存使用: {perf_metrics['memory_usage_mb']:.1f}MB")
                
            except Exception as e:
                scenario_result = {
                    'n_uavs': n_uavs,
                    'n_targets': n_targets,
                    'success': False,
                    'error': str(e)
                }
                
                results['scenarios'].append(scenario_result)
                results['failed_scenarios'].append((n_uavs, n_targets))
                
                print(f"  ✗ 失败 - {str(e)}")
            
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算性能指标
        successful_results = [r for r in results['scenarios'] if r['success']]
        if successful_results:
            inference_times = [r['metrics']['inference_time'] for r in successful_results]
            memory_usages = [r['metrics']['memory_usage_mb'] for r in successful_results]
            
            results['performance_metrics'] = {
                'avg_inference_time': np.mean(inference_times),
                'max_inference_time': np.max(inference_times),
                'avg_memory_usage': np.mean(memory_usages),
                'max_memory_usage': np.max(memory_usages),
                'success_rate': len(successful_results) / len(test_scenarios)
            }
        
        return results
    
    def _create_test_data(self, n_uavs: int, n_targets: int) -> Dict:
        """创建测试数据"""
        # 创建随机的图结构数据
        uav_features = torch.randn(1, n_uavs, 9, dtype=torch.float32, device=self.device)
        target_features = torch.randn(1, n_targets, 8, dtype=torch.float32, device=self.device)
        relative_positions = torch.randn(1, n_uavs, n_targets, 2, dtype=torch.float32, device=self.device)
        distances = torch.rand(1, n_uavs, n_targets, dtype=torch.float32, device=self.device)
        
        # 创建掩码（全部有效）
        uav_mask = torch.ones(1, n_uavs, dtype=torch.int32, device=self.device)
        target_mask = torch.ones(1, n_targets, dtype=torch.int32, device=self.device)
        
        return {
            "uav_features": uav_features,
            "target_features": target_features,
            "relative_positions": relative_positions,
            "distances": distances,
            "masks": {
                "uav_mask": uav_mask,
                "target_mask": target_mask
            }
        }
    
    def _run_performance_test(self, test_data: Dict, n_uavs: int, n_targets: int) -> Dict:
        """运行性能测试"""
        # 预热
        with torch.no_grad():
            _ = self.model(test_data)
        
        # 记录初始内存
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # 性能测试
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(test_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 记录最终内存
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        inference_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        
        # 检查时间和内存限制
        if inference_time > self.time_limit_seconds:
            raise RuntimeError(f"推理时间超限: {inference_time:.3f}s > {self.time_limit_seconds}s")
        
        if memory_usage > self.memory_limit_gb * 1024:
            raise RuntimeError(f"内存使用超限: {memory_usage:.1f}MB > {self.memory_limit_gb * 1024}MB")
        
        return {
            'inference_time': inference_time,
            'memory_usage_mb': memory_usage,
            'output_shape': list(output.shape),
            'n_actions': output.shape[1]
        }
    
    def find_maximum_scale(self, max_search_uavs: int = 50, max_search_targets: int = 50) -> Dict:
        """
        寻找最大支持规模
        
        使用二分搜索找到最大支持的UAV和目标数量
        """
        print("开始寻找最大支持规模...")
        
        # 寻找最大UAV数量（固定目标数量为10）
        max_uavs = self._binary_search_max_entities(
            entity_type='uavs',
            fixed_targets=10,
            max_search=max_search_uavs
        )
        
        # 寻找最大目标数量（固定UAV数量为10）
        max_targets = self._binary_search_max_entities(
            entity_type='targets',
            fixed_uavs=10,
            max_search=max_search_targets
        )
        
        # 寻找最大总实体数量
        max_total = self._binary_search_max_total_entities(max_search_uavs + max_search_targets)
        
        return {
            'max_uavs_with_10_targets': max_uavs,
            'max_targets_with_10_uavs': max_targets,
            'max_total_entities': max_total,
            'recommended_limits': {
                'conservative_max_uavs': max_uavs * 0.8,
                'conservative_max_targets': max_targets * 0.8,
                'production_ready_scale': {
                    'max_uavs': min(20, max_uavs * 0.7),
                    'max_targets': min(30, max_targets * 0.7)
                }
            }
        }
    
    def _binary_search_max_entities(self, entity_type: str, fixed_uavs: int = None, 
                                   fixed_targets: int = None, max_search: int = 50) -> int:
        """二分搜索最大实体数量"""
        left, right = 1, max_search
        max_successful = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if entity_type == 'uavs':
                n_uavs, n_targets = mid, fixed_targets
            else:  # targets
                n_uavs, n_targets = fixed_uavs, mid
            
            try:
                test_data = self._create_test_data(n_uavs, n_targets)
                self._run_performance_test(test_data, n_uavs, n_targets)
                
                max_successful = mid
                left = mid + 1
                print(f"  {entity_type} = {mid}: ✓")
                
            except Exception as e:
                right = mid - 1
                print(f"  {entity_type} = {mid}: ✗ ({str(e)[:50]}...)")
            
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return max_successful
    
    def _binary_search_max_total_entities(self, max_total: int) -> Dict:
        """二分搜索最大总实体数量"""
        successful_combinations = []
        
        # 测试不同的UAV:目标比例
        ratios = [
            (0.3, 0.7),  # 30% UAV, 70% 目标
            (0.4, 0.6),  # 40% UAV, 60% 目标
            (0.5, 0.5),  # 50% UAV, 50% 目标
            (0.6, 0.4),  # 60% UAV, 40% 目标
            (0.7, 0.3),  # 70% UAV, 30% 目标
        ]
        
        for uav_ratio, target_ratio in ratios:
            left, right = 10, max_total
            max_successful_total = 0
            best_combination = None
            
            while left <= right:
                mid_total = (left + right) // 2
                n_uavs = max(1, int(mid_total * uav_ratio))
                n_targets = max(1, int(mid_total * target_ratio))
                
                try:
                    test_data = self._create_test_data(n_uavs, n_targets)
                    self._run_performance_test(test_data, n_uavs, n_targets)
                    
                    max_successful_total = mid_total
                    best_combination = (n_uavs, n_targets)
                    left = mid_total + 1
                    
                except Exception:
                    right = mid_total - 1
                
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if best_combination:
                successful_combinations.append({
                    'ratio': f"{int(uav_ratio*100)}% UAV, {int(target_ratio*100)}% 目标",
                    'total_entities': max_successful_total,
                    'n_uavs': best_combination[0],
                    'n_targets': best_combination[1]
                })
        
        return {
            'combinations': successful_combinations,
            'best_total': max(c['total_entities'] for c in successful_combinations) if successful_combinations else 0
        }
    
    def generate_scalability_report(self, theoretical_analysis: Dict, 
                                  benchmark_results: Dict, max_scale_results: Dict) -> str:
        """生成可扩展性报告"""
        report = "ZeroShotGNN 可扩展性分析报告\n"
        report += "=" * 60 + "\n\n"
        
        # 理论分析
        report += "1. 理论分析\n"
        report += "-" * 20 + "\n"
        report += f"模型参数数量: {theoretical_analysis['model_parameters']:,}\n"
        report += f"嵌入维度: {theoretical_analysis['embedding_dimension']}\n"
        report += f"计算复杂度: {theoretical_analysis['computational_complexity']['complexity_order']}\n"
        report += f"理论最大UAV数量: {theoretical_analysis['theoretical_limits']['estimated_max_uavs']}\n"
        report += f"理论最大目标数量: {theoretical_analysis['theoretical_limits']['estimated_max_targets']}\n\n"
        
        # 实际测试结果
        report += "2. 实际测试结果\n"
        report += "-" * 20 + "\n"
        if benchmark_results.get('performance_metrics'):
            metrics = benchmark_results['performance_metrics']
            report += f"平均推理时间: {metrics['avg_inference_time']:.3f}s\n"
            report += f"最大推理时间: {metrics['max_inference_time']:.3f}s\n"
            report += f"平均内存使用: {metrics['avg_memory_usage']:.1f}MB\n"
            report += f"最大内存使用: {metrics['max_memory_usage']:.1f}MB\n"
            report += f"成功率: {metrics['success_rate']:.1%}\n\n"
        
        # 最大规模测试
        report += "3. 最大支持规模\n"
        report += "-" * 20 + "\n"
        report += f"最大UAV数量 (固定10个目标): {max_scale_results['max_uavs_with_10_targets']}\n"
        report += f"最大目标数量 (固定10个UAV): {max_scale_results['max_targets_with_10_uavs']}\n"
        
        if max_scale_results['max_total_entities']['combinations']:
            report += "\n最佳实体组合:\n"
            for combo in max_scale_results['max_total_entities']['combinations']:
                report += f"  - {combo['ratio']}: {combo['n_uavs']} UAV + {combo['n_targets']} 目标 = {combo['total_entities']} 总实体\n"
        
        # 生产环境推荐
        report += "\n4. 生产环境推荐\n"
        report += "-" * 20 + "\n"
        rec = max_scale_results['recommended_limits']['production_ready_scale']
        report += f"推荐最大UAV数量: {rec['max_uavs']}\n"
        report += f"推荐最大目标数量: {rec['max_targets']}\n"
        report += f"推荐总实体数量: {rec['max_uavs'] + rec['max_targets']}\n\n"
        
        # 性能优化建议
        report += "5. 性能优化建议\n"
        report += "-" * 20 + "\n"
        report += "- 使用GPU加速可显著提升性能\n"
        report += "- 批处理多个场景可提高吞吐量\n"
        report += "- 考虑使用模型量化减少内存占用\n"
        report += "- 对于超大规模场景，可考虑分层处理策略\n"
        
        return report

def run_full_scalability_analysis(model_path: str, device: str = 'cpu') -> Dict:
    """
    运行完整的可扩展性分析
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        
    Returns:
        Dict: 完整的分析结果
    """
    # 加载模型
    from networks import ZeroShotGNN
    model = ZeroShotGNN(input_dim=1, hidden_dims=[256, 128], output_dim=1000)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    
    # 创建分析器
    analyzer = ScalabilityAnalyzer(model, device)
    
    print("开始ZeroShotGNN可扩展性分析...")
    
    # 1. 理论分析
    print("\n1. 进行理论分析...")
    theoretical_analysis = analyzer.analyze_theoretical_limits()
    
    # 2. 基准测试
    print("\n2. 进行基准性能测试...")
    test_scenarios = [
        (2, 3), (4, 6), (6, 9), (8, 12), (10, 15),
        (12, 18), (15, 22), (18, 25), (20, 30)
    ]
    benchmark_results = analyzer.benchmark_performance(test_scenarios)
    
    # 3. 最大规模测试
    print("\n3. 寻找最大支持规模...")
    max_scale_results = analyzer.find_maximum_scale(max_search_uavs=30, max_search_targets=40)
    
    # 4
