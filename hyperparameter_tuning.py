import numpy as np
import pickle
import itertools
import csv
import time
from tqdm import tqdm
from uav_task_allocation_gcn_dl import UAV, Target, DirectedGraph, GraphRLSolver, UAVTaskEnv, set_chinese_font

def evaluate_solution(task_assignments, uavs):
    """
    评估一个解决方案的质量，综合考虑任务完成度和负载均衡度。
    """
    # 检查是否有任何任务被分配
    if not any(task_assignments.values()):
        return {'total_tasks': 0, 'imbalance': 100, 'combined_score': -1000}

    task_counts = [len(tasks) for tasks in task_assignments.values()]
    total_tasks = sum(task_counts)

    # 使用标准差作为不均衡度的衡量标准
    imbalance = np.std(task_counts) if len(task_counts) > 1 else 0

    # 综合得分：我们希望任务数多（奖励高），且不均衡度低（惩罚小）
    # 权重可以调整，这里我们假设任务总数的重要性是不均衡度的5倍
    combined_score = total_tasks - 5.0 * imbalance

    return {'total_tasks': total_tasks, 'imbalance': imbalance, 'combined_score': combined_score}


def tune_hyperparameters(dataset_path="uav_scenarios.pkl", num_scenarios_to_test=5):
    """
    在数据集上进行超参数网格搜索，并将结果保存到文件。
    注意：为了节省时间，我们只在数据集的前 N 个场景上进行测试。
    """
    # 1. 定义要搜索的超参数网格
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'load_balance_penalty': [0.2, 0.5, 1.0],
        'epsilon_decay': [0.995, 0.999]
    }

    # 2. 加载数据集
    try:
        with open(dataset_path, 'rb') as f:
            full_dataset = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 数据集文件 '{dataset_path}' 未找到。")
        print("请先运行 'generate_dataset.py' 来创建数据集。")
        return None

    # 为了效率，只取数据集的一个子集进行测试
    test_scenarios = full_dataset[:num_scenarios_to_test]
    print(f"已加载数据集，将在 {len(test_scenarios)} 个测试场景上进行超参数调优...")

    # 3. 创建所有超参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    best_score = -float('inf')
    best_params = None

    print(f"\n将要测试 {len(param_combinations)} 组超参数组合...")

    # 4. 遍历所有组合进行训练和评估（使用总进度条）
    for params in tqdm(param_combinations, desc="Total Tuning Progress"):
        scenario_scores = []
        for j, scenario_data in enumerate(test_scenarios):
            # 使用 tqdm.write 打印信息，避免弄乱进度条
            tqdm.write(f"  - Testing params: {params} on scenario {j + 1}/{num_scenarios_to_test}...")

            uav_objects = [UAV(**u) for u in scenario_data['uavs']]
            target_objects = [Target(**t) for t in scenario_data['targets']]

            graph = DirectedGraph(uav_objects, target_objects)

            # 精确计算状态维度
            temp_env = UAVTaskEnv(uav_objects, target_objects, graph)
            input_dim = len(temp_env.reset())
            output_dim = len(target_objects) * len(uav_objects) * graph.n_phi

            solver = GraphRLSolver(
                uavs=uav_objects,
                targets=target_objects,
                graph=graph,
                input_dim=input_dim,
                hidden_dim=128,
                output_dim=output_dim,
                **params
            )

            # 为了快速得到结果，训练轮数可以减少# 减少log打印
            solver.train(episodes=100, use_cache=False, early_stopping_patience=15, log_interval=100,
                         enable_plotting=False)
            assignments = solver.get_task_assignments()
            evaluation = evaluate_solution(assignments, uav_objects)
            scenario_scores.append(evaluation['combined_score'])

        avg_score = np.mean(scenario_scores)
        results.append({'params': params, 'avg_score': avg_score})
        tqdm.write(f"--- Params: {params}, Avg Score: {avg_score:.2f} ---")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # 5. 输出最终结果到控制台
    print("\n\n========== 超参数调优结果 ==========")
    print(f"在 {len(test_scenarios)} 个场景上测试了 {len(param_combinations)} 组参数。")
    print(f"\n最优综合得分: {best_score:.2f}")
    print(f"最优参数组合: {best_params}")

    sorted_results = sorted(results, key=lambda x: x['avg_score'], reverse=True)
    print("\n详细结果:")
    for res in sorted_results:
        print(f"  - 参数: {res['params']}, 平均得分: {res['avg_score']:.2f}")

    # 6. 将结果保存到文件
    report_filename = "tuning_report.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("========== 超参数调优结果报告 ==========\n\n")
        f.write(f"测试场景数量: {len(test_scenarios)}\n")
        f.write(f"测试参数组合数量: {len(param_combinations)}\n\n")
        f.write("--- 最优结果 ---\n")
        f.write(f"最优综合得分: {best_score:.4f}\n")
        f.write(f"最优参数组合: {best_params}\n\n")
        f.write("--- 详细结果列表 (按得分降序) ---\n")
        for res in sorted_results:
            f.write(f"参数: {res['params']}, 平均得分: {res['avg_score']:.4f}\n")
    print(f"\n详细报告已保存到: {report_filename}")

    # 7. 将结果保存到 CSV 文件
    csv_filename = "tuning_results.csv"
    header = list(param_grid.keys()) + ['avg_score']
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for res in sorted_results:
            row = list(res['params'].values()) + [f"{res['avg_score']:.4f}"]
            writer.writerow(row)
    print(f"CSV数据已保存到: {csv_filename}")

    return best_params


if __name__ == "__main__":
    # 确保中文字体已设置，以防万一
    set_chinese_font()
    tune_hyperparameters()