import numpy as np
# 从您现有的主文件中导入所有需要的类和函数
from uav_task_allocation_gcn_dl import (
    UAV,
    Target,
    DirectedGraph,
    GraphRLSolver,
    UAVTaskEnv,  # 确保导入 UAVTaskEnv
    visualize_task_assignments,
    set_chinese_font,
    DirectedGraph
)


def run_feasibility_test():
    """
    一个独立的测试函数，用于验证核心修改的可行性。
    """
    print("--- 开始核心逻辑可行性测试 ---")

    # 1. 定义一个固定的、小型的测试场景
    print("\n[步骤 1/5] 创建测试场景（2个无人机，5个目标）...")
    uavs = [
        UAV(id=1, position=[0, 0], heading=0, resources=[100], max_distance=100),
        UAV(id=2, position=[10, 0], heading=np.pi, resources=[120], max_distance=100)
    ]
    targets = [
        Target(id=1, position=[5, 5], resources=[20], value=10),
        Target(id=2, position=[8, 3], resources=[30], value=15),
        Target(id=3, position=[12, 7], resources=[25], value=12),
        Target(id=4, position=[3, 9], resources=[15], value=8),
        Target(id=5, position=[7, 12], resources=[22], value=11)
    ]
    print("场景创建成功。")

    # 2. 创建图对象，这将直接测试我们修复的 _get_position 方法
    print("\n[步骤 2/5] 创建图对象 (测试 DirectedGraph 核心修复)...")
    # 使用 cache_path=None 强制重新计算图，以确保测试的是新代码
    try:
        graph = DirectedGraph(uavs, targets, cache_path=None)
        print("图对象创建成功，邻接矩阵已生成。")
    except Exception as e:
        print(f"!!! 在创建图对象时发生严重错误: {e}")
        print("测试失败。请检查 uav_task_allocation_gcn_dl.py 中的 DirectedGraph 类。")
        return

    # 3. 创建求解器（Solver）
    print("\n[步骤 3/5] 初始化强化学习求解器...")
    try:
        # 为了测试，我们使用一组固定的超参数
        temp_env = UAVTaskEnv(uavs, targets, graph)
        input_dim = len(temp_env.reset())
        output_dim = len(targets) * len(uavs) * graph.n_phi

        solver = GraphRLSolver(
            uavs=uavs,
            targets=targets,
            graph=graph,
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=output_dim,
            learning_rate=0.0005,
            load_balance_penalty=1
        )
        print("求解器初始化成功。")
    except Exception as e:
        print(f"!!! 在初始化求解器时发生错误: {e}")
        print("测试失败。")
        return

    # 4. 进行一次短时间的训练
    print("\n[步骤 4/5] 开始进行短时训练 (5个 episodes)...")
    # 训练几轮就足以验证流程是否通畅
    solver.train(episodes=5, use_cache=False, early_stopping_patience=5, log_interval=1)
    print("短时训练完成。")

    # 5. 获取并打印结果
    print("\n[步骤 5/5] 获取并打印最终任务分配结果...")
    task_assignments = solver.get_task_assignments()

    print("\n--- 测试最终结果 ---")
    print("任务分配结果:")
    has_tasks = False
    for uav_id, tasks in task_assignments.items():
        print(f"无人机 {uav_id}: {[(t[0], f'{np.degrees(t[1]):.1f}°') for t in tasks]}")
        if tasks:
            has_tasks = True

    if has_tasks:
        print("\n[结论] 测试成功通过！端到端流程无崩溃，并成功生成了分配方案。")
        # 可视化结果以供检查
        set_chinese_font()
        visualize_task_assignments(task_assignments, uavs, targets)
    else:
        print("\n[结论] 测试运行完毕，但未分配任何任务。虽然没有崩溃，但请检查模型学习效果或参数。")


if __name__ == "__main__":
    run_feasibility_test()