# -*- coding: utf-8 -*-
import pro
from swarms_planning import UAV
from swarms_planning import DirectedGraph
from swarms_planning import Target
from swarms_planning import GeneticAlgorithm
from swarms_planning import PHCurve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from typing import List, Dict, Tuple, Set
import matplotlib.font_manager as fm
import matplotlib


# Note: The following classes are defined in the 'swarms_planning' library and are described here for context.
# class UAV:
#     self.position = np.array(position)  # Initial position [x, y]
#     self.heading = heading  # Initial heading (radians)
#     self.resources = np.array(resources)  # Available resources [Type1, Type2, ...]
#     self.initial_resources = np.array(resources)  # Initial resources
#     self.max_distance = max_distance  # Maximum flight distance
#     self.velocity = velocity  # Flight speed
#     self.task_sequence = []  # Task sequence [(target_ID, heading), ...]
#     self.current_distance = 0  # Distance flown
#     self.current_position = np.array(position)  # Current position

# Main function to solve the multi-UAV task allocation problem
def solve_multi_uav_task_allocation():
    """
    Initializes the scenario, runs the optimized genetic algorithm, and visualizes the results.
    """
    # 1. Initialize UAVs and Targets
    # 定义无人机和目标
    uavs = [
        UAV(id=1, position=[0, 0], heading=0, resources=[100], max_distance=100),
        UAV(id=2, position=[10, 0], heading=0, resources=[100], max_distance=100)
    ]
    targets = [
        Target(id=1, position=[5, 5], resources=[20], value=10),
        Target(id=2, position=[8, 3], resources=[30], value=15),
        Target(id=3, position=[12, 7], resources=[25], value=12),
        Target(id=4, position=[3, 9], resources=[15], value=8),
        Target(id=5, position=[7, 12], resources=[22], value=11)
    ]

    # 2. Create the directed graph model for path calculations
    graph = DirectedGraph(uavs, targets, n_phi=6)

    # 3. Create and configure the Genetic Algorithm instance
    base_mutation_rate = 0.1  # 基础变异率
    ga = GeneticAlgorithm(
        uavs=uavs,
        targets=targets,
        graph=graph,
        population_size=100,
        max_generations=200,  # 增加迭代代数，给算法更多时间收敛
        crossover_rate=0.8,
        mutation_rate=base_mutation_rate  # 使用基础变异率初始化
    )

    # 4. Run the Genetic Algorithm with Elitism and Adaptive Mutation
    best_fitness_history = []
    stagnation_counter = 0  # 停滞计数器

    print("Starting Genetic Algorithm optimization with Adaptive Mutation...")
    for generation in range(ga.max_generations):
        # --- Elitism: Preserve the best individual ---
        current_fitnesses = [ga.evaluate_fitness(chrom) for chrom in ga.population]
        best_index = np.argmax(current_fitnesses)
        elite_chromosome = ga.population[best_index].copy()
        current_best_fitness = current_fitnesses[best_index]

        # --- Adaptive Mutation Logic ---
        if generation > 0 and current_best_fitness <= best_fitness_history[-1]:
            stagnation_counter += 1
        else:
            stagnation_counter = 0  # 如果有提升，重置计数器

        if stagnation_counter >= 15:  # 如果连续15代没有提升
            print(f"Generation {generation}: Fitness stagnated. Increasing mutation rate!")
            ga.mutation_rate = 0.4  # 临时提高变异率以跳出局部最优
            stagnation_counter = 0  # 重置计数器，避免连续触发
        else:
            ga.mutation_rate = base_mutation_rate  # 恢复基础变异率

        best_fitness_history.append(current_best_fitness)

        # --- Standard GA Steps ---
        parents = ga.selection()
        offspring = ga.crossover(parents)
        offspring = ga.mutation(offspring)  # 变异操作会使用ga对象中当前的mutation_rate

        # --- Elitism: Replace worst offspring with the elite ---
        offspring_fitnesses = [ga.evaluate_fitness(chrom) for chrom in offspring]
        worst_index = np.argmin(offspring_fitnesses)
        offspring[worst_index] = elite_chromosome

        ga.population = offspring

        if generation % 20 == 0:
            print(
                f"Generation {generation}: Best Fitness = {best_fitness_history[-1]:.6f} (Mutation Rate: {ga.mutation_rate})")

    # 5. Get and display the best solution
    best_chromosome = ga.best_chromosome
    best_fitness = ga.best_fitness
    print("\nOptimization completed!")
    print(f"Final Best Fitness: {best_fitness:.6f}")

    # 6. Parse and visualize the results
    task_assignments = parse_chromosome(best_chromosome, uavs, targets)
    visualize_task_assignments(uavs, targets, task_assignments, graph)

    # 7. Visualize the fitness history
    visualize_fitness_history(best_fitness_history)

    return task_assignments


def parse_chromosome(chromosome: np.ndarray, uavs: List[UAV], targets: List[Target]) -> Dict:
    """Parses a chromosome to determine the task assignments for each UAV."""
    task_assignments = {uav.id: [] for uav in uavs}
    for i in range(chromosome.shape[1]):
        target_id = chromosome[0, i]
        uav_id = -chromosome[1, i] if chromosome[1, i] != 0 else None
        phi_idx = chromosome[2, i]
        if uav_id and uav_id in task_assignments:
            target = next(t for t in targets if t.id == target_id)
            heading = DirectedGraph(uavs, targets).phi_set[phi_idx]
            task_assignments[uav_id].append((target_id, heading))
    for uav_id in task_assignments:
        task_assignments[uav_id].sort(key=lambda x: x[0])
    return task_assignments


def visualize_task_assignments(uavs: List[UAV], targets: List[Target],
                               task_assignments: Dict, graph=None):
    """Visualizes the final task allocation results, showing paths for each UAV."""
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 10))

    target_x = [target.position[0] for target in targets]
    target_y = [target.position[1] for target in targets]
    plt.scatter(target_x, target_y, c='red', s=120, marker='o', label='Target')
    for i, target in enumerate(targets):
        plt.annotate(f'T{target.id}', (target.position[0], target.position[1]),
                     xytext=(8, 8), textcoords='offset points', fontsize=12, color='red', fontweight='bold')

    uav_x = [uav.position[0] for uav in uavs]
    uav_y = [uav.position[1] for uav in uavs]
    plt.scatter(uav_x, uav_y, c='blue', s=150, marker='^', label='UAV')

    for i, uav in enumerate(uavs):
        plt.annotate(f'U{uav.id}', (uav.position[0], uav.position[1]),
                     xytext=(8, 8), textcoords='offset points', fontsize=12, color='blue', fontweight='bold')
        if hasattr(uav, 'heading'):
            # --- VISUALIZATION FIX: Smaller arrow for initial heading ---
            arrow_length = 1.0
            head_width = 0.4
            head_length = 0.6
            dx = arrow_length * np.cos(uav.heading)
            dy = arrow_length * np.sin(uav.heading)
            plt.arrow(uav.position[0], uav.position[1], dx, dy,
                      head_width=head_width, head_length=head_length, fc='blue', ec='blue', alpha=0.7)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(uavs)))
    for uav_id, tasks in task_assignments.items():
        if not tasks: continue
        uav = next(u for u in uavs if u.id == uav_id)
        color = colors[uav_id - 1]
        path_x = [uav.position[0]]
        path_y = [uav.position[1]]
        for target_id, phi_rad in tasks:
            target = next(t for t in targets if t.id == target_id)
            path_x.append(target.position[0])
            path_y.append(target.position[1])
        plt.plot(path_x, path_y, '-', color=color, linewidth=2.0, label=f'UAV {uav_id} Path')
        for j, (x, y) in enumerate(zip(path_x[1:], path_y[1:])):
            target_id, phi_rad = tasks[j]
            plt.annotate(f'Task {j + 1}\n(T{target_id})', (x, y),
                         xytext=(-10, 15), textcoords='offset points', fontsize=9, color='white', fontweight='bold',
                         bbox=dict(boxstyle="circle,pad=0.3", fc=color, ec="none", alpha=0.8))
            # --- VISUALIZATION FIX: Smaller arrow for arrival heading ---
            arrow_length = 1.0
            head_width = 0.4
            head_length = 0.6
            dx = arrow_length * np.cos(phi_rad)
            dy = arrow_length * np.sin(phi_rad)
            plt.arrow(x, y, dx, dy, head_width=head_width, head_length=head_length, fc=color, ec=color, alpha=0.7)

    plt.legend(loc='upper left')
    plt.title('Optimized Multi-UAV Task Allocation', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_fitness_history(fitness_history: List[float]):
    """Visualizes the convergence of the fitness value over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness Value', fontsize=12)
    plt.title('Genetic Algorithm Optimization Process', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# The visualize_directed_graph function is omitted for brevity as it was not changed.


# Main execution block
if __name__ == "__main__":
    task_assignments = solve_multi_uav_task_allocation()

    # Print the detailed task assignments
    print("\n--- Final Task Assignment Details ---")
    all_tasks_assigned = True
    for uav_id, tasks in task_assignments.items():
        if tasks:
            print(f"UAV {uav_id}:")
            for i, (target_id, heading) in enumerate(tasks):
                print(
                    f"  - Task {i + 1}: Attack Target {target_id}, arrival heading {np.degrees(heading):.2f} degrees ({heading:.2f} rad)")
        else:
            print(f"UAV {uav_id}: No tasks assigned.")
            all_tasks_assigned = False

    if all_tasks_assigned:
        print("\nResult: All UAVs have been assigned tasks.")
    else:
        print("\nResult: Some UAVs remain idle. Consider further tuning of the fitness function.")