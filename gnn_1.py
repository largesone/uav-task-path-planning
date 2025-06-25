import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, FontManager


# 设置中文字体，增强错误处理
def set_chinese_font(preferred_fonts=None):
    """设置matplotlib支持中文显示的字体"""
    if preferred_fonts is None:
        preferred_fonts = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]

    try:
        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]

        for font in preferred_fonts:
            if font in available_fonts:
                plt.rcParams["font.family"] = font
                print(f"成功设置中文字体: {font}")
                return True
    except Exception as e:
        print(f"警告: 字体检测失败: {e}")

    try:
        default_font = findfont(FontProperties(family=['sans-serif']))
        plt.rcParams["font.family"] = default_font
        print(f"警告: 未找到中文字体，将使用默认字体: {default_font}")
        print("中文可能无法正常显示。若需显示中文，请安装中文字体并重新运行。")
        return False
    except Exception as e:
        print(f"错误: 字体设置失败: {e}")
        return False


def visualize_task_assignments(task_assignments, uavs, targets, show=True):
    """可视化多无人机任务分配结果"""
    font_set = set_chinese_font()

    plt.figure(figsize=(10, 8))

    # 绘制目标点
    target_positions = np.array([t['position'] for t in targets])
    plt.scatter(target_positions[:, 0], target_positions[:, 1], c='red', marker='o', s=100, label='目标')

    # 为每个目标添加标签
    for target in targets:
        plt.annotate(f"目标{target['id']}",
                     (target['position'][0], target['position'][1]),
                     fontsize=12,
                     xytext=(5, 5),
                     textcoords='offset points')

    # 绘制无人机起点
    uav_positions = np.array([u['position'] for u in uavs])
    plt.scatter(uav_positions[:, 0], uav_positions[:, 1], c='blue', marker='s', s=100, label='无人机起点')

    # 为每个无人机添加标签
    for uav in uavs:
        plt.annotate(f"无人机{uav['id']}",
                     (uav['position'][0], uav['position'][1]),
                     fontsize=12,
                     xytext=(5, 5),
                     textcoords='offset points')

    # 定义颜色列表
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 绘制每个无人机的任务路径
    for uav_id, tasks in task_assignments.items():
        uav_idx = next(i for i, u in enumerate(uavs) if u['id'] == uav_id)
        uav_color = colors[uav_idx % len(colors)]
        uav_start = uavs[uav_idx]['position']

        # 绘制从起点到第一个目标的路径
        if tasks:
            first_target_idx = tasks[0][0]
            first_target = next(t for t in targets if t['id'] == first_target_idx)
            plt.plot([uav_start[0], first_target['position'][0]],
                     [uav_start[1], first_target['position'][1]],
                     c=uav_color, linestyle='-', linewidth=1.5, alpha=0.7)

        # 绘制目标之间的路径
        for i in range(len(tasks) - 1):
            current_target_idx = tasks[i][0]
            next_target_idx = tasks[i + 1][0]
            current_target = next(t for t in targets if t['id'] == current_target_idx)
            next_target = next(t for t in targets if t['id'] == next_target_idx)

            plt.plot([current_target['position'][0], next_target['position'][0]],
                     [current_target['position'][1], next_target['position'][1]],
                     c=uav_color, linestyle='-', linewidth=1.5, alpha=0.7)

        # 绘制无人机航向角
        for target_idx, heading in tasks:
            target = next(t for t in targets if t['id'] == target_idx)
            heading_rad = np.radians(heading)
            arrow_length = 2.0
            plt.arrow(target['position'][0], target['position'][1],
                      arrow_length * np.cos(heading_rad),
                      arrow_length * np.sin(heading_rad),
                      head_width=0.5, head_length=0.7, fc=uav_color, ec=uav_color, alpha=0.8)

    # 添加图例和标题
    plt.legend(fontsize=12)
    plt.title("多无人机任务分配结果", fontsize=16)
    plt.xlabel("X坐标", fontsize=14)
    plt.ylabel("Y坐标", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    if show:
        plt.show()

    return plt.gcf()


# 简化版的任务分配算法
def greedy_task_allocation(uavs, targets):
    """使用贪心算法进行任务分配"""
    task_assignments = {uav['id']: [] for uav in uavs}
    remaining_targets = targets.copy()

    while remaining_targets:
        for uav in uavs:
            if not remaining_targets:
                break

            distances = [
                np.linalg.norm(np.array(uav['position']) - np.array(target['position']))
                for target in remaining_targets
            ]

            closest_idx = np.argmin(distances)
            closest_target = remaining_targets[closest_idx]

            # 为简化示例，随机分配一个航向角
            heading = np.random.choice([0, 90, 180, 270])

            task_assignments[uav['id']].append((closest_target['id'], heading))
            uav['position'] = closest_target['position'].copy()
            remaining_targets.pop(closest_idx)

    return task_assignments


# 主函数
def main():
    # 定义无人机和目标
    uavs = [
        {'id': 1, 'position': [0, 0]},
        {'id': 2, 'position': [10, 0]}
    ]

    targets = [
        {'id': 1, 'position': [5, 5]},
        {'id': 2, 'position': [8, 3]},
        {'id': 3, 'position': [12, 7]},
        {'id': 4, 'position': [3, 9]},
        {'id': 5, 'position': [7, 12]}
    ]

    # 使用贪心算法进行任务分配
    print("正在使用贪心算法进行任务分配...")
    task_assignments = greedy_task_allocation(uavs.copy(), targets.copy())

    # 可视化任务分配结果
    print("可视化任务分配结果...")
    visualize_task_assignments(task_assignments, uavs, targets)

    # 打印任务分配结果
    print("\n任务分配结果:")
    for uav_id, tasks in task_assignments.items():
        print(f"无人机{uav_id}:")
        for target_id, heading in tasks:
            print(f"  - 目标{target_id}, 航向角: {heading}度")


if __name__ == "__main__":
    main()