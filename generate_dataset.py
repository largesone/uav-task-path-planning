import numpy as np
import random
import pickle

#创建并保存一个包含500个随机场景的 .pkl 文件。每个场景都包含不同数量和随机属性的无人机和目标。
def generate_random_scenario(max_uavs=50, max_targets=100, area_size=100):
    """生成一个随机的无人机与目标场景"""
    num_uavs = random.randint(2, max_uavs)
    num_targets = random.randint(10, max_targets)

    uavs_data = []
    for i in range(1, num_uavs + 1):
        uav = {
            'id': i,
            'position': [random.uniform(0, area_size), random.uniform(0, area_size)],
            'heading': random.uniform(0, 2 * np.pi),
            'resources': [random.randint(80, 150)],
            'max_distance': random.uniform(100, 200)
        }
        uavs_data.append(uav)

    targets_data = []
    for i in range(1, num_targets + 1):
        target = {
            'id': i,
            'position': [random.uniform(0, area_size), random.uniform(0, area_size)],
            'resources': [random.randint(15, 40)],
            'value': random.uniform(5, 20)
        }
        targets_data.append(target)

    return {'uavs': uavs_data, 'targets': targets_data}


def create_dataset(num_samples=500, filename="uav_scenarios.pkl"):
    """创建并保存包含多个随机场景的数据集"""
    dataset = []
    print(f"开始生成 {num_samples} 个样本...")
    for i in range(num_samples):
        dataset.append(generate_random_scenario())
        if (i + 1) % 50 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本。")

    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n数据集已成功生成并保存到文件: {filename}")


if __name__ == "__main__":
    create_dataset(num_samples=500)