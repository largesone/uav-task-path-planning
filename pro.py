#import swarms_planning
import numpy as np

class HelloClass:
    def __init__(self,msg):
        self.msg = msg

    def outputmsg(self):
        print("输出消息：" +  self.msg)

#
# class ConfigParas:
#     def __init__(self, id):
#         self.id = id
#
#     def initpara(self):
#         # 创建无人机实例
#         uav1 = swarms_planning.UAV(
#             id=1,
#             position = np.array([0, 0]),  # 初始位置
#             heading=0,  # 初始航向角（弧度）
#             resources=np.array([100, 50, 20]),  # 携带资源 [弹药, 燃料, 侦察设备]
#             max_distance=1000  # 最大飞行距离
#         )
#
#         uav2 = swarms_planning.UAV(
#             id=2,
#             position=np.array([50, 0]),
#             heading=np.pi / 4,
#             resources=np.array([80, 70, 30]),
#             max_distance=1200
#         )
#
#         uav3 = swarms_planning.UAV(
#             id=3,
#             position=np.array([0, 50]),
#             heading=np.pi / 2,
#             resources=np.array([60, 90, 10]),
#             max_distance=800
#         )
#
#         # 创建目标实例
#         target1 = swarms_planning.Target(
#             id=1,
#             position=np.array([200, 200]),  # 目标位置
#             resources=np.array([50, 20, 10]),  # 所需资源 [攻击, 压制, 侦察]
#             value=100  # 目标价值
#         )
#
#         target2 = swarms_planning.Target(
#             id=2,
#             position=np.array([300, 100]),
#             resources=np.array([30, 40, 5]),
#             value=80
#         )
#
#         target3 = swarms_planning.Target(
#             id=3,
#             position=np.array([150, 300]),
#             resources=np.array([70, 10, 15]),
#             value=120
#         )
#
#         # 收集所有无人机和目标
#         uavs = [uav1, uav2, uav3]
#         targets = [target1, target2, target3]
#         print(uavs)
#
#
#
#     def sayHello():
#         # strinfo ="hello, the id is " + str( self.id )
#         strinfo = "hello, the id is success"
#         print(strinfo )
