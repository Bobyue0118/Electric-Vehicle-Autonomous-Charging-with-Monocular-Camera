import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# 得到calibrateHandEye()函数的相机棋盘格输入部分
R_board2camera = np.load('R_board2camera_4k_1212.npy')
T_board2camera = np.load('T_board2camera_4k_1212.npy')

# print('R_board2camera:', R_board2camera)
# print('T_board2camera:', T_board2camera)

# 得到calibrateHandEye()函数的机械臂部分

robot_poses = [
    (-0.365325,-0.009600,-0.716599,-69.626,-20.959,-0.222),
    (-0.391352,-0.065084,-0.569899,-90.904,-9.285,14.112),
    (-0.548677,-0.049253,-0.503067,-105.188,-35.088,-4.943),
    #(-0.713091,0.053381,-0.523648,-97.603,-64.032,-38.251),
    #(-0.700014,0.044063,-0.583858,-66.257,-63.266,-66.984),

    (-0.794186,0.105242,-0.756164,-23.965,-59.510,-117.553),
    (-0.659310,0.174398,-0.879781,-2.801,-39.860,-140.515),
    #(-0.515090,0.154076,-0.888023,15.704,-13.511,-165.893),
    #(-0.219004,0.144121,-0.818472,10.002,-41.350,-80.184),
    #(-0.221793,0.044100,-0.697230,-15.778,-63.236,-46.842),

    #(-0.245396,0.000306,-0.526201,-76.178,-70.026,0.995),
    #(-0.366722,0.007932,-0.368622,-125.827,-57.719,30.801),
    #(-0.399414,-0.048212,-0.501810,-80.746,-72.312,-15.500),
    (-0.543973,-0.045378,-0.543583,-59.567,-83.755,-50.850),
    (-0.722274,0.076062,-0.567743,-56.884,-66.443,-80.660),

    #(-0.705767,0.129953,-0.716527,-20.513,-54.973,-123.419),
    (-0.651864,0.113258,-0.789025,19.681,-37.881,-167.303),
    (-0.544269,0.133422,-0.790820,15.508,-25.854,-148.782),
    #(-0.458918,0.087518,-0.760553,15.086,-35.944,-126.981),
    (-0.338731,0.069921,-0.676522,1.299,-55.879,-82.913),

    (-0.209545,0.087643,-0.666035,-28.540,-60.187,-33.028),
    (-0.240384,0.019640,-0.538594,-96.836,-60.953,28.063),
    (-0.336785,-0.026613,-0.488747,-109.205,-67.802,25.544),
    (-0.313843,0.061963,-0.678079,-8.924,-62.125,-73.048),
    (-0.500566,0.051042,-0.789974,21.814,-42.010,-139.421),
]
#(-0.327435,-0.007205,-0.673388,-16.173,-65.699,-69.416),
#(-0.329988,0.001919,-0.519747,-86.564,-72.154,3.714),
#(-0.444832,0.069431,-0.730590,14.811,-47.900,-120.305),



R_gripper2base = []
for x, y, z, rx, ry, rz in robot_poses:
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    R_gripper2base.append(r.as_matrix())

T_gripper2base = []
for i in range(len(robot_poses)):
    tvec = np.array([robot_poses[i][0], robot_poses[i][1], robot_poses[i][2]])
    T_gripper2base.append(tvec)

R_gripper2base=np.array(R_gripper2base)
T_gripper2base=np.array(T_gripper2base)
T_gripper2base=T_gripper2base.reshape((T_gripper2base.shape[0],T_gripper2base.shape[1],1))
np.save("R_gripper2base_4k_1212", R_gripper2base)
np.save("T_gripper2base_4k_1212", T_gripper2base)
R_gripper2base = np.load('R_gripper2base_4k_1212.npy')
T_gripper2base = np.load('T_gripper2base_4k_1212.npy')

# print('R_gripper2base:', R_gripper2base)
# print('T_gripper2base:', T_gripper2base)

# 使用calibrateHandEye()函数求解
R_camera2gripper, T_camera2gripper = cv2.calibrateHandEye(R_gripper2base, T_gripper2base, R_board2camera, T_board2camera, method = None)

# print('R_camera2gripper:', R_camera2gripper)
# print('T_camera2gripper:', T_camera2gripper)

# 合并旋转矩阵和平移向量
R = np.array(R_camera2gripper)
T = np.array(T_camera2gripper)

M_camera2gripper = np.concatenate((R, T), axis=1)
M_camera2gripper = np.concatenate((M_camera2gripper, np.array([[0, 0, 0, 1]])), axis=0)

np.save("R_camera2gripper_4k_1212", R_camera2gripper)
np.save("T_camera2gripper_4k_1212", T_camera2gripper)
np.save("M_camera2gripper_4k_1212", M_camera2gripper)

# 输出结果
print("机械臂到相机的变换矩阵：")
print(M_camera2gripper)
# print(np.linalg.inv(M_camera2gripper))