
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def rot2euler(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = math.atan2(R[1, 0], R[0, 0]) * 180 / np.pi
    else:
        x = math.atan2(-R[1, 2], R[1, 1]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = 0

    return np.array([x, y, z])
T_board2camera_test = np.load('T_board2camera_4k_1212_test.npy')
R_board2camera_test = np.load('R_board2camera_4k_1212_test.npy')
T_board2camera = T_board2camera_test[1]
R_board2camera = R_board2camera_test[1]
M_board2camera = np.concatenate((R_board2camera, T_board2camera), axis=1)
M_board2camera = np.concatenate((M_board2camera, np.array([[0, 0, 0, 1]])), axis=0)
M_board2camera_INV = np.linalg.inv(M_board2camera)
print(M_board2camera @ M_board2camera_INV)
# robot_poses = [( 0.621585,0.193620,0.396984,-170.642 ,8.183,-91.717  )]
# robot_poses = [( 0.546531,0.154810,0.373327,-166.615 ,8.183,-91.717  )]

#720p 21-23
# robot_poses = [(0.679278,0.153706,0.581754,-175.890,1.501,-86.827)]
# robot_poses = [(0.713016,-0.028553,0.567242,-179.603,23.543,-104.563)]

robot_poses =[   (-0.444832,0.069431,-0.730590,14.811,-47.900,-120.305)]



R_gripper2base = []
for x, y, z, rx, ry, rz in robot_poses:
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    R_gripper2base.append(r.as_matrix())

T_gripper2base = []
for i in range(len(robot_poses)):
    tvec = np.array([robot_poses[i][0], robot_poses[i][1], robot_poses[i][2]])
    T_gripper2base.append(tvec)

R_gripper2base = np.array(R_gripper2base)
T_gripper2base = np.array(T_gripper2base)
R_gripper2base = R_gripper2base.reshape(3, 3)
T_gripper2base = T_gripper2base.reshape(3, 1)
M_gripper2base = np.concatenate((R_gripper2base, T_gripper2base), axis=1)
M_gripper2base = np.concatenate((M_gripper2base, np.array([[0, 0, 0, 1]])), axis=0)
M_gripper2base_INV = np.linalg.inv(M_gripper2base)
print(M_gripper2base_INV @ M_gripper2base)

M_camera2gripper = np.load('M_camera2gripper_4k_1212.npy')
M_camera2gripper_INV = np.linalg.inv(M_camera2gripper)
print(M_camera2gripper_INV @ M_camera2gripper)
print(M_board2camera_INV @ M_camera2gripper_INV @ M_gripper2base_INV)
# print(np.linalg.inv(M_board2camera_INV @ M_camera2gripper_INV @ M_gripper2base_INV))
R_base2board = (M_board2camera_INV @ M_camera2gripper_INV @ M_gripper2base_INV)[:3, :3]
euler_base2virtual = rot2euler(R_base2board)
print(euler_base2virtual)
# T_virtual2base = [M_board2camera_INV @ M_camera2gripper_INV @ M_gripper2base_INV][:3, 3]
# print(T_virtual2base)


