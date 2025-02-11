from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

# def euler2rot(euler):
#     r = R.from_euler('xyz', euler, degrees=True)
#     rotation_matrix = r.as_matrix()
#     return rotation_matrix

rotvector = np.array([ [-3.13523199, -0.04427474 ,-0.0090753] ])
R_object2camera = cv2.Rodrigues(rotvector)[0]
T_object2camera=np.array([[ -1.91113821 *1e-3],[  3.65913151  *1e-3],[     389.0899501 *1e-3]])
#[ -9.73491332,  -5.56974882, 389.28851456]

M_object2camera = np.concatenate((R_object2camera, T_object2camera), axis=1)
M_object2camera = np.concatenate((M_object2camera, np.array([[0, 0, 0, 1]])), axis=0)
M_camera2gripper = np.load('1212/parameters/M_camera2gripper_4k_1212.npy')
M_object2gripper=M_camera2gripper@M_object2camera
#6Dposes to M_gripper2base
# poses_gripper2base = [ ( 0.175391,0.402607,0.373370,-110,0,0 )]
poses_gripper2base = [ ( 0.165352 ,0.523510,0.311762, -116.369, -1.311, -1.090 )]
R_gripper2base = []
for x, y, z, rx, ry, rz in poses_gripper2base:
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    R_gripper2base.append(r.as_matrix())

T_gripper2base = []
for i in range(len(poses_gripper2base)):
    tvec = np.array([poses_gripper2base[i][0], poses_gripper2base[i][1], poses_gripper2base[i][2]])
    T_gripper2base.append(tvec)
R_gripper2base = np.array(R_gripper2base)
T_gripper2base = np.array(T_gripper2base)
R_gripper2base = R_gripper2base.reshape(3, 3)
T_gripper2base = T_gripper2base.reshape(3, 1)
M_gripper2base = np.concatenate((R_gripper2base, T_gripper2base), axis=1)
M_gripper2base = np.concatenate((M_gripper2base, np.array([[0, 0, 0, 1]])), axis=0)

M_object2base=M_gripper2base@M_object2gripper
np.save("M_object2base_xi_1018", M_object2base)



poses_virtual2base = [(( 0.171884 ,0.838771,0.288446, -116.369, -1.311, -1.090 ))]
R_virtual2base = []
for x, y, z, rx, ry, rz in poses_virtual2base:
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    R_virtual2base.append(r.as_matrix())

T_virtual2base = []
for i in range(len(poses_virtual2base)):
    tvec = np.array([poses_virtual2base[i][0], poses_virtual2base[i][1], poses_virtual2base[i][2]])
    T_virtual2base.append(tvec)
R_virtual2base = np.array(R_virtual2base)
T_virtual2base = np.array(T_virtual2base)
R_virtual2base = R_virtual2base.reshape(3, 3)
T_virtual2base = T_virtual2base.reshape(3, 1)
M_virtual2base = np.concatenate((R_virtual2base, T_virtual2base), axis=1)
M_virtual2base = np.concatenate((M_virtual2base, np.array([[0, 0, 0, 1]])), axis=0)
# 中心点和虚拟点的变换关系
M_object2base = np.load('M_object2base_xi_1018.npy')
M_virtual2object=np.linalg.inv(M_object2base)@M_virtual2base

np.save('M_virtual2object_xi_1018.npy',M_virtual2object)
print(M_virtual2object)