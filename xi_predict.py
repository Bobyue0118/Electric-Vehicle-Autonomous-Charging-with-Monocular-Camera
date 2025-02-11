import cv2
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



loc = np.array([  -28.68855812, -26.0513807 , 493.42825546])
ori =  np.array([ 3.12757436 ,-0.01688149 , 0.14208626])
def xi_predict(loc,ori,pose):
    R_object2camera = cv2.Rodrigues(ori)
    M_object2camera = np.r_[np.c_[R_object2camera[0], loc * 1e-3], np.array([[0, 0, 0, 1]])]
    euler_virtual2base1 = rot2euler(R_object2camera[0])
    M_camera2gripper = np.load('parameters/M_camera2gripper_4k_1212.npy')
    M_object2gripper = M_camera2gripper@M_object2camera
    # 6Dposes to M_gripper2base
    # poses_gripper2base = [( 0.182036,0.406740,0.375143,-110.464,-0.224,0.980 )]
    poses_gripper2base = pose

    # poses_gripper2base = [ ( 0.175391,0.402607,0.373370,-110,0,0 )]
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

    M_object2base = M_gripper2base@M_object2gripper
    M_virtual2object = np.load('1018/parameters/M_virtual2object_xi_1018.npy')
    M_virtual2base = M_object2base@ M_virtual2object
    R_virtual2base = M_virtual2base[:3, :3]
    euler_virtual2base = rot2euler(R_virtual2base)
    T_virtual2base = M_virtual2base[:3, 3]
    return T_virtual2base,euler_virtual2base

if __name__ == '__main__':
    loc = np.array([-9.74707091,  -5.30691014, 389.73975052])
    ori = np.array([-3.13443982e+00, -1.96388619e-02, -9.12282928e-04])
    pose = [ ( 0.166856 ,0.491754,0.343027, -116.192, -1.246, -2.258 )]
    print(xi_predict(loc,ori,pose))
