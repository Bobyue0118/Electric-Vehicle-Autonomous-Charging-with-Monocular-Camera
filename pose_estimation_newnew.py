'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils_aruco import ARUCO_DICT
import argparse
import time
from utils_aruco import ARUCO_DICT, aruco_display
import pyzed.sl as zed
import json
import copy
from PIL import Image
from matplotlib import pyplot as plt
from dvp import *  # 将对应操作系统的dvp.pyd或dvp.so放入python安装目录下的Lib目录或者工程目录
import numpy as np  # 用pip命令安装numpy库
import cv2  # 用pip命令安装opencv-python库

from scipy.spatial.transform import Rotation as R
from  math import sin,cos

from MvCameraControl_class import *
import msvcrt

def frame2mat(frameBuffer):
    frame, buffer = frameBuffer
    bits = np.uint8 if (frame.bits == Bits.BITS_8) else np.uint16
    shape = None
    convertType = None
    if (frame.format >= ImageFormat.FORMAT_MONO and frame.format <= ImageFormat.FORMAT_BAYER_RG):
        shape = 1
    elif (frame.format == ImageFormat.FORMAT_BGR24 or frame.format == ImageFormat.FORMAT_RGB24):
        shape = 3
    elif (frame.format == ImageFormat.FORMAT_BGR32 or frame.format == ImageFormat.FORMAT_RGB32):
        shape = 4
    else:
        return None

    mat = np.frombuffer(buffer, bits)
    mat = mat.reshape(frame.iHeight, frame.iWidth, shape)  # 转换维度
    return mat
def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    frame1=copy.deepcopy(frame)
    gray=frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    Roc=0
    OC=0
    P1=0
    quar=0
    quar=0

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.035, matrix_coefficients,
                                                                       distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error

            # The rotation vector is Rodriguez's angles between the camera and marker center.
            # 根据右手法则，红轴是x，绿轴是y，蓝轴是z
            R = cv2.Rodrigues(rvec)
            # print("id is",ids[i])
            # print("num is",len(ids))
            # print("rotation vec is",rvec)
            # print("translation vec is",tvec)
            # print("rotation matrix is\n",R[0])

            # Camera point = [[R[0],tvec],[0,0,0,1]] * Marker centroid
            #Mark到相机
            Rt = np.r_[np.c_[R[0],tvec[0].T],np.array([[0,0,0,1]])]
            # print("4*4从Marker坐标系到相机坐标系的变换矩阵是\n", Rt, "\n********************************************")
            #物体到mark
            # Rom=np.array([[1,0,0,0.1385],[0,1,0,0.031],[0,0,1,0.01035],[0,0,0,1]])
            # Rom = np.array([[1, 0, 0, 0.115], [0, 1, 0, 0.0785], [0, 0, 1, 0.01035], [0, 0, 0, 1]])
            # Rom = np.array([[1, 0, 0, 0.1145], [0, 1, 0, 0.0795], [0, 0, 1, 0.01035], [0, 0, 0, 1]])
            # Rom = np.array([[1, 0, 0, 0.09], [0, 1, 0, 0.076], [0, 0, 1, 0], [0, 0, 0, 1]])
            # Rom = np.array([[1, 0, 0, 0.12], [0, 1, 0, 0.02975], [0, 0, 1, -0.001], [0, 0, 0, 1]])


            # Rom = np.array([[1, 0, 0, 0.119], [0, 1, 0, 0.0245], [0, 0, 1, 0.0072], [0, 0, 0, 1]])
            # Rom = np.array([[1, 0, 0, 0.118], [0, 1, 0, 0.0175], [0, 0, 1, 0.007], [0, 0, 0, 1]])
            # Rom = np.array([[1, 0, 0, 0], [0, 1, 0, 0.00], [0, 0, 1, 0.0], [0, 0, 0, 1]])
            Rom = np.array([[1, 0, 0, 0.1125], [0, 1, 0, 0.0215], [0, 0, 1, 0.007], [0, 0, 0, 1]])


            Rco = np.array([[1, 0, 0, 0], [0, cos(-20/180*3.1415926), -sin(-20/180*3.1415926), +0.0406*(sin(20/180*3.1415926))], [0, sin(-20/180*3.1415926), cos(-20/180*3.1415926), -0.0416*(1-cos(-20/180*3.1415926))], [0, 0, 0, 1]])

            Rom=Rom@ Rco
            #物体到相机
            Roc=Rt@Rom
            # print("Roc is",Roc)
            # print(Roc[:3,:3])
            quar = Rotation_to_Quarternion(Roc[:3,:3])
            # print("4*4从Marker坐标系到相机坐标系的四元数是\n", quar, "\n********************************************")
            #相机到成像平面q
            # Rf= np.array([[0.003828,0,0,0],[0,0.003828,0,0],[0,0,1,0]])

            # 焦距
            # Rf= np.array([[0.003837,0,0,0],[0,0.003835,0,0],[0,0,1,0]])
            Rf= np.array([[0.0035165737185000003,0,0,0],[0,0.003514988271,0,0],[0,0,1,0]])
            # # 焦距0.0024mm*fx 4K相机
            # Rf= np.array([[0.02629,0,0,0],[0,0.0263499,0,0],[0,0,1,0]])
            # Rf = np.array([[0.01686, 0, 0, 0], [0, 0.01686, 0, 0], [0, 0, 1, 0]])

            #焦距720p
            # Rf= np.array([[0.0038302900,0,0,0],[0,0.00382988395,0,0],[0,0,1,0]])


            #成像平面到像素
            # Rp = np.array([[500000, 0,  1.11609913e+03], [0, 500000, 6.82553672e+02], [0, 0, 1]])

            # Rp = np.array([[250000, 0, 660.10099434], [0, 250000, 355.23392565], [0, 0, 1]])

            Rp = np.array([[289855, 0, 7.29052191e+02], [0,289855, 5.65643620e+02], [0, 0, 1]])

            # Rp = np.array([[416666, 0,  2.84791857e+03], [0, 416666, 1.98002195e+03], [0, 0, 1]])

            # Rp = np.array([[250000, 0, 653.0003357], [0, 250000, 356.52366972], [0, 0, 1]])

            # a=0.0445
            # a = 0.045

            # O=np.array([[-a, a, h, 1],[a, a, h, 1],[a, -a, h, 1],[-a, -a, h, 1],
            #             [-a, a, -h, 1],[a, a, -h, 1],[a, -a, -h, 1],[-a, -a, -h, 1]])
            # O=np.array([[0, 0.021, h, 1],[0.017, 0, h, 1],[0, -0.021, h, 1],[-0.017, 0, h, 1],
            #             [0, 0.021, -h, 1],[0.017, 0, -h, 1],[0, -0.021, -h, 1],[-0.017, 0, -h, 1]])

            # h=0.01035
            # h = 13.1
            h=5
            g=0.3
            O=(np.array([[-17, -1.2+g, 0+h, 1000],[-12, 21+g, -10+h, 1000],[0, 21+g, 0+h, 1000],[12, 21+g, -10+h, 1000],
                        [17, -1.2+g,0+h, 1000],[14.25, -21+g, -10+h, 1000],[0, -21+g, 0+h, 1000],[-14.25,-21+g, -10+h, 1000]]))*1e-3
            OC=Roc@O.T
            OC=OC.T
            OC=OC[:,:3]
            W=np.zeros((8,4))
            for i in range(8):
                W[i]=Rom@O[i].T
                W[i]=W[i].T


            P = np.zeros((8, 3))
            Z = np.zeros(8)

            for i in range(8):
                #Z_c
                Z[i]=(Rt@(W[i].T))[2]

                # print(np.dot(Rp@Rf@Rt,np.array(W[i].T))/Z[i])
                P[i]=np.dot(Rp@Rf@Rt,np.array(W[i].T))/Z[i]


            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            P1=P
            P=P.astype(int)

            # for i in range(3):
            #     cv2.circle(frame, tuple(P[i,:2]), 1, (0, 0, 255))
            #     cv2.line(frame, tuple(P[i,:2]),tuple(P[i+1,:2]), (0, 255, 0), 1)
            # cv2.circle(frame, tuple(P[3, :2]), 1, (0, 0, 255))
            # cv2.line(frame, tuple(P[3, :2]), tuple(P[0, :2]), (0, 255, 0), 1)
            # for i in range(4,7):
            #     cv2.circle(frame, tuple(P[i,:2]), 1, (0, 0, 255))
            #     cv2.line(frame, tuple(P[i,:2]),tuple(P[i+1,:2]), (255, 0, 0), 1)
            # cv2.circle(frame, tuple(P[7, :2]), 1, (0, 0, 255))
            # cv2.line(frame, tuple(P[7, :2]), tuple(P[4, :2]), (255, 0, 0), 1)
            # for i in range(4):
            #     cv2.line(frame, tuple(P[i,:2]),tuple(P[i+4,:2]), (255, 0, 0), 1)
            for i in range(7):
                cv2.circle(frame, tuple(P[i, :2]), 1, (0, 0, 255))
                cv2.line(frame, tuple(P[i, :2]), tuple(P[i + 1, :2]), (0, 255, 0), 1)
            cv2.circle(frame, tuple(P[7, :2]), 1, (0, 0, 255))
            cv2.line(frame, tuple(P[7, :2]), tuple(P[0, :2]), (0, 255, 0), 1)

    return frame,Roc,OC,P1,quar,frame1

def func():
    a = 0
    return a

def Rotation_to_Quarternion(rvec):

    w = np.sqrt(rvec.trace() + 1) / 2
    x = (rvec[2][1] - rvec[1][2]) / (4 * w)
    y = (rvec[0][2] - rvec[2][0]) / (4 * w)
    z = (rvec[1][0] - rvec[0][1]) / (4 * w)
    quar = np.array([w, x, y, z])

    return quar

def image_show(image, name):
    image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    name = str(name)
    cv2.imshow(name, image)
    cv2.imwrite("name.bmp", image)
    k = cv2.waitKey(1) & 0xff
def image_control(data, stFrameInfo):
    print(stFrameInfo.enPixelType)
    if stFrameInfo.enPixelType == 17301505:
        image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
        image_show(image=image, name=stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 17301514:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
        image_show(image=image, name=stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 35127316:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        image_show(image=image, name=stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 34603039:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
        image_show(image=image, name=stFrameInfo.nHeight)
    else:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)
        image_show(image=image, name=stFrameInfo.nHeight)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default= '0717/calibration_matrix_720p_0718.npy' ,help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff",default ="0717/distortion_coefficients_720p_0718.npy",help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())


    if ARUCO_DICT.get(args["type"], None) is None:
        # print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    #标定时注释：
    # k = np.load(calibration_matrix_path)
    # d = np.load(distortion_coefficients_path)



    # cam = zed.Camera()
    # input_type = zed.InputType()
    # init = zed.InitParameters(input_t=input_type)
    # init.camera_resolution = zed.RESOLUTION.HD720# HD720,HD1080,HD2K
    # init.coordinate_units = zed.UNIT.MILLIMETER
    # init.camera_fps = 15
    # cam.open(init)
    # image_size = cam.get_camera_information().camera_resolution
    # image_zed = zed.Mat(image_size.width, image_size.height, zed.MAT_TYPE.U8_C4)
    # num_l = 1
    # num_r = 1

    camera = Camera(0)
    camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ             #启用消除50HZ的频闪
    camera.AeTarget =30    #设置需要调节到的目标亮度
    camera.AeMode = AeMode.AE_MODE_AE_ONLY                  #仅仅自动调节曝光时间
    camera.AeOperation = AeOperation.AE_OP_CONTINUOUS       #启动自动曝光
    camera.Start()  # 启动视频流


    # deviceList = MV_CC_DEVICE_INFO_LIST()
    # tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    # # ch:枚举设备 | en:Enum device
    # ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    # nConnectionNum=0
    # # ch:创建相机实例 | en:Creat Camera Object
    # cam = MvCamera()
    # # ch:选择设备并创建句柄 | en:Select device and create handle
    # stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
    # cam.MV_CC_CreateHandle(stDeviceList)
    #
    # # ch:打开设备 | en:Open device
    # cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

    # ret = cam.MV_CC_SetFloatValue("ExposureTime", 2000)
    # if ret != 0:
    #     print("Set ExposureTime fail! ret[0x%x]" % ret)
    #     sys.exit()


    # ch:开始取流 | en:Start grab image
    # cam.MV_CC_StartGrabbing()
    # stOutFrame = MV_FRAME_OUT()
    # memset(byref(stOutFrame), 0, sizeof(stOutFrame))







    cnt =0



    while True:
        # cam.grab()
        # image_sl_left = zed.Mat()  # left_img
        # cam.retrieve_image(image_sl_left, zed.VIEW.LEFT)
        # image_cv_left = image_sl_left.get_data()
        # image_cv_left = cv2.cvtColor(image_cv_left, 1)


        frame = camera.GetFrame(4000)  # 从相机采集图像数据，超时时间为4000毫秒
        image_cv_left = frame2mat(frame)  # 转换为标准数据格式


        # ret = cam.MV_CC_GetImageBuffer(stOutFrame, 10000)
        # nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        # pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
        # cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
        #                    stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
        # data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
        #                      dtype=np.uint8)
        # data = data.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, -1)
        # image_cv_left = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)







        # output,pose_transform,cuboid,projected_cuboid,quaternion_xyzw,original= pose_esitmation(image_cv_left, aruco_dict_type, k, d)



        cv2.namedWindow("Estimated Pose", 0)
        # cv2.resizeWindow("Estimated Pose", 640,360)
        cv2.resizeWindow("Estimated Pose",685, 456)



        # 标定用:
        cv2.imshow('Estimated Pose', image_cv_left)


        #做数据集用:
        # cv2.imshow('Estimated Pose', image_cv_left)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('l'):
            cnt += 1
            # 标定参数用：
            # cv2.imwrite('1018/xidingwei/' + '{:0>10d}_rgb.png'.format(cnt), image_cv_left)
            # cv2.imwrite('1018/jixian/' + '{:0>10d}_rgb.png'.format(cnt), image_cv_left)
            cv2.imwrite('1212/calibration/' + '{:0>10d}_rgb.png'.format(cnt), image_cv_left)
            # cv2.imwrite('0731/calibration_xi/' + '{:0>10d}_rgb.png'.format(cnt), image_cv_left)
            # cv2.imwrite('0717/biaoding/' + '{:0>10d}_rgb.png'.format(cnt), image_cv_left)
            # cv2.imwrite('cudingwei/' + '{:0>10d}_rgb.png'.format(cnt), image_cv_left)
            # cv2.imwrite('0609
            # /object2base/'+'{:0>10d}_rgb_blue.png'.format(cnt), output)
            # cv2.imwrite('0609/object2base/' + '{:0>10d}_rgb.png'.format(cnt), original)


            # 保存数据集：720p
            # cv2.imwrite('720p_blue/'+'{:0>10d}_rgb.png'.format(cnt), output)
            # cv2.imwrite('720p/' + '{:0>10d}_rgb.png'.format(cnt), original)

            # 保存数据集：2k
            # cv2.imwrite('xidingwei_blue/'+'{:0>10d}_rgb.png'.format(cnt), output)
            # cv2.imwrite('xidingwei/' + '{:0>10d}_rgb.png'.format(cnt), original)


            # a={"objects":[{}]}
            # a["objects"][0]['class']='anmuxi'
            # a["objects"][0]['cuboid'] = cuboid.tolist()
            # a["objects"][0]['cuboid_centroid'] = cuboid.mean(0).tolist()
            # a["objects"][0]['label']=13
            # a["objects"][0]['location']=pose_transform.T[3][:3].tolist()
            # a["objects"][0]['num_pixels'] = 4656.0
            # a["objects"][0]['pose_transform'] = pose_transform.tolist()
            # a["objects"][0]['projected_cuboid'] = projected_cuboid[:,:2].tolist()
            # a["objects"][0]['projected_cuboid_centroid'] = projected_cuboid[:,:2].mean(0).tolist()
            # a["objects"][0]['quaternion_xyzw']=quaternion_xyzw.tolist()

            # with open('720p/'+'{:0>10d}_rgb.json'.format(cnt), 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(a, ensure_ascii=False, indent=4))



            # 保存720pjson
            # with open('720p/'+'{:0>10d}_rgb.json'.format(cnt), 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(a, ensure_ascii=False, indent=4))
            # 保存2k json
            # with open('xidingwei/'+'{:0>10d}_rgb.json'.format(cnt), 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(a, ensure_ascii=False, indent=4))


            #
            # M_object2camera=pose_transform
            #
            # M_camera2gripper = np.load('0609/M_camera2gripper_0609.npy')
            # M_object2gripper=M_camera2gripper@M_object2camera
            # #6Dposes to M_gripper2base
            # poses_gripper2base = [ ( -0.598726,0.037771,0.038589,-89.527,0.001,93.3 )]
            # R_gripper2base = []
            # for x, y, z, rx, ry, rz in poses_gripper2base:
            #     r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
            #     R_gripper2base.append(r.as_matrix())
            #
            # T_gripper2base = []
            # for i in range(len(poses_gripper2base)):
            #     tvec = np.array([poses_gripper2base[i][0], poses_gripper2base[i][1], poses_gripper2base[i][2]])
            #     T_gripper2base.append(tvec)
            # R_gripper2base = np.array(R_gripper2base)
            # T_gripper2base = np.array(T_gripper2base)
            # R_gripper2base = R_gripper2base.reshape(3, 3)
            # T_gripper2base = T_gripper2base.reshape(3, 1)
            # M_gripper2base = np.concatenate((R_gripper2base, T_gripper2base), axis=1)
            # M_gripper2base = np.concatenate((M_gripper2base, np.array([[0, 0, 0, 1]])), axis=0)
            #
            # M_object2base=M_gripper2base@M_object2gripper
            # np.save('M_object2base_0609.npy', M_object2base)





    # video.release()
    cv2.destroyAllWindows()