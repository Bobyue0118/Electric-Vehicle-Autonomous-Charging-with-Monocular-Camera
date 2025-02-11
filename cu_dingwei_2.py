#求z
# -- coding: utf-8 --

import cv2
import numpy as np
from skimage.draw import polygon
from skimage import feature
from math import  *
from copy import deepcopy
from  predict_cu import predict
from PIL import Image

# 粗定位前矫正，使得充电口在视野中央
def cudingwei(img,image_name):
    result=predict(img,image_name)
    print(result)
    lens=len(result)/5
    list_b=[]
    for i in range(int(lens)):
        if result[5*i]=='small':
            tem0=int((result[5*i+2]+result[5*i+4])/2)
            tem1=int((result[5*i+1]+result[5*i+3])/2)
            list_b.append((tem0,tem1))
    if len(list_b)==2:
        box1=list_b[0]
        box2=list_b[1]
        cameraMatrix = np.load(r'parameters/calibration_matrix_4k_1212.npy')
        a = list(box1)
        a.append(1)
        b = list(box2)
        b.append(1)
        fx = cameraMatrix[0, 0] * 2.4 * 1e-6
        # 成像平面到像素
        Rp = np.array([[1000 / 2.4, 0, cameraMatrix[0, 2]], [0, 1000 / 2.4, cameraMatrix[1, 2]], [0, 0, 1]])
        # 像素到成像平面
        Rp_inv = np.linalg.inv(Rp)
        A = Rp_inv @ np.array(a)
        B = Rp_inv @ np.array(b)
        l = abs(A[0] - B[0])
        L = 0.034
        D = L * fx / l * 1e6


        move_z = D - 350

        send_z = "102,3,1000,0,0,%.3f,0,0,0,0,0,0,0" % (move_z)
        return send_z
    else:
        print('找不到1，2号大圆')
        send_z = "102,3,0,0,0,0,0,0,0,0,0,0,0"
        return send_z




if __name__ == '__main__':
    # img = cv2.imread(r"D:\aruco1\aruco\0822\cudingwei_jiaozheng\1023_rgb.png")
    img = Image.open(r"E:\Projects\AutoCharge\0822\1110\all\5_5_5.png")
    image_name='5_5_5.png'
    cudingwei(img,image_name)