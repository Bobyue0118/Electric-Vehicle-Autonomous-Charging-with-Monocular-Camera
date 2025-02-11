#求xy
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
def cu_jiaozheng_qian(img,image_name):
    result=predict(img,image_name)
    print(result)
    lens=len(result)/5
    list_a=[]
    distance_tem=[]
    for i in range(int(lens)):
        if result[5*i]=='Big':
            tem0=int((result[5*i+2]+result[5*i+4])/2)
            tem1=int((result[5*i+1]+result[5*i+3])/2)
            list_a.append((tem0,tem1))
            distance_tem.append(abs(result[5*i+4]-result[5*i+2]))
    if len(list_a)==1:
        center=list_a[0]
        distance=distance_tem[0]
        ratio = distance / 66

        # # 粗定位时，移动相机使得充电口在视野中央
        move_x = (-2740 + center[0]) / ratio
        move_y = (-1824 + center[1]) / ratio

        send_xy = "102,3,1000,%.3f,%.3f,0,0,0,0,0,0,0,0" % (move_x, move_y)
        return send_xy
    else:
        print('找不到外框')
        send_xy = "102,3,0,0,0,0,0,0,0,0,0,0,0"
        return send_xy






if __name__ == '__main__':
    # img = cv2.imread(r"D:\aruco1\aruco\0822\cudingwei_jiaozheng\1023_rgb.png")
    img = Image.open(r"D:\aruco1\aruco\0822\cudingwei_jiaozheng\1023_rgb.png")
    # image_name='5_5_5.png'
    cu_jiaozheng_qian(img,'1023_rgb_1_1.png')