#求xy
# -- coding: utf-8 --

import cv2
import numpy as np
from skimage.draw import polygon
from skimage import feature
from math import  *
from copy import deepcopy
from  predict_xi import predict
from PIL import Image

# 粗定位前矫正，使得充电口在视野中央
def xi_jiaozheng_0(img,image_name):
    result=predict(img,image_name)
    print(result)
    lens=len(result)/6
    list_a=[]
    for i in range(int(lens)):
        if result[6*i]=='a':
            tem0=int((result[6*i+2]+result[6*i+4])/2)
            tem1=int((result[6*i+1]+result[6*i+3])/2)
            list_a.append((tem0,tem1))
    if len(list_a)==2:
        box1=list_a[0]
        box2=list_a[1]
        distance = np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)
        ratio = distance / 34
        center = []
        center.append(int((box1[0] + box2[0]) / 2))
        center.append(int((box1[1] + box2[1]) / 2))
        move_x = (-2740 + center[0]) / ratio
        move_y = (-1824 + center[1]) / ratio

        send_xy = "102,5,1000,%.3f,%.3f,0,0,0,0,0,0,0,0" % (move_x, move_y)

        return send_xy
    else:
        print('找不到1，2号大圆')
        send_xy = "102,5,0,0,0,0,0,0,0,0,0,0,0"

        return send_xy








if __name__ == '__main__':
    # img = cv2.imread(r"D:\aruco1\aruco\0822\cudingwei_jiaozheng\1023_rgb.png")
    img = Image.open(r"E:\Projects\AutoCharge\0822\1110\all\5_5_5.png")
    image_name='5_5_5.png'
    xi_jiaozheng_0(img,image_name)