#!/usr/bin/env python
# -- coding: utf-8 --
# @Time   : 2023/04/04 12:32
# @Author : Yue Bo
# @File   : udp_client.py
import socket
import  copy
from dvp import *  # 将对应操作系统的dvp.pyd或dvp.so放入python安装目录下的Lib目录或者工程目录
import numpy as np  # 用pip命令安装numpy库
import cv2  # 用pip命令安装opencv-python库
from cu_jiaozheng_qian_2 import  *
from cu_dingwei_2 import  *
from xi_jiaozheng_0 import *
from xi_jiaozheng_1 import  *
from xi_dingwei import *
from cu_predict import *
from xi_predict import *
import time
import datetime
from PIL import Image

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

port = 5
Server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

ip_port=("192.168.1.85",port)
# print(s.getpeername())
Server.bind(ip_port)
Server.listen(5)



while True:

    socket, addr = Server.accept()
    remsg = socket.recv(1024).decode('utf-8')
    print(remsg )
    remsg = remsg.split(',')
    if remsg[0] == '101' and remsg[1] == '3':
        camera = Camera(0)
        camera.TriggerState = False
        camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ  # 启用消除50HZ的频闪
        camera.AeTarget =25# 设置需要调节到的目标亮度
        camera.AeMode = AeMode.AE_MODE_AE_ONLY  # 仅仅自动调节曝光时间
        camera.AeOperation = AeOperation.AE_OP_CONTINUOUS

        # camera.SaveConfig("example.ini")
        camera.Start()  # 启动视频流
        time.sleep(1)
        print('相机已启动_粗定位校正')
        msg = "101,3,1000,0,0,0,0,0,0,0,0,0,0"
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '102' and remsg[1] == '3':

        frame = camera.GetFrame(4000)  # 从相机采集图像数据，超时时间为4000毫秒
        image_cv_left = frame2mat(frame)
        current_time=datetime.datetime.now()
        hour=current_time.hour
        minute=current_time.minute
        cv2.imwrite('0822/cudingwei_jiaozheng/' + '1023_rgb_{}_{}.png'.format(hour,minute), image_cv_left)
        camera.Stop()
        camera.Close()


        img =  Image.open('0822/cudingwei_jiaozheng/' + '1023_rgb_{}_{}.png'.format(hour,minute))
        img2 =  Image.open('0822/cudingwei_jiaozheng/' + '1023_rgb_{}_{}.png'.format(hour,minute))
        # msg = cu_jiaozheng(img, img2)
        msg=cu_jiaozheng_qian(img, '1023_rgb_{}_{}.png'.format(hour,minute))  ### 传入图片名
        print(msg)
        # msg = "102,3,1000,%.3f,%.3f,0,0,0,0,0,0,0,0" % (0, 0)
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '101' and remsg[1] == '1':
        camera = Camera(0)
        camera.TriggerState = False
        camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ  # 启用消除50HZ的频闪
        camera.AeTarget =30# 设置需要调节到的目标亮度
        camera.AeMode = AeMode.AE_MODE_AE_ONLY  # 仅仅自动调节曝光时间
        camera.AeOperation = AeOperation.AE_OP_CONTINUOUS
        # camera.SaveConfig("example.ini")
        camera.Start()  # 启动视频流
        time.sleep(1)
        print('相机已启动_粗定位')
        msg = "101,1,1000,0,0,0,0,0,0,0,0,0,0"
        socket.send(msg.encode('utf-8'))
        continue

    elif remsg[0] == '102' and remsg[1] == '1':
        pose = []
        temp = []
        for i in range(3):
            temp.append(float(remsg[2 + i])/1000)
        for i in range(3,6):
            temp.append(float(remsg[2 + i]) )
        temp = tuple(temp)
        pose.append(temp)
        frame = camera.GetFrame(4000)  # 从相机采集图像数据，超时时间为4000毫秒
        image_cv_left = frame2mat(frame)

        cv2.imwrite('0822/cudingwei/' + '1021_rgb_{}_{}.png'.format(hour,minute), image_cv_left)
        img = Image.open('0822/cudingwei/' + '1021_rgb_{}_{}.png'.format(hour,minute))
        img2 = Image.open('0822/cudingwei/' + '1021_rgb_{}_{}.png'.format(hour,minute))
        camera.Stop()
        camera.Close()
        msg = cudingwei(img, '1023_rgb_{}_{}.png'.format(hour, minute)) ### 传入图片名
        print(msg)
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '101' and remsg[1] == '5':
        camera = Camera(0)
        camera.TriggerState = False
        camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ  # 启用消除50HZ的频闪
        camera.AeTarget =25# 设置需要调节到的目标亮度
        camera.AeMode = AeMode.AE_MODE_AE_ONLY  # 仅仅自动调节曝光时间
        camera.AeOperation = AeOperation.AE_OP_CONTINUOUS

        # camera.SaveConfig("example.ini")
        camera.Start()  # 启动视频流
        time.sleep(1)
        print('相机已启动_细定位校正0')
        msg = "101,5,1000,0,0,0,0,0,0,0,0,0,0"
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '102' and remsg[1] == '5':

        frame = camera.GetFrame(4000)  # 从相机采集图像数据，超时时间为4000毫秒
        image_cv_left = frame2mat(frame)
        current_time=datetime.datetime.now()
        cv2.imwrite('0822/xidingwei_jiaozheng0/' + '1023_rgb_{}_{}.png'.format(hour,minute), image_cv_left)
        camera.Stop()
        camera.Close()
        img = Image.open('0822/xidingwei_jiaozheng0/' + '1023_rgb_{}_{}.png'.format(hour,minute))
        img2 = Image.open('0822/xidingwei_jiaozheng0/' + '1023_rgb_{}_{}.png'.format(hour,minute))
        # msg = cu_jiaozheng(img, img2)
        msg=xi_jiaozheng_0(img, '1023_rgb_{}_{}.png'.format(hour,minute))  ### 传入图片名
        # msg = "102,5,1000,%.3f,%.3f,0,0,0,0,0,0,0,0" % (0, 0)
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '101' and remsg[1] == '4':
        camera = Camera(0)
        camera.TriggerState = False
        camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ  # 启用消除50HZ的频闪
        camera.AeTarget = 40  # 设置需要调节到的目标亮度
        camera.AeMode = AeMode.AE_MODE_AE_ONLY  # 仅仅自动调节曝光时间
        camera.AeOperation = AeOperation.AE_OP_CONTINUOUS
        camera.Start()  # 启动视频流
        time.sleep(5)
        print('相机已启动_细定位校正1')
        msg = "101,4,1000,0,0,0,0,0,0,0,0,0,0"
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '102' and remsg[1] == '4':
        pose = []
        temp = []
        for i in range(3):
            temp.append(float(remsg[2 + i])/1000)
        for i in range(3,6):
            temp.append(float(remsg[2 + i]) )
        temp = tuple(temp)
        pose.append(temp)

        frame = camera.GetFrame(4000)  # 从相机采集图像数据，超时时间为4000毫秒
        image_cv_left = frame2mat(frame)
        cv2.imwrite('0822/xidingwei_jiaozheng1/' + '1024_rgb_{}_{}.png'.format(hour,minute), image_cv_left)
        image_cv_left = Image.open('0822/xidingwei_jiaozheng1/' + '1024_rgb_{}_{}.png'.format(hour,minute))
        camera.Stop()
        camera.Close()
        img = copy.deepcopy(image_cv_left)
        img2 = copy.deepcopy(image_cv_left)
        loc,ori= xi_jiaozheng_1(img, img2,'1024_rgb_{}_{}.png'.format(hour,minute))
        T_virtual2base, euler_virtual2base = cu_predict(loc, ori, pose)
        print(T_virtual2base)
        print(euler_virtual2base)
        msg = "102,4,1000,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,0,0,0,0" % (
            T_virtual2base[0] * 1000, T_virtual2base[1] * 1000, T_virtual2base[2] * 1000, euler_virtual2base[0],
            euler_virtual2base[1],
            euler_virtual2base[2])
        # msg = "102,4,1000,%.3f,%.3f,0,0,0,0,0,0,0,0" % (0, 0)
        print('细定位矫正1结果')
        print(msg)
        socket.send(msg.encode('utf-8'))
        socket.close()
        continue

    elif remsg[0] == '101' and remsg[1] == '2':
        camera = Camera(0)
        camera.TriggerState = False
        camera.AntiFlick = AntiFlick.ANTIFLICK_50HZ  # 启用消除50HZ的频闪
        camera.AeTarget = 40  # 设置需要调节到的目标亮度
        camera.AeMode = AeMode.AE_MODE_AE_ONLY  # 仅仅自动调节曝光时间
        camera.AeOperation = AeOperation.AE_OP_CONTINUOUS
        camera.Start()  # 启动视频流
        time.sleep(5)
        print('相机已启动_细定位')
        msg = "101,1,1000,0,0,0,0,0,0,0,0,0,0"
        socket.send(msg.encode('utf-8'))
        continue

    elif remsg[0] == '102' and remsg[1] == '2':
        pose = []
        temp = []
        for i in range(3):
            temp.append(float(remsg[2 + i]) / 1000)
        for i in range(3, 6):
            temp.append(float(remsg[2 + i]))
        temp = tuple(temp)
        pose.append(temp)

        frame = camera.GetFrame(4000)  # 从相机采集图像数据，超时时间为4000毫秒
        image_cv_left = frame2mat(frame)
        cv2.imwrite('0822/xidingwei/' + '1022_rgb_{}_{}.png'.format(hour,minute), image_cv_left)
        img = Image.open('0822/xidingwei/' + '1022_rgb_{}_{}.png'.format(hour,minute))
        img2 = Image.open('0822/xidingwei/' + '1022_rgb_{}_{}.png'.format(hour,minute))
        camera.Stop()
        camera.Close()
        loc, ori = xi_dingwei(img, img2,'1022_rgb_{}_{}.png'.format(hour,minute))
        print('细定位结果')
        print(loc,ori)
        T_virtual2base, euler_virtual2base = xi_predict(loc, ori, pose)
        print(T_virtual2base,euler_virtual2base)
        msg = "102,2,1000,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,0,0,0,0" % (
            T_virtual2base[0]*1000, T_virtual2base[1]*1000, T_virtual2base[2]*1000, euler_virtual2base[0], euler_virtual2base[1],
            euler_virtual2base[2])
        socket.send(msg.encode('utf-8'))
        continue





#conn.close()
# s.sendall("111111".encode())
# reply=s.recv(1024)
# print(reply.decode())
# s.close()
#f = open("result.txt",encoding="utf-8")
#s.sendto(f.read().encode(),(host,port))