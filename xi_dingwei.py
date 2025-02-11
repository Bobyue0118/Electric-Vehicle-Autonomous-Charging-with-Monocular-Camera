#求位姿
# -- coding: utf-8 --

import cv2
from copy import deepcopy
import numpy as np
from skimage.draw import polygon
from skimage import feature
import datetime
from math import *
import time
import os
from PIL import Image
from predict_xi import  predict


def dist(pointa, pointb):
    distance = np.sqrt((pointa[0] - pointb[0])**2 + (pointa[1] - pointb[1])**2)
    return distance

def get_png_paths(folder_path):
    png_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                file_name = os.path.splitext(relative_path)[0]  # 去除文件后缀
                png_paths.append(file_name)

    return png_paths

# 保存图像函数，自动创建文件夹
def save_image_with_folder(image_path, image):
    folder_path = os.path.dirname(image_path)
    os.makedirs(folder_path, exist_ok=True)  # 创建文件夹，如果已存在则不会重复创建
    cv2.imwrite(image_path, image)

# 12圆排序
def order12(box1, box2):
    if box2[0] < box1[0]:
        box = box2
        box2 = box1
        box1 = box
    return box1, box2

# 4-9号圆排序
def order4to9(box4, box5, box6, box7, box8, box9, one1, two1):
    four1 = None
    five1 = None
    six1 = None
    seven1 = None
    eight1 = None
    nine1 = None

    if two1[0] > one1[0]:  # 还未区分1号圆和2号圆
        pass
    else:
        intermediate = two1
        two1 = one1
        one1 = intermediate
    theta = np.arctan((two1[1] - one1[1]) / (two1[0] - one1[0]))
    center = ((one1[0]+two1[0])/2, (one1[1]+two1[1])/2)
    ratio = (two1[0]-one1[0])/34

    box4to9 = [box4, box5, box6, box7, box8, box9]
    while None in box4to9:
        box4to9.remove(None)
    circle78 = [i for i in box4to9 if i[1] > center[1]]
    circle4569 = [i for i in box4to9 if i[1] < center[1]]
    if len(circle78) != 2: print("78圆yolo识别出现少圆")
    if len(circle4569) != 4: print("4569圆yolo识别出现少圆")

    if len(circle78) == 2:
        if circle78[0][0] < circle78[1][0]:
            seven1 = (circle78[0][0], circle78[0][1])
            eight1 = (circle78[1][0], circle78[1][1])
        else:
            seven1 = (circle78[1][0], circle78[1][1])
            eight1 = (circle78[0][0], circle78[0][1])
    elif len(circle78) == 1:
        if circle78[0][0] < center[0]:
            seven1 = (circle78[0][0], circle78[0][1])
            print("8号圆yolo识别失败")
        else:
            eight1 = (circle78[0][0], circle78[0][1])
            print("7号圆yolo识别失败")

    x = 6000
    y = -1
    if len(circle4569) == 4:
        for circle in circle4569:
            if circle[0] < x:
                five1 = circle
                x = circle[0]
            if circle[0] > y:
                six1 = circle
                y = circle[0]

        circle49 = [i for i in circle4569 if i != five1 and i != six1]
        if len(circle49) != 2: print("49圆识别错误")
        if circle49[0][1] < circle49[1][1]:
            four1 = (circle49[0][0], circle49[0][1])
            nine1 = (circle49[1][0], circle49[1][1])
        else:
            four1 = (circle49[1][0], circle49[1][1])
            nine1 = (circle49[0][0], circle49[0][1])
    elif len(circle4569) == 3:
        print('theta',theta)
        four11 = (center[0] + ratio * (22.2) * sin(theta), center[1] - ratio * (22.2) * cos(theta))
        x = 6000
        for i in circle4569:
            print('dist(i, four11)',dist(i, four11))
            if dist(i, four11) < x and dist(i, four11) < 90:
                x = dist(i, four11)
                four1 = (i[0], i[1])
        print('four1',four1)
        if four1 != None:
            circle4569.remove(four1)
            alpha = np.arctan((circle4569[0][1]-circle4569[1][1])/(circle4569[0][0]-circle4569[1][0]))
            if abs(alpha-theta) < 0.15:# 说明是5号圆和6号圆
                if circle4569[0][0] < circle4569[1][0]:
                    five1 = (circle4569[0][0],circle4569[0][1])
                    six1 = (circle4569[1][0],circle4569[1][1])
                else:
                    five1 = (circle4569[1][0], circle4569[1][1])
                    six1 = (circle4569[0][0], circle4569[0][1])
            else:# 说明包含9号圆
                if circle4569[0][1] > circle4569[1][1]:
                    nine1 = (circle4569[0][0], circle4569[0][1])
                else:
                    nine1 = (circle4569[1][0], circle4569[1][1])
                circle4569.remove(nine1)
                if circle4569[0][0] < center[0]:
                    five1 = (circle4569[0][0],circle4569[0][1])
                else: six1 = (circle4569[0][0], circle4569[0][1])
        else:
            x = 6000
            for circle in circle4569:
                if circle[0] < x:
                    five1 = circle
                    x = circle[0]
                if circle[0] > y:
                    six1 = circle
                    y = circle[0]
            circle4569.remove(five1)
            circle4569.remove(six1)
            nine1 = circle4569[0]
    elif len(circle4569) == 2 or len(circle4569) == 1:
        four11 = (center[0] + ratio * 0 * cos(theta) - ratio * (-21) * sin(theta), center[1] + ratio * 0 * sin(theta) + ratio * (-21) * cos(theta))
        x = 6000
        for i in circle4569:
            #print('dist(i, four1)',dist(i, four1))
            if dist(i, four11) < x and dist(i, four11) < 30: #可能阈值设置的过于严苛，导致4圆没选上
                x = dist(i, four11)
                four1 = (i[0],i[1])

    return four1, five1, six1, seven1, eight1, nine1


def dist(pointa, pointb):
    distance = np.sqrt((pointa[0] - pointb[0])**2 + (pointa[1] - pointb[1])**2)
    return distance


# *******************************************************主函数：细定位***************************************************************** #
def xi_dingwei(img, img2, name):
    image_name = name+'.png'
    result=predict(img,image_name)
    print("yolo识别输出：", result)
    lens=len(result)/6
    list_a=[]
    list_b=[]
    list_c=[]
    dict_c={}
    threshold_of_yolo_small = 0.8
    for i in range(int(lens)):
        if result[6*i]=='a':
            tem0=int((result[6*i+2]+result[6*i+4])/2)
            tem1=int((result[6*i+1]+result[6*i+3])/2)
            list_a.append((tem0,tem1))
        elif result[6*i]=='b':
            tem0=int((result[6*i+2]+result[6*i+4])/2)
            tem1=int((result[6*i+1]+result[6*i+3])/2)
            list_b.append((tem0,tem1))
        else:
            #print("prob",result[6*i+5])
            if result[6*i+5]>threshold_of_yolo_small:
                tem0=int((result[6*i+2]+result[6*i+4])/2)
                tem1=int((result[6*i+1]+result[6*i+3])/2)
                list_c.append((tem0,tem1))
                dict_c[result[6*i+5]]=(tem0,tem1)
    if len(list_c) < 6:
        num_of_term = 6-len(list_c)
        print("满足阈值：{}的456789圆数量为{}".format(threshold_of_yolo_small,6-num_of_term))
    elif len(list_c) > 6:
        list_c=[]
        key_of_dict_c = sorted(dict_c,reverse=True)
        for i in range(6):
            list_c.append(dict_c[key_of_dict_c[i]])
        print("满足阈值：{}的456789圆数量已被缩减为{}".format(threshold_of_yolo_small,len(list_c)))
        #print(key_of_dict_c)

    if len(list_a)==2:
        box1=list_a[0]
        box2=list_a[1]
    else:
        print('找不到1，2号大圆')
    if len(list_b)==1:
        box3=list_b[0]
    else:
        print('找不到3号圆')
    if len(list_c)==6:
        box4 = list_c[0]
        box5 = list_c[1]
        box6 = list_c[2]
        box7 = list_c[3]
        box8 = list_c[4]
        box9 = list_c[5]
    elif len(list_c)==5:
        box4 = list_c[0]
        box5 = list_c[1]
        box6 = list_c[2]
        box7 = list_c[3]
        box8 = list_c[4]
        box9 = None
    elif len(list_c)==4:
        box4 = list_c[0]
        box5 = list_c[1]
        box6 = list_c[2]
        box7 = list_c[3]
        box8 = None
        box9 = None
    elif len(list_c) == 3:
        box4 = list_c[0]
        box5 = list_c[1]
        box6 = list_c[2]
        box7 = None
        box8 = None
        box9 = None
    elif len(list_c) == 2:
        box4 = list_c[0]
        box5 = list_c[1]
        box6 = None
        box7 = None
        box8 = None
        box9 = None

    # 得到9个圆的锚点, box1, ..., box9
    one1, two1 = order12(box1, box2)

    three1 = (box3[0], box3[1])

    four1, five1, six1, seven1, eight1, nine1 = order4to9(box4, box5, box6, box7, box8, box9, one1, two1)
    img = np.array(img)  # 先转换为数组   H W C
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2 = np.array(img2)  # 先转换为数组   H W C
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)



    center12 = ((one1[0]+two1[0])/2, (one1[1]+two1[1])/2)
    ratio_s = dist(one1, two1)/34

    # 仅保留充电口区域
    rr, cc = polygon(np.array([center12[0]-35*ratio_s, center12[0]-35*ratio_s, center12[0]+35*ratio_s, center12[0]+35*ratio_s]),
                     np.array([center12[1]+32*ratio_s, center12[1]-32*ratio_s, center12[1]-32*ratio_s, center12[1]+32*ratio_s]))  # 30 -> 32
    mask = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
    mask[rr, cc] = 1
    match = cv2.bitwise_and(img2, img2, mask=mask.T)  # 得到原图像上的ROI

    # 成功匹配之后，图像圆检测
    match2 = deepcopy(match)
    prepare = cv2.cvtColor(match, cv2.COLOR_RGB2GRAY)
    prepare = cv2.GaussianBlur(prepare, (3, 3), 1.3)
    # prepare = cv2.medianBlur(prepare, 3)
    x = cv2.Sobel(prepare, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(prepare, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    sob = cv2.addWeighted(absX, 1, absY, 1, 0)

    imgPoints = []

    one = (0, 0)
    two = (0, 0)
    three = (0, 0)
    four = (0, 0)
    five = (0, 0)
    six = (0, 0)
    seven = (0, 0)
    eight = (0, 0)
    nine = (0, 0)
    num_detected_1 = 0
    num_detected_2 = 0
    num_detected_3 = 0
    num_detected_4 = 0
    num_detected_5 = 0
    num_detected_6 = 0
    num_detected_7 = 0
    num_detected_8 = 0
    num_detected_9 = 0
    r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = 0
    cnt = False

    # 画出九个圆的锚点
    cv2.circle(match, tuple(np.int0(one1)), 7, (0, 255, 0), 15)
    cv2.putText(match, '1', tuple(np.int0(one1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.circle(match, tuple(np.int0(two1)), 7, (0, 255, 0), 15)
    cv2.putText(match, '2', tuple(np.int0(two1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.circle(match, tuple(np.int0(three1)), 7, (0, 255, 0), 15)
    cv2.putText(match, '3', tuple(np.int0(three1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    if four1!=None:
        cv2.circle(match, tuple(np.int0(four1)), 7, (0, 255, 0), 15)
        cv2.putText(match, '4', tuple(np.int0(four1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    if five1!=None:
        cv2.circle(match, tuple(np.int0(five1)), 7, (0, 255, 0), 15)
        cv2.putText(match, '5', tuple(np.int0(five1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    if six1!=None:
        cv2.circle(match, tuple(np.int0(six1)), 7, (0, 255, 0), 15)
        cv2.putText(match, '6', tuple(np.int0(six1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    if seven1!=None:
        cv2.circle(match, tuple(np.int0(seven1)), 7, (0, 255, 0), 15)
        cv2.putText(match, '7', tuple(np.int0(seven1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    if eight1!=None:
        cv2.circle(match, tuple(np.int0(eight1)), 7, (0, 255, 0), 15)
        cv2.putText(match, '8', tuple(np.int0(eight1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)
    if nine1!=None:
        cv2.circle(match, tuple(np.int0(nine1)), 7, (0, 255, 0), 15)
        cv2.putText(match, '9', tuple(np.int0(nine1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)

    # cv2.namedWindow("Detected Circle", 0)
    # cv2.resizeWindow("Detected Circle", 685, 456)
    # cv2.imshow("Detected Circle", match)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for threshold in [250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 75, 70,
                      65, 60, 55, 50, 45, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if num_detected_1 == 1 and num_detected_2 == 1:
            break
        ret, thresh1 = cv2.threshold(sob, threshold, 255, cv2.THRESH_BINARY)  # 阈值在这里
        print("12圆阈值是", ret)

        if num_detected_1 == 0 or num_detected_2 == 0:
            p2 = 30
            min = 100
            max = 150

        detect_circle = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=50,
                                         param2=p2, minRadius=min, maxRadius=max)

        if detect_circle is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.int0(detect_circle)

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # 检测点1和点2
                if one == (0, 0) and abs(a - one1[0]) < 50 and abs(
                        b - one1[1]) < 50 and 6.1 * ratio_s - 18 < r < 6.1 * ratio_s + 18:  #20,30 -> 18
                    print('6.1*ratio_s:', 6.1 * ratio_s)
                    one = (a, b)
                    num_detected_1 = 1
                    r1 = r
                    print("banjing1", r, abs(a - one1[0]),a,b)
                    cv2.circle(match, (a, b), r, (0, 255, 0), 5)
                    cv2.circle(match, (a, b), 3, (0, 0, 255), 15)
                    #cv2.circle(match, tuple(np.int0(one1)), 3, (255, 0, 0), 5)
                    #cv2.putText(match, '1', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    continue
                if two == (0, 0) and abs(a - two1[0]) < 50 and abs(
                        b - two1[1]) < 50 and 6.1 * ratio_s - 18 < r < 6.1 * ratio_s + 18:  #20,30 -> 18
                    two = (a, b)
                    print("banjing2", r, abs(a - two1[0]),a,b)
                    num_detected_2 = 1
                    r2 = r
                    cv2.circle(match, (a, b), r, (0, 255, 0), 5)
                    cv2.circle(match, (a, b), 3, (0, 0, 255), 15)
                    #cv2.circle(match, tuple(np.int0(two1)), 3, (255, 0, 0), 5)
                    #cv2.putText(match, '2', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    continue

    for threshold in [250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 95, 90, 85, 80,
                      75, 70, 65, 60, 55, 50, 45, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:

        if num_detected_3 == 1:
            break
        ret, thresh1 = cv2.threshold(sob, threshold, 255, cv2.THRESH_BINARY)  # 阈值在这里
        print("3圆阈值是", ret)

        # p2 = 50
        # min = 100
        # max = 170

        p2 = 30
        min = 40
        max = 75

        detect_circle = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=50,
                                         param2=p2, minRadius=min, maxRadius=max)

        if detect_circle is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.int0(detect_circle)

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # 检测点3
                if one != (0, 0) and two != (0, 0):   # 与1、2同面，所以限制得更加严苛
                    if three == (0, 0) and abs(a - three1[0]) < 50 and abs(
                            b - three1[1]) < 50 and ratio_s * 3.3 - 18 < r < ratio_s * 3.3 + 18:
                        print('ratio_s*3.3:', ratio_s*3.3)
                        three = (a, b)
                        print("banjing3", r, abs(b - three1[1]),a,b)
                        num_detected_3 = 1
                        r3 = r
                        cv2.circle(match, (a, b), r, (0, 255, 0), 5)
                        cv2.circle(match, (a, b), 3, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(three1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '3', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue

    for threshold in [250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 95, 90, 85, 80,
                      75, 70, 65, 60, 55, 50, 45, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:

        ret, thresh1 = cv2.threshold(sob, threshold, 255, cv2.THRESH_BINARY)  # 阈值在这里
        print("456789圆阈值是", ret)
        if (num_detected_3 == 1) and ((num_detected_4 == 1)+(four1==None)) and ((num_detected_5 == 1)+(five1==None)) and ((num_detected_6 == 1)+(six1==None)) and ((num_detected_7 == 1)+(seven1==None)) and ((num_detected_8 == 1)+(eight1==None)) and ((num_detected_9 == 1)+(nine1==None)):  # early stop
            break

        p2 =50
        min = 20
        max = 100
        # else:
        #     p2=30
        #     min=10
        #     max=100

        detect_circle = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=50,
                                         param2=p2, minRadius=min, maxRadius=max)

        if detect_circle is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.int0(detect_circle)

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # 检测点4
                if one != (0, 0) and two != (0, 0):  # 与1、2同面，所以限制得更加严苛
                    if four1!=None and four == (0, 0) and abs(a - four1[0]) < 50 and abs(
                            b - four1[1]) < 50 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                        print('ratio_s * 4.55:',ratio_s * 4.55)
                        four = (a, b)
                        num_detected_4 = 1
                        r4 = r
                        print("banjing4", r, abs(a - four1[0]),a,b)
                        cv2.circle(match, (a, b), r, (0, 255, 0), 5)
                        cv2.circle(match, (a, b), 3, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(four1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '4', (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue

                # 检测点5和点6
                if one != (0, 0) and two != (0, 0):
                    if five1!=None and five == (0, 0) and abs(a - five1[0]) < 50 and abs(
                            b - five1[1]) < 50 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                        five = (a, b)
                        print('ratio_s * 4.55:', ratio_s * 4.55)
                        print("banjing5", r, abs(a - five1[0]),a,b)
                        num_detected_5 = 1
                        r5 = r
                        cv2.circle(match, five, r5, (0, 255, 0), 5)
                        cv2.circle(match, five, 3, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(five1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '5', five, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue
                    if six1!=None and six == (0, 0) and abs(a - six1[0]) < 50 and abs(
                            b - six1[1]) < 50 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                        # print(ratio_s*4.2)
                        six = (a, b)
                        print("banjing6", r, abs(a - six1[0]),a,b)
                        num_detected_6 = 1
                        r6 = r
                        cv2.circle(match, six, r6, (0, 255, 0), 5)
                        cv2.circle(match, six, 3, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(six1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '6', six, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue

                # 检测点7和点8
                if one != (0, 0) and two != (0, 0):
                    if seven1!=None and seven == (0, 0) and abs(a - seven1[0]) < 50 and abs(
                            b - seven1[1]) < 50 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                        seven = (a, b)
                        print("banjing7", r, abs(a - seven1[0]),a,b)
                        num_detected_7 = 1
                        r7 = r
                        cv2.circle(match, seven, r7, (0, 255, 0), 5)
                        cv2.circle(match, seven, 1, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(seven1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '7', seven, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue
                    if eight1!=None and eight == (0, 0) and abs(a - eight1[0]) < 50 and abs(
                            b - eight1[1]) < 50 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                        eight = (a, b)
                        print("banjing8", r, abs(a - eight1[0]),a,b)
                        num_detected_8 = 1
                        r8 = r
                        cv2.circle(match, eight, r8, (0, 255, 0), 5)
                        cv2.circle(match, eight, 1, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(eight1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '8', eight, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue

                # 检测点9
                if one != (0, 0) and two != (0, 0):
                    if nine1!=None and nine == (0, 0) and abs(a - nine1[0]) < 50 and abs(
                            b - nine1[1]) < 50 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                    # if nine == (0, 0) and (((a - nine1[0] > -50)and(a - nine1[0] < 0)) or((a - nine1[0] > 0)and(a - nine1[0] < 100))) and abs(
                    #         b - nine1[1]) < 100 and ratio_s * 4.55 - 18 < r < ratio_s * 4.55 + 18:
                        nine = (a, b)
                        print("banjing9", r, abs(a - nine1[0]),a,b)
                        num_detected_9 = 1
                        r9 = r
                        cv2.circle(match, nine, r9, (0, 255, 0), 5)
                        cv2.circle(match, nine, 1, (0, 0, 255), 15)
                        #cv2.circle(match, tuple(np.int0(nine1)), 3, (255, 0, 0), 5)
                        #cv2.putText(match, '9', nine, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        continue

    cv2.namedWindow("Detected Circle", 0)
    cv2.resizeWindow("Detected Circle", 685, 456)
    cv2.imshow("Detected Circle", match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    objPoints1 = np.array([[-17, -1.2, 5],
                           [17, -1.2, 5],
                           [0, -21, 5],
                           [0, 21, 5],
                           [-12, 21, -5],
                           [12, 21, -5],
                           [-14.25, -21, -5],
                           [14.25, -21, -5],
                           [0, 9, -5]], dtype=np.float64)

    objPoints = []
    flag_num = 0
    r_list = []
    if one != (0, 0):
        imgPoints.append(one)
        objPoints.append(objPoints1[0])
        flag_num += 1
        r_list.append(r1)
    if two != (0, 0):
        imgPoints.append(two)
        objPoints.append(objPoints1[1])
        flag_num += 1
        r_list.append(r2)
    if three != (0, 0):
        imgPoints.append(three)
        objPoints.append(objPoints1[2])
        flag_num += 1
        r_list.append(r3)
    if four != (0, 0):
        imgPoints.append(four)
        objPoints.append(objPoints1[3])
        flag_num += 1
        r_list.append(r4)
    if five != (0, 0):
        imgPoints.append(five)
        objPoints.append(objPoints1[4])
        flag_num += 1
        r_list.append(r5)
    if six != (0, 0):
        imgPoints.append(six)
        objPoints.append(objPoints1[5])
        flag_num += 1
        r_list.append(r6)
    if seven != (0, 0):
        imgPoints.append(seven)
        objPoints.append(objPoints1[6])
        flag_num += 1
        r_list.append(r7)
    if eight != (0, 0):
        imgPoints.append(eight)
        objPoints.append(objPoints1[7])
        flag_num += 1
        r_list.append(r8)
    if nine != (0, 0):
        imgPoints.append(nine)
        objPoints.append(objPoints1[8])
        flag_num += 1
        r_list.append(r9)

    # PnP位姿解算至少需要6个点
    if flag_num < 0:
        print("PnP位姿结算点数量少于6个！！！！！！！！！！！！\n请重新进行细定位拍照！！！！！！！！！！！")
    else:
        # 椭圆检测
        prepare2 = cv2.cvtColor(match2, cv2.COLOR_RGB2GRAY)
        prepare2 = cv2.GaussianBlur(prepare2, (3, 3), 1.3)
        # prepare2 = cv2.medianBlur(prepare2, 3)
        canny2 = cv2.Canny(prepare2, 10, 40)
        contours2, _ = cv2.findContours(canny2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("椭圆检测矫正前的imgPoints",imgPoints)
        # print("轮廓数量是是",len(contours2))
        for cnt in contours2:
            if len(cnt) > 60:
                # 椭圆拟合
                ellipse = cv2.fitEllipse(cnt)
                # 筛选不符合条件的椭圆，ellipse包括中心点，（长轴长度，短轴长度），旋转角度
                if ellipse[1][0] != 0 and 1 < ellipse[1][1] / ellipse[1][0] < 1.05:
                    # 根据8个特征圆圆心坐标把符合条件的椭圆筛选出来，用椭圆圆心修正坐标
                    for k in range(len(imgPoints)):
                        # if k == 0 or k == 1:
                        if False:
                            pass
                        else:
                            imgPoint = imgPoints[k]
                            if abs(ellipse[0][0] - imgPoint[0]) < 20 and abs(ellipse[0][1] - imgPoint[1]) < 20 and \
                                    2.1 * r_list[k] > ellipse[1][1] > 2.01 * r_list[k]:
                                cv2.ellipse(match, ellipse, (0, 0, 255), 5)
                                imgPoints[k] = (ellipse[0][0], ellipse[0][1])
                                cv2.circle(match, (np.int0(ellipse[0][0]), np.int0(ellipse[0][1])), 3, (0, 255, 255),
                                           15)
        #
        # save_image_with_folder('E:/Projects/AutoCharge/0822/1110_result/1106_result10/' + name + '.png', match)
        # print(name + ' have done')

        # 解算位姿，2D和3D必须一一对应，且坐标矩阵中每个元素的值必须是np.float32
        # print("经椭圆检测矫正后的imgpoints", imgPoints)
        cameraMatrix = np.load(r"parameters\calibration_matrix_4k_1212.npy")
        distCoeffs = np.load(r"parameters\distortion_coefficients_4k_1212.npy")

        imgPoints = np.array(imgPoints, dtype=np.double)
        objPoints = np.array(objPoints, dtype=np.double)
        # print("objPoints",objPoints)
        # print("imgPoints",imgPoints)
        _, rvec, tvec, _ = cv2.solvePnPRansac(objPoints, imgPoints, cameraMatrix, distCoeffs, flags=3)
        print("rvec is", rvec.reshape(1,3))
        print("tvec is", tvec.reshape(1,3))
        return np.array(tvec.reshape(1, 3).tolist()[0]), np.array(rvec.reshape(1, 3).tolist()[0])



if __name__ == '__main__':
    img1 = Image.open(r"0822/xidingwei/2023-12-12_15-40-37_132.png")
    img2 = Image.open(r"0822/xidingwei/2023-12-12_15-40-37_132.png")
    image_name='5_5_5.png'
    xi_dingwei(img1,img2,image_name)

###*************************************************************************************************************
    # directory = r'E:\Projects\AutoCharge\0822\1106\all'
    # filenames = get_png_paths(directory)
    # print(filenames)
    # names = ['-15_15_-15']
    #
    # names = filenames
    # for name in names:
    #     filename = name + '.png'
    #     full_filename = os.path.join(directory, filename)
    #     img1 = Image.open(full_filename)
    #     img2 = Image.open(full_filename)
    #     xi_dingwei(img1, img2, name)