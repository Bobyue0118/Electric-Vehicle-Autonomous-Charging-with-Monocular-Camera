'''
Sample Usage:-
python calibration.py --dir calibration_checkerboard/ --square_size 0.024
'''

import numpy as np
import cv2
import os
import argparse
def calibrate_parameter(dirpath, square_size, width, height, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = os.listdir(dirpath)

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
        if visualize:
            cv2.namedWindow('img'+fname,0)
            cv2.resizeWindow("img"+fname, 685, 456)
            cv2.imshow('img'+fname,img)
            cv2.waitKey(0)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return [ret, mtx, dist, rvecs, tvecs]

def calibrate_eyeandhand(dirpath, square_size, mtx,dist, width, height,visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    R=[]
    T=[]
    images = os.listdir(dirpath)

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            rvec, _ = cv2.Rodrigues(rvec)

            R.append(rvec)
            T.append(tvec)
        if visualize:
            cv2.namedWindow('img',0)
            cv2.resizeWindow("img", 685, 456)
            cv2.imshow('img',img)

            cv2.waitKey(0)

    return [ret_pnp, R,T]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=False,default=r'1212/calibration', help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=9)",  default=11)
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=8)
    ap.add_argument("-s", "--square_size", type=float, default=0.01, help="Length of one edge (in metres)")
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")
    args = vars(ap.parse_args())
    
    dirpath = args['dir']
    # 2.4 cm == 0.024 m
    # square_size = 0.024
    square_size = args['square_size']

    width = args['width']
    height = args['height']

    if args["visualize"].lower() == "true":
        visualize = True
    else:
        visualize = False

    ret1, mtx, dist, rvecs, tvecs = calibrate_parameter(dirpath, square_size, visualize=visualize, width=width,
                                                        height=height)
    #
    print(mtx)
    print(dist)
    #
    np.save("calibration_matrix_4k_1212", mtx)
    np.save("distortion_coefficients_4k_1212", dist)

    # #
    mtx1 = np.load('calibration_matrix_4k_1212.npy')
    dist1 = np.load('distortion_coefficients_4k_1212.npy')
    # # print('mtx:', mtx1)
    # # print('dist:',dist1)
    ret2, R, T = calibrate_eyeandhand(dirpath, square_size,mtx1,dist1, visualize=visualize, width=width, height=height)
    # # print('R:', R)
    # # print('T:', T)
    # np.save("R_board2camera_4k_1018", R)
    # np.save("T_board2camera_4k_1018", T)
    # # # # # # # # 标定时使用
    np.save("R_board2camera_4k_1212", R)
    np.save("T_board2camera_4k_1212", T)

