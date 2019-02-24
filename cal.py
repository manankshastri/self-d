import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

objpoints = []
imgpoints = []

obj = np.zeros((6*8,3), np.float32)
obj[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

img = cv2.imread('calibration_test.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

if ret == True:
    imgpoints.append(corners)
    objpoints.append(obj)
    
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.show()

