import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

objpoints = []
imgpoints = []

obj = np.zeros((6*9,3), np.float32)
obj[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

img = cv2.imread('calibration1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
print("image size: ", img.shape)
#print("\n", corners)

if ret == True:
    imgpoints.append(corners)
    objpoints.append(obj)
    
    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.show()

