import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nx = 8 # number of inside corners in x
ny = 6 # number of inside corners in y

# Make a list of calibration images
fname = 'test_image2.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
print(corners)
"""
print("0", corners[0])
print("7", corners[6])
print("42", corners[42])
print("49", corners[48])
"""
src = np.float32([[corners[0,0,0], corners[0,0,1]], [corners[7,0,0],corners[7,0,1]], [corners[40,0,0],corners[40,0,1]], [corners[47,0,0],corners[47,0,1]]])

print("0:", corners[0,0,0], corners[0,0,1])
print("1:", corners[1,0,0], corners[1,0,1])
print("7:", corners[8,0,0], corners[8,0,1])


# If found, draw corners

if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()
