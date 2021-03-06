import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

img = mpimg.imread('solidWhiteRight.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)

z = sobelx + sobely

abs_sobelx = np.absolute(z)

scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <=thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
plt.show()