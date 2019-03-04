import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('warped-example.jpg')
nwindows = 9

height = np.int(img.shape[0]//nwindows)
"""
for windows in range(nwindows):
    low = img.shape[0] - (windows+1)*height
    high = img.shape[0] - windows*height
    print(windows, ": low: ", low, ": high: ", high, "\n")
"""

non = img.nonzero()[0]
#print(non)

h = np.array([1, 2, 2, 2, 0, 0, 0])
print(h.nonzero()[0])