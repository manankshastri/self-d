import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = mpimg.imread("warped-example.jpg")/255

# Take a histogram of the bottom half of the image
i = np.sum(image[image.shape[0]//2:, :], axis=0)

# Create an output image to draw on and visualize the result
out = np.dstack((image, image, image))*255

# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
print(i.shape)
mid = np.int(i.shape[0]//2)

leftx_base = np.argmax(i[:mid])
rightx_base = np.argmax(i[mid:]) + mid

#plt.plot(i)
#plt.imshow(out)
#plt.show()

