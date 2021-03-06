import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)

red_threshold = 200
blue_threshold = 200
green_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])

color_select[thresholds] = [0,0,0]


#plt.subplot(221), plt.imshow(image)
#plt.axis('off')
#plt.subplot(222), plt.imshow(color_select)
#plt.axis('off')
#plt.show()
print(xsize)
print(int(xsize/13))
print(int((xsize/13)))
