import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('test_image2.png')

plt.imshow(img)
plt.show()
plt.plot(166, 166, '+')