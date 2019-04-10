"""
manual vehicle detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for b in bboxes:
        cv2.rectangle(draw_img, b[0], b[1], color, thick)
    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((850, 511), (1146, 695)), ((265, 500), (387, 582)), ((479, 515), (561, 570)), ((598, 516), (652, 561)), 
          ((652, 518), (692, 547)), ((551, 520), (588, 549))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()