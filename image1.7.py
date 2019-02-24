import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def d(img, x, y, color, thickness):
    if len(x)==0:
        return
    
    m, b = np.polyfit(x, y, 1)
    #A = np.vstack([x, np.ones(len(x))]).T
    #m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y1 = img.shape[0]
    x1 = int((y1 - b)/m - 0.01)
    y2 = 330
    x2 = int((y2 - b)/m)
    
    cv2.line(img, (x1,y1), (x2,y2), color, thickness)   
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lx = []
    ly = []
    rx = []
    ry = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = ((y2 - y1)/(x2 - x1))
            if m < 0:
                #cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 8)
                lx.append(x1)
                lx.append(x2)
                ly.append(y1)
                ly.append(y2)
            else:
                #cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 8)
                rx.append(x1)
                rx.append(x2)
                ry.append(y1)
                ry.append(y2)
                
    d(img, lx, ly, (255,255,255), 8)
    d(img, rx, ry, (0,255,0), 8)
    """
    
    x_left = []
y_left = []
x_right = []
y_right = []
y3 = 320
y4 = 540
left = True
right = True
img_x_center = image.shape[1] / 2
print(lines)

#315 425 394 372
#segment = np.polyfit((315,394),(425,372),1)
#print(segment)

for line in lines:
    for x1,y1,x2,y2 in line:
        #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        if x1==x2:
            continue
        line_seg = np.polyfit((x1,x2),(y1,y2),1)

        if line_seg[0]<0 and abs(line_seg[0])>0.5 and x1<img_x_center and x2<img_x_center:
            x_left.append(x1)
            x_left.append(x2)
            y_left.append(y1)
            y_left.append(y2)

        if line_seg[0]>0 and abs(line_seg[0])>0.5:
            x_right.append(x1)
            x_right.append(x2)
            y_right.append(y1)
            y_right.append(y2)

print(x_left)
if len(x_left)>0:
    line_left = np.polyfit(x_left,y_left,1)
else:
    left = False

if len(x_right)>0:
    line_right = np.polyfit(x_right,y_right,1)
else:
    right = False

if left:
    x3 = (y3-line_left[1])/line_seg[0]
    x4 = (y4-line_left[1])/line_seg[0]
    x3 = abs(x3)
    x4 = abs(x4)
    cv2.line(img, (int(x3), y3), (int(x4), y4), color, 10)

if right:
    x3 = (y3-line_right[1])/line_seg[0]
    x4 = (y4-line_right[1])/line_seg[0]
    cv2.line(img, (int(x3), y3), (int(x4), y4), color, 10)
    """
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
"""
def image_pipeline(path):
    gray = grayscale(path)
    blur_gray = gaussian_blur(gray, 15)
    edges = canny(blur_gray, 20, 100)
    im = image.shape
    vertices = np.array([[(0,im[0]),(450, 320), (500, 320),(im[1], im[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    lines = hough_lines(masked_edges, 2, np.pi/180, 20, 25, 10)
    lines_edges = weighted_img(lines, image)
    return lines_edges
"""
image = mpimg.imread("solidWhiteRight.jpg")
gray = grayscale(image)
blur = gaussian_blur(gray, 1)
edges = canny(blur, 10, 200)
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(0, 322), (960, 322), (imshape[1],imshape[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)
line_img = hough_lines(masked_edges, 1, np.pi/180, 25, 4, 30)
lines_edges = weighted_img(line_img, image) 
plt.imshow(lines_edges)
plt.show()