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


def iu(img,x,y,color,thickness):
    """
    if len(x) ==0:
        return 
    k = np.polyfit(x,y,1)
    m = k[0]
    b = k[1]
    
    Y1 = img.shape[0]
    Y2 = 320
    
    X1 = int((Y1-b)/m)
    X2 = int((Y2-b)/m)
    cv2.line(img,(X1,Y1),(X2,Y2),color,thickness)
    """
    if len(x) == 0: 
        return
    
    lineParameters = np.polyfit(x, y, 1) 
    
    m = lineParameters[0]
    b = lineParameters[1]
    
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1 = maxY
    x1 = int((y1 - b)/m)
    y2 = int((maxY/2)) + 60
    x2 = int((y2 - b)/m)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
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
    Rx = []
    Ry = []
    Lx = []
    Ly = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            m = ((y1 - y2)/(x1 - x2))
            if m<0:
                Lx.append(x1)
                Ly.append(y1)
                Lx.append(x2)
                Ly.append(y2)
            else:
                Rx.append(x1)
                Ry.append(y1)
                Rx.append(x2)
                Ry.append(y2)
                
    iu(img,Lx,Ly,(255,255,255),20)
    iu(img,Rx,Ry,(0,255,0),20)

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


image = mpimg.imread('solidWhiteRight.jpg')
gray = grayscale(image)
blur_gray = gaussian_blur(gray, 5)
edges = canny(blur_gray, 50, 120)
im = image.shape
#vertices = np.array([[(0,im[0]), (400,320), (150,320), (im[1], im[0])]], dtype=np.int32)
vertices = np.array([[(130,im[0]),(im[1]/2, (im[0]/2) + 10), (870, im[0])]], dtype=np.int32)
#vertices = np.array([[(0,im[0]),(450, 320), (500, 320),(im[1], im[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)
lines = hough_lines(masked_edges, 1, np.pi/180, 90, 15, 18)
lines_edges = weighted_img(lines, image)
plt.imshow(lines_edges)
plt.show()