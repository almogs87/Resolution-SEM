import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import cv2
import os
from scipy.signal import convolve2d
from scipy.ndimage import median_filter

identity = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

sharpen = np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]])*0.25

edge = np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

guassian = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])
guassian = guassian/16

box_size = 8
box_blur = np.ones((box_size,box_size))/box_size**2

sobel_x = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

sobel_x_left = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])
sobel_y = sobel_x.T

sobel_xy = np.sqrt(sobel_x**2 + sobel_y**2)

blur = np.ones([3,3])/9

kernel = sobel_x

path = 'H:\My Drive\תמונות\Marina'
filename = 'IMG-20231211-WA0006.jpg'

# img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)
img = mpimg.imread(os.path.join(path,filename))
img = rgb2gray(img)
img_diff = np.diff(img)
# img_kernel = cv2.filter2D(img,ddepth=-1,sobel_x)
print(img.shape)
# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# img = rgb2gray(img)
# print(img.shape)
# img = mpimg.imread(os.path.join(path,filename))
# img_diff = np.diff(img)


# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,150,param1=100, param2=30, minRadius=0, maxRadius=0)
# circles = np.uint16(np.around(circles))
#
# for i in circles[0,:]:
#     cv2.circle(cimg,(i[0],i[1]), i[2], (0,255,0),2)

# cv2.imshow('title',cimg)
# cv2.waitKey(0)

img2 = convolve2d(img,sharpen)
img2_title = 'sharpen derivative only'
print(img2.shape)

lambd = .5


img3 = convolve2d(img,box_blur)
lambd = img - img3[:-7,:-7]
img3 = img + lambd * img2[:-2,:-2]
img3_title = 'box_blur 3x3'


f,arr = plt.subplots(1,3)
arr[0].imshow(img, cmap='gray')
arr[0].set_title('original image')
arr[1].imshow(img2, cmap='gray')
arr[1].set_title(img2_title)
arr[2].imshow(img3, cmap='gray')
arr[2].set_title(img3_title)

plt.show()