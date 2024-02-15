import numpy as np
import cv2
import os
import scipy.signal as sig
import matplotlib.pyplot as plt

path = './res'
file = 'res1.png'

fullpath = os.path.join(path,file)

img = cv2.imread(fullpath,0)

title = 'original'
cv2.imshow(title,img)
cv2.waitKey(0)

kernel_sharp = np.array(
                    [[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]])

# rotation_max = np.array(
#                     [[ 1,  0,  -1],
#                      [ 2,  0,  -2],
#                      [ 1,  0,  -1]])
theta = 1

kernel_sobel_x = theta*np.array(
                    [[ 1,  0,  -1],
                     [ 2,  0,  -2],
                     [ 1,  0,  -1]])
kernel_sobel_y = kernel_sobel_x.T

title = 'diff'
img_k1 = np.diff(img)
img_k1 = img_k1.astype('uint8')
cv2.imshow(title,img_k1)
cv2.waitKey(0)

ker = kernel_sharp
title = 'kernel_sharp'
img_k1 = sig.convolve2d(img,ker,mode='same')
img_k1 = img_k1.astype('uint8')
cv2.imshow(title,img_k1)
cv2.waitKey(0)


ker = kernel_sobel_x
title = 'kernel_sobel_x'
img_k1 = sig.convolve2d(img,ker,mode='same')
img_k1 = img_k1.astype('uint8')
cv2.imshow(title,img_k1)
cv2.waitKey(0)

ker = kernel_sobel_y
title = 'kernel_sobel_y'
img_k2 = sig.convolve2d(img,ker,mode='same')
img_k2 = img_k2.astype('uint8')
cv2.imshow(title,img_k2)
cv2.waitKey(0)


edges = (img_k1.astype('uint32')**2 + img_k2.astype('uint32')**2)**0.5
title = 'before canny'
edges = edges.astype('uint8')
cv2.imshow(title,edges)
cv2.waitKey(0)

edges = cv2.Canny(edges,100,200)
title = 'canny edges'
cv2.imshow(title,edges)
cv2.waitKey(0)