import numpy as np
import cv2
import os
import scipy.signal as sig
import matplotlib.pyplot as plt

path = './res'
file = 'res3.png'

fullpath = os.path.join(path,file)

img = cv2.imread(fullpath,0)

title = 'original'
cv2.imshow(title,img)

kernel_sharp = np.array(
                    [[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]])

kernel_gauss = (1/16)*np.array(
                    [[ 1,  2,  1],
                     [ 2,  4,  2],
                     [ 1,  2,  1]])

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
# cv2.imshow(title,img_k1)
# cv2.waitKey(0)

ker = kernel_sharp
title = 'kernel_sharp'
img_k1 = sig.convolve2d(img,ker,mode='same')
img_k1 = img_k1.astype('uint8')
# cv2.imshow(title,img_k1)
# cv2.waitKey(0)

ker = kernel_gauss
title = '3x3 guass filtered'
img_gauss =  sig.convolve2d(img,ker,mode='same')
img_gauss= cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
# img_gauss = img_gauss.astype('uint8')
cv2.imshow(title,img_gauss)

ker = kernel_sobel_x
title = 'kernel_sobel_x'
img_k1 = sig.convolve2d(img_gauss,ker,mode='same')
img_k1 = img_k1.astype('uint8')
# cv2.imshow(title,img_k1)
# cv2.waitKey(0)

ker = kernel_sobel_y
title = 'kernel_sobel_y'
img_k2 = sig.convolve2d(img_gauss,ker,mode='same')
img_k2 = img_k2.astype('uint8')
# cv2.imshow(title,img_k2)
# cv2.waitKey(0)


edges = (img_k1.astype('uint32')**2 + img_k2.astype('uint32')**2)**0.5
edges = np.diff(img)
title = 'manually canny'
cv2.imshow(title,edges)

eps=1e-7
angle = np.arctan(img_k2.astype('float')/(img_k1.astype('float') + eps))
angle = angle/(np.max(angle))*255
angle = angle.astype('uint8')
title = 'angle tg-1'
cv2.imshow(title,angle)

line = 40
fig, axs = plt.subplots(3)
title_big = 'comparison of line ' + str(line)
fig.suptitle(title_big)
axs[0].plot(img[line,:])
title_0 = 'original image'
axs[0].set_title(title_0)
title_1 = 'manual canny'
axs[1].plot(edges[line,:])
axs[1].set_title(title_1)
title_2 = 'manual angle'
axs[2].plot(angle[line,:])
axs[2].set_title(title_2)


# edges = edges.astype('uint8')
# cv2.imshow(title,edges)
# cv2.waitKey(0)

# title = 'before canny'
# edges = edges.astype('uint8')
# cv2.imshow(title,edges)
# cv2.waitKey(0)
L2Gradient = False # Boolean

aperture_size = 3
edges = cv2.Canny(img,100,150,apertureSize=aperture_size, L2gradient =L2Gradient)
title = 'canny edges'
cv2.imshow(title,edges)
L2Gradient = True # Boolean
edges = cv2.Canny(img,100,150,apertureSize=aperture_size, L2gradient =L2Gradient)
title = 'canny edges, L2Gradient'
cv2.imshow(title,edges)

cv2.waitKey(0)
plt.show()