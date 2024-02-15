import numpy as np
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def contrast(img, center=False, coord = [-1,-1], window = 16):
    if len(img.shape) == 3:
        img = rgb2gray(img)*255
    H,W = img.shape

    if (center > 0):
        if(coord[0] > 0):
            H = 2*coord[0]
            W = 2*coord[1]
        x_start = int(W/2 - window)
        x_end = int(W/2 + window)
        y_start = int(H/2 - window)
        y_end = int(H/2 + window)
        window_vec = [x_start,x_end,y_start,y_end]
    else:
        x_start = 0
        x_end = W
        y_start = 0
        y_end = H
        window_vec = [0, W, 0, H]

    print(window_vec)


    diff_total = 0
    for j in range(y_start,y_end):
        diff_line = np.sum( img[j,x_start+1:x_end] - img[j,x_start:x_end-1] )
        diff_total += np.abs(diff_line)

    return diff_total

# path = './focus/'
# file = 'ex1_fig3s.jpg'
# path = 'H:/My Drive/GitHub/ApertureAlign/images/AA/x/'
# file = '12_x.tif'
# file = ['09_x.tif','10_x.tif','11_x.tif','12_x.tif','13_x.tif','14_x.tif','15_x.tif']

path = './focus/'
file = ['teapot_02.jpg','teapot_03.jpg','teapot_04.jpg']

contrast_vec = np.array(())
for k in file:
    filename = os.path.join(path,k)
    img = mpimg.imread(filename)
    img_contrast = contrast(img, center=True, coord=[150,307])
    contrast_vec = np.append(contrast_vec,img_contrast)
    print(str(k) + ' total contrast diff is: ' + str(img_contrast))


plt.imshow(img)
# plt.plot(contrast_vec)
# plt.show()