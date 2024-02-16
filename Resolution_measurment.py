import numpy as np
import cv2
import os
import scipy.signal as sig
import matplotlib.pyplot as plt

def sigmoid(x):
    z = 1/(1+np.exp(-x))
    return z

def res_calc(img, axis ):

    w_size = 30
    window = np.ones((w_size, 1))

    a = sigmoid(img)
    a = (a - np.min(a)) / (np.max(a) - np.min(a)) * 255

    mean = np.mean(a)
    std = np.std(a)
    low = mean - normal_k * std
    high = mean + normal_k * std
    # plt.hist(np.reshape(a, (680 ** 2, 1)), bins=round(8 * std), range=(mean - 2 * std, mean + 2 * std))
    # plt.show()

    b = np.bitwise_or((a < low), (a > high)) * 255
    b = b.astype('uint8')

    # cv2.imshow(title, b)
    res_total = np.array([])


    frame_size = img.shape[axis]
    pix_res = FOV / frame_size
    height = width

    for line in range(img.shape[axis]):
        if axis == 0:
            b_line = np.reshape(b[line, :] / 255, (b.shape[1], 1))
        else:
            b_line = np.reshape(b[:,line] / 255, (b.shape[0], 1))

        b_corr = np.round(sig.correlate(b_line, window))
        b_res = sig.find_peaks(np.reshape(b_corr, (len(b_corr))), height=height)
        b_res = np.array(b_res[1]['peak_heights'])
        res_total = np.append(res_total, np.array(b_res))
        resolution_estimation = np.round(pix_res * np.mean(res_total), 2)
        print('L#' + str(line) + ' res_x= ' + str(np.round(np.mean(res_total), 2)) + '[pix] ,res_x= ' + str(
            resolution_estimation) + '[nm] ,len = ' + str(len(res_total)) + ', BuiltGauss(' + str(
            gauss_size) + ') = ' + str(BuiltGauss) + ', witdh_Thrshld = ' + str(height))
    return resolution_estimation, b

normal_k=1.5
FOV = 250
gauss_size = 3
BuiltGauss = True
width = 0



path = './res'
file = 'res2.png'
fullpath = os.path.join(path,file)
img = cv2.imread(fullpath,0)

# title = 'original'
# cv2.imshow(title,img)


kernel_gauss = (1/16)*np.array(
                    [[ 1,  2,  1],
                     [ 2,  4,  2],
                     [ 1,  2,  1]])

theta = 1

kernel_sobel_x = theta*np.array(
                    [[ 1,  0,  -1],
                     [ 2,  0,  -2],
                     [ 1,  0,  -1]])
kernel_sobel_y = kernel_sobel_x.T


ker = kernel_gauss
title = '3x3 guass filtered'

''' CV2 gaussian blur normalizing output to [0,255] int range '''
if BuiltGauss==True:
    img_gauss= cv2.GaussianBlur(img, (gauss_size, gauss_size), cv2.BORDER_DEFAULT)
else:
    img_gauss = sig.convolve2d(img, ker, mode='same')

# img_gauss = img_gauss.astype('uint8')
# cv2.imshow(title,img_gauss)

ker = kernel_sobel_x
title = 'kernel_sobel_x'
img_k1 = sig.convolve2d(img_gauss,ker,mode='same')
img_k1 = (img_k1 - np.min(img_k1))/( np.max(img_k1) - np.min(img_k1))
# img_k1 = img_k1.astype('uint8')
# cv2.imshow(title,img_k1)

title = 'sigmoid on sobel-x'



# a=sigmoid(img_k1)
# a = (a - np.min(a))/( np.max(a) - np.min(a))*255
# b = a.astype('uint8')
#
#
# mean = np.mean(a)
# std = np.std(a)
# low = mean-normal_k*std
# high = mean+normal_k*std
# plt.hist(np.reshape(a,(680**2,1)),bins=round(8*std) ,range=(mean-2*std,mean+2*std))
# # plt.show()
#
# b=np.bitwise_or((a<low),(a>high))*255
# b = b.astype('uint8')
#
# cv2.imshow(title,b)
# res_total = np.array([])
# w_size =30
# window = np.ones((w_size,1))
#
# frame_size_x = img.shape[0]
# pix_res=FOV/frame_size_x
# height = width

# for line in range(img.shape[0]):
#
#     b_line = np.reshape(b[line,:]/255, (b.shape[0],1))
#     b_corr = np.round(sig.correlate(b_line,window))
#     b_res = sig.find_peaks(np.reshape(b_corr, (len(b_corr))), height=height)
#     b_res = np.array(b_res[1]['peak_heights'])
#     res_total=np.append(res_total,np.array(b_res))
#     reslution_estimation = np.round(pix_res * np.mean(res_total),2)
#     print('L#' + str(line) + ' res_x= ' + str(np.round(np.mean(res_total),2)) + '[pix] ,res_x= ' + str(reslution_estimation) + '[nm] ,len = ' +str(len(res_total)) + ', BuiltGauss(' + str(gauss_size) + ') = ' + str(BuiltGauss) + ', witdh_Thrshld = ' +str(height))
#

ker = kernel_sobel_y
title = 'kernel_sobel_y'
img_k2 = sig.convolve2d(img_gauss,ker,mode='same')
img_k2 = (img_k2 - np.min(img_k2))/( np.max(img_k2) - np.min(img_k2))

res_y_est,edges_y = res_calc(img_k2,1)
res_x_est,edges_x = res_calc(img_k1,0)
print('res x = ' + str(res_x_est) + '[nm], res_y = ' + str(res_y_est) + '[nm]. Ratio Y/X =' + str(np.round((res_y_est/res_x_est),2)) )



fig, axs = plt.subplots(1,3, figsize=(12, 4))
title_big = 'Resolution measurement'
fig.suptitle(title_big)

title_0 = 'original image'
axs[0].imshow(img, cmap='grey')
axs[0].set_title(title_0)

title_1 = 'x res:' + str(res_x_est) + '[nm]'
axs[1].imshow(0.5*img+edges_x, cmap = 'summer')
axs[1].set_title(title_1)

title_2 = 'y res:' + str(res_y_est) + '[nm]'
axs[2].imshow(0.5*img+edges_y, cmap = 'hot')
axs[2].set_title(title_2)
plt.show()


# img_k2 = img_k2.astype('uint8')
# cv2.imshow(title,img_k2)
# cv2.waitKey(0)

# im_xy_res = abs(img_k2-img_k1)
# im_xy_res = im_xy_res.astype('uint8')
# title = 'im_xy_res'
#
# cv2.imshow(title,im_xy_res)
# cv2.waitKey(0)


edges = (img_k1.astype('uint32')**2 + img_k2.astype('uint32')**2)**0.5
edges = (edges - np.min(edges))/( np.max(edges) - np.min(edges))*255
edges = edges.astype('uint8')


# title = 'manually canny'
# cv2.imshow(title,edges)

# eps=1e-7
# angle = np.arctan(img_k2.astype('float')/(img_k1.astype('float') + eps))
# angle = (angle - np.min(angle))/( np.max(angle) - np.min(angle))*255
# angle = angle.astype('uint8')
# title = 'angle tg-1'
# cv2.imshow(title,angle)

# fig, axs = plt.subplots(4)
# title_big = 'comparison of line ' + str(line)
# fig.suptitle(title_big)
# title_0 = 'original image'
# axs[0].plot(img[line,:])
# axs[3].grid()
# axs[0].set_title(title_0)
# title_1 = 'manual canny'
# axs[1].plot(edges[line,:])
# axs[3].grid()
# axs[1].set_title(title_1)
# title_2 = 'edge detection almog'
# axs[2].plot(b[line,:]/255)
# axs[3].grid()
# axs[2].set_title(title_2)
# title_3 = 'corrolation resolution'
# axs[3].plot(b_corr)
#
# axs[3].set_title(title_3)
# axs[3].grid()

# edges = edges.astype('uint8')
# cv2.imshow(title,edges)
# cv2.waitKey(0)

# title = 'before canny'
# edges = edges.astype('uint8')
# cv2.imshow(title,edges)
# cv2.waitKey(0)
# L2Gradient = False # Boolean

# aperture_size = 3
# edges = cv2.Canny(img,100,150,apertureSize=aperture_size, L2gradient =L2Gradient)
# title = 'canny edges'
# cv2.imshow(title,edges)
# L2Gradient = True # Boolean
# edges = cv2.Canny(img,100,150,apertureSize=aperture_size, L2gradient =L2Gradient)
# title = 'canny edges, L2Gradient'
# cv2.imshow(title,edges)

cv2.waitKey(0)
# plt.show()