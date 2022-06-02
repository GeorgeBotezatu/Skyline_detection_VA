import math

import cv2
from scipy.signal import medfilt
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt



def gaussianKernel(size, sigma, twoDimensional=True):
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1*(x-(size-1)/2)**2) / (2*sigma**2)), (size,))
    return kernel / np.sum(kernel)


# functie pentru evidentierea cerului
def cal_skyline(mask):
    #salvam dimensiunile mastii
    h, w = mask.shape
    for i in range(w):

        raw = mask[:, i]
        #aplicam un filtru median
        after_median = medfilt(raw, 19)
        try:
            # extragem valorile mastii unde valorile mediane sunt egale cu 0
            first_zero_index = np.where(after_median == 0)[0][0]
            # extragem valorile mastii unde valorile mediane sunt egale cu 1
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 20:
                #salvam valorile
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
    return mask


def detect_sky(img):


    # Convertim imaginea in grayScale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("img_gray1", img_gray)
    # // varianta mai simpla
    # blured_image = cv2.blur(img_gray, (9, 3))

    gaussian_kernel_x = gaussianKernel(3,1, True)
    print('Kernel(x): ',gaussian_kernel_x)
    gaussian_kernel_y = gaussian_kernel_x.reshape(-1, 1)
    print('Kernel(y): ', gaussian_kernel_y)
    # Adaugam filtrul de blur
    blured_image = cv2.filter2D(img_gray, -1, gaussian_kernel_x)
    blured_image = cv2.filter2D(blured_image, -1, gaussian_kernel_y)


    cv2.imshow("img_blur2",  blured_image)

    #Adaugam Filtrul laplacian pentru a determina marginile
    lap = cv2.Laplacian( blured_image, cv2.CV_8U)

    cv2.imshow("lap", lap)
    #creare masca
    gradient_mask = (lap < 6).astype(np.uint8)

    # creare kernel pentru acoperire
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9))


    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)


    mask = cal_skyline(mask)

    after_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("img",after_img)
    cv2.waitKey(0)


    return after_img
