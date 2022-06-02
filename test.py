from sky_detector import detector
import cv2
from matplotlib import pyplot as plt 
import os


img = cv2.imread("sample/p8.jpg")[:, :, ::-1]
plt.figure(2)
plt.subplot(2,1,1)
plt.imshow(img)

img_sky = detector.detect_sky(img)
plt.figure(2)
plt.subplot(2,1,2)
plt.imshow(img_sky)
plt.show()
