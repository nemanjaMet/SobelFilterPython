import numpy as np
from random import randint
import cv2
#import matplotlib.pyplot as plt
import time
#from PIL import Image
def sobel_filter(im, k_size):
     
    im = im.astype(np.float)
    #width, height, c = im.shape
    width, height = im.shape
    c = 0
    if c > 1:
        img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
    else:
        img = im
     
    assert(k_size == 3 or k_size == 5);
     
    if k_size == 3:
        kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
        kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)
    else:
        kh = np.array([[-1, -2, 0, 2, 1], 
                   [-4, -8, 0, 8, 4], 
                   [-6, -12, 0, 12, 6],
                   [-4, -8, 0, 8, 4],
                   [-1, -2, 0, 2, 1]], dtype = np.float)
        kv = np.array([[1, 4, 6, 4, 1], 
                   [2, 8, 12, 8, 2],
                   [0, 0, 0, 0, 0], 
                   [-2, -8, -12, -8, -2],
                   [-1, -4, -6, -4, -1]], dtype = np.float)
     
    #gx = signal.convolve2d(img, kh, mode='same', boundary = 'symm', fillvalue=0)
    #gy = signal.convolve2d(img, kv, mode='same', boundary = 'symm', fillvalue=0)
    gx = cv2.filter2D(img,-1,kh)
    gy = cv2.filter2D(img,-1,kv)
 
    g = np.sqrt(gx * gx + gy * gy)
    g *= 255.0 / np.max(g)
   
    #plt.figure()
    #plt.imshow(g, cmap=plt.cm.gray)      
   
    return g
slika = cv2.imread("gajba1.jpg",0)
sobelSlika = sobel_filter(slika, 3)
width, height = slika.shape
#size = (w,h,channels) = (width,height,1)
#blackAndWhite = np.zeros(size,np.uint8)
cv2.imwrite("gajba1Sobel3.jpg", sobelSlika)
