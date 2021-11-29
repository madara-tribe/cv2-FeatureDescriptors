import cv2
import numpy as np
def gamma(img, gamma = 1.8):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
         gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    gamma_img = cv2.LUT(img, gamma_cvt)
    return gamma_img

def resize(img):
    expand_query = 0.5
    query_img = cv2.resize(img, (int(img.shape[1] * expand_query),
                       int(img.shape[0] * expand_query)))
    return query_img
