import cv2
import numpy as np

def binary_threshold(color_image):
    grayed = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    under_thresh = 105
    upper_thresh = 145
    maxValue = 255
    th, drop_back = cv2.threshold(grayed, under_thresh, maxValue, cv2.THRESH_BINARY)
    th, clarify_born = cv2.threshold(grayed, upper_thresh, maxValue, cv2.THRESH_BINARY_INV)
    merged = np.minimum(drop_back, clarify_born)
    return cv2.cvtColor(merged,cv2.COLOR_GRAY2RGB)