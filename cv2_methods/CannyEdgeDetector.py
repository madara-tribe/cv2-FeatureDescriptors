import numpy as np
import cv2

def canny_edge_detector(color_image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(color_image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(color_image, lower, upper)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)