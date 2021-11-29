import cv2


def cv_segmentation(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    thresh, bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return cv2.cvtColor(bin_img,cv2.COLOR_GRAY2RGB)
    