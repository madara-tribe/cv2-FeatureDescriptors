import numpy as np 
import cv2

def square_resize(img):
    max_length = max(img.shape)

    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(max_length)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = max_length - new_size[1]
    delta_h = max_length - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    ret_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    #ret_img = tf.image.resize_with_pad(img, max_length, max_length, method=ResizeMethod.BILINEAR, antialias=False)
    return ret_img
