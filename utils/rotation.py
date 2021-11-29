import os
import numpy as np
from skimage import data, color, exposure
import cv2

def rotation(imgs):
    SHAPE=imgs[1].shape[1]
    # 画像サイズの取得(横, 縦)
    size = tuple(np.array([SHAPE, SHAPE]))
    # 回転させたい角度
    rad=np.pi/20
    # x軸方向に平行移動させたい距離
    move_x = 20
    # y軸方向に平行移動させたい距離
    move_y = SHAPE* -0.1
    matrix = [[np.cos(rad), -1 * np.sin(rad), move_x],
               [np.sin(rad), np.cos(rad), move_y]]

    affine_matrix = np.float32(matrix)

    afn = [cv2.warpAffine(i, affine_matrix, size, flags=cv2.INTER_LINEAR) for i in imgs]
    return np.array(afn)
