import sys
sys.path.append('../utils')
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

from utils import homography_util

THRESHOLD = 0.9

def akaze_descriptor(img):
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(img, None)
    return kp, des
    
def akaze_homography(t1, t2):
    kp1, des1 = akaze_descriptor(t1)
    kp2, des2 = akaze_descriptor(t2)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good1 = [[m] for m,n in matches if m.distance < THRESHOLD*n.distance]
    matches = bf.knnMatch(des2,des1, k=2)

    good2 = [[m] for m,n in matches if m.distance < THRESHOLD*n.distance]
    good=[]
    src, dst = [], []
    for i in good1:
        (x1,y1)=kp1[i[0].queryIdx].pt
        (x2,y2)=kp2[i[0].trainIdx].pt

        for j in good2:
            (a1,b1)=kp2[j[0].queryIdx].pt
            (a2,b2)=kp1[j[0].trainIdx].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                src.append(kp1[i[0].queryIdx].pt)
                dst.append(kp2[i[0].trainIdx].pt)
                good.append(i)
                
    src_pts = np.float32([m for m in src]).reshape(-1,1,2)
    dst_pts = np.float32([m for m in dst]).reshape(-1,1,2)
    cv2_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    homography_matrix = cv2_homography
    matches_img = cv2.drawMatchesKnn(t1,kp1,t2,kp2,good,None,[0,0,255],flags=2)

    return matches_img, homography_matrix

