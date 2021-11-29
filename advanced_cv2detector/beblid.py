import sys
sys.path.append('../utils')
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from math import sqrt
from utils.homography_util import svd_findHomography
THRESHOLD = 0.5
INLINER_THRESHOLD = 0.9

def beblid_detector(t1, detector):
    kpts1 = detector.detect(t1, None)
    reblid = cv2.xfeatures2d.BEBLID_create(0.75)
    kp, des = reblid.compute(t1, kpts1)
    return kp, des

def beblid_findHomography(src, dst):
    src_pts = np.float32([m.pt for m in src]).reshape(-1,1,2)
    dst_pts = np.float32([m.pt for m in dst]).reshape(-1,1,2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return homography
    
def beblid_homography(img1, img2):
    t1, t2 = img1, img2
    detector = cv2.ORB_create(10000)
    kp1, des1 = beblid_detector(t1, detector)
    kp2, des2 = beblid_detector(t1, detector)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(des1, des2, 2)
    good = []
    for m, n in nn_matches:
        if m.distance < THRESHOLD * n.distance:
            good.append([m])
    point_num = 100
    good = sorted(good, key=lambda x: x[0].distance)
    #if len(good)>point_num:
     #   good = good[:point_num]

    matched1 = []
    matched2 = []
    for m in good:
    #for m, n in nn_matches:
        #if m.distance < THRESHOLD * n.distance:
        matched1.append(kp1[m[0].queryIdx])
        matched2.append(kp2[m[0].trainIdx])
    homography = beblid_findHomography(matched1, matched2)
    
    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = INLINER_THRESHOLD  # Distance threshold to identify inliers with homography check
    for i, m in enumerate(matched1):
        # Create the homogeneous point
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        # Project from image 1 to image 2
        col = np.dot(homography, col)
        col /= col[2, 0]
        # Calculate euclidean distance
        dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                    pow(col[1, 0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
    beblid_homography = beblid_findHomography(inliers1, inliers2)
    res = np.empty((max(t1.shape[0], t2.shape[0]), t1.shape[1] + t2.shape[1], 3), dtype=np.uint8)
    matches_img =cv2.drawMatches(t1, inliers1, t2, inliers2, good_matches, res)
    return matches_img, beblid_homography

