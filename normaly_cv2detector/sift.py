import sys
sys.path.append('../utils')
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from utils import homography_util


MIN_MATCH_COUNT = 5
THRESHOLD = 0.85

def sift_descriptor(grayscale_img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(grayscale_img, None)
    return kp, des

# Initiate SIFT detector
def sift_homography(img1, img2):

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift_descriptor(img1)
    kp2, des2 = sift_descriptor(img2)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < THRESHOLD * n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        ####### cal homography
        cv2_homography, svd_homography, mask = homography_util.calHomography(good, kp1, kp2, knnbase=True)
        homography_matrix = cv2_homography
        #######
        
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, homography_matrix)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        homography_matrix = None
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask, # draw only inliers
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    matches_img = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    if homography_matrix is not None:
        transformed_image = homography_util.homography_transfrom(img2, homography_matrix)
        #cv2.imwrite(os.path.join(OUTPUT_DIR, "tranformed_"+out_name+".png"), transformed_image)
    return homography_matrix, matches_img


