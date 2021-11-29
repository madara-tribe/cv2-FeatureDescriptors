import math
import numpy as np
import cv2

def homography_transfrom(img, homography):
    angle = math.atan2(homography[0,1], homography[0,0])
    height, width = img.shape[:2]
    rotation_matrix=cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    transfrom_img = cv2.warpAffine(img, rotation_matrix,(width,height))
    print(angle)
    return transfrom_img

def svd_findHomography(good, kp1, kp2, knnbase=True):
    if knnbase:
        p2 = [kp2[m.trainIdx].pt for m in good]
        p1 = [kp1[m.queryIdx].pt for m in good]
    else:
        p1, p2 = kp1, kp2
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

def calHomography(good, kp1, kp2, knnbase):
    if knnbase:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        cv2_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        svd_homography = svd_findHomography(good, kp1, kp2, knnbase=knnbase)
    else:
        points1 = np.zeros((len(good), 2), dtype=np.float32)
        points2 = np.zeros((len(good), 2), dtype=np.float32)

        for i, match in enumerate(good):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
        cv2_homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        svd_homography = svd_findHomography(good, points1, points2, knnbase=knnbase)
    return cv2_homography, svd_homography, mask
