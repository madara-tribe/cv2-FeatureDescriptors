import sys
import numpy as np
import cv2
import os
import json 
from utils import preprocess, square_resize
from advanced_cv2detector import beblid
from cv2_methods import CannyEdgeDetector
from utils.preprocess import gamma, resize
np.set_printoptions(suppress=True)

DIRS = 'images'
OUTPUT_DIR = os.path.join('results', DIRS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def round(matrix, dim=5):
    return np.round(matrix, decimals=dim)

def load_path(path1, path2):
    path1 = os.path.join(DIRS, path1)
    path2 = os.path.join(DIRS, path2)
    img1, img2 = cv2.imread(path1, 0), cv2.imread(path2, 0)
    return img1, img2

def main(path1, path2, out_name):
    ##### detect homography ####
    img1, img2 = load_path(path1, path2)
    img1, img2 = gamma(img1), gamma(img2)
    img1, img2 = resize(img1), resize(img2)
    beblid_matches, beblid_homography = beblid.beblid_homography(img1, img2)


    #### saving ####
    cv2.imwrite(os.path.join(OUTPUT_DIR, "beblid_"+out_name+".png"),beblid_matches)
    result = {'beblid':str(beblid_homography)}
    with open(os.path.join(OUTPUT_DIR, 'beblidi_'+out_name+"_homografies.txt"), 'w') as file:
        file.write(json.dumps(result))
    print('beblid_homography', beblid_homography)

if __name__=='__main__':
    args = sys.argv
    if len(args)>4:
        print('python3 ~.py input_img1 input_img2 output_name')
    else:
        path1 = str(args[1])
        path2 = str(args[2])
        out_name = str(args[3])
        main(path1, path2, out_name)
