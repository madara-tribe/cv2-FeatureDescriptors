# BEBLID_cv2HomographyDetector

# Versions
- pyhton 3.7.3
- opencv 4.5.1
- opencv-contrib-python

# Results-1

## Normal SIFT (left).     &&.    Advanced SIFT (Right)
<img src="https://user-images.githubusercontent.com/48679574/118399190-f4012f00-b696-11eb-9f41-26c629b75efd.png" width="400px" title="Normal SIFT"><img src="https://user-images.githubusercontent.com/48679574/118399195-fbc0d380-b696-11eb-9b8a-cc7829925284.png" width="400px" title="Advanced SIFT">

## Normal SURF (left).     &&     Advanced SURF (Right)
<img src="https://user-images.githubusercontent.com/48679574/118399211-12ffc100-b697-11eb-9e6c-14445bafec77.png" width="400px" title="Normal SURF"><img src="https://user-images.githubusercontent.com/48679574/118399213-15621b00-b697-11eb-8989-8a8ae518c1b6.png" width="400px" title="Advanced SURF">


# Result-2
# BEBLID Detector for Levi Ackerman　(from opencv 4.5.1)

## BEBLID

<img src="https://user-images.githubusercontent.com/48679574/118399382-d97b8580-b697-11eb-8690-98d00d09b0f7.png" title="Levi Ackerman" width="700px">

```
# beblid_homography 
[[ 1. -0.  0.]
 [-0.  1.  0.]
 [ 0. -0.  1.]]
```

## Advanced SIFT

<img src="https://user-images.githubusercontent.com/48679574/118399387-ded8d000-b697-11eb-98b3-ca5f28a05f20.png" width="700px">

```
# sift_homography 
[[ 0.65544  0.07709 24.86545]
 [-0.05323  0.77897 60.19786]
 [-0.00042  0.00018  1.     ]]
```

## Advanced SURF

<img src="https://user-images.githubusercontent.com/48679574/118399405-f0ba7300-b697-11eb-924f-cedcac8727c7.png" width="700px">

```
# surf_homography 
[[    53.42561     24.04696 -19285.00593]
 [    16.87045     53.18019 -14406.12753]
 [     0.0335       0.05074      1.     ]]
```

## Advanced AKAZE

<img src="https://user-images.githubusercontent.com/48679574/118399414-fb750800-b697-11eb-9051-5fe6484c3d6b.png" width="700px">

```
# akaze_homography
[[  -1.71304    5.38194  -48.70431]
 [  -0.66598    3.33049 -221.6983 ]
 [  -0.00683    0.01311    1.     ]]
```

# opencv methods

## normal image

![Elevator_shoot](https://user-images.githubusercontent.com/48679574/118398920-9fa97f80-b695-11eb-9c1d-a77e32a50415.jpeg)


## Thresholding (type: A, B, C, D)

<img src="https://user-images.githubusercontent.com/48679574/118398922-a1734300-b695-11eb-8739-0b3df05c66ff.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/118398925-a2a47000-b695-11eb-88b8-e4b9afd36ab6.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/118398927-a46e3380-b695-11eb-8228-b2afe7d07691.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/118398929-a637f700-b695-11eb-9e21-3d6c8692883d.png" width="200px">


## canny_edge_detector.  &&. reduce_color (num_color = 2)

<img src="https://user-images.githubusercontent.com/48679574/118398991-f9aa4500-b695-11eb-9f1a-94a6878abb19.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/118398995-fe6ef900-b695-11eb-91b6-8c014f871af4.png" width="400px">

## segmentation && masking

<img src="https://user-images.githubusercontent.com/48679574/118399007-0cbd1500-b696-11eb-9219-e2d5628bfe44.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/118399009-0f1f6f00-b696-11eb-8dc2-87341b575a72.png" width="400px">



## FeatureMatch_similarity of two images

![similaity](https://user-images.githubusercontent.com/48679574/118399014-121a5f80-b696-11eb-9495-d4ebff19dd69.png)


# References
- [OpenCVで他のどの記事よりも頑強に特徴量マッチングしてみた(Python, AKAZE)](https://qiita.com/grouse324/items/74988134a9073568b32d)
- [Detecting garbage homographies from findHomography in OpenCV?](https://stackoverflow.com/questions/10972438/detecting-garbage-homographies-from-findhomography-in-opencv)
