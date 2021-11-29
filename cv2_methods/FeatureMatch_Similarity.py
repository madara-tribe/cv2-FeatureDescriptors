import numpy as np
import cv2

def FeatureMatch_similarity(query, original):
    MAX_FEATURES = 500
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    im1Gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    detector = cv2.ORB_create(MAX_FEATURES)
    (target_kp, target_des) = detector.detectAndCompute(im1Gray, None)

    try:
        original_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        (comparing_kp, comparing_des) = detector.detectAndCompute(original_img, None)
        matches = bf.match(target_des, comparing_des)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
    except cv2.error:
        print('error, try again')

    print(ret)

def hist_similarity(query, original):
    ch_names = {0: 'Hue', 1: 'Saturation', 2: 'Brightness'}

    # convert to HSV
    hsv1 = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # caliculate histgram similarity at each channels
    scores, hists1, hists2 = [], [], []
    for ch in ch_names:
        h1 = cv2.calcHist(
            [hsv1], [ch], None, histSize=[256], ranges=[0, 256])
        h2 = cv2.calcHist(
            [hsv2], [ch], None, histSize=[256], ranges=[0, 256])
        score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        hists1.append(h1)
        hists2.append(h2)
        scores.append(score)
    mean = np.mean(scores)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for [axL, axR], hist, img in zip(axes, [hists1, hists2], [query, original]):
        # draw image
        axL.imshow(img[..., ::-1])
        axL.axis('off')
        # make histgram
        for i in range(3):
            axR.plot(hist[i], label=ch_names[i])
        axR.legend()
    fig.suptitle('similarity={:.2f}'.format(mean))
    plt.show()


# difine target image
def main(target, query):
    th, tw = target.shape[:2]
    print(th, tw)

    hs, ws = query.shape[:2]

    print('resize query or target')
    if hs<th and ws<tw:
        query = cv2.resize(query, (th, tw))
    else:
        target = cv2.resize(target, (hs, ws))

    # featues similarity
    FeatureMatch_similarity(query, target)

    # hist similarity
    hist_similarity(query, target)
    
if __name__=='__main__':
    img1 = cv2.imread('2s/origin.png')
    img2 = cv2.imread('movie.jpeg')
    main(img1, img2)

