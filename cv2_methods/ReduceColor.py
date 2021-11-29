import cv2
from sklearn.cluster import MiniBatchKMeans

def reduce_color(color_image, num_color=16):
    (h, w) = color_image.shape[:2]
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters = num_color)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)