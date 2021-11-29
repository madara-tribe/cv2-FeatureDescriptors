import cv2
import matplotlib.pyplot as plt

def Thresholding(color_image, method='A'):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    if method=='A':
        ret,thresh = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)        
    elif method=='B':
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    elif method=='C':
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    elif method=='D':
        ret1,thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    return thresh

    
if __name__=='__main__':
    stack, stack2 = [], []
    plt.imshow(img),plt.show()
    plt.imshow(image),plt.show()
    methods = ['A', 'B', 'C', 'D']
    for method in methods:
        print(method)
        im1 = Thresholding(img, method=method)
        im2 = Thresholding(image, method=method)
        stack.append(im1)
        stack2.append(im2)
        plt.imshow(im1),plt.show()
        plt.imshow(im2),plt.show()
    concat = np.hstack(stack)
    concat2 = np.hstack(stack2)
    plt.imshow(concat),plt.show()
    plt.imshow(concat2),plt.show()