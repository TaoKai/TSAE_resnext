import numpy as np
import cv2
from skimage import transform as trans

def faceAlign(img, points):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    shp = [112, 96]
    if M is None:
        return None
    else:
        warped = cv2.warpAffine(img, M, (shp[1],shp[0]), borderValue = 0.0)
        return warped

def distort():
    path = 'rzr.png'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h = img.shape[0]
    w = img.shape[1]
    print('w', w, 'h', h)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    distort()