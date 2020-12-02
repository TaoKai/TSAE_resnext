import random
from skimage import transform as trans
import numpy as np
import cv2

def random_warp(image):
    assert image.shape==(112,96,3)
    num = 9
    rangeX = np.linspace(4, 92, num)
    mapx = np.broadcast_to(rangeX, (num, num))
    rangeY = np.linspace(4, 108, num)
    mapy = np.broadcast_to(rangeY, (num, num)).T
    mapx = mapx + np.random.normal(size=(num, num), scale=2)
    mapy = mapy + np.random.normal(size=(num, num), scale=2)
    interp_mapx = cv2.resize(mapx, (88, 104)).astype('float32')
    interp_mapy = cv2.resize(mapy, (88, 104)).astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    warped = image.copy()
    warped[4:108, 4:92, :] = warped_image
    return warped, image

def random_warp_legacy(image):
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - 80, 128 + 80, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)
    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (64, 64))
    return warped_image, target_image

def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T

def faceAlign(img, points):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2,:]

    shp = [112, 96]
    if M is None:
        return None
    else:
        warped = cv2.warpAffine(img, M, (shp[1],shp[0]), borderValue = 0.0)
        return warped

def random_facepair(img):
    warp = None
    if random.random()>0.5:
        warp, img = random_warp(img)
    else:
        warp = img.copy()
    if random.random()>0.5:
        warp = cv2.flip(warp, 1)
        img = cv2.flip(img, 1)
    return warp, img
    

if __name__=='__main__':
    path = 'obm.jpg'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    while True:
        warp, img = random_facepair(img)
        out = np.concatenate([warp, img], axis=1)
        cv2.imshow('out', out)
        cv2.waitKey(500)
