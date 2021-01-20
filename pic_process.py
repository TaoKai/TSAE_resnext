import cv2
import os, sys, shutil
from codecs import open
import numpy as np
import torch
import random
from face_distort import random_facepair, random_facepair_68
from mtcnn import DetectFace
from dlib_util import test_68points, get_det_pred, show_68points
from skimage import transform as trans

detect = DetectFace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det68, pred68 = get_det_pred()

landmarks_2D = np.array([
[ 0.000213256,  0.106454  ], #17
[ 0.0752622,    0.038915  ], #18
[ 0.18113,      0.0187482 ], #19
[ 0.29077,      0.0344891 ], #20
[ 0.393397,     0.0773906 ], #21
[ 0.586856,     0.0773906 ], #22
[ 0.689483,     0.0344891 ], #23
[ 0.799124,     0.0187482 ], #24
[ 0.904991,     0.038915  ], #25
[ 0.98004,      0.106454  ], #26
[ 0.490127,     0.203352  ], #27
[ 0.490127,     0.307009  ], #28
[ 0.490127,     0.409805  ], #29
[ 0.490127,     0.515625  ], #30
[ 0.36688,      0.587326  ], #31
[ 0.426036,     0.609345  ], #32
[ 0.490127,     0.628106  ], #33
[ 0.554217,     0.609345  ], #34
[ 0.613373,     0.587326  ], #35
[ 0.121737,     0.216423  ], #36
[ 0.187122,     0.178758  ], #37
[ 0.265825,     0.179852  ], #38
[ 0.334606,     0.231733  ], #39
[ 0.260918,     0.245099  ], #40
[ 0.182743,     0.244077  ], #41
[ 0.645647,     0.231733  ], #42
[ 0.714428,     0.179852  ], #43
[ 0.793132,     0.178758  ], #44
[ 0.858516,     0.216423  ], #45
[ 0.79751,      0.244077  ], #46
[ 0.719335,     0.245099  ], #47
[ 0.254149,     0.780233  ], #48
[ 0.726104,     0.780233  ], #54
], dtype=np.float32)

def getFileList(path):
    files = os.listdir(path)
    files = [path+'/'+fp for fp in files]
    return files

def innerGen68(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    dic = test_68points(img, det68, pred68)
    if dic is None:
        return None
    points = []
    for _, v in dic.items():
        points += v
    points = points[17:49]+points[54:55]
    points = np.array(points, dtype=np.float32)
    src = landmarks_2D*np.array([128, 128])+np.array([16, 16])
    tform = trans.SimilarityTransform()
    tform.estimate(points, src)
    M = tform.params[0:3,:]
    if M is None:
        return None
    else:
        # warped = cv2.warpAffine(img, M[:2], (64,64), borderValue = 0.0)
        # cv2.imshow('img', warped)
        # cv2.waitKey(1)
        ml = list(M[:2].reshape(-1))
        str_ml = ''
        for m in ml:
            str_ml += str(m)+' '
        return str_ml.strip()

def generateAB68():
    pathA = 'faces/yueyunpeng'
    pathB = 'faces/dilireba_adv'
    Afiles = getFileList(pathA)
    Bfiles = getFileList(pathB)
    A_str = ''
    B_str = ''
    for a in Afiles:
        m_str = innerGen68(a)
        if m_str is not None:
            A_str += a+' '+m_str+'\n'
            print(a, m_str)
    for b in Bfiles:
        m_str = innerGen68(b)
        if m_str is not None:
            B_str += b+' '+m_str+'\n'
            print(b, m_str)
    open('A_68.txt', 'w', 'utf-8').write(A_str.strip())
    open('B_68.txt', 'w', 'utf-8').write(B_str.strip())



def generateABCoord():
    pathA = 'faces/andy'
    pathB = 'faces/obama'
    Afiles = getFileList(pathA)
    Bfiles = getFileList(pathB)
    A_str = ''
    B_str = ''
    for a in Afiles:
        img = cv2.imread(a, cv2.IMREAD_COLOR)
        if img is None:
            continue
        _, data = detect(img)
        if data[1].shape[0]==0:
            continue
        points = data[1][0]
        points = points.min(axis=0)-np.array([13, 14], dtype=np.float32)
        lt = points.clip(0, 96).astype(np.int32)
        A_str += a+' '+str(lt[0])+' '+str(lt[1])+'\n'
        print(a, lt)
        cut = img[lt[1]:lt[1]+64, lt[0]:lt[0]+64, :]
        cv2.imshow('cut', cut)
        cv2.waitKey(1)
    for b in Bfiles:
        img = cv2.imread(b, cv2.IMREAD_COLOR)
        if img is None:
            continue
        _, data = detect(img)
        if data[1].shape[0]==0:
            continue
        points = data[1][0]
        points = points.min(axis=0)-np.array([13, 14], dtype=np.float32)
        lt = points.clip(0, 96).astype(np.int32)
        B_str += b+' '+str(lt[0])+' '+str(lt[1])+'\n'
        print(b, lt)
        cut = img[lt[1]:lt[1]+64, lt[0]:lt[0]+64, :]
        cv2.imshow('cut', cut)
        cv2.waitKey(1)
    open('A_c.txt', 'w', 'utf-8').write(A_str.strip())
    open('B_c.txt', 'w', 'utf-8').write(B_str.strip())

def generateAB():
    pathA = 'faces/andy'
    pathB = 'faces/obama'
    Afiles = getFileList(pathA)
    Bfiles = getFileList(pathB)
    A_str = ''
    B_str = ''
    for a in Afiles:
        A_str += a+'\n'
    for b in Bfiles:
        B_str += b+'\n'
    open('A.txt', 'w', 'utf-8').write(A_str.strip())
    open('B.txt', 'w', 'utf-8').write(B_str.strip())

class FaceData(object):
    def __init__(self, pathA, pathB, batch_size):
        self.batch_size = batch_size
        self.Alist = self.getPaths(pathA)
        self.Blist = self.getPaths(pathB)
        self.cur_A = 0
        self.cur_B = 0
        self.len_A = len(self.Alist)
        self.len_B = len(self.Blist)
        print('caculate mean color.')
        self.mean_A = self.get_mean_color(self.Alist)
        self.mean_B = self.get_mean_color(self.Blist)
        print('A', self.mean_A, 'B', self.mean_B)
    
    def getAffineMat(self, strs):
        mat = []
        strs = strs[1:]
        for s in strs:
            mat.append(float(s))
        mat = np.array(mat, dtype=np.float32).reshape(2, 3)
        return mat

    def getTestBatch(self):
        lst = self.Blist.copy()
        random.shuffle(lst)
        lst = lst[:64]
        imgs = []
        for l in lst:
            ls = l.split(' ')
            p = ls[0]
            M = self.getAffineMat(ls)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.warpAffine(img, M, (144, 144), borderValue=0.0)
                imgs.append(img)
        imgs = np.array(imgs, dtype=np.uint8)
        return imgs

    def get_mean_color(self, pics):
        cnt = 0
        mean_color = np.zeros(3, dtype=np.float32)
        for pp in pics:
            ls = pp.split(' ')
            p = ls[0]
            M = self.getAffineMat(ls)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.warpAffine(img, M, (144, 144), borderValue=0.0)
            img = img/255.0
            mean_color += img.mean(axis=(0, 1))
            cnt += 1
        return mean_color/cnt

    def getTrainBatch(self, batch):
        imgs = []
        warps = []
        for fp in batch:
            ls = fp.split(' ')
            p = ls[0]
            M = self.getAffineMat(ls)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            warp, img = random_facepair_68(img, M)
            imgs.append(img)
            warps.append(warp)
        imgs = np.array(imgs, dtype=np.float32)/255
        warps = np.array(warps, dtype=np.float32)/255
        return warps, imgs

    def to_tensor(self, mat):
        mat = torch.from_numpy(mat).float()
        mat = mat.permute(0, 3, 1, 2)
        return mat

    def next(self):
        if self.cur_A+self.batch_size<=self.len_A:
            batchA = self.Alist[self.cur_A:self.cur_A+self.batch_size]
            self.cur_A += self.batch_size
        else:
            self.cur_A = 0
            random.shuffle(self.Alist)
            batchA = self.Alist[self.cur_A:self.cur_A+self.batch_size]
            self.cur_A += self.batch_size
        if self.cur_B+self.batch_size<=self.len_B:
            batchB = self.Blist[self.cur_B:self.cur_B+self.batch_size]
            self.cur_B += self.batch_size
        else:
            self.cur_B = 0
            random.shuffle(self.Blist)
            batchB = self.Blist[self.cur_B:self.cur_B+self.batch_size]
            self.cur_B += self.batch_size
        warpsA, imgsA = self.getTrainBatch(batchA)
        warpsB, imgsB = self.getTrainBatch(batchB)
        warpsA += self.mean_B-self.mean_A
        imgsA += self.mean_B-self.mean_A
        warpsA = warpsA.clip(0, 1)
        imgsA = imgsA.clip(0, 1)
        warpsA = self.to_tensor(warpsA)
        imgsA = self.to_tensor(imgsA)
        warpsB = self.to_tensor(warpsB)
        imgsB = self.to_tensor(imgsB)
        return warpsA, imgsA, warpsB, imgsB
    
    def getPaths(self, path):
        lines = open(path, 'r', 'utf-8').read().strip().split('\n')
        return lines

def sharp(image):
    kernel = np.array(
        [[-1, 0, -1], 
        [0, 5, 0], 
        [-1, 0, -1]],
        dtype=np.float32
    )
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

if __name__ == "__main__":
    generateAB68()
    face = FaceData('A_68.txt', 'B_68.txt', 6)
    while True:
        warpsA, imgsA, warpsB, imgsB = face.next()
        print(face.cur_A, face.cur_B, warpsA.shape, imgsA.shape, warpsB.shape, imgsB.shape)
        wa = warpsA.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 144, 3)*255
        wa = wa.astype(np.uint8)
        ia = imgsA.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 144, 3)*255
        ia = ia.astype(np.uint8)
        wb = warpsB.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 144, 3)*255
        wb = wb.astype(np.uint8)
        ib = imgsB.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 144, 3)*255
        ib = ib.astype(np.uint8)
        out = np.concatenate([wa, ia, wb, ib], axis=1)
        cv2.imshow('out', out)
        cv2.waitKey(100)