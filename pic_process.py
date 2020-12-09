import cv2
import os, sys, shutil
from codecs import open
import numpy as np
import torch
import random
from face_distort import random_facepair, random_facepair_crop
from mtcnn import DetectFace

detect = DetectFace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getFileList(path):
    files = os.listdir(path)
    files = [path+'/'+fp for fp in files]
    return files

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
    
    def getTestBatch(self):
        lst = self.Blist.copy()
        random.shuffle(lst)
        lst = lst[:64]
        imgs = []
        for l in lst:
            ls = l.split(' ')[0]
            p = ls[0]
            x = int(ls[1])
            y = int(ls[2])
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            img = img[y:y+64, x:x+64, :]
            if img is not None:
                imgs.append(img)
        imgs = np.array(imgs, dtype=np.uint8)
        return imgs

    def get_mean_color(self, pics):
        cnt = 0
        mean_color = np.zeros(3, dtype=np.float32)
        for pp in pics:
            ls = pp.split(' ')
            p = ls[0]
            x = int(ls[1])
            y = int(ls[2])
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = img[y:y+64, x:x+64, :]
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
            x = int(ls[1])
            y = int(ls[2])
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            warp, img = random_facepair_crop(img, x, y)
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
    # generateABCoord()
    # path = 'test_out.jpg'
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = sharp(img)
    # cv2.imwrite('sharp.jpg', img)
    face = FaceData('A_c.txt', 'B_c.txt', 6)
    while True:
        warpsA, imgsA, warpsB, imgsB = face.next()
        print(face.cur_A, face.cur_B, warpsA.shape, imgsA.shape, warpsB.shape, imgsB.shape)
        wa = warpsA.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 64, 3)*255
        wa = wa.astype(np.uint8)
        ia = imgsA.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 64, 3)*255
        ia = ia.astype(np.uint8)
        wb = warpsB.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 64, 3)*255
        wb = wb.astype(np.uint8)
        ib = imgsB.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 64, 3)*255
        ib = ib.astype(np.uint8)
        out = np.concatenate([wa, ia, wb, ib], axis=1)
        cv2.imshow('out', out)
        cv2.waitKey(10)
        