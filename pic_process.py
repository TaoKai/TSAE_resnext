import cv2
import os, sys, shutil
from codecs import open
import numpy as np
import torch
import random
from face_distort import random_facepair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getFileList(path):
    files = os.listdir(path)
    files = [path+'/'+fp for fp in files]
    return files

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
            img = cv2.imread(l, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
        imgs = np.array(imgs, dtype=np.uint8)
        return imgs

    def get_mean_color(self, pics):
        cnt = 0
        mean_color = np.zeros(3, dtype=np.float32)
        for p in pics:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = img/255.0
            mean_color += img.mean(axis=(0, 1))
            cnt += 1
        return mean_color/cnt

    def getTrainBatch(self, batch):
        imgs = []
        warps = []
        for fp in batch:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            warp, img = random_facepair(img)
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
    generateAB()
    # path = 'test_out.jpg'
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = sharp(img)
    # cv2.imwrite('sharp.jpg', img)