import cv2
import os, sys, shutil
from codecs import open
import numpy as np
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getFileList(path):
    files = os.listdir(path)
    files = [path+'/'+fp for fp in files]
    return files

def split_train_and_test(path, ratio):
    lines = open(path, 'r', 'utf-8').read().strip().split('\n')
    random.shuffle(lines)
    train_str = ''
    test_str = ''
    for i, l in enumerate(lines):
        if i%ratio==0:
            test_str += l+'\n'
        else:
            train_str += l+'\n'
    open('train.txt', 'w', 'utf-8').write(train_str.strip())
    open('test.txt', 'w', 'utf-8').write(test_str.strip())
    print('data splited.')

class FaceData(object):
    def __init__(self, train_path, test_path, batch_size):
        self.train_data = self.getPaths(train_path)
        self.test_data = self.getPaths(test_path)
        self.tr_cur = 0
        self.te_cur = 0
        self.batch_size = batch_size
        self.tr_len = len(self.train_data)
        self.te_len = len(self.test_data)
    
    def img_prepare(self, img):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img/255-mean)/std
        return img

    def next_train(self):
        if self.tr_cur+self.batch_size<=self.tr_len:
            data = self.train_data[self.tr_cur:self.tr_cur+self.batch_size]
            batch = self.getBatch(data)
            self.tr_cur += self.batch_size
            return batch
        else:
            self.tr_cur = 0
            random.shuffle(self.train_data)
            return self.next_train()
    
    def next_test(self):
        if self.te_cur+self.batch_size<=self.te_len:
            data = self.test_data[self.te_cur:self.te_cur+self.batch_size]
            batch = self.getBatch(data)
            self.te_cur += self.batch_size
            return batch
        else:
            self.te_cur = 0
            return self.next_test()

    def getBatch(self, data):
        imgs = []
        for dp in data:
            img = cv2.imread(dp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = self.img_prepare(img)
            imgs.append(img)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(device)
        return imgs

    def getPaths(self, path):
        lines = open(path, 'r', 'utf-8').read().strip().split('\n')
        return lines

if __name__ == "__main__":
    split_train_and_test('allFaces.txt', 50)