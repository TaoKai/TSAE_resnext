import cv2
import torch
from SAE_resnext import SAE_DECODER
import os, sys
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_prepare(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img/255-mean)/std
    img = img.transpose([2, 0, 1])
    return img

def img_rollback(img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.transpose([1, 2, 0])
    img = (img*std+mean)*255
    img = img.astype(np.uint8)
    return img

def init_model():
    model_path = 'faceTSAE_model.pth'
    model = SAE_DECODER(encoder_grad=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict():
    path = 'tmp'
    pics = [path+'/'+fp for fp in os.listdir(path)]
    random.shuffle(pics)
    model = init_model()
    for p in pics:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        orig = img.copy()
        img = img_prepare(img)
        img = torch.tensor([img]).float()
        out = model(img)[0]
        out = img_rollback(out.detach().numpy())
        cat_img = np.concatenate([orig, out], axis=1)
        cv2.imshow('cat', cat_img)
        cv2.waitKey(0)

if __name__ == "__main__":
    predict()

    
