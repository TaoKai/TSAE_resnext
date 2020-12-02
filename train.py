import torch
import torch.optim as optim
import torch.nn as nn
from SAE_resnext import AutoEncoder
from pic_process import FaceData
from codecs import open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, batch_size):
    model_path = 'faceTSAE_model.pth'
    model = AutoEncoder()
    faceData = FaceData('A.txt', 'B.txt', batch_size)
    criterion = nn.L1Loss().to(device)
    model.to(device)
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoderA.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoderB.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    for i in range(epoch):
        model.train()
        for j in range(faceData.len_A//batch_size+1):
            wa, ia, wb, ib = faceData.next()
            wa = wa.to(device)
            ia = ia.to(device)
            wb = wb.to(device)
            ib = ib.to(device)
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            outA = model(wa)
            outB = model(wb)
            loss1 = criterion(outA, ia)
            loss2 = criterion(outB, ib)
            loss1.backward()
            loss2.backward()
            optimizer_1.step()
            optimizer_2.step()
            print('epoch', i, 'step', j, 'loss A', loss1.item(), 'loss B', loss2.item())


if __name__ == "__main__":
    train(10000, 8)