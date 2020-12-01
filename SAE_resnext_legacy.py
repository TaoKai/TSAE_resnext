import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAE_RESNEXT_ENCODER(nn.Module):
    def __init__(self):
        super(SAE_RESNEXT_ENCODER, self).__init__()
        self.resnext50 = models.resnext50_32x4d(pretrained=True)
        self.conv1 = self.resnext50.conv1
        self.bn1 = self.resnext50.bn1
        self.relu = self.resnext50.relu
        self.maxpool = self.resnext50.maxpool
        self.L1 = self.resnext50.layer1
        self.L2 = self.resnext50.layer2
        self.L3 = self.resnext50.layer3
        self.L4 = self.resnext50.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1_out = self.L1(x)
        l2_out = self.L2(l1_out)
        l3_out = self.L3(l2_out)
        l4_out = self.L4(l3_out)
        return l4_out, l3_out, l2_out, l1_out

class UPSAMPLE_LAYER(nn.Module):
    def __init__(self, feature_in, feature_out):
        super(UPSAMPLE_LAYER, self).__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.conv = nn.Conv2d(feature_in, feature_out, (1, 1))
        self.bn = nn.BatchNorm2d(feature_out)
        self.activ = nn.ReLU()
    
    def forward(self, layer_d, layer_u):
        layer_d = self.conv(layer_d)
        layer_u = F.upsample(layer_d, size=layer_u.shape[2:], mode='bilinear')+layer_u
        layer_u = self.bn(layer_u)
        layer_u = self.activ(layer_u)
        return layer_u

class SAE_DECODER(nn.Module):
    def __init__(self, encoder_grad=True):
        super(SAE_DECODER, self).__init__()
        self.encoder_grad = encoder_grad
        self.encoder = SAE_RESNEXT_ENCODER()
        self.up_layer43 = UPSAMPLE_LAYER(2048, 1024)
        self.up_layer32 = UPSAMPLE_LAYER(1024, 512)
        self.up_layer21 = UPSAMPLE_LAYER(512, 256)
        self.conv_u0 = nn.Conv2d(256, 32, (1, 1))
        self.conv_u0_3x3 = nn.Conv2d(256, 32, (3, 3), padding=1, padding_mode='reflect')
        self.bn_u0 = nn.BatchNorm2d(32)
        self.conv_u1 = nn.Conv2d(32, 3, (1, 1))
        self.conv_u1_3x3 = nn.Conv2d(32, 3, (3, 3), padding=1, padding_mode='reflect')
        self.bn_u1 = nn.BatchNorm2d(3)
        self.activ = nn.ReLU()
        self.loss_func = nn.MSELoss()
    
    def forward(self, x):
        if self.encoder_grad:
            l4_out, l3_out, l2_out, l1_out = self.encoder(x)
        else:
            with torch.no_grad():
                l4_out, l3_out, l2_out, l1_out = self.encoder(x)
        l3_out = self.up_layer43(l4_out, l3_out)
        l2_out = self.up_layer32(l3_out, l2_out)
        l1_out = self.up_layer21(l2_out, l1_out)
        u0_layer = F.upsample(l1_out, size=(int(x.shape[2]/2), int(x.shape[3]/2)), mode='bilinear')
        u03_layer = self.conv_u0_3x3(u0_layer)
        u0_layer = self.activ(self.bn_u0(self.conv_u0(u0_layer)+u03_layer))
        u1_layer = F.upsample(u0_layer, size=x.shape[2:], mode='bilinear')
        u13_layer = self.conv_u1_3x3(u1_layer)
        u1_layer = self.bn_u1(self.conv_u1(u1_layer)+u13_layer)
        return u1_layer
    
    def loss(self, x, u1_layer):
        cost = self.loss_func(u1_layer, x)
        return cost

if __name__ == "__main__":
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    x = torch.randint(0, 256, (5, 112, 96, 3)).float()/255
    x = (x-mean)/std
    x = x.permute(0, 3, 1, 2)
    sae = SAE_DECODER(encoder_grad=True)
    out = sae(x)
    cost = sae.loss(x, out)
    print(cost, cost.shape)
    torch.save(sae.state_dict(), 'tmp.pth')
