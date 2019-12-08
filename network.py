import torch
import numpy as np
from torch import nn
from config import *

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # mean: 0, std: 0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # mean: 1, std: 0.02
        m.bias.data.fill_(0)
    else:
        #use default initializer
        pass

class IdeaMaker(nn.Module):
    def __init__(self):
        super(IdeaMaker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(NOISE_DIM, IM_HIDDEN_UNIT_NUM),
            nn.BatchNorm1d(IM_HIDDEN_UNIT_NUM),
            nn.ReLU(),
            nn.Linear(IM_HIDDEN_UNIT_NUM, NOISE_DIM)
        )
        self.apply(init_weights)


    def forward(self, x):
        return self.network(x)

class Encoder(nn.Module):
    def __init__(self, params=None, layernorm=True):
        super(Encoder, self).__init__()
        if params is None:
            # (ch, kernel, stride, padding)
            params = [(NDF*1, 4, 2, 1), (NDF*2, 4, 2, 1), (NDF*4, 4, 2, 1), (NDF*8, 4, 2, 1)]
        in_channels = IMG_CH
        new_size = IMG_SIZE

        self.convs = nn.ModuleList()
        for i, (out_channels, kernel_size, stride, padding) in enumerate(params):
            new_size = int((new_size - kernel_size + 2*padding)/stride) + 1
            self.convs.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size, stride, padding, bias=False))  # all layers
            if layernorm and i != 0:  
                self.convs.append(nn.LayerNorm((out_channels, new_size, new_size)))  # except input layer
            self.convs.append(nn.LeakyReLU(0.2, inplace=True))  # all layers
            in_channels = out_channels
        self.fc = nn.Linear(in_channels*new_size*new_size, NOISE_DIM)  # output layer
        self.apply(init_weights)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, params=None):
        super(Decoder, self).__init__()
        if params is None:
            # (ch, kernel, stride, padding)
            params = [(NGF*4, 4, 1, 0),  (NGF*2, 4, 2, 1), (NGF*1, 4, 2, 1), (IMG_CH, 4, 2, 1)]
        in_channels = NOISE_DIM

        self.convs = nn.ModuleList()
        for i, (out_channels, kernel_size, stride, padding) in enumerate(params):             
            self.convs.append(nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size, stride, padding, bias=False))  # all layers
            if i < len(params) - 1:     
                self.convs.append(nn.BatchNorm2d(out_channels))  # except output layer
                self.convs.append(nn.ReLU())  # except output layer
            else:                       
                self.convs.append(nn.Tanh())  # output layer
            in_channels = out_channels
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, NOISE_DIM, 1, 1)
        for layer in self.convs:
            x = layer(x)
        return x

class Little_D(nn.Module):
    def __init__(self):
        super(Little_D, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(NOISE_DIM, CLASS_NUM + 1),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.network(x)


class Big_D(nn.Module):
    def __init__(self, params=None, layernorm=True):
        super(Big_D, self).__init__()
        if params is None:
            # (ch, kernel, stride, padding)
            params = [(NDF*1, 4, 2, 1), (NDF*2, 4, 2, 1), (NDF*4, 4, 2, 1), (NDF*8, 4, 2, 1)]
        in_channels = IMG_CH
        new_size = IMG_SIZE

        self.convs = nn.ModuleList()
        for i, (out_channels, kernel_size, stride, padding) in enumerate(params):
            new_size = int((new_size - kernel_size + 2*padding)/stride) + 1
            self.convs.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size, stride, padding, bias=False))  # all layers
            if layernorm and i != 0:  
                self.convs.append(nn.LayerNorm((out_channels, new_size, new_size)))  # except input layer
            self.convs.append(nn.LeakyReLU(0.2, inplace=True))  # all layers
            in_channels = out_channels
        self.fc = nn.Linear(in_channels*new_size*new_size, 1)  # output layer
        self.apply(init_weights)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

if __name__ == '__main__':
    im = IdeaMaker()
    de = Decoder()
    en = Encoder()
    big_d = Big_D()
    little_d = Little_D()

    
    tmp = torch.randn(10, NOISE_DIM)
    print(little_d(en(de(im(tmp)))).size())