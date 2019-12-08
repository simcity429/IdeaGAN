from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from torch import nn
from torchsummary import summary


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # mean: 0, std: 0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # mean: 1, std: 0.02
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, params, noise_dim):
        super(Generator, self).__init__()
        self.params = params
        in_channels = noise_dim
        self.noise_dim = noise_dim

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
        x = x.view(x.size(0), self.noise_dim, 1, 1)
        for module in self.convs:
            x = module(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, params, img_ch, img_size, layernorm=False):
        super(Discriminator, self).__init__()
        self.params = params
        in_channels = img_ch
        new_size = img_size

        self.convs = nn.ModuleList()
        for i, (out_channels, kernel_size, stride, padding) in enumerate(self.params):
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
        for module in self.convs:
            x = module(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # test models
    noise_dim = 100
    img_size = 32
    ngf = 64
    ndf = 64
    img_ch = 1

    # (ch, kernel, stride, padding)
    # generator : new_size = (size -1)*s + k - 2p
    # discriminator : new_size = (size - k + 2p)/s + 1 ... output_layer : size - k + 2p = 0, stride=1
    #   scaling factor = params[][2], where (params[][1] - params[][2]) / 2 = params[][3]
    G_params = [(ngf*4, 4, 1, 0),  (ngf*2, 4, 2, 1), (ngf*1, 4, 2, 1), (img_ch, 4, 2, 1)]
    D_params = [(ndf*1, 4, 2, 1), (ndf*2, 4, 2, 1), (ndf*4, 4, 2, 1), (ndf*8, 4, 2, 1)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(G_params, noise_dim)
    netD = Discriminator(D_params, img_ch, img_size,layernorm=True)
    netG.cuda()
    netD.cuda()
    print( summary(netG, input_size=(noise_dim,)) )
    print( summary(netD, input_size=(img_ch, img_size, img_size)) )
