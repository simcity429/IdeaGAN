import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
        self.opt = optim.RMSprop(self.parameters(), lr=BIG_D_LR)
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

class IdeaGAN(nn.Module):
    def __init__(self):
        self.idea_maker = IdeaMaker()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.little_d = Little_D()
        self.big_d = Big_D()

    def make_fake_img(self):
        noise = torch.randn(BATCH_SIZE, NOISE_DIM)
        idea = self.idea_maker(noise)
        fake_img = self.decoder(idea)
        return fake_img.detach()

    def g_fake_pass(self):
        noise = torch.randn(BATCH_SIZE, NOISE_DIM)
        idea = self.idea_maker(noise)
        little_d_out = self.little_d(idea)
        fake_img = self.decoder(idea)
        detached_fake_img = fake_img.detach()
        big_d_out = self.big_d(fake_img)
        return little_d_out[:, -1], big_d_out, detached_fake_img

    def g_real_pass(self, real_img):
        #real_img::(BATCH_SIZE, IMG_CH, IMG_SIZE, IMG_SIZE), torch.FloatTensor
        idea = self.encoder(real_img)
        little_d_out = self.little_d(idea)
        reconstructed = self.decoder(idea)
        return little_d_out, reconstructed

    def d_pass(self, real_img, fake_img):
        real_out = torch.mean(self.big_d(real_img))
        fake_out = torch.mean(self.big_d(fake_img))
        return real_out - fake_out

if __name__ == '__main__':
    print(DEVICE)
    transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.ToTensor(),  # convert in the range [0.0, 1.0]
                                transforms.Normalize([0.5], [0.5])])  # (ch - m) / s -> [-1, 1]
    mnist = datasets.MNIST('./mnist_data', download=True, train=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=1, shuffle=True, num_workers=1)
    for x, y in data_loader:
        print(x.type())
        save_image(x, './tmp/tmp.png', normalize=True)
        break
    print(x.to(DEVICE))

