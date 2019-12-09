import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.utils import save_image
from itertools import chain
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
        self.cov = torch.eye(NOISE_DIM)
        self.network = nn.Sequential(
            nn.Linear(NOISE_DIM, IM_HIDDEN_UNIT_NUM),
            nn.BatchNorm1d(IM_HIDDEN_UNIT_NUM),
            nn.ReLU(),
            nn.Linear(IM_HIDDEN_UNIT_NUM, NOISE_DIM)
        )
        self.apply(init_weights)
        self.opt = optim.Adam(self.parameters(), IDEAMAKER_LR)


    def forward(self, x):
        m = self.network(x)
        dist = MultivariateNormal(m, self.cov)
        return dist.rsample()

class Encoder(nn.Module):
    def __init__(self, params=None, layernorm=True):
        super(Encoder, self).__init__()
        self.cov = torch.eye(NOISE_DIM)
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
        self.opt = optim.Adam(self.parameters(), ENCODER_LR)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1)
        m = self.fc(x)
        dist = MultivariateNormal(m, self.cov)
        return dist.rsample()

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
        self.opt = optim.Adam(self.parameters(), DECODER_LR)

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
        self.opt = optim.RMSprop(self.parameters(), LITTLE_D_LR)

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
        self.opt = optim.RMSprop(self.parameters(), lr=BIG_D_LR)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class IdeaGAN(nn.Module):
    def __init__(self):
        super(IdeaGAN, self).__init__()
        self.idea_maker = IdeaMaker().to(DEVICE)
        self.encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.little_d = Little_D().to(DEVICE)
        self.big_d = Big_D().to(DEVICE)
        self.logit_bceloss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def make_fake_img(self):
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
        idea = self.idea_maker(noise)
        fake_img = self.decoder(idea)
        return fake_img.detach().to(DEVICE)

    def make_idea(self, real_img):
        real_idea = self.encoder(real_img)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
        fake_idea = self.idea_maker(noise)
        return real_idea.detach(), fake_idea.detach()

    def make_restored_img(self, real_img):
        real_idea = self.encoder(real_img)
        reconstructed = self.decoder(real_idea)
        return reconstructed.detach().to(DEVICE)

    def g_fake_pass(self):
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
        fake_idea = self.idea_maker(noise)
        little_d_out = self.little_d(fake_idea)
        fake_img = self.decoder(fake_idea)
        big_d_out = self.big_d(fake_img)
        adv_big_d_loss = -torch.mean(big_d_out)
        real_label = torch.ones(BATCH_SIZE).to(DEVICE)
        adv_little_d_loss = self.logit_bceloss(little_d_out[:, -1], real_label)
        return adv_little_d_loss, adv_big_d_loss, fake_img.detach(), fake_idea.detach()

    def g_real_pass(self, real_img, answer):
        #real_img::(BATCH_SIZE, IMG_CH, IMG_SIZE, IMG_SIZE), torch.FloatTensor
        real_idea = self.encoder(real_img)
        little_d_out = self.little_d(real_idea)
        reconstructed = self.decoder(real_idea)
        recon_loss = self.l1_loss(real_img, reconstructed)
        classifier_loss = self.crossentropy_loss(little_d_out[:, :-1], answer)
        return recon_loss, classifier_loss, real_idea.detach()

    def big_d_pass(self, real_img, fake_img):
        real_out = self.big_d(real_img)
        fake_out = self.big_d(fake_img)
        return torch.mean(real_out - fake_out)

    def little_d_pass(self, real_idea, fake_idea, answer):
        real_out = self.little_d(real_idea)
        fake_out = self.little_d(fake_idea)
        classifier_loss = self.crossentropy_loss(real_out[:, :-1], answer)
        little_d_loss = torch.mean(real_out - fake_out)
        return classifier_loss, little_d_loss

    def d_only_update(self, real_img, answer):
        #big_d update
        fake_img = self.make_fake_img()
        big_d_loss = self.big_d_pass(real_img, fake_img)
        self.zero_grad()
        big_d_loss.backward()
        self.big_d.opt.step()
        for p in self.big_d.parameters():
            p.data.clamp_(-0.01, 0.01)
        #little_d update
        real_idea, fake_idea = self.make_idea(real_img)
        classifier_loss, little_d_loss = self.little_d_pass(real_idea, fake_idea, answer)
        little_d_loss = CLASSIFIER_COEF*classifier_loss + little_d_loss
        self.zero_grad()
        little_d_loss.backward()
        self.little_d.opt.step()
        for p in self.big_d.parameters():
            p.data.clamp_(-0.01, 0.01)

    def update_all(self, real_img, answer):
        #encoder, decoder, idea_maker update
        adv_little_d_loss, adv_big_d_loss, fake_img, fake_idea = self.g_fake_pass()
        recon_loss, classifier_loss, real_idea = self.g_real_pass(real_img, answer)
        loss = adv_little_d_loss + adv_big_d_loss + recon_loss + CLASSIFIER_COEF*classifier_loss
        print(loss)
        self.zero_grad()
        loss.backward()
        self.idea_maker.opt.step()
        self.encoder.opt.step()
        self.decoder.opt.step()
        #little_d_update
        little_classifier_loss, little_d_loss = self.little_d_pass(real_idea, fake_idea, answer)
        loss = CLASSIFIER_COEF*little_classifier_loss + little_d_loss
        print(loss)
        self.zero_grad()
        loss.backward()
        self.little_d.opt.step()
        #big_d_update
        loss = self.big_d_pass(real_img, fake_img)
        print(loss)
        self.zero_grad()
        loss.backward()
        self.big_d.opt.step()
        for p in self.big_d.parameters():
            p.data.clamp_(-0.01, 0.01)

