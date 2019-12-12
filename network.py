import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
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
        self.cov = COV_COEF*torch.eye(NOISE_DIM).to(DEVICE)
        zero_m = torch.zeros(NOISE_DIM).to(DEVICE)
        self.target_dist = MultivariateNormal(zero_m, self.cov)
        self.network = nn.Sequential(
            nn.Linear(NOISE_DIM, IM_HIDDEN_UNIT_NUM),
            nn.BatchNorm1d(IM_HIDDEN_UNIT_NUM),
            nn.PReLU(),
            nn.Linear(IM_HIDDEN_UNIT_NUM, IM_HIDDEN_UNIT_NUM),
            nn.BatchNorm1d(IM_HIDDEN_UNIT_NUM),
            nn.PReLU(),
            nn.Linear(IM_HIDDEN_UNIT_NUM, NOISE_DIM)
        )
        self.apply(init_weights)
        self.opt = optim.Adam(self.parameters(), IDEAMAKER_LR)


    def forward(self, x):
        m = self.network(x)
        dist = MultivariateNormal(m, self.cov)
        return dist.mean, torch.mean(kl_divergence(self.target_dist, dist))

class Encoder(nn.Module):
    def __init__(self, params=None, layernorm=True):
        super(Encoder, self).__init__()
        self.cov = COV_COEF*torch.eye(NOISE_DIM).to(DEVICE)
        zero_m = torch.zeros(NOISE_DIM).to(DEVICE)
        self.target_dist = MultivariateNormal(zero_m, self.cov)
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
        ret = torch.clamp(m, -1, 1)
        dist = MultivariateNormal(m, self.cov)
        return ret, torch.mean(kl_divergence(self.target_dist, dist))

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
                self.convs.append(nn.PReLU())  # except output layer
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
'''
            nn.Conv1d(NOISE_DIM, FILTER_NUM, 2),
            #(batch, 16, 13)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(FILTER_NUM*15, 1),
'''
'''
self.disc_network = nn.Sequential(
            nn.Conv1d(NOISE_DIM, 2*FILTER_NUM, 4),
            #(batch, FILTER_NUM, int(BATCH_SIZE/4) - 3)
            nn.PReLU(),
            nn.Conv1d(2*FILTER_NUM, FILTER_NUM, 4),
            #(batch, FILTER_NUM, int(BATCH_SIZE/4) - 6)
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(FILTER_NUM*(int(BATCH_SIZE/4)-6), 1),
        )
'''
'''
self.disc_network = nn.Sequential(
            nn.Linear(NOISE_DIM, 2*NOISE_DIM),
            nn.PReLU(),
            nn.Linear(2*NOISE_DIM, 2*NOISE_DIM),
            nn.PReLU(),
            nn.Linear(2*NOISE_DIM, 1),
        )
'''
class Little_D(nn.Module):
    def __init__(self):
        super(Little_D, self).__init__()
        self.class_network = nn.Sequential(
            nn.Linear(NOISE_DIM, CLASS_NUM)
        )
        encoder_layer = nn.modules.transformer.TransformerEncoderLayer(NOISE_DIM, 5, dim_feedforward=4*NOISE_DIM)
        self.transformer_network = nn.modules.transformer.TransformerEncoder(encoder_layer, 1)
        self.disc_network = nn.Sequential(
            nn.Linear(NOISE_DIM, 4*NOISE_DIM),
            nn.PReLU(),
            nn.Linear(4*NOISE_DIM, 2*NOISE_DIM),
            nn.PReLU(),
            nn.Linear(2*NOISE_DIM, 1),
        )
        self.d_out = nn.Linear(NOISE_DIM*int(BATCH_SIZE/BATCH_DIV), 1)
        self.apply(init_weights)
        self.opt = optim.RMSprop(self.parameters(), LITTLE_D_LR)

    def forward(self, x):
        c = self.class_network(x)
        if MODE == 'TRANSFORMER':
            x = x.view(int(BATCH_SIZE/BATCH_DIV), -1, NOISE_DIM)
            d = self.transformer_network(x)
            d = nn.functional.relu(d.view(BATCH_DIV, -1))
            d = self.d_out(d)
        else:
            d = self.disc_network(x)
        return c, d



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
        idea, _ = self.idea_maker(noise)
        fake_img = self.decoder(idea)
        return fake_img.detach().to(DEVICE)

    def make_fake_img_by_noise(self):
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
        fake_img = self.decoder(noise)
        return fake_img.detach().to(DEVICE)

    def make_idea(self, real_img):
        real_idea, _ = self.encoder(real_img)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
        fake_idea, _ = self.idea_maker(noise)
        return real_idea.detach(), fake_idea.detach()

    def make_restored_img(self, real_img):
        real_idea, _ = self.encoder(real_img)
        reconstructed = self.decoder(real_idea)
        return reconstructed.detach().to(DEVICE)

    def g_fake_pass(self):
        noise = torch.randn(BATCH_SIZE, NOISE_DIM).to(DEVICE)
        fake_idea, kl = self.idea_maker(noise)
        _, d = self.little_d(fake_idea)
        fake_img = self.decoder(fake_idea.detach())
        big_d_out = self.big_d(fake_img)
        adv_big_d_loss = -torch.mean(big_d_out)
        adv_little_d_loss = -torch.mean(d) + KL_COEF*kl
        return adv_little_d_loss, adv_big_d_loss, fake_img.detach(), fake_idea.detach()

    def g_real_pass(self, real_img, answer):
        #real_img::(BATCH_SIZE, IMG_CH, IMG_SIZE, IMG_SIZE), torch.FloatTensor
        real_idea, kl = self.encoder(real_img)
        c, _ = self.little_d(real_idea)
        reconstructed = self.decoder(real_idea)
        recon_loss = self.l1_loss(real_img, reconstructed) + KL_COEF*kl
        classifier_loss = self.crossentropy_loss(c, answer)
        return recon_loss, classifier_loss, real_idea.detach()

    def big_d_pass(self, real_img, fake_img):
        real_out = self.big_d(real_img)
        fake_out = self.big_d(fake_img)
        return -torch.mean(real_out - fake_out)

    def little_d_pass(self, real_idea, fake_idea, answer):
        real_c, real_d = self.little_d(real_idea)
        fake_c, fake_d = self.little_d(fake_idea)
        classifier_loss = self.crossentropy_loss(real_c, answer)
        little_d_loss = -torch.mean(real_d - fake_d)
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
        classifier_loss, _little_d_loss = self.little_d_pass(real_idea, fake_idea, answer)
        little_d_loss = CLASSIFIER_COEF*classifier_loss + _little_d_loss
        self.zero_grad()
        little_d_loss.backward()
        self.little_d.opt.step()
        for p in self.big_d.parameters():
            p.data.clamp_(-0.01, 0.01)
        return _little_d_loss

    def update_all(self, real_img, answer):
        #encoder, decoder, idea_maker update
        adv_little_d_loss, adv_big_d_loss, fake_img, fake_idea = self.g_fake_pass()
        recon_loss, classifier_loss, real_idea = self.g_real_pass(real_img, answer)
        loss = adv_little_d_loss + 0*adv_big_d_loss + RECON_COEF*recon_loss + CLASSIFIER_COEF*classifier_loss
        print('adv_little_d_loss: ',adv_little_d_loss.data, 'recon: ', recon_loss.data, 'classifier: ', classifier_loss.data)
        self.zero_grad()
        loss.backward()
        self.idea_maker.opt.step()
        self.encoder.opt.step()
        self.decoder.opt.step()
        #little_d_update
        little_classifier_loss, little_d_loss = self.little_d_pass(real_idea, fake_idea, answer)
        loss = CLASSIFIER_COEF*little_classifier_loss + little_d_loss
        print('little_d_loss: ', little_d_loss.data)
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

