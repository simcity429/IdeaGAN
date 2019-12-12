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
from network import init_weights
from config import *


class Tester(nn.Module):
    def __init__(self, load=True, layernorm=True):
        super(Tester, self).__init__()
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
        self.fc = nn.Linear(in_channels*new_size*new_size, CLASS_NUM)  # output layer
        self.apply(init_weights)
        self.crossentropy_loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.parameters(), TESTER_LR)
        self.oracle_dist = torch.distributions.Categorical(logits=torch.ones(CLASS_NUM).to(DEVICE))
        self.to(DEVICE)
        if load:
            self.load_model()

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit

    def train(self, x, answer):
        logit = self.forward(x)
        loss = self.crossentropy_loss(logit, answer)
        self.zero_grad()
        loss.backward()
        self.opt.step()

    def test(self, x, answer):
        logit = self.forward(x)
        pred = torch.argmax(logit, dim=1)
        return np.sum(np.where(torch.eq(answer, pred).cpu().numpy() == True, 1, 0))/BATCH_SIZE

    def gen_test(self, generated):
        logit = self.forward(generated)
        prob = nn.functional.softmax(logit, dim=1)
        max_prob = torch.mean(torch.max(prob, dim=1).values)
        dist_prob = torch.mean(prob, dim=0)
        dist = torch.distributions.Categorical(probs=dist_prob)
        kl_1 = kl_divergence(dist, self.oracle_dist)
        kl_2 = kl_divergence(self.oracle_dist, dist)
        jsd = (kl_1 + kl_2)/2
        return max_prob.detach(), jsd.detach()
        

    def save(self):
        torch.save(self.state_dict(), TESTER_PATH)

    def load_model(self):
        self.load_state_dict(torch.load(TESTER_PATH))

if __name__ == '__main__':
    mode = 'test'
    transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                    transforms.ToTensor(),  # convert in the range [0.0, 1.0]
                                    transforms.Normalize([0.5], [0.5])])  # (ch - m) / s -> [-1, 1]
    mnist = datasets.MNIST('./mnist_data', download=True, train=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_set = datasets.MNIST('./mnist_data', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    tester = Tester().to(DEVICE)
    if mode == 'train':
        EPOCH = 10
        for e in range(EPOCH):
            print('epoch: ', e)
            for x, answer in data_loader:
                x = x.to(DEVICE)
                answer = answer.to(DEVICE)
                tester.train(x, answer)
            cnt = 0
            answer_rate = 0
            for x, answer in test_loader:
                x = x.to(DEVICE)
                answer = answer.to(DEVICE)
                answer_rate += tester.test(x, answer)
                cnt += 1
            print('answer_rate: ', answer_rate/cnt)
            tester.save()
    else:
        tester.load_model()
        cnt = 0
        answer_rate = 0
        for x, answer in test_loader:
            x = x.to(DEVICE)
            answer = answer.to(DEVICE)
            max_prob, jsd = tester.gen_test(x)
            print(max_prob, jsd)