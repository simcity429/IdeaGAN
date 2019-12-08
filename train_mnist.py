from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from GANs.WGANGP.models import Generator, Discriminator
import argparse

num_workers = 5
n_critic = 5
n_epochs = 500
BATCH_SIZE = 64
noise_dim = 100
img_size = 32
img_ch = 1
ngf = 64
ndf = 64
lambda_term = 10
lr = 1e-4

USE_CUDA = torch.cuda.is_available()
torch.backends.cudnn.benchmark=True

SEED = 3
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)

SAVE_PATH = 'C:/vscode/runs/wgan-gp-mnist-11(no-ln)'
save_dir = Path(SAVE_PATH)
save_dir.mkdir(exist_ok=True)
sample_dir = save_dir / 'samples'
sample_dir.mkdir(exist_ok=True)  # mkdir -p
weight_dir = save_dir / 'weights'
weight_dir.mkdir(exist_ok=True)
log_dir = save_dir / 'log_files'
log_dir.mkdir(exist_ok=True)


def adjust_lr_linearly(optimizer, current_step, end_step, end_value=0, eps=1e-8):
    new_lr = (lr - end_value + eps) * (end_step - current_step) / (end_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def gradient_penalty(x, x_tilde, onesided=False, square=True, get_norms=False):
    # one-sided penalty: lambda*E[ max(0, ||grad_D(x_hat)||_2 - 1)^2 ]
    if onesided:  # penalize 'gradients-1' larger than 0
        clip_grad = lambda t: t.clamp(min=1)
    else:
        clip_grad = lambda t: t

    eps = torch.rand(BATCH_SIZE, 1, 1, 1)  # uniform distribution
    if USE_CUDA:
        eps = eps.cuda()
    eps = eps.expand(x.size(0), x.size(1), x.size(2), x.size(3))
    x_hat = eps*x + (1-eps)*x_tilde
    if USE_CUDA:
        x_hat = x_hat.cuda()
    x_hat.requires_grad_()
    D_x_hat = netD(x_hat)

    grads, = torch.autograd.grad(D_x_hat, x_hat,
                                 grad_outputs=torch.ones(D_x_hat.size()).cuda(),
                                 create_graph=True)
    grads_norm = (grads.view(x.size(0), -1)).norm(dim=1)
    clipped_norm = clip_grad(grads_norm)

    if square:
        grad_penalty = ( clipped_norm-1 )**2
    else:
        grad_penalty = ( clipped_norm-1 )

    return grad_penalty, grads_norm


if __name__ == "__main__":
    writer = SummaryWriter(log_dir)
    trans = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor(),  # convert in the range [0.0, 1.0]
                                transforms.Normalize([0.5], [0.5])])  # (ch - m) / s -> [-1, 1]
    mnist = datasets.MNIST('C:/vscode/ml/my_datasets', train=True, transform=trans)
    data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    G_params = [(ngf*4, 4, 1, 0), (ngf*2, 4, 2, 1), (ngf*1, 4, 2, 1), (img_ch, 4, 2, 1)]
    D_params = [(ndf*1, 4, 2, 1), (ndf*2, 4, 2, 1), (ndf*4, 4, 2, 1), (ndf*8, 4, 2, 1)]

    netG = Generator(G_params, noise_dim)
    netD = Discriminator(D_params, img_ch, img_size, layernorm=False)
    fixed_noise = torch.FloatTensor(64, noise_dim).normal_(0, 1)
    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(0, 0.9))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(0, 0.9))
    if USE_CUDA:
        netG.cuda()
        netD.cuda()
        fixed_noise = fixed_noise.cuda()

    start_epoch = 0
    g_train_step = 0
    critic_count = 0
    LOAD = False
    if LOAD:
        load_epoch = 500
        print('load', weight_dir / 'checkpoint_epoch{:05}.pth.tar'.format(load_epoch))
        checkpoint = torch.load( weight_dir / 'checkpoint_epoch{:05}.pth.tar'.format(load_epoch) )
        start_epoch = checkpoint['start_epoch']
        g_train_step = checkpoint['g_train_step'] 
        netG.load_state_dict(checkpoint['g_state_dict'])
        netD.load_state_dict(checkpoint['d_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optim_g_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optim_d_state_dict'])

    netG.train(), netD.train()
    for epoch in range(1 + start_epoch, n_epochs + 1 + start_epoch):
        for iter, (x, _) in enumerate(data_loader):
            if x.size(0) < BATCH_SIZE:
                continue
            # train critic n times
            optimizer_D.zero_grad()

            x = torch.Tensor(x)
            z = torch.randn(x.size(0), noise_dim)
            if USE_CUDA:
                x = x.cuda()
                z = z.cuda()
            x_tilde = netG(z)

            D_x = netD(x).mean()
            D_Gz = netD(x_tilde.detach()).mean()

            grad_penalty, norms = gradient_penalty(x, x_tilde,
                                        onesided=False, square=True, get_norms=True)
            # max_D E[D(x)] - E[D(G(z))] + gradient norm -> 1
            loss_D = D_Gz - D_x + lambda_term*grad_penalty.mean()
            loss_D.backward()
            optimizer_D.step()

            critic_count += 1
            if (critic_count % n_critic) != 0:
                continue

            # train generator
            optimizer_G.zero_grad()

            z = torch.randn(x.size(0), noise_dim)
            if USE_CUDA:
                z = z.cuda()

            fake = netG(z)
            D_fake = netD(fake)
            # min_G E[D(x)] - E[D(G(z))]
            loss_G = -torch.mean(D_fake)
            loss_G.backward()
            optimizer_G.step()
            
            g_train_step += 1
            # adjust_lr_linearly(optimizer_D, g_train_step, end_step=100000)
            # adjust_lr_linearly(optimizer_G, g_train_step, end_step=100000)
            if g_train_step % 10 == 0:
                loss_d = loss_D.detach().cpu().numpy()
                loss_g = loss_G.detach().cpu().numpy()
                dx_dgz_err = D_x - D_Gz
                gp_term = lambda_term*grad_penalty.mean()
                norms_mean = norms.mean()
                for param_group in optimizer_D.param_groups:
                    lr_d = param_group['lr']
                for param_group in optimizer_G.param_groups:
                    lr_g = param_group['lr']
                writer.add_scalar('losses/loss_d', loss_d, g_train_step)
                writer.add_scalar('losses/loss_g', loss_g, g_train_step)
                writer.add_scalar('losses/loss_d_gp_term', gp_term, g_train_step)
                writer.add_scalar('data/approx_EMD', dx_dgz_err, g_train_step)
                writer.add_scalar('data/norm_mean', norms_mean, g_train_step)
                writer.add_scalar('lr/lr_d', lr_d, g_train_step)
                writer.add_scalar('lr/lr_g', lr_g, g_train_step)
                print('[Epoch {:03d}: {:03d}/{:03d}, g_train: {:07d}], loss_D: {:.4f}, loss_G: {:.4f} | (approx) emd: {:.4f}, gp_term: {:.4f}, norm_mean: {:.4f}, lr_g: {:.7f}, lr_d {:.7f}'
                    .format(epoch, iter, len(data_loader), g_train_step, loss_d, loss_g, dx_dgz_err, gp_term, norms_mean, lr_g, lr_d ))

        if (epoch) % 2 == 0:
            fake = netG(fixed_noise)
            save_image(fake.data, sample_dir / "WGAN_GP_MNIST_EPOCH{:03}.png".format(epoch), normalize=True)

        if (epoch) % 100 == 0:
            torch.save({'g_state_dict': netG.state_dict(),
                        'd_state_dict': netD.state_dict(),
                        'optim_g_state_dict': optimizer_G.state_dict(),
                        'optim_d_state_dict': optimizer_D.state_dict(),
                        'g_train_step': g_train_step,
                        'start_epoch': epoch
                        }, weight_dir / 'checkpoint_epoch{:05}.pth.tar'.format(epoch) )
    writer.close()