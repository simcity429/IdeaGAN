import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from network import IdeaGAN
from tester import Tester
from config import *

if __name__ == '__main__':
    max_prob_list = []
    jsd_list = []
    idea_gan = IdeaGAN()
    tester = Tester()
    transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.ToTensor(),  # convert in the range [0.0, 1.0]
                                transforms.Normalize([0.5], [0.5])])  # (ch - m) / s -> [-1, 1]
    mnist = datasets.MNIST('./mnist_data', download=True, train=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    big_d_update_cnt = 0
    step_cnt = 0
    little_d_sum = 0
    for e in range(EPOCH):
        for real_img, answer in data_loader:
            real_img = real_img.to(DEVICE)
            answer = answer.to(DEVICE)
            if real_img.size(0) < BATCH_SIZE:
                continue
            if big_d_update_cnt < BIG_D_UPDATE_NUM:
                big_d_update_cnt += 1
                little_d_loss = idea_gan.d_only_update(real_img, answer)
                little_d_sum += little_d_loss
                if little_d_sum > 0 and big_d_update_cnt == BIG_D_UPDATE_NUM - 1:
                    big_d_update_cnt = 0
                    little_d_sum = 0
            else:
                big_d_update_cnt = 0
                little_d_sum = 0
                print(step_cnt)
                idea_gan.update_all(real_img, answer)
            step_cnt += 1
            if step_cnt % 10 == 0:
                fake_img = idea_gan.make_fake_img()
                max_prob, jsd = tester.gen_test(fake_img)
                max_prob = max_prob.cpu().numpy()
                jsd = jsd.cpu().numpy()
                max_prob_list.append(max_prob)
                jsd_list.append(jsd)
                indice = [i for i in range(len(jsd_list))]
                plt.plot(indice, jsd_list)
                plt.plot(indice, max_prob_list)
                noise_fake_img = idea_gan.make_fake_img_by_noise()[:64]
                real_img = idea_gan.make_restored_img(real_img)[:64]
                if MODE == 'TRANSFORMER':
                    save_image(make_grid(noise_fake_img, normalize=True), "./tmp/noise_fake.png")
                    save_image(make_grid(fake_img[:64], normalize=True), "./tmp/fake_transformer.png")
                    save_image(make_grid(real_img, normalize=True), "./tmp/restored.png")
                    plt.savefig('./tmp/transformer.jpg')
                else:
                    save_image(make_grid(noise_fake_img, normalize=True), "./tmp/noise_fake_base.png")
                    save_image(make_grid(fake_img[:64], normalize=True), "./tmp/fake_base.png")
                    save_image(make_grid(real_img, normalize=True), "./tmp/restored_base.png")
                    plt.savefig('./tmp/baseline.jpg')
    print('mode: ', MODE)
    print('last_100_jsd_mean: ', np.mean(jsd_list[-100:]))
    print('last_100_probmax_mean: ', np.mean(max_prob_list[-100:]))
