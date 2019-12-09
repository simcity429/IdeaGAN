import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from network import IdeaGAN
from config import *

if __name__ == '__main__':
    idea_gan = IdeaGAN()
    transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.ToTensor(),  # convert in the range [0.0, 1.0]
                                transforms.Normalize([0.5], [0.5])])  # (ch - m) / s -> [-1, 1]
    mnist = datasets.MNIST('./mnist_data', download=True, train=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    big_d_update_cnt = 0
    step_cnt = 0
    for e in range(EPOCH):
        for real_img, answer in data_loader:
            real_img = real_img.to(DEVICE)
            answer = answer.to(DEVICE)
            if real_img.size(0) < BATCH_SIZE:
                continue
            if big_d_update_cnt < BIG_D_UPDATE_NUM:
                big_d_update_cnt += 1
                idea_gan.big_d_only_update(real_img)
            else:
                big_d_update_cnt = 0
                idea_gan.update_all(real_img, answer)
            step_cnt += 1
            print(step_cnt)
            if step_cnt % 10 == 0:
                fake_img = idea_gan.make_fake_img()
                real_img = idea_gan.make_restored_img(real_img)
                save_image(make_grid(fake_img, normalize=True), "./tmp/fake.png")
                save_image(make_grid(real_img, normalize=True), "./tmp/restored.png")
