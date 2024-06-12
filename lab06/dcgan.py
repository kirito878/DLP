import argparse
import os
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import torchvision.utils as vutils
import cv2
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class iclevr_dataset(Dataset):
    def __init__(self, img_dir, ann_file, objects, transform=None):
        super().__init__()
        self.transform = transform
        self.img_dir = img_dir
        self.ann_file = ann_file
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_names = list(self.annotations.keys())
        self.objects = objects

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(image_name).convert('RGB')

        labels = self.annotations[self.image_names[idx]]
        one_hot_labels = self.label_2_one_hot(labels)
        if self.transform:
            image = self.transform(image)
        return image, one_hot_labels

    def label_2_one_hot(self, labels):
        num_class = len(self.objects)
        one_hot_array = np.zeros(num_class, dtype=np.int16)
        for label in labels:
            object_idx = self.objects[label]
            one_hot_array[object_idx] = 1.0
        return torch.tensor(one_hot_array, dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes):
        super(Generator, self).__init__()
        self.embedding = nn.Linear(num_classes, nz)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, noise, labels):
        labels = self.embedding(labels)
        labels = labels.unsqueeze(-1).unsqueeze(-1)
        input = torch.cat((noise, labels), dim=1)

        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.embedding = nn.Linear(num_classes, ndf)
        self.conv1 = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main = nn.Sequential(

            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf+ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),)
        self.adv_output = nn.Sequential(
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.class_output = nn.Sequential(
            nn.Linear(ndf*8*2*2, 512),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = self.embedding(labels)
        img = self.conv1(img)
        labels = labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -
                                                           1, img.size(2), img.size(3))
        input = torch.cat((img, labels), dim=1)
        x = self.main(input)
        class_x = x.clone()
        adv_output = self.adv_output(x)
        class_x = self.pool(class_x)
        class_x = class_x.view(labels.size(0), -1)

        class_output = self.class_output(class_x)
        return adv_output.view(-1), class_output.view(labels.size(0), self.num_classes)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main(args):

    device = args.device
    obj_file = args.obj_file
    img_dir = args.img_dir
    ann_file = args.train_json
    batch_size = args.batch_size
    nc = 3
    nz = 100
    ngf = 256
    ndf = 256
    num_epochs = args.epochs
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    with open(obj_file, 'r') as f:
        objects = json.load(f)
    num_classes = len(objects)
    train_dataset = iclevr_dataset(
        img_dir=img_dir, ann_file=ann_file, objects=objects, transform=transform)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    netG = Generator(nz, ngf, nc, num_classes).to(device)
    

    netD = Discriminator(ndf, nc, num_classes).to(device)
    if args.resume:
        netG.load_state_dict(torch.load(os.path.join(args.load_ckpt, "netG.pth")))
        netD.load_state_dict(torch.load(os.path.join(args.load_ckpt, "netD.pth")))
        print("resume training")
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    criterion = nn.BCELoss()
    class_criterion = nn.BCELoss()
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(b1, b2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))

    os.makedirs(args.ckpt_path, exist_ok=True)
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(train_data_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)

            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), 1., device=device)
            fake_labels = torch.full((batch_size,), 0., device=device)

            for step_d in range(3):
                # Train Discriminator
                netD.zero_grad()
                adv_output, class_output = netD(real_images, labels)
                errD_real = criterion(
                    adv_output, real_labels) + class_criterion(class_output, labels) * args.class_weight
                errD_real.backward()

                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images = netG(noise, labels)
                adv_output, class_output = netD(fake_images.detach(), labels)
                errD_fake = criterion(
                    adv_output, fake_labels) + class_criterion(class_output, labels) * args.class_weight
                errD_fake.backward()
                optimizerD.step()

            # Train Generator
            for step_g in range(3):
                netG.zero_grad()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images = netG(noise, labels)
                adv_output, class_output = netD(fake_images, labels)
                errG = criterion(adv_output, real_labels) + \
                    class_criterion(class_output, labels) * args.class_weight
                errG.backward()
                optimizerG.step()

            if i % 50 == 0:
                print(
                    f'[{epoch}/{num_epochs}][{i}/{len(train_data_loader)}] Loss_D: {errD_real.item() + errD_fake.item()} Loss_G: {errG.item()} ')
        torch.save(netG.state_dict(), os.path.join(args.ckpt_path, f"netG_{epoch}.pth"))
        torch.save(netD.state_dict(), os.path.join(args.ckpt_path, f"netD_{epoch}.pth"))
    torch.save(netG.state_dict(), os.path.join(args.ckpt_path, f"netG.pth"))
    torch.save(netD.state_dict(), os.path.join(args.ckpt_path, "netD.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lab06")
    parser.add_argument('--device', type=str, default="cuda",
                        help='Which device the training is on.')  # cuda
    parser.add_argument('--img_dir', type=str, default="/home/wujh1123/DLP/lab06/iclevr",
                        help='img folder')
    parser.add_argument('--train_json', type=str, default="/home/wujh1123/DLP/lab06/train.json",
                        help='train_json')
    parser.add_argument('--obj_file', type=str, default="/home/wujh1123/DLP/lab06/objects.json",
                        help='object file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='epochs')
    parser.add_argument('--ckpt_path', "-c", type=str, default="/home/wujh1123/DLP/lab06/weight",
                        help='netG weight')
    parser.add_argument('--class_weight', type=int, default=10,
                        help='class weight')
    parser.add_argument('--load_ckpt', "-l", type=str, default="/home/wujh1123/DLP/lab06/weight",
                        help='netG weight')
    parser.add_argument('--resume',         action='store_true')
    args = parser.parse_args()
    main(args)
