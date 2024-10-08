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
import evaluator
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




class iclevr_dataset(Dataset):
    def __init__(self, test_object, objects):
        super().__init__()
        self.annotations = test_object
        self.objects = objects

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        labels = self.annotations[idx]
        one_hot_labels = self.label_2_one_hot(labels)
        return one_hot_labels

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




def main(args):

    device = args.device
    obj_file = args.obj_file
    nc = 3
    nz = 100
    ngf = 256

    with open(obj_file, 'r') as f:
        objects = json.load(f)
    num_classes = len(objects)

    test_one = "/home/wujh1123/DLP/lab06/test.json"
    with open(test_one, 'r') as f:
        test_object = json.load(f) 
    batch_size = len(test_object)
    dataset = iclevr_dataset(test_object=test_object, objects=objects)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    netG = Generator(nz, ngf, nc, num_classes).to(device)
    netG.load_state_dict(torch.load(args.netG))
    noise = torch.randn(batch_size, nz, 1, 1, device=device)  # 生成噪聲
    with torch.no_grad():
        for i, labels in enumerate(data_loader):
            labels = labels.to(device)
            fake_images = netG(noise, labels)
            eval_model = evaluator.evaluation_model()
            accuracy = eval_model.eval(fake_images, labels)
            print(f"test Accuracy: {accuracy}")
            img_grid = vutils.make_grid(fake_images.detach().cpu(), padding=2, normalize=True)
            # Convert the image grid to a NumPy array and save using OpenCV
            img_np = img_grid.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
            img_np = (img_np * 255).astype(np.uint8)  # Denormalize to 0-255
            # Convert RGB to BGR for OpenCV
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'gan_test.png', img_np)
    test_one = "/home/wujh1123/DLP/lab06/new_test.json"
    with open(test_one, 'r') as f:
        test_object = json.load(f) 
    batch_size = len(test_object)
    dataset = iclevr_dataset(test_object=test_object, objects=objects)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    netG = Generator(nz, ngf, nc, num_classes).to(device)
    netG.load_state_dict(torch.load(args.netG))
    noise = torch.randn(batch_size, nz, 1, 1, device=device)  # 生成噪聲
    with torch.no_grad():
        for i, labels in enumerate(data_loader):
            labels = labels.to(device)
            fake_images = netG(noise, labels)
            eval_model = evaluator.evaluation_model()
            accuracy = eval_model.eval(fake_images, labels)
            print(f"new test Accuracy: {accuracy}")
            img_grid = vutils.make_grid(fake_images.detach().cpu(), padding=2, normalize=True)
            # Convert the image grid to a NumPy array and save using OpenCV
            img_np = img_grid.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
            img_np = (img_np * 255).astype(np.uint8)  # Denormalize to 0-255
            # Convert RGB to BGR for OpenCV
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'gan_new_test.png', img_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lab06")
    parser.add_argument('--device', type=str, default="cuda",
                        help='Which device the training is on.')  # cuda
    parser.add_argument('--obj_file', type=str, default="/home/wujh1123/DLP/lab06/objects.json",
                        help='object file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs')
    parser.add_argument('--netG', type=str, default="/home/wujh1123/DLP/lab06/weight/netG.pth",
                        help='netG weight')
    args = parser.parse_args()
    main(args)
