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
from diffusers import DDPMScheduler, UNet2DModel
import json
from tqdm import tqdm
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


class Unet(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=64,           # the target image resolution
            # Additional input channels for class cond.
            in_channels=3 + num_classes,
            out_channels=3,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            time_embedding_type="positional",
            block_out_channels=(224, 448, 672, 896),
            down_block_types=(
                "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"
            ),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D",
                            "AttnUpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t, class_labels):
        bs, c, w, h = x.shape
        labal = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(
            bs, class_labels.shape[1], w, h)
        unet_input = torch.cat((x, labal), 1)

        return self.model(unet_input, t).sample


def main(args):

    device = args.device
    obj_file = args.obj_file
    img_dir = args.img_dir
    ann_file = args.train_json
    batch_size = args.batch_size
    lr = 0.0001
    num_train_timestamps = 1000

    n_epochs = args.epochs

    with open(obj_file, 'r') as f:
        objects = json.load(f)
    num_classes = len(objects)
    train_dataset = iclevr_dataset(
        img_dir=img_dir, ann_file=ann_file, objects=objects, transform=transform)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timestamps, beta_schedule='squaredcos_cap_v2')
    net = Unet(num_classes=num_classes).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    os.makedirs(args.ckpt_path, exist_ok=True)
    for epoch in range(n_epochs):
        for x, y in tqdm(train_data_loader):
            x = x.to(device) 
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            pred = net(noisy_x, timesteps, y)
            loss = loss_fn(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg_loss = sum(losses[-100:])/100
        print(
            f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
        torch.save(net.state_dict(), os.path.join(args.ckpt_path, "ddpm.pth"))


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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=150,
                        help='epochs')
    parser.add_argument('--ckpt_path', "-c", type=str, default="/home/wujh1123/DLP/lab06/weight",
                        help='ddpm weight')
    args = parser.parse_args()
    main(args)
