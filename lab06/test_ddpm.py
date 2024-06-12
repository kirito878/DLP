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
import evaluator
import ddpm
import warnings

warnings.filterwarnings("ignore")


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


class Evaluate():
    def __init__(self, args):
        self.device = args.device
        self.obj_file = args.obj_file
        self.num_train_timestamps = 1000
        self.ckpt_path = args.ckpt_path
        with open(self.obj_file, 'r') as f:
            self.objects = json.load(f)
        self.num_classes = len(self.objects)

    def open_test_json(self, data):
        with open(data, 'r') as f:
            self.test_object = json.load(f)
        self.batch_size = len(self.test_object)

    def eval(self, data, data_name):
        self.open_test_json(data)
        self.dataset = iclevr_dataset(
            test_object=self.test_object, objects=self.objects)
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False)
        ddpm_model = ddpm.Unet(num_classes=self.num_classes).to(self.device)
        ddpm_model.load_state_dict(torch.load(
            os.path.join(self.ckpt_path, "ddpm.pth")))
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timestamps, beta_schedule='squaredcos_cap_v2')
        with torch.no_grad():
            for i, labels in enumerate(self.data_loader):
                labels = labels.to(self.device)
                x = torch.randn(self.batch_size, 3, 64, 64).to(self.device)
                for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                    pred_noise = ddpm_model(x, t, labels)
                    x = noise_scheduler.step(pred_noise, t, x).prev_sample
                eval_model = evaluator.evaluation_model()
                accuracy = eval_model.eval(x.detach(), labels)
                print(f"{data_name} Accuracy: {accuracy}")
                img_grid = vutils.make_grid(
                    x.detach().cpu(), padding=2, normalize=True)
                # Convert the image grid to a NumPy array and save using OpenCV
                img_np = img_grid.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
                # Denormalize to 0-255
                img_np = (img_np * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'ddpm_{data_name}.png', img_np)

    def process(self):
        ddpm_model = ddpm.Unet(
            num_classes=self.num_classes).to(self.device)
        ddpm_model.load_state_dict(torch.load(
            os.path.join(self.ckpt_path, "ddpm.pth")))
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timestamps, beta_schedule='squaredcos_cap_v2')
        num_class = len(self.objects)
        one_hot_array = np.zeros(num_class, dtype=np.int16)
        labels = ["red sphere", "yellow cube", "cyan cylinder"]
        for label in labels:
            object_idx = self.objects[label]
            one_hot_array[object_idx] = 1.0
        label_vector = torch.tensor(
            one_hot_array, dtype=torch.float32).to(self.device).unsqueeze(0)
        x = torch.randn(1, 3, 64, 64).to(self.device)
        img_list = []
        for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
            with torch.no_grad():
                pred_noise = ddpm_model(x, t, label_vector)
            x = noise_scheduler.step(pred_noise, t, x).prev_sample
            if (t % 50 == 0):
                img_list.append(((x.detach()+1)/2).clamp(0, 1))
        img_list = torch.stack(img_list).squeeze(dim=1).detach().cpu()
        print(img_list.shape)
        img_grid = vutils.make_grid(
            img_list, padding=2, nrow=10)
        # Convert the image grid to a NumPy array and save using OpenCV
        img_np = img_grid.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
        # Denormalize to 0-255
        img_np = (img_np * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'ddpm_process.png', img_np)


def main(args):
    eval_model = Evaluate(args=args)
    if args.generate:
        print("denoising process image")
        eval_model.process()
    test_one = "/home/wujh1123/DLP/lab06/test.json"
    eval_model.eval(test_one, "Test")
    test_two = "/home/wujh1123/DLP/lab06/new_test.json"
    eval_model.eval(test_two, "New Test")


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
    parser.add_argument('--ckpt_path', "-c", type=str, default="/home/wujh1123/DLP/lab06",
                        help='ddpm weight')
    parser.add_argument("--generate", "-g", action='store_true')
    args = parser.parse_args()
    main(args)
