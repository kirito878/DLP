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


def main(args):

    device = args.device
    obj_file = args.obj_file
    num_train_timestamps = 1000

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
    ddpm_model = ddpm.Unet(num_classes=num_classes).to(device)
    ddpm_model.load_state_dict(torch.load(os.path.join(args.ckpt_path,"ddpm.pth")))
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timestamps, beta_schedule='squaredcos_cap_v2')
    with torch.no_grad():
        for i, labels in enumerate(data_loader):
            labels = labels.to(device)
            x = torch.randn(batch_size, 3, 64, 64).to(device)
            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):

                pred_noise = ddpm_model(x, t, labels)
                x = noise_scheduler.step(pred_noise, t, x).prev_sample
            eval_model = evaluator.evaluation_model()

            accuracy = eval_model.eval(x.detach(), labels)
            print(f"test Accuracy: {accuracy}")
            img_grid = vutils.make_grid(
                x.detach().cpu(), padding=2, normalize=True)
            # Convert the image grid to a NumPy array and save using OpenCV
            img_np = img_grid.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
            img_np = (img_np * 255).astype(np.uint8)  # Denormalize to 0-255
            # Convert RGB to BGR for OpenCV
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'ddpm_test.png', img_np)
    test_one = "/home/wujh1123/DLP/lab06/new_test.json"
    with open(test_one, 'r') as f:
        test_object = json.load(f)
    batch_size = len(test_object)
    dataset = iclevr_dataset(test_object=test_object, objects=objects)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    ddpm_model = ddpm.Unet(num_classes=num_classes).to(device)
    ddpm_model.load_state_dict(torch.load(os.path.join(args.ckpt_path,"ddpm.pth")))
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timestamps, beta_schedule='squaredcos_cap_v2')
    with torch.no_grad():
        for i, labels in enumerate(data_loader):
            labels = labels.to(device)
            x = torch.randn(batch_size, 3, 64, 64).to(device)
            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):

                pred_noise = ddpm_model(x, t, labels)
                x = noise_scheduler.step(pred_noise, t, x).prev_sample
            eval_model = evaluator.evaluation_model()

            accuracy = eval_model.eval(x.detach(), labels)
            print(f"test Accuracy: {accuracy}")
            img_grid = vutils.make_grid(
                x.detach().cpu(), padding=2, normalize=True)
            # Convert the image grid to a NumPy array and save using OpenCV
            img_np = img_grid.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC
            img_np = (img_np * 255).astype(np.uint8)  # Denormalize to 0-255
            # Convert RGB to BGR for OpenCV
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'ddpm_new_test.png', img_np)


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
    parser.add_argument('--ckpt_path', "-c", type=str, default="/home/wujh1123/DLP/lab06/weight",
                        help='ddpm weight')
    args = parser.parse_args()
    main(args)
