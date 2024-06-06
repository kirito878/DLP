import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

# TODO2 step1-4: design the transformer training strategy


class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(
            MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.args = args
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()

    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        # pass
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"Epoch {epoch}")

        for i, img in progress_bar:
            img = img.to(self.args.device)
            # logits : ([10, 256, 1025])
            # target : ([10, 256])
            logits, target = self.model(img)

            # logits : ([2560, 1025])
            # target : ([2560])
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
            loss = F.cross_entropy(logits, target)
            total_loss += loss

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return total_loss / len(train_loader)

    def eval_one_epoch(self, val_loader):
        # pass
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            for i, img in enumerate(val_loader):
                img = img.to(self.args.device)
                logits, target = self.model(img)
                # import ipdb
                # ipdb.set_trace()
                logits = logits.reshape(-1, logits.shape[-1])
                target = target.reshape(-1)
                loss = F.cross_entropy(logits, target)
                total_loss += loss
        return total_loss / len(val_loader)

    def configure_optimizers(self):
        scheduler = None
        optimizer = torch.optim.Adam(
            self.model.transformer.parameters(), lr=self.args.learning_rate)
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct
    parser.add_argument('--train_d_path', type=str,
                        default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str,
                        default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str,
                        default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--accum-grad', type=int, default=10,
                        help='Number for gradient accumulation.')

    # you can modify the hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1,
                        help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int,
                        default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int,
                        default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float,
                        default=0.0001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml',
                        help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)

# TODO2 step1-5:
    min_loss = 999999
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        val_loss = train_transformer.eval_one_epoch(val_loader)
        print(train_loss)
        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join(
                'transformer_checkpoints', f'epoch_{epoch}_ckpt.pth'))
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(train_transformer.model.transformer.state_dict(
            ), os.path.join('transformer_checkpoints', f'best_ckpt.pth'))
