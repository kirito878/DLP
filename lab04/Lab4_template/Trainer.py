import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10


def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(
        imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        if self.kl_anneal_type == "None":
            self.beta = self.kl_anneal_ratio
        else:
            self.beta = 0.05
        # raise NotImplementedError

    def update(self):
        # TODO
        self.current_epoch += 1
        if self.kl_anneal_type == "Cyclical":
            self.frange_cycle_linear(
                self.current_epoch, 0.01, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
        elif self.kl_anneal_type == "Monotonic":
            beta = (self.current_epoch/self.kl_anneal_cycle) * \
                self.kl_anneal_ratio
            self.beta = min(beta, self.kl_anneal_ratio)
        # raise NotImplementedError

    def get_beta(self):
        # TODO
        return self.beta
        # raise NotImplementedError

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        tmp_beta = 0
        tmp_beta = ((n_iter % n_cycle) / n_cycle) * ratio
        beta = np.clip(tmp_beta, start, stop)
        self.beta = beta
        # raise NotImplementedError


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        # normalization
        if self.args.norm == "Batch":
            norm_layer = nn.BatchNorm2d
        elif self.args.norm == "Instance":
            norm_layer = nn.InstanceNorm2d
        elif self.args.norm == "Group":
            def get_group_norm(num_channels):
                return nn.GroupNorm(num_channels=num_channels, num_groups=2)
            norm_layer = get_group_norm
        else:
            raise ValueError("norm layer error")
        self.label_transformation = Label_Encoder(3, args.L_dim, norm_layer)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    def plot(self, epoch_list, target_list, Title, Target_name, save_name, per_frame=False):
        plt.plot(epoch_list, target_list, linestyle='-')
        plt.title(Title)
        if per_frame == True:
            x_name = "Index"
        else:
            x_name = "Epochs"
        plt.xlabel(x_name)
        plt.ylabel(Target_name)
        path = os.path.join(self.args.save_root, f"{save_name}_curve.png")
        plt.savefig(path)
        print(f"Success save graph to {path}")
        plt.close()

    def training_stage(self):
        epoch_list = []
        tfr_list = []
        beta_list = []
        loss_list = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            total_loss = 0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(
                        self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(
                        self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                total_loss += loss.detach().cpu()
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root,
                          f"epoch={self.current_epoch}.ckpt"))

            epoch_list.append(self.current_epoch)
            tfr_list.append(self.tfr)
            beta_list.append(self.kl_annealing.get_beta())
            loss_list.append(total_loss/len(train_loader))

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

        self.plot(epoch_list, tfr_list,
                  "Teacher Forcing Ratio", "Ratio", "tfr")
        self.plot(epoch_list, beta_list, self.kl_annealing.kl_anneal_type,
                  "Beta", f"{self.kl_annealing.kl_anneal_type}_beta")
        self.plot(epoch_list, loss_list, "Loss Curve", "Loss", "loss")

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(),
                          lr=self.scheduler.get_last_lr()[0])

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        self.frame_transformation.train()
        self.label_transformation.train()
        self.Gaussian_Predictor.train()
        self.Decoder_Fusion.train()
        self.Generator.train()

        loss_mse = 0.0
        loss_kl = 0.0
        next_predicted_frame = img[:, 0]
        for i in range(self.train_vi_len-1):
            # pose
            current_pose = label[:, i]
            next_pose = label[:, i+1]

            # frame
            current_frame = next_predicted_frame
            if adapt_TeacherForcing:
                current_frame = img[:, i]
            next_frame = img[:, i+1]

            # part kl loss
            encode_frame_next = self.frame_transformation(next_frame)
            encode_pose_next = self.label_transformation(next_pose)
            z, mu, logvar = self.Gaussian_Predictor(
                encode_frame_next, encode_pose_next)
            loss_kl += kl_criterion(mu, logvar, self.batch_size)

            # part mse loss
            encode_frame_current = self.frame_transformation(current_frame)
            decode_feature = self.Decoder_Fusion(
                encode_frame_current, encode_pose_next, z)
            next_predicted_frame = self.Generator(decode_feature)
            loss_mse += self.mse_criterion(next_predicted_frame, next_frame)

        beta = self.kl_annealing.get_beta()
        loss_sum = loss_mse + beta*loss_kl
        loss_sum.backward()
        self.optimizer_step()
        self.optim.zero_grad()

        return loss_sum
        # raise NotImplementedError

    def val_one_step(self, img, label):
        # TODO
        self.frame_transformation.eval()
        self.label_transformation.eval()
        self.Decoder_Fusion.eval()
        self.Generator.eval()

        loss_mse = 0
        next_predicted_frame = img[:, 0]
        psnr_total = 0
        psnr_list = []
        iteration = []
        for i in range(self.val_vi_len-1):

            current_pose = label[:, i]
            next_pose = label[:, i+1]

            current_frame = next_predicted_frame
            next_frame = img[:, i+1]

            encode_frame_current = self.frame_transformation(current_frame)
            encode_pose_next = self.label_transformation(next_pose)

            z = torch.cuda.FloatTensor(
                1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_().to(self.args.device)

            decode_feature = self.Decoder_Fusion(
                encode_frame_current, encode_pose_next, z)
            next_predicted_frame = self.Generator(decode_feature)

            loss_mse += self.mse_criterion(next_predicted_frame, next_frame)

            psnr_per_frame = Generate_PSNR(
                next_predicted_frame, next_frame).item()
            psnr_total += psnr_per_frame
            iteration.append(i)
            psnr_list.append(psnr_per_frame)
        print(f"\nAverage PSNR: {psnr_total / self.val_vi_len:.5f}")
        self.plot(iteration, psnr_list, "PSNR-per frame",
                  "PSNR", "psnr", per_frame=True)
        return loss_mse
        # raise NotImplementedError

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(img_name, format="GIF", append_images=new_list,
                         save_all=True, duration=40, loop=0)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len,
                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform,
                                mode='val', video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                drop_last=True,
                                shuffle=False)
        return val_loader

    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.tfr_sde:
            tmp_tfr = self.tfr-self.tfr_d_step
            self.tfr = max(tmp_tfr, 0)
        # raise NotImplementedError

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),
            "lr": self.scheduler.get_last_lr()[0],
            "tfr":   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']

            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()


def main(args):

    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,
                        default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str,
                        choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true',
                        help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str,
                        required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str,
                        required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int,
                        default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,
                        help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,
                        help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int,
                        default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int,
                        default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,
                        help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,
                        help="Width input image to be resize")

    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,
                        help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,
                        help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int,
                        default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,
                        help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float,
                        default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,
                        help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,
                        help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,
                        default=None, help="The path of your checkpoints")

    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,
                        help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,
                        help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str,
                        default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int,
                        default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float,
                        default=1,              help="")

    # Normalization
    parser.add_argument('--norm', type=str, default="Batch")

    args = parser.parse_args()

    main(args)
