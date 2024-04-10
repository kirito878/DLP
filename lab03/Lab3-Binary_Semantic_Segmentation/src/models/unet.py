# Implement your UNet model here

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Double_Conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Double_Conv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(
                                             input_channels, output_channels, kernel_size=3, padding=1, stride=1),
                                         nn.ReLU(inplace=True),
                                         )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class Unet_model(nn.modules):
    def __init__(self, input_channels=3, num_classes=1):
        super(Unet_model, self).__init__()
        self.down_conv_1 = Double_Conv(input_channels, 64)
        self.down_conv_2 = Double_Conv(64, 128)
        self.down_conv_3 = Double_Conv(128, 256)
        self.down_conv_4 = Double_Conv(256, 512)
        self.down_conv_5 = Double_Conv(512, 1024)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
