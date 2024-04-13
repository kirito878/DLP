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
                                             output_channels, output_channels, kernel_size=3, padding=1, stride=1),
                                         nn.ReLU(inplace=True),
                                         )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class Unet_model(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(Unet_model, self).__init__()
        self.down_conv_1 = Double_Conv(input_channels, 64)
        self.down_conv_2 = Double_Conv(64, 128)
        self.down_conv_3 = Double_Conv(128, 256)
        self.down_conv_4 = Double_Conv(256, 512)
        self.down_conv_5 = Double_Conv(512, 1024)

        self.up_conv_4 = Double_Conv(1024, 512)
        self.up_conv_3 = Double_Conv(512, 256)
        self.up_conv_2 = Double_Conv(256, 128)
        self.up_conv_1 = Double_Conv(128, 64)
        self.max_pooling = nn.MaxPool2d(2)

        self.up_sample_4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_sample_3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_sample_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_sample_1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        conv_1 = self.down_conv_1(x)
        x = self.max_pooling(conv_1)

        conv_2 = self.down_conv_2(x)
        x = self.max_pooling(conv_2)

        conv_3 = self.down_conv_3(x)
        x = self.max_pooling(conv_3)

        conv_4 = self.down_conv_4(x)
        x = self.max_pooling(conv_4)

        x = self.down_conv_5(x)

        x = self.up_sample_4(x)
        x = torch.cat([x, conv_4], dim=1)
        x = self.up_conv_4(x)

        x = self.up_sample_3(x)
        x = torch.cat([x, conv_3], dim=1)
        x = self.up_conv_3(x)

        x = self.up_sample_2(x)
        x = torch.cat([x, conv_2], dim=1)
        x = self.up_conv_2(x)

        x = self.up_sample_1(x)
        x = torch.cat([x, conv_1], dim=1)
        x = self.up_conv_1(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    model = Unet_model().to('cuda')
    summary(model, (3, 512, 512))
