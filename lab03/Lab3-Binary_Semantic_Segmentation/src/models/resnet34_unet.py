import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1):
        super(Block, self).__init__()
        self.stride = stride
        self.conv_1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(output_channels)

        self.conv_2 = nn.Conv2d(output_channels, output_channels,
                                kernel_size=3, padding=1, stride=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or input_channels != output_channels*self.expansion:
            self.downsample = nn.Sequential(nn.Conv2d(
                input_channels, output_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels*self.expansion),
            )

    def forward(self, x):
        identity = x.clone()

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        # print(identity.shape)
        x = self.conv_2(x)
        x = self.bn_2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class Double_Conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Double_Conv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1),
                                         nn.BatchNorm2d(output_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(
                                             output_channels, output_channels, kernel_size=3, padding=1, stride=1),
                                         nn.BatchNorm2d(output_channels),
                                         nn.ReLU(inplace=True),
                                         )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class Resnet34_unet(nn.Module):
    def __init__(self, block, num_classes=1):
        super(Resnet34_unet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inital_input_channels = 64
        #  [3,4,6,3]
        self.layer_1 = self._make_layer(block, 64, 3, stride=1)
        self.layer_2 = self._make_layer(block, 128, 4, stride=2)
        self.layer_3 = self._make_layer(block, 256, 6, stride=2)
        self.layer_4 = self._make_layer(block, 512, 3, stride=2)
        self.layer_5 = Double_Conv(512, 256)

        self.up_sample_1 = nn.ConvTranspose2d(768, 384, 2, 2)
        self.up_conv_1 = Double_Conv(384, 32)
        self.skip_conv = nn.Conv2d(256, 512, 3, 1, 1, bias=False)

        self.up_sample_2 = nn.ConvTranspose2d(544, 272, 2, 2)
        self.up_conv_2 = Double_Conv(272, 32)

        self.up_sample_3 = nn.ConvTranspose2d(160, 80, 2, 2)
        self.up_conv_3 = Double_Conv(80, 32)

        self.up_sample_4 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.up_conv_4 = Double_Conv(48, 32)

        self.up_sample_5 = nn.ConvTranspose2d(32, 32, 2, 2)
        self.up_conv_5 = Double_Conv(32, 32)

        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_layer(self, block, output_channels, num_layers, stride=1):
        layers = []
        layers.append(block(
            self.inital_input_channels, output_channels, stride))
        self.inital_input_channels = output_channels*block.expansion

        for i in range(1, num_layers):
            layers.append(block(
                self.inital_input_channels, output_channels))

        layer = nn.Sequential(*layers)
        return layer

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        conv_1 = self.layer_1(x)
        conv_2 = self.layer_2(conv_1)
        conv_3 = self.layer_3(conv_2)
        conv_4 = self.layer_4(conv_3)
        x = self.layer_5(conv_4)
        x = torch.cat([x, conv_4], dim=1)
        x = self.up_sample_1(x)
        x = self.up_conv_1(x)
        conv_3 = self.skip_conv(conv_3)
        x = torch.cat([x, conv_3], dim=1)
        x = self.up_sample_2(x)
        x = self.up_conv_2(x)
        x = torch.cat([x, conv_2], dim=1)
        x = self.up_sample_3(x)
        x = self.up_conv_3(x)
        x = torch.cat([x, conv_1], dim=1)
        x = self.up_sample_4(x)
        x = self.up_conv_4(x)
        x = self.up_sample_5(x)
        x = self.up_conv_5(x)
        x = self.output(x)
        # print(x.shape)
        return x


if __name__ == "__main__":
    resnet34 = Resnet34_unet(Block).to('cuda')
    summary(resnet34, (3, 256, 256))
