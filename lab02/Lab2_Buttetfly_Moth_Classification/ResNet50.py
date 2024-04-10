import torch
import torch.nn as nn
from torchsummary import summary


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channels, output_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv_1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, stride=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(output_channels)

        self.conv_2 = nn.Conv2d(output_channels, output_channels,
                                kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        self.conv_3 = nn.Conv2d(
            output_channels, output_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(output_channels*self.expansion)
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

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.bn_3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Resnet50_net(nn.Module):
    def __init__(self, bottleneck, classes=100):
        super(Resnet50_net, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inital_input_channels = 64
        #  [3,4,6,3]
        self.layer_1 = self._make_layer(bottleneck, 64, 3, stride=1)
        self.layer_2 = self._make_layer(bottleneck, 128, 4, stride=2)
        self.layer_3 = self._make_layer(bottleneck, 256, 6, stride=2)
        self.layer_4 = self._make_layer(bottleneck, 512, 3, stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*bottleneck.expansion, classes)

    def _make_layer(self, bottleneck, output_channels, num_layers, stride=1):
        layers = []
        layers.append(bottleneck(
            self.inital_input_channels, output_channels, stride))
        self.inital_input_channels = output_channels*bottleneck.expansion

        for i in range(1, num_layers):
            layers.append(bottleneck(
                self.inital_input_channels, output_channels))

        layer = nn.Sequential(*layers)
        return layer

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        # print(self.layer_1)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avg_pooling(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    resnet50 = Resnet50_net(Bottleneck, classes=1000).to('cuda')
    summary(resnet50, (3, 224, 224))
