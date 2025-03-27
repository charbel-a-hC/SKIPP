import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetNoSkip, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(256, 512),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x, goal_pose):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        x6 = self.encoder[5](x5)
        x7 = self.encoder[6](x6)

        # print(x7.shape)

        x = self.decoder[0](x7)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        x = self.decoder[5](x)
        x = self.decoder[6](x)

        return x
