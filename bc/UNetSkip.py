import torch
import torch.nn as nn

from configs import BCSweepConfig

SIZE_CONFIG = {
    "large": {
        "down_channels": tuple(reversed((1024, 512, 256, 128, 64, 32, 16))),
        "up_channels": (1024, 512, 256, 128, 64, 32, 16),
    },
    "medium": {
        "up_channels": (256, 128, 64, 32),
        "down_channels": tuple(reversed((256, 128, 64, 32))),
    },
    "small": {
        "up_channels": (64, 32, 16),
        "down_channels": tuple(reversed((64, 32, 16))),
    },
}


class ConvBlockBC(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.bn1(self.relu(self.conv1(x)))
        h = self.bn2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class UnetBC(nn.Module):
    """
    A simplified Unet architecture.
    """

    @classmethod
    def from_config(cls, config, out_channels):
        bc_backbone, obs = config.bc_backbone, config.obs

        backbone_size = bc_backbone.split("-")[-1]
        down_channels, up_channels = (
            SIZE_CONFIG[backbone_size]["down_channels"],
            SIZE_CONFIG[backbone_size]["up_channels"],
        )

        if obs == "full":
            in_channels = 3
        elif obs == "semi":
            in_channels = 2
        else:
            in_channels = 1

        return cls(
            in_channels=in_channels,
            down_channels=down_channels,
            up_channels=up_channels,
            out_channels=out_channels,
        )

    def __init__(
        self,
        in_channels=3,
        down_channels=(64, 128, 256),
        up_channels=(256, 128, 64),
        out_channels=3,
    ):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                ConvBlockBC(down_channels[i], down_channels[i + 1])
                for i in range(len(down_channels) - 1)
            ]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [
                ConvBlockBC(up_channels[i], up_channels[i + 1], up=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        self.output = nn.Conv2d(
            up_channels[-1], kernel_size=1, out_channels=out_channels
        )

    def forward(self, x):
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)

        return nn.Sigmoid()(self.output(x))
