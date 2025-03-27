from bc.UNetNoSkip import UNetNoSkip
from bc.UNetSkip import UnetBC

MODELS = {
    "unet-skip-large": {
        "model": UnetBC,
        "down_channels": tuple(reversed((1024, 512, 256, 128, 64, 32, 16))),
        "up_channels": (1024, 512, 256, 128, 64, 32, 16),
        "params": {
            "out_channels": 1,
        },
    },
    "unet-skip-medium": {
        "model": UnetBC,
        "up_channels": (256, 128, 64, 32),
        "down_channels": tuple(reversed((256, 128, 64, 32))),
        "params": {
            "out_channels": 1,
        },
    },
    "unet-skip-small": {
        "model": UnetBC,
        "up_channels": (64, 32, 16),
        "down_channels": tuple(reversed((64, 32, 16))),
        "params": {
            "out_channels": 1,
        },
    },
    "unet-no-skip": {"model": UNetNoSkip, "params": {"out_channels": 1}},
}
