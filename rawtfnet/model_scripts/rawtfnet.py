import torch.nn as nn
from .blocks.frontend import DWS_Frontend_SE, DWS_Frontend_SE_small
from .blocks.classifier import TfSepNet
import torch
from torchinfo import summary


class RawTFNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sample_rate = args.sample_rate

        self.front_end = DWS_Frontend_SE(
            sinc_kernel_size=128,
            sample_rate=self.sample_rate
        )

        self.classifier = TfSepNet(depth=10, width=32, dropout_rate=0.2,
                                   shuffle=True, shuffle_groups=8)

    def forward(self, x):
        return self.classifier(self.front_end(x))


class RawTFNet_small(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sample_rate = args.sample_rate

        self.front_end = DWS_Frontend_SE_small(
            sinc_kernel_size=128,
            sample_rate=self.sample_rate
        )

        self.classifier = TfSepNet(depth=18, width=16,
                                   dropout_rate=0.2,
                                   shuffle=True,
                                   shuffle_groups=8)

    def forward(self, x):
        return self.classifier(self.front_end(x))