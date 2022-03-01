from typing import Tuple

import torch
from torch import nn, Tensor

__all__ = ["ResidualConv2d"]


class ResidualConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int] = (3, 3),
                 stride: Tuple[int] = (2, 2), padding: Tuple[int] = (1, 1)):
        super(ResidualConv2d, self).__init__()
        if in_channels <= out_channels:
            self.operate1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            )
        else:
            self.operate1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, output_padding=(1, 1)),
                nn.ReLU()
            )

        self.operate2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x1: Tensor):
        x2 = self.operate1(x1)
        x3 = self.operate2(x2)
        x3 = x2 + x3

        return x3


# if __name__ == '__main__':
#     a = torch.ones(3, 10, 256, 256)
#     net = ResidualConv2d(in_channels=10, out_channels=64)
#     r = net(a)
#     print(r.shape)
#
#     b = torch.ones(5, 512, 4, 4)
#     net2 = ResidualConv2d(in_channels=512, out_channels=256)
#     t = net2(b)
#     print(t.shape)
