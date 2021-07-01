from typing import Sequence

from torch import nn, Tensor

__all__ = ["i_RevNet_Block"]


# noinspection PyShadowingNames
class i_RevNet_Block(nn.Module):
    """
    经过 i-RevNet 后，通道数和尺寸相等
    """

    def __init__(self, channels: int):
        super(i_RevNet_Block, self).__init__()

        self.channels = channels

        if channels // 4 == 0:
            ch = 1
        else:
            ch = channels // 4

        self.bottleneck_block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1), bias=False),

            nn.GroupNorm(num_groups=1, num_channels=ch, affine=True),

            nn.ReLU(),

            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1), bias=False),

            nn.GroupNorm(num_groups=1, num_channels=ch, affine=True),

            nn.ReLU(),

            nn.Conv2d(in_channels=ch, out_channels=channels, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1), bias=False),

            nn.GroupNorm(num_groups=1, num_channels=channels, affine=True),

            nn.ReLU()
        )

        self.bottleneck_block2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1), bias=False),

            nn.GroupNorm(num_groups=1, num_channels=ch, affine=True),

            nn.ReLU(),

            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1), bias=False),

            nn.GroupNorm(num_groups=1, num_channels=ch, affine=True),

            nn.ReLU(),

            nn.Conv2d(in_channels=ch, out_channels=channels, kernel_size=(3, 3),
                      padding=(1, 1), stride=(1, 1), bias=False),

            nn.GroupNorm(num_groups=1, num_channels=channels, affine=True),

            nn.ReLU()
        )

    def forward(self, x: Sequence[Tensor]):
        x1 = x[0]  # 前一时刻输入
        x2 = x[1]  # 后一时刻输入

        Fx1 = self.bottleneck_block1(x1)
        x2 = x2 + Fx1
        Fx2 = self.bottleneck_block2(x2)
        x1 = x1 + Fx2
        return x1, x2

    def inverse(self, x: Sequence[Tensor]):
        x1 = x[0]  # 前一时刻输出
        x2 = x[1]  # 后一时刻输出
        Fx2 = self.bottleneck_block2(x2)
        x1 = x1 - Fx2
        Fx1 = self.bottleneck_block1(x1)
        x2 = x2 - Fx1
        return x1, x2
