from typing import Tuple

import torch
from torch import nn, Tensor

__all__ = ["CubicLSTM"]


class ConvLSTM(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        r"""
        :param in_channels:          输入通道数
        :param hidden_channels:      隐藏层通道数
        :param kernel_size:          卷积核尺寸
        :param forget_bias:          偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)  # 大多数都是stride=1，并且是same卷积
        kernel_size = (kernel_size, kernel_size)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels * 4,
                              kernel_size=kernel_size, stride=(1, 1), padding=padding)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        :param inputs:     一个 batch 的 一帧输入
        :return:           c, h
        """
        gates = self.conv(inputs)
        gates = torch.layer_norm(gates, gates.shape[1:])
        ii, ff, oo, cc = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(ii)
        f = torch.sigmoid(ff + self.forget_bias)
        o = torch.sigmoid(oo)
        c = torch.tanh(cc)

        c = f * c + i * c

        h = o * torch.tanh(c)

        return c, h


class CubicLSTM(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size_x: int = 3, kernel_size_y: int = 1,
                 kernel_size_z: int = 5, forget_bias: float = 0.01):
        r"""
        :param in_channels:             输入通道数
        :param hidden_channels:         隐藏层通道数
        :param kernel_size_x:           时间分支
        :param kernel_size_y:           输出分支
        :param kernel_size_z:           空间分支
        :param forget_bias:             偏移量
        """
        super().__init__()
        self.hidden_channels = hidden_channels

        self.branch_x = ConvLSTM(in_channels=in_channels + hidden_channels * 2, hidden_channels=hidden_channels,
                                 kernel_size=kernel_size_x, forget_bias=forget_bias)

        self.branch_z = ConvLSTM(in_channels=in_channels + hidden_channels * 2, hidden_channels=hidden_channels,
                                 kernel_size=kernel_size_z, forget_bias=forget_bias)

        self.branch_y = nn.Conv2d(in_channels=hidden_channels * 2, out_channels=hidden_channels,
                                  kernel_size=(kernel_size_y, kernel_size_y), stride=(1, 1),
                                  padding=(kernel_size_y // 2, kernel_size_y // 2))

    def forward(self, x: Tensor, state_x: Tensor, state_z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        c_x, h_x = torch.split(state_x, self.hidden_channels, dim=1)
        c_z, h_z = torch.split(state_z, self.hidden_channels, dim=1)

        inputs = torch.cat([x, h_x, h_z], dim=1)

        c_x, h_x = self.branch_x(inputs)
        c_z, h_z = self.branch_z(inputs)

        h = torch.cat([h_x, h_z], dim=1)
        y = self.branch_y(h)
        state_x = torch.cat([c_x, h_x], dim=1)

        return y, state_x, state_z


# if __name__ == '__main__':
#     cubic = CubicLSTM(in_channels=8, hidden_channels=32, kernel_size_x=3, kernel_size_y=1, kernel_size_z=5).to("cuda")
#     inputs = torch.ones(2, 8, 128, 128).to("cuda")
#     states_x = torch.ones(2, 64, 128, 128).to("cuda")
#     states_z = torch.ones(2, 64, 128, 128).to("cuda")
#
#     result = cubic(inputs, states_x, states_z)
#     for x in result:
#         print(x.shape)