from typing import Tuple

import torch
from torch import nn, Tensor

__all__ = ["ConvLSTMCell"]


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, size: Tuple[int, int],
                 kernel_size: int = 3, forget_bias: float = 0.01):
        """
        :param in_channels:       输入的通道数
        :param hidden_channels:   隐藏层通道数
        :param size:              输入的二维尺寸，即 (Height, Width)
        :param kernel_size:       卷积核尺寸
        :param forget_bias:       偏置
        """
        super(ConvLSTMCell, self).__init__()

        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        padding = tuple([f // 2 for f in kernel_size])

        self.conv_x = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels * 4,
                                kernel_size=kernel_size, padding=padding, stride=(1, 1))

        self.conv_h = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 4,
                                kernel_size=kernel_size, padding=padding, stride=(1, 1))

        self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, *size), requires_grad=True)
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, *size), requires_grad=True)
        self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, *size), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x:      x 是输入的一个 batch 的某一时序，shape应该是 (B, in_channels, H, W)
        :param h:      h 是隐藏层，shape应该是 (B, hidden_channels, H, W)
        :param c:      c 是 cell 记忆的载体，shape应该是 (B, hidden_channels, H, W)
        :return:       更新过的 h 和 c
        """
        if x is None and (h is None or c is None):
            raise ValueError("x 和 [h, c] 不能同时为 None")

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)

        i = torch.sigmoid(i_x + i_h + self.w_ci * c)
        f = torch.sigmoid(f_x + f_h + self.forget_bias)
        c = f * c + i * torch.tanh(c_x + c_h)
        o = torch.sigmoid(o_x + o_h + self.w_co * c)
        h = o * torch.tanh(c)

        return h, c


# if __name__ == '__main__':
#     device = "cpu"
#     cell = ConvLSTMCell(in_channels=64, hidden_channels=96, size=(50, 50)).to(device)
#     x = torch.ones(3, 64, 50, 50).to(device)
#     h = torch.zeros(3, 96, 50, 50).to(device)
#     c = torch.zeros(3, 96, 50, 50).to(device)
#     hh, cc = cell(x, h, c)
#     print(hh.shape)
#     print(cc.shape)
#     hh.sum().backward()
