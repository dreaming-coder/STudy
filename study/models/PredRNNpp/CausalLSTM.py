from typing import Tuple

import torch
from torch import nn, Tensor

__all__ = ["CausalLSTM"]


class CausalLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int,
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:      输入通道数
        :param hidden_channels:  隐藏层通道数
        :param out_channels:     输出通道数
        :param kernel_size:      卷积核尺寸
        :param forget_bias:      偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)
        kernel_size = (kernel_size, kernel_size)

        self.conv_w1 = nn.Conv2d(
            in_channels=in_channels + hidden_channels * 2, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, stride=(1, 1), padding=padding
        )

        self.conv_w2 = nn.Conv2d(
            in_channels=in_channels + hidden_channels * 2, out_channels=out_channels * 3,
            kernel_size=kernel_size, stride=(1, 1), padding=padding
        )

        self.conv_w3 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=out_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.conv_w4 = nn.Conv2d(
            in_channels=in_channels + hidden_channels + out_channels, out_channels=hidden_channels,
            kernel_size=kernel_size, stride=(1, 1), padding=padding
        )

        self.conv_w5 = nn.Conv2d(
            in_channels=hidden_channels + out_channels, out_channels=hidden_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, x: Tensor, h: Tensor, c: Tensor, m: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        :param x:     输入图像
        :param h:     时间信息记忆
        :param c:     cell记忆
        :param m:     空间信息记忆
        :return:      h, c, m
        """
        if x is None and (h is None or c is None or m is None):
            raise ValueError("x 和 [h, c, m] 不能同时为 None")

        w1_concat = torch.cat([x, h, c], dim=1)
        w1_out = self.conv_w1(w1_concat)
        w1_out = torch.layer_norm(w1_out, w1_out.shape[1:])
        g_w1, i_w1, f_w1 = torch.split(w1_out, self.hidden_channels, dim=1)
        g = torch.tanh(g_w1)
        i = torch.sigmoid(i_w1)
        f = torch.sigmoid(f_w1 + self.forget_bias)

        c = f * c + i * g

        w2_concat = torch.cat([x, c, m], dim=1)
        w2_out = self.conv_w2(w2_concat)
        w2_out = torch.layer_norm(w2_out, w2_out.shape[1:])
        gg_w2, ii_w2, ff_w2 = torch.split(w2_out, self.out_channels, dim=1)

        gg = torch.tanh(gg_w2)
        ii = torch.sigmoid(ii_w2)
        ff = torch.sigmoid(ff_w2 + self.forget_bias)

        m = ff * torch.tanh(self.conv_w3(m)) + ii * gg

        w4_concat = torch.cat([x, c, m], dim=1)

        w4_out = self.conv_w4(w4_concat)

        w4_out = torch.layer_norm(w4_out, w4_out.shape[1:])
        o = torch.tanh(w4_out)

        w5_concat = torch.cat([c, m], dim=1)
        w5_out = self.conv_w5(w5_concat)
        w5_out = torch.layer_norm(w5_out, w5_out.shape[1:])
        h = o * torch.tanh(w5_out)

        return h, c, m


# if __name__ == '__main__':
#     lstm = CausalLSTM(in_channels=1, hidden_channels=64, out_channels=32, kernel_size=5).to("cuda")
#     x = torch.ones(2, 1, 100, 100).to("cuda")
#     h = torch.ones(2, 64, 100, 100).to("cuda")
#     c = torch.ones(2, 64, 100, 100).to("cuda")
#     m = torch.ones(2, 64, 100, 100).to("cuda")
#
#     result = lstm(x, h, c, m)
#     for x in result:
#         print(x.shape)
