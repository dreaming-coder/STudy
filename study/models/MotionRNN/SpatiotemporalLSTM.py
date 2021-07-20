from typing import Tuple

import torch
from torch import nn, Tensor

__all__ = ["SpatiotemporalLSTM"]


class SpatiotemporalLSTM(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        r"""
        :param in_channels:         输入通道数
        :param hidden_channels:     隐藏层数，这里也到做输出通道数，因为蚊帐里都是每层的隐藏层数一样
        :param kernel_size:         卷积核尺寸
        :param forget_bias:         偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)
        # 对 x 的卷积
        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 7,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        # 对 h 的卷积
        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        # 对 m 的卷积
        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        # 将对更新后的 c 和 m 的卷积合在一起
        self.conv_o = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        # 最后一行公式的 1x1 卷积
        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, x: Tensor, h: Tensor, c: Tensor, m: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        :param x:   输入的图像，shape 为 (B, in_channels, H, W)
        :param h:   时间方向隐藏状态，shape 为 (B, hidden_channels, H, W)
        :param c:   cell记忆，shape 为 (B, hidden_channels, H, W)
        :param m:   空间方向隐藏状态，shape 为 (B, hidden_channels, H, W)
        :return:    h, c, m
        """
        if x is None and (h is None or c is None or m is None):
            raise ValueError("x 和 [h, c, m] 不能同时为 None")

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, f_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, f_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self.forget_bias)

        c = f * c + i * g

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m)

        m = ff * m + ii * gg

        states = torch.cat([c, m], dim=1)

        o = torch.sigmoid(o_x + o_h + self.conv_o(states))
        h = o * torch.tanh(self.conv1x1(states))

        return h, c, m, o


# if __name__ == '__main__':
#     lstm = SpatiotemporalLSTM(in_channels=1, hidden_channels=64, kernel_size=3).cuda()
#     x = torch.ones(2, 1, 100, 100).cuda()
#     h = torch.ones(2, 64, 100, 100).cuda()
#     c = torch.ones(2, 64, 100, 100).cuda()
#     m = torch.ones(2, 64, 100, 100).cuda()
#
#     result = lstm(x, h, c, m)
#
#     for x in result:
#         print(x.shape)