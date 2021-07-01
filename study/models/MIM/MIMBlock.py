import torch
from torch import nn, Tensor
from typing import Tuple, Union

__all__ = ["MIMBlock"]


class MIMBlock(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        r"""
        :param in_channels:           输入通道数
        :param hidden_channels:       隐藏层通道数
        :param kernel_size:           卷积核尺寸
        :param forget_bias:           偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        self.mim_n = MIMN(in_channels=in_channels, hidden_channels=hidden_channels, kernel_size=kernel_size,
                          forget_bias=forget_bias)

        self.mim_s = MIMS(hidden_channels=hidden_channels, kernel_size=kernel_size, forget_bias=forget_bias)

        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 6,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_o_c = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_o_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )

    def forward(self, x_: Tensor, x: Tensor, c: Tensor, h: Tensor,
                m: Tensor, n: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        :param x_:   h_{t-1}^{l-1}
        :param x:    h_{t-1}^l
        :param c:    cell记忆信息
        :param h:    时间方向记忆Tensor
        :param m:    空间方向记忆Tensor
        :param n:    MIM-N 记忆Tensor
        :param s:    MIM-S 记忆Tensor
        :return:     更新后的 c, h, m, n, s
        """
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h)

        h_diff = x - x_
        n, d = self.mim_n(h_diff, n)
        s, t = self.mim_s(d, c, s)

        c = t + i * g

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m + self.forget_bias)

        m = ff * m + ii * gg

        o = torch.sigmoid(o_x + o_h + self.conv_o_c(c) + self.conv_o_m(m))

        states = torch.cat([c, m], dim=1)

        h = o * torch.tanh(self.conv1x1(states))

        return c, h, m, n, s


class MIMN(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: Union[int, tuple],
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:              输入通道数
        :param hidden_channels:          隐藏层通道数
        :param kernel_size:              卷积核尺寸
        :param forget_bias:              偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)

        self.conv_h_diff = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_n = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_w_no = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

    def forward(self, h_diff: Tensor, n: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        :param h_diff:     输入的隐藏层的差分值
        :param n:          状态Tensor
        :return:           n, d
        """
        h_diff_concat = self.conv_h_diff(h_diff)
        h_diff_concat = torch.layer_norm(h_diff_concat, h_diff_concat.shape[1:])
        n_concat = self.conv_n(n)
        n_concat = torch.layer_norm(n_concat, n_concat.shape[1:])

        g_h, i_h, f_h, o_h = torch.split(h_diff_concat, self.hidden_channels, dim=1)
        g_n, i_n, f_n = torch.split(n_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_h + g_n)
        i = torch.sigmoid(i_h + i_n)
        f = torch.sigmoid(f_h + f_n + self.forget_bias)

        n = f * n + i * g

        o_n = self.conv_w_no(n)
        o_n = torch.layer_norm(o_n, o_n.shape[1:])
        o = torch.sigmoid(o_h + o_n)
        d = o * torch.tanh(n)

        return n, d


class MIMS(nn.Module):

    def __init__(self, hidden_channels: int, kernel_size: Union[int, tuple], forget_bias: float = 0.01):
        r"""
        :param hidden_channels:     通道数
        :param kernel_size:         卷积核尺寸
        :param forget_bias:         偏移量
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)

        self.conv_d = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )
        self.conv_c = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_w_so = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

    def forward(self, d: Tensor, c: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        :param d:     差分信息
        :param c:     状态记忆Tensor
        :param s:     MIMS记忆Tensor
        :return:      s, t
        """
        d_concat = self.conv_d(d)
        c_concat = self.conv_c(c)

        d_concat = torch.layer_norm(d_concat, d_concat.shape[1:])
        c_concat = torch.layer_norm(c_concat, c_concat.shape[1:])

        g_d, i_d, f_d, o_d = torch.split(d_concat, self.hidden_channels, dim=1)
        g_c, i_c, f_c, o_c = torch.split(c_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_d + g_c)
        i = torch.sigmoid(i_d + i_c)
        f = torch.sigmoid(f_d + f_c + self.forget_bias)

        s = f * s + i * g

        o_s = self.conv_w_so(s)
        o_s = torch.layer_norm(o_s, o_s.shape[1:])

        o = torch.sigmoid(o_d + o_c + o_s)

        t = o * torch.tanh(s)

        return s, t


# if __name__ == '__main__':
#     mimn = MIMBlock(in_channels=64, hidden_channels=64, kernel_size=3).to("cuda")
#     x_ = torch.ones(2, 64, 100, 100).to("cuda")
#     x = torch.ones(2, 64, 100, 100).to("cuda")
#     c = torch.ones(2, 64, 100, 100).to("cuda")
#     h = torch.ones(2, 64, 100, 100).to("cuda")
#     m = torch.ones(2, 64, 100, 100).to("cuda")
#     n = torch.ones(2, 64, 100, 100).to("cuda")
#     s = torch.ones(2, 64, 100, 100).to("cuda")
#
#     result = mimn(x_, x, c, h, m, n, s)
#     for ret in result:
#         print(ret.shape)
