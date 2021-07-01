from typing import Tuple, List

import torch
from torch import nn, Tensor

__all__ = ["Eidetic3DLSTMCell"]


# noinspection PyTypeChecker
class Eidetic3DLSTMCell(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, depth: int, kernel_size: Tuple,
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:
        :param hidden_channels:
        :param depth:
        :param kernel_size:
        :param forget_bias:
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.forget_bias = forget_bias

        D, H, W = kernel_size
        padding = [0, H // 2, W // 2]  # 深度方向不padding，stride=1， 通过输入的深度控制输出的深度和隐藏层深度一致

        self.conv3d_x = nn.Conv3d(in_channels=in_channels, out_channels=hidden_channels * 7,
                                  kernel_size=kernel_size, stride=1, padding=padding)

        self.conv3d_h = nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels * 4,
                                  kernel_size=kernel_size, stride=1, padding=padding)

        self.conv3d_m = nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels * 3,
                                  kernel_size=kernel_size, stride=1, padding=padding)

        self.conv3d_oc = nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels,
                                   kernel_size=kernel_size, stride=1, padding=padding)

        self.conv3d_om = nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels,
                                   kernel_size=kernel_size, stride=1, padding=padding)

        self.soft_max = nn.Softmax(dim=2)

        self.conv1x1x1 = nn.Conv3d(in_channels=hidden_channels * 2, out_channels=hidden_channels,
                                   kernel_size=1, stride=1, padding=0)

    def __recall(self, r: Tensor, c_states: Tensor) -> Tensor:
        batch = r.shape[0]
        height = r.shape[-2]
        width = r.shape[-1]
        r = r.reshape(batch, -1, self.hidden_channels)  # [batch, DHW, C]
        c_states = c_states.reshape(batch, -1, self.hidden_channels)  # [batch, tDHW, C]
        temp = r @ c_states.transpose(1, 2)  # [batch, DHW, tDHW], @ 可以用 torch.bmm(这是带batch的矩阵乘法) 替代
        soft_max = self.soft_max(temp)  # [batch, DHW, tDHW]
        recall = soft_max @ c_states  # [batch, DHW, C]

        return recall.reshape(batch, self.hidden_channels, -1, height, width)

    @staticmethod
    def __layer_norm(input: Tensor) -> Tensor:
        shape = input.shape[1:]
        normalized = torch.layer_norm(input, shape)
        return normalized

    def forward(self, x: Tensor, h: Tensor, c_states: List[Tensor], m: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 3D 卷积输入的 shape 为 [batch, channel, depth, height, width]
        assert x is not None and h is not None and c_states is not None and m is not None, "输入不能为空"

        c_states = torch.cat(c_states, dim=2)
        c_: Tensor = c_states[:, :, -self.depth:]

        x_concat = self.conv3d_x(x)
        h_concat = self.conv3d_h(h)
        m_concat = self.conv3d_m(m)

        x_concat = self.__layer_norm(x_concat)
        h_concat = self.__layer_norm(h_concat)
        m_concat = self.__layer_norm(m_concat)

        r_x, i_x, g_x, ii_x, gg_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        r_h, i_h, g_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        ii_m, gg_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        r = torch.sigmoid(r_x + r_h)
        i = torch.sigmoid(i_x + i_h)
        g = torch.tanh(g_x + g_h)

        recall = self.__recall(r, c_states)

        c = i * g + self.__layer_norm(c_ + recall)

        ii = torch.sigmoid(ii_x + ii_m)
        gg = torch.tanh(gg_x + gg_m)
        ff = torch.sigmoid(ff_x + ff_m + self.forget_bias)
        m = ii * gg + ff * m

        o = torch.sigmoid(o_x + o_h + self.conv3d_oc(c) + self.conv3d_om(m))

        cm_concat = torch.cat([c, m], dim=1)  # 在通道维度上 concatenate
        h = o * torch.tanh(self.conv1x1x1(cm_concat))

        return c, h, m


# if __name__ == '__main__':
#     lstm = Eidetic3DLSTMCell(in_channels=1, hidden_channels=8, depth=2, kernel_size=(2, 5, 5)).to("cuda")
#     x = torch.zeros(1, 1, 2, 25, 25).to("cuda")  # 3D 卷积输入的 shape 为 [batch, channel, depth, height, width]
#     h = torch.ones(1, 8, 2, 25, 25).to("cuda")
#     c = [torch.ones(1, 8, 2, 25, 25).to("cuda")] * 19
#     m = torch.ones(1, 8, 2, 25, 25).to("cuda")
#
#     b = lstm(x, h, c, m)
#     for x in b:
#         print(x.shape)